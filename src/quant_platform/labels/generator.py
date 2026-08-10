"""Leakage-safe label generation (spec section 13).

The label is the security's forward return *relative to the benchmark* over a fixed number
of trading sessions. Four details separate a correct implementation from a subtly broken one:

1. **Sessions, not calendar days.** "20 days ahead" drifts with weekends and holidays;
   "20 sessions ahead" does not.
2. **Execution delay.** The strategy cannot trade at the close that produced the signal, so
   the label measures the return actually *capturable*: from the delayed entry price forward,
   not from the signal price.
3. **Delistings.** A security that dies inside the window has no future price. Dropping it
   silently deletes exactly the worst outcomes -- so the delisting return is booked instead.
4. **Window bookkeeping.** ``window_start``/``window_end`` are stored so the walk-forward
   splitter can purge training samples whose windows reach into the test period.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_platform.config.models import LabelConfig
from quant_platform.domain.enums import TargetType
from quant_platform.utilities.calendar import TradingCalendar
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("labels")


class LabelGenerator:
    """Builds forward benchmark-relative labels from a price panel.

    All target types (spec section 3) come from this one interface so they cannot drift out
    of sync with each other.
    """

    def __init__(
        self,
        config: LabelConfig,
        calendar: TradingCalendar,
        execution_delay_sessions: int = 1,
    ) -> None:
        self.config = config
        self.calendar = calendar
        if execution_delay_sessions < 1:
            raise ValueError(
                f"execution_delay_sessions must be >= 1, got {execution_delay_sessions}: a "
                f"label measured from the signal close assumes an impossible fill"
            )
        self.execution_delay = execution_delay_sessions

    def generate(
        self,
        prices: pl.DataFrame,
        benchmark: pl.DataFrame,
        securities: pl.DataFrame,
        signal_dates: list[date] | None = None,
    ) -> pl.DataFrame:
        """Generate labels for every (security, signal date) pair.

        Returns a frame with the label value, its forward window bounds, and the components
        (raw and benchmark return) so results can be audited rather than taken on faith.
        """
        horizon = self.config.forecast_horizon_sessions
        delay = self.execution_delay

        px = prices.select(["security_id", "observation_date", "adjusted_close"]).sort(
            ["security_id", "observation_date"]
        )

        # Reindex onto the trading calendar so "n sessions ahead" is exact even where a
        # security has missing bars. Forward-filling the price is deliberate: a stale mark is
        # the honest treatment of a non-trading day, whereas dropping the row would silently
        # shorten the horizon.
        sessions = pl.DataFrame({"observation_date": self.calendar.sessions})
        sec_ids = px["security_id"].unique().sort()
        grid = sessions.join(pl.DataFrame({"security_id": sec_ids}), how="cross")
        panel = (
            grid.join(px, on=["security_id", "observation_date"], how="left")
            .sort(["security_id", "observation_date"])
            .with_columns(
                pl.col("adjusted_close").forward_fill().over("security_id").alias("px_ffill")
            )
        )

        # Entry price is the close `delay` sessions after the signal; exit is `horizon`
        # sessions after entry. Both are shifts *backward* in the frame (i.e. looking ahead
        # in time) applied per security.
        panel = panel.with_columns(
            [
                pl.col("px_ffill").shift(-delay).over("security_id").alias("entry_px"),
                pl.col("px_ffill").shift(-(delay + horizon)).over("security_id").alias("exit_px"),
                pl.col("observation_date").shift(-delay).over("security_id").alias("window_start"),
                pl.col("observation_date")
                .shift(-(delay + horizon))
                .over("security_id")
                .alias("window_end"),
            ]
        )

        bench = benchmark.select(
            ["observation_date", pl.col("adjusted_close").alias("bench_px")]
        ).sort("observation_date")
        bench = bench.with_columns(
            [
                pl.col("bench_px").shift(-delay).alias("bench_entry"),
                pl.col("bench_px").shift(-(delay + horizon)).alias("bench_exit"),
            ]
        )
        panel = panel.join(
            bench.select(["observation_date", "bench_entry", "bench_exit"]),
            on="observation_date",
            how="left",
        )

        panel = panel.with_columns(
            [
                (pl.col("exit_px") / pl.col("entry_px") - 1.0).alias("raw_return"),
                (pl.col("bench_exit") / pl.col("bench_entry") - 1.0).alias("benchmark_return"),
            ]
        )

        panel = self._apply_delisting_returns(panel, securities, horizon, delay)

        panel = panel.with_columns(
            (pl.col("raw_return") - pl.col("benchmark_return")).alias("excess_return")
        )

        # A label needs a real forward window; without one it must be dropped, never
        # back-filled or clamped (either would fabricate a return).
        panel = panel.filter(
            pl.col("raw_return").is_not_null()
            & pl.col("benchmark_return").is_not_null()
            & pl.col("window_start").is_not_null()
            & pl.col("window_end").is_not_null()
        )

        if signal_dates is not None:
            panel = panel.filter(pl.col("observation_date").is_in(signal_dates))

        # Cross-sectional rank: the primary portfolio signal. Percentile within each date, so
        # it is scale-free and comparable across regimes.
        panel = panel.with_columns(
            (
                pl.col("excess_return").rank("average").over("observation_date")
                / pl.len().over("observation_date")
            ).alias("excess_return_rank")
        )
        panel = panel.with_columns(
            (pl.col("excess_return") > 0).cast(pl.Float64).alias("outperform_probability")
        )

        return panel.select(
            [
                "security_id",
                pl.col("observation_date").alias("as_of"),
                "window_start",
                "window_end",
                "raw_return",
                "benchmark_return",
                "excess_return",
                "excess_return_rank",
                "outperform_probability",
                "is_delisted_in_window",
            ]
        ).sort(["as_of", "security_id"])

    def _apply_delisting_returns(
        self, panel: pl.DataFrame, securities: pl.DataFrame, horizon: int, delay: int
    ) -> pl.DataFrame:
        """Book the delisting return for securities that die inside the label window.

        Without this the panel simply has a null exit price for dead names and they get
        filtered out -- which is delisting bias: the bankruptcies vanish and only survivors
        train the model.
        """
        delist = securities.filter(pl.col("delisting_date").is_not_null()).select(
            ["security_id", "delisting_date", "delisting_return"]
        )
        if delist.is_empty():
            return panel.with_columns(pl.lit(False).alias("is_delisted_in_window"))

        panel = panel.join(delist, on="security_id", how="left")

        # The window is [window_start, window_end]; but for dead names window_end is null
        # (no price that far out), so compare against the *intended* end date derived from
        # the calendar instead.
        intended_end = {d: self.calendar.shift(d, delay + horizon) for d in self.calendar.sessions}
        panel = panel.with_columns(
            pl.col("observation_date")
            .replace_strict(intended_end, default=None)
            .alias("_intended_end")
        )

        dies_in_window = (
            pl.col("delisting_date").is_not_null()
            & (pl.col("delisting_date") >= pl.col("observation_date"))
            & (
                pl.col("_intended_end").is_null()
                | (pl.col("delisting_date") <= pl.col("_intended_end"))
            )
        )

        panel = panel.with_columns(
            [
                dies_in_window.alias("is_delisted_in_window"),
                # Where the security dies in the window, the realized return IS the delisting
                # return (e.g. -1.0 for a wipeout), overriding the missing price path.
                pl.when(dies_in_window & pl.col("delisting_return").is_not_null())
                .then(pl.col("delisting_return"))
                .otherwise(pl.col("raw_return"))
                .alias("raw_return"),
                # Give the label a real window end so it can still be purged correctly.
                pl.when(dies_in_window & pl.col("window_end").is_null())
                .then(pl.col("delisting_date"))
                .otherwise(pl.col("window_end"))
                .alias("window_end"),
                pl.when(dies_in_window & pl.col("window_start").is_null())
                .then(pl.col("observation_date"))
                .otherwise(pl.col("window_start"))
                .alias("window_start"),
            ]
        )
        return panel.drop(["delisting_date", "delisting_return", "_intended_end"])

    def target_column(self) -> str:
        """The column matching the configured target type."""
        mapping = {
            TargetType.FORWARD_RETURN: "raw_return",
            TargetType.EXCESS_RETURN: "excess_return",
            TargetType.EXCESS_RETURN_RANK: "excess_return_rank",
            TargetType.OUTPERFORM_PROBABILITY: "outperform_probability",
        }
        return mapping[self.config.target_type]
