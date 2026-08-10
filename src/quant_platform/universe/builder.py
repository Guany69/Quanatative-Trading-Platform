"""Point-in-time universe construction (spec sections 4.1 and 11).

The universe on date ``t`` must contain exactly the securities that a manager standing on
date ``t`` could have known were eligible. Three separate mistakes break this, and each is
guarded here:

1. **Survivorship bias** -- building history from *today's* index members. Companies that
   went bankrupt are missing, so the backtest only ever trades winners. Guarded by requiring
   real membership intervals, and by refusing to fall back to a current snapshot unless the
   user explicitly opts in (which then marks every result as biased).
2. **Look-ahead on membership** -- using the effective date of an index change instead of its
   announcement. Guarded via ``available_at`` on membership records.
3. **Look-ahead on eligibility** -- filtering on adjusted prices or on a full-sample liquidity
   average. Guarded by using unadjusted prices and strictly trailing windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import polars as pl

from quant_platform.config.models import UniverseConfig
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("universe")


@dataclass
class UniverseSnapshot:
    """The eligible securities on one date, with the reason each name was excluded."""

    as_of: date
    security_ids: list[str]
    exclusion_counts: dict[str, int] = field(default_factory=dict)
    is_survivorship_biased: bool = False

    def __len__(self) -> int:
        return len(self.security_ids)

    def __contains__(self, security_id: str) -> bool:
        return security_id in self.security_ids


class UniverseBuilder:
    """Resolves the eligible universe for any date from point-in-time inputs."""

    def __init__(
        self,
        config: UniverseConfig,
        securities: pl.DataFrame,
        membership: pl.DataFrame,
        prices: pl.DataFrame,
    ) -> None:
        self.config = config
        self.securities = securities
        self.membership = membership
        self.prices = prices
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Refuse to build on a survivorship-biased base unless explicitly permitted."""
        if "is_survivorship_biased" in self.membership.columns:
            biased = bool(self.membership["is_survivorship_biased"].any())
            if biased and not self.config.allow_survivorship_biased_fallback:
                raise ValueError(
                    "membership data is flagged survivorship-biased (reconstructed from a "
                    "current snapshot rather than true historical constituents). Refusing to "
                    "build a universe that would silently present it as real history. Set "
                    "universe.allow_survivorship_biased_fallback=true to proceed with results "
                    "explicitly marked as biased."
                )
            self._is_biased = biased
        else:
            self._is_biased = False

    @property
    def is_survivorship_biased(self) -> bool:
        return self._is_biased

    def members_on(self, as_of: date) -> set[str]:
        """Index members on ``as_of``, gated by announcement knowledge.

        Two filters, not one: the membership interval must *contain* ``as_of`` (so future
        members are absent and removed members are gone), and the change must have been
        *announced* by ``as_of`` (so a pending addition is not tradeable early).
        """
        as_of_dt = datetime.combine(as_of, datetime.max.time())
        df = self.membership.filter(
            (pl.col("universe") == self.config.name)
            & (pl.col("start_date") <= as_of)
            & (pl.col("end_date") > as_of)  # half-open: removal effective date excluded
        )
        if "available_at" in df.columns:
            df = df.filter(pl.col("available_at") <= as_of_dt)
        return set(df["security_id"].to_list())

    def _eligible_by_type_and_listing(self, as_of: date) -> tuple[set[str], dict[str, int]]:
        """Filter on security type, sector exclusions, and listing status."""
        counts: dict[str, int] = {}
        df = self.securities

        before = df.height
        df = df.filter(pl.col("security_type").is_in(self.config.allowed_security_types))
        counts["not_allowed_security_type"] = before - df.height

        if self.config.exclude_sectors:
            before = df.height
            df = df.filter(~pl.col("sector").is_in(self.config.exclude_sectors))
            counts["excluded_sector"] = before - df.height

        # Not yet listed on as_of.
        before = df.height
        df = df.filter(pl.col("first_trade_date").is_null() | (pl.col("first_trade_date") <= as_of))
        counts["not_yet_listed"] = before - df.height

        # Already delisted. Note this removes the name only *going forward* from its
        # delisting; its historical bars remain and are still traded in the backtest.
        before = df.height
        df = df.filter(pl.col("delisting_date").is_null() | (pl.col("delisting_date") >= as_of))
        counts["already_delisted"] = before - df.height

        return set(df["security_id"].to_list()), counts

    def _liquidity_and_price_screen(
        self, as_of: date, candidates: set[str]
    ) -> tuple[set[str], dict[str, int]]:
        """Apply price, liquidity, and history screens using only trailing data.

        The price screen uses the **unadjusted** close deliberately: the $5 rule is about the
        price that actually traded. Applying it to a split-adjusted series would retroactively
        disqualify securities that were never actually penny stocks (or admit ones that were).
        """
        counts: dict[str, int] = {}
        lookback = self.config.dollar_volume_lookback_sessions

        window = self.prices.filter(
            (pl.col("observation_date") <= as_of) & pl.col("security_id").is_in(list(candidates))
        )
        if window.is_empty():
            return set(), {"no_price_history": len(candidates)}

        # Trailing stats per security, strictly on or before as_of.
        stats = (
            window.sort("observation_date")
            .group_by("security_id")
            .agg(
                [
                    pl.col("close").last().alias("last_close"),
                    pl.col("observation_date").last().alias("last_date"),
                    pl.col("dollar_volume").tail(lookback).mean().alias("adv"),
                    pl.len().alias("n_history"),
                ]
            )
        )

        before = stats.height
        stats = stats.filter(pl.col("n_history") >= self.config.min_history_sessions)
        counts["insufficient_history"] = before - stats.height

        before = stats.height
        stats = stats.filter(pl.col("last_close") >= self.config.min_price)
        counts["below_min_price"] = before - stats.height

        before = stats.height
        stats = stats.filter(pl.col("adv") >= self.config.min_dollar_volume)
        counts["below_min_dollar_volume"] = before - stats.height

        # A security whose last bar is far behind as_of is stale/suspended, not tradeable.
        before = stats.height
        stats = stats.filter((pl.lit(as_of) - pl.col("last_date")).dt.total_days() <= 10)
        counts["stale_price"] = before - stats.height

        eligible = set(stats["security_id"].to_list())

        if self.config.max_securities is not None and len(eligible) > self.config.max_securities:
            # Keep the most liquid names, a rule knowable at as_of.
            top = (
                stats.sort("adv", descending=True)
                .head(self.config.max_securities)["security_id"]
                .to_list()
            )
            counts["over_max_securities"] = len(eligible) - len(top)
            eligible = set(top)

        return eligible, counts

    def build(self, as_of: date) -> UniverseSnapshot:
        """Resolve the eligible universe on ``as_of``."""
        members = self.members_on(as_of)
        if not members:
            return UniverseSnapshot(
                as_of=as_of,
                security_ids=[],
                exclusion_counts={"no_members": 0},
                is_survivorship_biased=self._is_biased,
            )

        typed, type_counts = self._eligible_by_type_and_listing(as_of)
        candidates = members & typed

        eligible, liq_counts = self._liquidity_and_price_screen(as_of, candidates)

        counts = {**type_counts, **liq_counts}
        counts = {k: v for k, v in counts.items() if v}

        return UniverseSnapshot(
            as_of=as_of,
            security_ids=sorted(eligible),
            exclusion_counts=counts,
            is_survivorship_biased=self._is_biased,
        )

    def build_panel(self, dates: list[date]) -> pl.DataFrame:
        """Build snapshots for many dates as a long frame (security_id, as_of)."""
        rows: list[dict[str, object]] = []
        for d in dates:
            snap = self.build(d)
            rows.extend({"as_of": d, "security_id": sid} for sid in snap.security_ids)
        if not rows:
            return pl.DataFrame(
                {"as_of": [], "security_id": []},
                schema={"as_of": pl.Date, "security_id": pl.Utf8},
            )
        return pl.DataFrame(rows)
