"""Data validation (spec section 10.5).

Every check returns ``DataQualityIssue`` records rather than raising, so one run surfaces
*all* problems instead of stopping at the first. CRITICAL issues then fail the pipeline via
``raise_if_critical`` -- the distinction matters because a duplicate row is fatal (it silently
double-weights a security) while an occasional stale price is a warning worth seeing but not
worth aborting a research run.

The most valuable checks here are the availability ones. A record whose ``available_at``
precedes its ``observation_date``, or sits in the future, is a look-ahead bug that produces
beautiful backtests and worthless conclusions.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import UTC, date, datetime

import polars as pl

from quant_platform.domain.enums import IssueSeverity
from quant_platform.domain.portfolio import DataQualityIssue
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("data.validation")

MAX_PLAUSIBLE_DAILY_RETURN = 1.0  # +100% in one session: possible, but worth flagging
STALE_PRICE_SESSIONS = 10


def _issue(
    check: str,
    severity: IssueSeverity,
    message: str,
    dataset: str,
    n: int = 1,
    **context: object,
) -> DataQualityIssue:
    return DataQualityIssue(
        check_name=check,
        severity=severity,
        message=message,
        dataset=dataset,
        n_affected=n,
        detected_at=datetime.now(UTC),
        context=dict(context),
    )


# --------------------------------------------------------------------------- price checks
def check_duplicate_records(
    frame: pl.DataFrame, keys: list[str], dataset: str
) -> list[DataQualityIssue]:
    """Duplicate keys silently double-count a security in every downstream aggregation."""
    if frame.is_empty() or not set(keys).issubset(frame.columns):
        return []
    dupes = frame.group_by(keys).agg(pl.len().alias("_n")).filter(pl.col("_n") > 1)
    if dupes.is_empty():
        return []
    return [
        _issue(
            "duplicate_records",
            IssueSeverity.CRITICAL,
            f"{dupes.height} duplicated {keys} combinations in {dataset}; each would be "
            f"counted more than once downstream",
            dataset,
            n=dupes.height,
            examples=dupes.head(5).to_dicts(),
        )
    ]


def check_invalid_prices(frame: pl.DataFrame, dataset: str = "prices") -> list[DataQualityIssue]:
    """Non-positive prices and negative volume are impossible."""
    issues: list[DataQualityIssue] = []
    if frame.is_empty():
        return issues

    for col in ("close", "adjusted_close", "open", "high", "low"):
        if col not in frame.columns:
            continue
        bad = frame.filter(pl.col(col).is_not_null() & (pl.col(col) <= 0))
        if not bad.is_empty():
            issues.append(
                _issue(
                    "invalid_price",
                    IssueSeverity.CRITICAL,
                    f"{bad.height} rows with non-positive {col}",
                    dataset,
                    n=bad.height,
                    column=col,
                )
            )

    if "volume" in frame.columns:
        bad = frame.filter(pl.col("volume") < 0)
        if not bad.is_empty():
            issues.append(
                _issue(
                    "invalid_volume",
                    IssueSeverity.CRITICAL,
                    f"{bad.height} rows with negative volume",
                    dataset,
                    n=bad.height,
                )
            )

    # OHLC coherence: high must bound low.
    if {"high", "low"}.issubset(frame.columns):
        bad = frame.filter(
            pl.col("high").is_not_null()
            & pl.col("low").is_not_null()
            & (pl.col("high") < pl.col("low"))
        )
        if not bad.is_empty():
            issues.append(
                _issue(
                    "ohlc_incoherent",
                    IssueSeverity.CRITICAL,
                    f"{bad.height} rows where high < low",
                    dataset,
                    n=bad.height,
                )
            )
    return issues


def check_extreme_returns(
    frame: pl.DataFrame, threshold: float = MAX_PLAUSIBLE_DAILY_RETURN, dataset: str = "prices"
) -> list[DataQualityIssue]:
    """Flag implausible one-session moves.

    These are usually unadjusted splits rather than real returns: a 2-for-1 split that was not
    adjusted looks exactly like a -50% day and will poison momentum and volatility features.
    """
    if frame.is_empty() or "adjusted_close" not in frame.columns:
        return []
    ret = (
        frame.sort(["security_id", "observation_date"])
        .with_columns(
            (pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1.0)
            .over("security_id")
            .alias("_ret")
        )
        .filter(pl.col("_ret").is_not_null() & (pl.col("_ret").abs() > threshold))
    )
    if ret.is_empty():
        return []
    return [
        _issue(
            "extreme_return",
            IssueSeverity.WARNING,
            f"{ret.height} single-session returns exceed +/-{threshold:.0%}. These are often "
            f"unadjusted corporate actions rather than genuine moves.",
            dataset,
            n=ret.height,
            examples=ret.select(["security_id", "observation_date", "_ret"]).head(5).to_dicts(),
        )
    ]


def check_adjustment_monotonicity(
    frame: pl.DataFrame, dataset: str = "prices"
) -> list[DataQualityIssue]:
    """Cumulative adjustment factors must not increase going forward in time.

    Backward adjustment restates history downward as splits accumulate, so the factor series
    is non-increasing. An increase means the adjustment series is corrupted, which silently
    distorts every return computed from it.
    """
    if frame.is_empty() or "cumulative_adjustment_factor" not in frame.columns:
        return []
    bad = (
        frame.sort(["security_id", "observation_date"])
        .with_columns(
            (
                pl.col("cumulative_adjustment_factor")
                - pl.col("cumulative_adjustment_factor").shift(1)
            )
            .over("security_id")
            .alias("_delta")
        )
        .filter(pl.col("_delta") > 1e-9)
    )
    if bad.is_empty():
        return []
    return [
        _issue(
            "non_monotonic_adjustment",
            IssueSeverity.CRITICAL,
            f"{bad.height} rows where the cumulative adjustment factor increases over time; "
            f"the adjustment series is corrupted",
            dataset,
            n=bad.height,
        )
    ]


def check_stale_observations(
    frame: pl.DataFrame, calendar_sessions: list[date], dataset: str = "prices"
) -> list[DataQualityIssue]:
    """Detect securities whose prices stop updating while still expected to trade."""
    if frame.is_empty() or not calendar_sessions:
        return []
    last_session = max(calendar_sessions)
    per_security = frame.group_by("security_id").agg(
        pl.col("observation_date").max().alias("last_date")
    )
    cutoff_idx = max(0, len(calendar_sessions) - STALE_PRICE_SESSIONS - 1)
    cutoff = calendar_sessions[cutoff_idx]
    stale = per_security.filter(pl.col("last_date") < cutoff)
    if stale.is_empty():
        return []
    return [
        _issue(
            "stale_observations",
            IssueSeverity.WARNING,
            f"{stale.height} securities have no prices in the last {STALE_PRICE_SESSIONS} "
            f"sessions before {last_session} (delisted, suspended, or a feed gap)",
            dataset,
            n=stale.height,
            examples=stale.head(5).to_dicts(),
        )
    ]


# --------------------------------------------------------------------------- PIT checks
def check_availability_dates(
    frame: pl.DataFrame, dataset: str, now: datetime | None = None
) -> list[DataQualityIssue]:
    """The core point-in-time integrity check.

    Two impossibilities:
      1. ``available_at < observation_date`` -- the data was "known" before the period it
         describes existed. This is look-ahead bias in its purest form.
      2. ``available_at`` in the future -- the record claims to become knowable later, which
         means it should not be usable at all yet.
    """
    issues: list[DataQualityIssue] = []
    if frame.is_empty() or "available_at" not in frame.columns:
        return issues
    if "observation_date" not in frame.columns:
        return issues

    impossible = frame.filter(pl.col("available_at").cast(pl.Date) < pl.col("observation_date"))
    if not impossible.is_empty():
        issues.append(
            _issue(
                "impossible_availability",
                IssueSeverity.CRITICAL,
                f"{impossible.height} rows in {dataset} have available_at BEFORE "
                f"observation_date: the data claims to have been knowable before the period it "
                f"describes. This is look-ahead bias.",
                dataset,
                n=impossible.height,
                examples=impossible.select(
                    [
                        c
                        for c in ("security_id", "series_id", "observation_date", "available_at")
                        if c in impossible.columns
                    ]
                )
                .head(5)
                .to_dicts(),
            )
        )

    reference = now or datetime.now()
    future = frame.filter(pl.col("available_at") > pl.lit(reference))
    if not future.is_empty():
        issues.append(
            _issue(
                "future_dated_information",
                IssueSeverity.WARNING,
                f"{future.height} rows in {dataset} have available_at in the future relative "
                f"to {reference.date()}; they must not be used yet",
                dataset,
                n=future.height,
            )
        )
    return issues


def check_missing_identifiers(frame: pl.DataFrame, dataset: str) -> list[DataQualityIssue]:
    """Records without a permanent identifier cannot be joined safely."""
    key = "security_id" if "security_id" in frame.columns else None
    if key is None or frame.is_empty():
        return []
    missing = frame.filter(pl.col(key).is_null() | (pl.col(key).cast(pl.Utf8).str.len_chars() == 0))
    if missing.is_empty():
        return []
    return [
        _issue(
            "missing_identifier",
            IssueSeverity.CRITICAL,
            f"{missing.height} rows in {dataset} have no {key}",
            dataset,
            n=missing.height,
        )
    ]


def check_benchmark_gaps(
    benchmark: pl.DataFrame, calendar_sessions: list[date]
) -> list[DataQualityIssue]:
    """Missing benchmark sessions break excess-return alignment.

    Every label is a difference against the benchmark, so a gap does not merely lose a data
    point -- it silently misaligns the comparison for every security on that date.
    """
    if benchmark.is_empty() or not calendar_sessions:
        return []
    have = set(benchmark["observation_date"].to_list())
    expected = set(calendar_sessions)
    missing = sorted(expected - have)
    if not missing:
        return []
    severity = (
        IssueSeverity.CRITICAL if len(missing) > len(expected) * 0.01 else IssueSeverity.WARNING
    )
    return [
        _issue(
            "benchmark_gaps",
            severity,
            f"benchmark is missing {len(missing)} of {len(expected)} trading sessions; "
            f"excess-return labels cannot be aligned on those dates",
            "benchmark",
            n=len(missing),
            first_missing=str(missing[0]),
            last_missing=str(missing[-1]),
        )
    ]


def check_membership_overlaps(membership: pl.DataFrame) -> list[DataQualityIssue]:
    """A security must not hold two overlapping membership intervals in one universe."""
    if membership.is_empty():
        return []
    issues: list[DataQualityIssue] = []
    overlapping = 0
    for (sec_id, universe), group in membership.group_by(["security_id", "universe"]):
        if group.height < 2:
            continue
        rows = group.sort("start_date").select(["start_date", "end_date"]).to_dicts()
        for prev, nxt in itertools.pairwise(rows):
            # Half-open intervals: an overlap needs next.start STRICTLY before prev.end.
            if nxt["start_date"] < prev["end_date"]:
                overlapping += 1
                issues.append(
                    _issue(
                        "membership_overlap",
                        IssueSeverity.CRITICAL,
                        f"{sec_id} has overlapping {universe} membership intervals "
                        f"[{prev['start_date']}..{prev['end_date']}) and "
                        f"[{nxt['start_date']}..{nxt['end_date']}); it would be counted twice",
                        "membership",
                        security_id=str(sec_id),
                    )
                )
    if overlapping:
        logger.warning("found %d overlapping membership intervals", overlapping)
    return issues


def check_corporate_actions(actions: pl.DataFrame) -> list[DataQualityIssue]:
    """Splits need positive ratios; renames need a target symbol."""
    if actions.is_empty():
        return []
    issues: list[DataQualityIssue] = []

    needs_value = ["stock_split", "reverse_split", "cash_dividend", "special_dividend"]
    if "value" in actions.columns:
        bad = actions.filter(
            pl.col("action_type").is_in(needs_value)
            & (pl.col("value").is_null() | (pl.col("value") <= 0))
        )
        if not bad.is_empty():
            issues.append(
                _issue(
                    "invalid_corporate_action",
                    IssueSeverity.CRITICAL,
                    f"{bad.height} splits/dividends have a null or non-positive value; the "
                    f"adjustment cannot be applied",
                    "corporate_actions",
                    n=bad.height,
                )
            )
    if "new_symbol" in actions.columns:
        bad = actions.filter(
            (pl.col("action_type") == "symbol_change") & pl.col("new_symbol").is_null()
        )
        if not bad.is_empty():
            issues.append(
                _issue(
                    "invalid_symbol_change",
                    IssueSeverity.CRITICAL,
                    f"{bad.height} symbol changes have no new_symbol",
                    "corporate_actions",
                    n=bad.height,
                )
            )
    return issues


def check_calendar_alignment(
    frame: pl.DataFrame, calendar_sessions: list[date], dataset: str
) -> list[DataQualityIssue]:
    """Observations on non-trading days indicate a calendar or timezone mismatch."""
    if frame.is_empty() or not calendar_sessions:
        return []
    sessions = set(calendar_sessions)
    lo, hi = min(sessions), max(sessions)
    off = frame.filter(
        (pl.col("observation_date") >= lo)
        & (pl.col("observation_date") <= hi)
        & ~pl.col("observation_date").is_in(list(sessions))
    )
    if off.is_empty():
        return []
    return [
        _issue(
            "calendar_mismatch",
            IssueSeverity.WARNING,
            f"{off.height} rows in {dataset} fall on non-trading days; likely a timezone or "
            f"exchange-calendar mismatch",
            dataset,
            n=off.height,
            examples=off["observation_date"].unique().head(5).to_list(),
        )
    ]


# --------------------------------------------------------------------------- orchestration
class ValidationReport:
    """Collected findings from a validation run."""

    def __init__(self, issues: list[DataQualityIssue]) -> None:
        self.issues = issues

    @property
    def critical(self) -> list[DataQualityIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def warnings(self) -> list[DataQualityIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def is_clean(self) -> bool:
        return not self.critical

    def to_frame(self) -> pl.DataFrame:
        if not self.issues:
            return pl.DataFrame(
                schema={
                    "check_name": pl.Utf8,
                    "severity": pl.Utf8,
                    "dataset": pl.Utf8,
                    "n_affected": pl.Int64,
                    "message": pl.Utf8,
                }
            )
        return pl.DataFrame(
            [
                {
                    "check_name": i.check_name,
                    "severity": i.severity.value,
                    "dataset": i.dataset or "",
                    "n_affected": i.n_affected,
                    "message": i.message,
                }
                for i in self.issues
            ]
        )

    def raise_if_critical(self) -> None:
        """Fail fast on critical findings (spec section 10.5)."""
        if self.critical:
            details = "\n".join(f"  - [{i.check_name}] {i.message}" for i in self.critical)
            raise DataQualityError(
                f"{len(self.critical)} CRITICAL data quality issue(s); refusing to continue "
                f"because results computed on this data would be misleading:\n{details}"
            )

    def summary(self) -> str:
        return (
            f"{len(self.critical)} critical, {len(self.warnings)} warning(s), "
            f"{len(self.issues)} total"
        )


class DataQualityError(RuntimeError):
    """Raised when critical data quality issues are found."""


def validate_dataset(
    prices: pl.DataFrame | None = None,
    benchmark: pl.DataFrame | None = None,
    membership: pl.DataFrame | None = None,
    fundamentals: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    corporate_actions: pl.DataFrame | None = None,
    calendar_sessions: list[date] | None = None,
    now: datetime | None = None,
) -> ValidationReport:
    """Run every applicable check across the supplied datasets."""
    issues: list[DataQualityIssue] = []
    sessions = calendar_sessions or []

    if prices is not None and not prices.is_empty():
        issues += check_duplicate_records(prices, ["security_id", "observation_date"], "prices")
        issues += check_missing_identifiers(prices, "prices")
        issues += check_invalid_prices(prices)
        issues += check_extreme_returns(prices)
        issues += check_adjustment_monotonicity(prices)
        issues += check_availability_dates(prices, "prices", now)
        issues += check_stale_observations(prices, sessions)
        issues += check_calendar_alignment(prices, sessions, "prices")

    if benchmark is not None and not benchmark.is_empty():
        issues += check_duplicate_records(benchmark, ["observation_date"], "benchmark")
        issues += check_benchmark_gaps(benchmark, sessions)

    if membership is not None and not membership.is_empty():
        issues += check_membership_overlaps(membership)

    if fundamentals is not None and not fundamentals.is_empty():
        issues += check_duplicate_records(
            fundamentals, ["security_id", "observation_date", "available_at"], "fundamentals"
        )
        issues += check_missing_identifiers(fundamentals, "fundamentals")
        issues += check_availability_dates(fundamentals, "fundamentals", now)

    if macro is not None and not macro.is_empty():
        issues += check_duplicate_records(
            macro, ["series_id", "observation_date", "revision_id"], "macro"
        )
        issues += check_availability_dates(macro, "macro", now)

    if corporate_actions is not None and not corporate_actions.is_empty():
        issues += check_corporate_actions(corporate_actions)

    report = ValidationReport(issues)
    logger.info("data validation: %s", report.summary())
    return report


ALL_CHECKS: dict[str, Callable[..., list[DataQualityIssue]]] = {
    "duplicate_records": check_duplicate_records,
    "invalid_prices": check_invalid_prices,
    "extreme_returns": check_extreme_returns,
    "adjustment_monotonicity": check_adjustment_monotonicity,
    "availability_dates": check_availability_dates,
    "missing_identifiers": check_missing_identifiers,
    "benchmark_gaps": check_benchmark_gaps,
    "membership_overlaps": check_membership_overlaps,
    "corporate_actions": check_corporate_actions,
    "stale_observations": check_stale_observations,
    "calendar_alignment": check_calendar_alignment,
}
