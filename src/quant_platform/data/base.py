"""Provider-neutral data source interfaces (spec section 10.1).

Every adapter returns a Polars frame with a documented schema, so the pipeline never depends
on which vendor supplied the data. The contract each source must honour:

* one row per (key, observation_date), no duplicates;
* an ``available_at`` column carrying real publication semantics -- an adapter that cannot
  determine publication time must say so explicitly via ``PROVENANCE`` rather than silently
  stamping ``available_at = observation_date`` (which would fabricate perfect foresight);
* a ``source`` column identifying the adapter.

``SourceProvenance`` travels with every adapter so reports can state which limitations apply
to the numbers being shown. That is the difference between "Sharpe 1.2" and "Sharpe 1.2 on
survivorship-biased data with no delisting returns".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

import polars as pl


@dataclass(frozen=True)
class SourceProvenance:
    """Honest description of what a data source can and cannot guarantee.

    Defaults are pessimistic on purpose: an adapter must *opt in* to claiming completeness,
    so a new adapter that forgets to describe itself is treated as incomplete rather than
    silently trusted.
    """

    name: str
    description: str = ""
    # Does the source provide true historical constituents, or only today's members?
    historical_constituents_complete: bool = False
    # Are delisting returns present, or do dead securities just stop having prices?
    delisting_returns_complete: bool = False
    # Are fundamentals stamped with real filing dates, or only fiscal period ends?
    point_in_time_fundamentals_complete: bool = False
    # Are prices adjusted for splits/dividends, and is the adjustment series trustworthy?
    corporate_actions_complete: bool = False
    # Do macro series carry vintages, or only the latest revised values?
    macro_vintages_complete: bool = False
    requires_credentials: bool = False
    requires_network: bool = False
    known_limitations: list[str] = field(default_factory=list)

    def limitation_summary(self) -> list[str]:
        """Human-readable limitations, for reports and the CLI."""
        out = list(self.known_limitations)
        if not self.historical_constituents_complete:
            out.append(
                "Historical index constituents are NOT complete: results may be "
                "survivorship-biased."
            )
        if not self.delisting_returns_complete:
            out.append(
                "Delisting returns are NOT complete: losses from failed companies may be "
                "understated."
            )
        if not self.point_in_time_fundamentals_complete:
            out.append(
                "Fundamentals are NOT point-in-time: filing dates may be approximated, "
                "risking look-ahead."
            )
        if not self.macro_vintages_complete:
            out.append(
                "Macro data has no revision vintages: revised values may be used before they "
                "were published."
            )
        return out


# --------------------------------------------------------------------------- protocols
@runtime_checkable
class PriceDataSource(Protocol):
    """Daily OHLCV bars.

    Required columns: security_id, observation_date, close, adjusted_close, volume,
    available_at, source.
    """

    PROVENANCE: SourceProvenance

    def load_prices(
        self,
        security_ids: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame: ...


@runtime_checkable
class FundamentalDataSource(Protocol):
    """Quarterly/annual financials.

    ``available_at`` MUST reflect the filing date, not the fiscal period end.
    """

    PROVENANCE: SourceProvenance

    def load_fundamentals(
        self,
        security_ids: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame: ...


@runtime_checkable
class ConstituentDataSource(Protocol):
    """Index membership intervals.

    Must set ``is_survivorship_biased`` truthfully; the universe builder refuses biased data
    unless the user explicitly opts in.
    """

    PROVENANCE: SourceProvenance

    def load_membership(
        self, universe: str, start: date | None = None, end: date | None = None
    ) -> pl.DataFrame: ...


@runtime_checkable
class MacroDataSource(Protocol):
    """Macroeconomic series, ideally with revision vintages."""

    PROVENANCE: SourceProvenance

    def load_series(
        self, series_ids: list[str], start: date | None = None, end: date | None = None
    ) -> pl.DataFrame: ...


@runtime_checkable
class CorporateActionDataSource(Protocol):
    """Splits, dividends, renames, delistings."""

    PROVENANCE: SourceProvenance

    def load_actions(
        self,
        security_ids: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame: ...


# --------------------------------------------------------------------------- helpers
REQUIRED_PRICE_COLUMNS = frozenset(
    {
        "security_id",
        "observation_date",
        "close",
        "adjusted_close",
        "volume",
        "available_at",
        "source",
    }
)
REQUIRED_FUNDAMENTAL_COLUMNS = frozenset(
    {"security_id", "observation_date", "fiscal_period_end", "available_at", "source"}
)
REQUIRED_MEMBERSHIP_COLUMNS = frozenset(
    {"security_id", "universe", "start_date", "end_date", "is_survivorship_biased"}
)
REQUIRED_MACRO_COLUMNS = frozenset(
    {"series_id", "observation_date", "value", "available_at", "revision_id", "source"}
)
REQUIRED_ACTION_COLUMNS = frozenset(
    {"security_id", "action_type", "ex_date", "available_at", "source"}
)


class SchemaError(ValueError):
    """Raised when an adapter returns a frame that violates its contract."""


def validate_schema(frame: pl.DataFrame, required: frozenset[str], label: str) -> pl.DataFrame:
    """Fail fast when an adapter returns the wrong shape.

    Catching this at the adapter boundary keeps a malformed frame from travelling deep into
    the pipeline, where a missing ``available_at`` would silently become a look-ahead bug.
    """
    missing = required - set(frame.columns)
    if missing:
        raise SchemaError(
            f"{label} frame is missing required columns {sorted(missing)}; got {frame.columns}"
        )
    return frame


def filter_window(
    frame: pl.DataFrame,
    start: date | None,
    end: date | None,
    column: str = "observation_date",
) -> pl.DataFrame:
    """Restrict a frame to a date window."""
    if start is not None:
        frame = frame.filter(pl.col(column) >= start)
    if end is not None:
        frame = frame.filter(pl.col(column) <= end)
    return frame


def filter_securities(frame: pl.DataFrame, security_ids: list[str] | None) -> pl.DataFrame:
    if security_ids is None:
        return frame
    return frame.filter(pl.col("security_id").is_in(security_ids))
