"""Security master records: identity, classification, membership, and corporate actions.

Identity is the foundation of every bias control in this platform. Tickers are *not*
identity: they get reassigned between companies, changed by the same company, and recycled
after delisting. Keying on a ticker silently merges unrelated economic securities and
splits a single one across a rename. Every record therefore keys on ``security_id``, a
permanent internal identifier that never changes for the life of the security.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field, model_validator

from quant_platform.domain.base import FrozenModel, PointInTimeRecord
from quant_platform.domain.enums import (
    CorporateActionType,
    DelistingReason,
    Exchange,
    Sector,
    SecurityType,
)

# A sentinel far-future date meaning "still in effect". Using a real date rather than None
# keeps interval comparisons total, so no branch has to special-case an open end.
FAR_FUTURE = date(9999, 12, 31)


class DateInterval(FrozenModel):
    """A half-open interval ``[start_date, end_date)``.

    Half-open is deliberate: it makes adjacent intervals composable without an off-by-one
    overlap on the boundary day, which is where membership bugs hide.
    """

    start_date: date
    end_date: date = FAR_FUTURE

    @model_validator(mode="after")
    def _check_order(self) -> DateInterval:
        if self.end_date < self.start_date:
            raise ValueError(f"end_date {self.end_date} precedes start_date {self.start_date}")
        return self

    def contains(self, as_of: date) -> bool:
        return self.start_date <= as_of < self.end_date

    def overlaps(self, other: DateInterval) -> bool:
        return self.start_date < other.end_date and other.start_date < self.end_date

    @property
    def is_open_ended(self) -> bool:
        return self.end_date == FAR_FUTURE


class SecurityIdentifier(FrozenModel):
    """A time-bounded mapping from an external identifier to a permanent security_id.

    Symbol history lives here. Resolving "which security was trading as AAPL on 1998-06-01"
    is an interval lookup, not a dictionary lookup.
    """

    security_id: str
    identifier_type: str = Field(description="e.g. 'ticker', 'cusip', 'isin', 'figi', 'cik'.")
    identifier_value: str
    interval: DateInterval
    is_primary: bool = Field(
        default=True, description="Whether this was the primary identifier of its type."
    )

    def active_on(self, as_of: date) -> bool:
        return self.interval.contains(as_of)


class Security(FrozenModel):
    """A permanent economic security.

    ``security_id`` is stable for the security's whole life, across renames and exchange
    moves. ``delisting_date`` being set does not remove the security from history: a
    survivorship-free backtest must keep trading it up to its delisting and then book the
    delisting return.
    """

    security_id: str = Field(description="Permanent internal identifier; never reused.")
    security_type: SecurityType = SecurityType.COMMON_STOCK
    primary_symbol: str | None = Field(
        default=None, description="Most recent known ticker. Convenience only, not a key."
    )
    name: str | None = None
    exchange: Exchange = Exchange.UNKNOWN
    sector: Sector = Sector.UNKNOWN
    industry: str | None = None
    country: str = "US"
    currency: str = "USD"
    first_trade_date: date | None = None
    delisting_date: date | None = None
    delisting_reason: DelistingReason | None = None
    delisting_return: float | None = Field(
        default=None,
        description="Terminal return booked on delisting (e.g. -1.0 for a total loss, or the "
        "merger consideration). Omitting this is the classic delisting bias: dead securities "
        "silently vanish at their last good price and losses never land.",
    )

    @model_validator(mode="after")
    def _check_dates(self) -> Security:
        if (
            self.first_trade_date is not None
            and self.delisting_date is not None
            and self.delisting_date < self.first_trade_date
        ):
            raise ValueError(
                f"delisting_date {self.delisting_date} precedes first_trade_date "
                f"{self.first_trade_date} for {self.security_id}"
            )
        if self.delisting_reason is not None and self.delisting_date is None:
            raise ValueError(f"{self.security_id} has a delisting_reason but no delisting_date")
        return self

    @property
    def is_common_equity(self) -> bool:
        return self.security_type == SecurityType.COMMON_STOCK

    def is_listed_on(self, as_of: date) -> bool:
        """Whether the security was tradeable on ``as_of``."""
        if self.first_trade_date is not None and as_of < self.first_trade_date:
            return False
        return not (self.delisting_date is not None and as_of > self.delisting_date)


class UniverseMembership(FrozenModel):
    """An index/universe membership interval for one security.

    ``available_at`` matters here as much as for fundamentals: index changes are announced
    before they take effect. Using the *effective* date as the knowledge date would let the
    strategy trade an addition before it was public.
    """

    security_id: str
    universe: str = Field(description="Universe name, e.g. 'sp500'.")
    interval: DateInterval
    available_at: datetime | None = Field(
        default=None,
        description="When the membership change became public. Defaults to the effective "
        "start when the provider gives no announcement date.",
    )
    source: str = "unknown"
    is_survivorship_biased: bool = Field(
        default=False,
        description="True when this membership was reconstructed from a current snapshot "
        "rather than true historical constituents. Must be surfaced in every report that "
        "depends on it -- never silently treated as real history.",
    )

    def active_on(self, as_of: date) -> bool:
        return self.interval.contains(as_of)


class PriceBar(PointInTimeRecord):
    """A daily OHLCV bar for one security.

    Both adjusted and unadjusted closes are carried deliberately. Eligibility rules (such as
    a $5 minimum price) must use the *unadjusted* close, because that is the price that
    actually traded; applying a split-adjusted price to a historical filter rewrites history.
    Return calculations use the adjusted series.
    """

    security_id: str
    open: float | None = Field(default=None, ge=0)
    high: float | None = Field(default=None, ge=0)
    low: float | None = Field(default=None, ge=0)
    close: float = Field(gt=0, description="Unadjusted close: the price that actually traded.")
    adjusted_close: float = Field(
        gt=0, description="Close adjusted for splits and dividends; use for returns."
    )
    volume: float = Field(default=0.0, ge=0, description="Shares traded.")
    dollar_volume: float | None = Field(default=None, ge=0)
    shares_outstanding: float | None = Field(default=None, ge=0)
    market_cap: float | None = Field(default=None, ge=0)
    cumulative_adjustment_factor: float = Field(
        default=1.0,
        gt=0,
        description="Cumulative split/dividend factor. Must be non-increasing going forward "
        "in time; a jump signals a corrupted adjustment series.",
    )
    is_halted: bool = False

    @model_validator(mode="after")
    def _check_ohlc_coherence(self) -> PriceBar:
        """High must bound low and both must bound open/close, else the bar is corrupt."""
        if self.high is not None and self.low is not None and self.high < self.low:
            raise ValueError(
                f"high {self.high} < low {self.low} on {self.observation_date} "
                f"for {self.security_id}"
            )
        for name, value in (("open", self.open), ("close", self.close)):
            if self.high is not None and value is not None and value > self.high * 1.0001:
                raise ValueError(
                    f"{name} {value} exceeds high {self.high} on {self.observation_date} "
                    f"for {self.security_id}"
                )
            if self.low is not None and value is not None and value < self.low * 0.9999:
                raise ValueError(
                    f"{name} {value} below low {self.low} on {self.observation_date} "
                    f"for {self.security_id}"
                )
        return self


class CorporateAction(PointInTimeRecord):
    """A split, dividend, rename, or delisting.

    ``ex_date`` (not the announcement) drives price adjustment, while ``available_at``
    drives what the model may know. The two differ and conflating them leaks information.
    """

    security_id: str
    action_type: CorporateActionType
    ex_date: date = Field(description="Date the price adjusts for this action.")
    record_date: date | None = None
    payment_date: date | None = None
    value: float | None = Field(
        default=None,
        description="Cash amount per share for dividends; ratio for splits (2.0 = 2-for-1).",
    )
    old_symbol: str | None = None
    new_symbol: str | None = None
    delisting_reason: DelistingReason | None = None

    @model_validator(mode="after")
    def _check_payload(self) -> CorporateAction:
        """Each action type has required fields; a split without a ratio is unusable."""
        needs_positive_value = {
            CorporateActionType.STOCK_SPLIT,
            CorporateActionType.REVERSE_SPLIT,
            CorporateActionType.CASH_DIVIDEND,
            CorporateActionType.SPECIAL_DIVIDEND,
        }
        if self.action_type in needs_positive_value and (self.value is None or self.value <= 0):
            raise ValueError(
                f"{self.action_type} for {self.security_id} on {self.ex_date} requires a "
                f"positive value, got {self.value}"
            )
        if self.action_type == CorporateActionType.SYMBOL_CHANGE and not self.new_symbol:
            raise ValueError(f"symbol_change for {self.security_id} requires new_symbol")
        return self
