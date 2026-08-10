"""Fundamental, macro, feature, and label observations.

Fundamentals and macro data are where look-ahead bias is most often introduced, because the
period a number describes and the date it was published can be months apart. Both record
types below force those apart: ``observation_date`` is the fiscal/reference period, and
``available_at`` is publication. Macro data adds a third wrinkle -- revisions -- handled via
``revision_id`` vintages.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field, model_validator

from quant_platform.domain.base import FrozenModel, PointInTimeRecord
from quant_platform.domain.enums import TargetType


class FundamentalObservation(PointInTimeRecord):
    """One fiscal period's reported financials for a security.

    ``observation_date`` is the fiscal period end; ``available_at`` is the filing/publication
    timestamp. A Q4 that ends 31 Dec is typically not public until well into February, so a
    feature computed in January must not see it.
    """

    security_id: str
    fiscal_period_end: date = Field(description="Fiscal period end (== observation_date).")
    fiscal_year: int | None = None
    fiscal_quarter: int | None = Field(default=None, ge=1, le=4)
    period_type: str = Field(default="quarterly", description="'quarterly' or 'annual'.")
    filing_date: date | None = Field(
        default=None, description="Date the statement was filed; basis for available_at."
    )

    # Income statement
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    interest_expense: float | None = None
    eps_basic: float | None = None
    eps_diluted: float | None = None

    # Balance sheet
    total_assets: float | None = Field(default=None, ge=0)
    total_liabilities: float | None = Field(default=None, ge=0)
    total_equity: float | None = None
    cash_and_equivalents: float | None = Field(default=None, ge=0)
    total_debt: float | None = Field(default=None, ge=0)
    shares_outstanding: float | None = Field(default=None, ge=0)

    # Cash flow
    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None
    free_cash_flow: float | None = None
    dividends_paid: float | None = None
    share_repurchase: float | None = None

    # Analyst context (used only for surprise-style features)
    consensus_eps: float | None = None

    @model_validator(mode="after")
    def _check_period(self) -> FundamentalObservation:
        if self.fiscal_period_end != self.observation_date:
            raise ValueError(
                f"fiscal_period_end {self.fiscal_period_end} must equal observation_date "
                f"{self.observation_date} for {self.security_id}"
            )
        if self.period_type not in {"quarterly", "annual"}:
            raise ValueError(f"period_type must be 'quarterly' or 'annual', got {self.period_type}")
        # A filing cannot predate the period it reports on.
        if self.filing_date is not None and self.filing_date < self.fiscal_period_end:
            raise ValueError(
                f"filing_date {self.filing_date} precedes fiscal_period_end "
                f"{self.fiscal_period_end} for {self.security_id}"
            )
        return self

    @property
    def reporting_lag_days(self) -> int | None:
        """Days between period end and publication. Useful for validating PIT plausibility."""
        if self.filing_date is None:
            return None
        return (self.filing_date - self.fiscal_period_end).days


class MacroObservation(PointInTimeRecord):
    """One vintage of one macroeconomic series value.

    Macro series are revised. GDP for a quarter is printed, then revised repeatedly. Each
    vintage is a separate record sharing ``observation_date`` but differing in
    ``revision_id`` and ``available_at``. Backtests must select "latest vintage whose
    available_at <= t" -- using the final revised value is a subtle, powerful leak, because
    the revised number was literally unknowable at the time.
    """

    series_id: str = Field(description="e.g. 'DGS10', 'CPIAUCSL'.")
    value: float | None = None
    units: str | None = None
    frequency: str | None = Field(default=None, description="e.g. 'daily', 'monthly'.")
    is_revision: bool = Field(
        default=False, description="True when this supersedes an earlier vintage."
    )

    @model_validator(mode="after")
    def _check_revision_flag(self) -> MacroObservation:
        if self.is_revision and self.revision_id == 0:
            raise ValueError(
                f"{self.series_id} on {self.observation_date} is flagged as a revision but has "
                f"revision_id=0 (0 denotes the original print)"
            )
        return self


class FeatureObservation(FrozenModel):
    """One feature value for one security at one timestamp.

    ``as_of`` is the feature timestamp. Everything feeding this value must have had
    ``available_at <= as_of``. ``feature_version`` participates in reproducibility: changing
    a formula must change the version so stale artifacts cannot be silently mixed in.
    """

    security_id: str
    as_of: date
    feature_name: str
    value: float | None
    feature_version: str = "1.0.0"
    is_imputed: bool = False
    is_winsorized: bool = False


class LabelObservation(FrozenModel):
    """One supervised target for one security at one timestamp.

    The label window ``[window_start, window_end]`` is stored explicitly because purging
    depends on it: a training sample whose forward window overlaps the test period leaks
    test-period information into training, even though its ``as_of`` looks safely in the past.
    """

    security_id: str
    as_of: date = Field(description="Timestamp the label is attached to (signal date).")
    target_type: TargetType
    value: float | None
    horizon_sessions: int = Field(gt=0)
    window_start: date = Field(description="First session of the forward return window.")
    window_end: date = Field(description="Last session of the forward return window.")
    benchmark_return: float | None = None
    raw_return: float | None = None
    is_delisted_in_window: bool = Field(
        default=False,
        description="True when the security delisted inside the window; the label then uses "
        "the delisting return rather than silently dropping the observation.",
    )
    label_version: str = "1.0.0"

    @model_validator(mode="after")
    def _check_window(self) -> LabelObservation:
        if self.window_end < self.window_start:
            raise ValueError(
                f"window_end {self.window_end} precedes window_start {self.window_start} "
                f"for {self.security_id} as_of {self.as_of}"
            )
        if self.window_start < self.as_of:
            raise ValueError(
                f"window_start {self.window_start} precedes as_of {self.as_of} for "
                f"{self.security_id}: a forward label cannot start in the past"
            )
        return self

    def overlaps_window(self, start: date, end: date) -> bool:
        """Whether this label's forward window touches ``[start, end]`` -- the purge test."""
        return self.window_start <= end and start <= self.window_end


class ModelPrediction(FrozenModel):
    """One model's prediction for one security at one timestamp."""

    security_id: str
    as_of: date
    model_name: str
    model_version: str = "1.0.0"
    target_type: TargetType
    prediction: float
    cross_sectional_rank: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Percentile rank within the date's cross-section (1.0 = most attractive). "
        "This, not the raw prediction, is the primary portfolio signal.",
    )
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    generated_at: datetime | None = None
