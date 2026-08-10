"""Strongly validated configuration models.

Configuration is the platform's main safety surface: nearly every bias control is a setting
that could be quietly turned off. These models therefore reject incoherent configurations at
load time rather than letting them produce plausible-looking but invalid results. The
validation rules implement spec section 28.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_platform.domain.enums import BenchmarkSource, TargetType


class StrictConfig(BaseModel):
    """Base for config models: unknown keys are errors, not silently ignored.

    A typo'd YAML key that is ignored would leave the user believing a setting took effect
    when it did not -- exactly how a bias control gets silently disabled.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


class DateRange(StrictConfig):
    start: date
    end: date

    @model_validator(mode="after")
    def _check_order(self) -> DateRange:
        if self.end < self.start:
            raise ValueError(f"date range end {self.end} precedes start {self.start}")
        return self

    def overlaps(self, other: DateRange) -> bool:
        return self.start <= other.end and other.start <= self.end


class UniverseConfig(StrictConfig):
    """Eligibility rules for the investable universe (spec section 4.1)."""

    name: str = "sp500"
    allowed_security_types: list[str] = Field(default=["common_stock"])
    min_price: float = Field(default=5.0, ge=0, description="Minimum UNADJUSTED close.")
    min_dollar_volume: float = Field(default=5_000_000.0, ge=0)
    dollar_volume_lookback_sessions: int = Field(default=21, gt=0)
    min_history_sessions: int = Field(
        default=252, gt=0, description="Required price history before a security is eligible."
    )
    max_securities: int | None = Field(default=None, gt=0)
    allow_survivorship_biased_fallback: bool = Field(
        default=False,
        description="If a provider cannot supply true historical constituents, allow a "
        "current-membership fallback. Results are then explicitly marked survivorship-biased; "
        "they are never silently presented as real history.",
    )
    exclude_sectors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_types(self) -> UniverseConfig:
        if not self.allowed_security_types:
            raise ValueError("allowed_security_types must not be empty")
        return self


class BenchmarkConfig(StrictConfig):
    preferred_source: BenchmarkSource = BenchmarkSource.SP500_TOTAL_RETURN
    fallback_source: BenchmarkSource = BenchmarkSource.SPY_ADJUSTED
    symbol: str = "SPY"


class LabelConfig(StrictConfig):
    """Prediction target definition (spec section 3)."""

    target_type: TargetType = TargetType.EXCESS_RETURN_RANK
    forecast_horizon_sessions: int = Field(default=20, gt=0)
    label_version: str = "1.0.0"

    @model_validator(mode="after")
    def _check_target(self) -> LabelConfig:
        if self.target_type not in set(TargetType):
            raise ValueError(f"unsupported target type: {self.target_type}")
        return self


class RebalanceConfig(StrictConfig):
    """When signals are formed and when they may be traded (spec section 4.4)."""

    frequency: Literal["daily", "weekly", "monthly"] = "weekly"
    signal_day: Literal["week_start", "week_end", "month_end"] = "week_end"
    execution_delay_sessions: int = Field(
        default=1,
        ge=1,
        description="Sessions between signal and fill. Must be >= 1: filling at the same close "
        "that produced the signal is not achievable in reality.",
    )


class PortfolioConfig(StrictConfig):
    """Portfolio construction constraints (spec section 4.5)."""

    long_only: bool = True
    fully_invested: bool = True
    allow_leverage: bool = False
    min_holdings: int = Field(default=50, gt=0)
    max_holdings: int = Field(default=100, gt=0)
    max_position_weight: float = Field(default=0.02, gt=0, le=1.0)
    min_position_weight: float = Field(default=0.0025, ge=0, le=1.0)
    max_sector_deviation: float = Field(default=0.03, ge=0, le=1.0)
    min_beta: float = Field(default=0.90, gt=0)
    max_beta: float = Field(default=1.10, gt=0)
    max_turnover_per_rebalance: float = Field(default=0.15, ge=0, le=1.0)
    max_adv_participation: float = Field(default=0.02, gt=0, le=1.0)
    cash_buffer: float = Field(default=0.0, ge=0, lt=1.0)
    min_trade_size: float = Field(default=0.0, ge=0, description="Minimum trade notional.")
    restricted_securities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_coherence(self) -> PortfolioConfig:
        if self.min_holdings > self.max_holdings:
            raise ValueError(
                f"min_holdings ({self.min_holdings}) exceeds max_holdings ({self.max_holdings})"
            )
        if self.min_position_weight > self.max_position_weight:
            raise ValueError(
                f"min_position_weight ({self.min_position_weight}) exceeds max_position_weight "
                f"({self.max_position_weight})"
            )
        if self.min_beta > self.max_beta:
            raise ValueError(f"min_beta ({self.min_beta}) exceeds max_beta ({self.max_beta})")
        # Feasibility: the cap must admit enough names to reach full investment.
        investable = 1.0 - self.cash_buffer
        if self.max_position_weight * self.max_holdings < investable:
            raise ValueError(
                f"infeasible: max_holdings ({self.max_holdings}) * max_position_weight "
                f"({self.max_position_weight}) = "
                f"{self.max_position_weight * self.max_holdings:.3f} cannot reach the required "
                f"invested fraction ({investable:.3f})"
            )
        # Feasibility: the floor must not force more than 100% invested.
        if self.min_position_weight * self.min_holdings > investable:
            raise ValueError(
                f"infeasible: min_holdings ({self.min_holdings}) * min_position_weight "
                f"({self.min_position_weight}) = "
                f"{self.min_position_weight * self.min_holdings:.3f} exceeds the investable "
                f"fraction ({investable:.3f})"
            )
        if self.long_only and self.allow_leverage:
            raise ValueError("allow_leverage is not supported for the long_only MVP portfolio")
        return self


class CommissionConfig(StrictConfig):
    model: Literal["per_share", "per_order", "basis_points", "zero"] = "basis_points"
    per_share: float = Field(default=0.005, ge=0)
    per_order: float = Field(default=1.0, ge=0)
    basis_points: float = Field(default=1.0, ge=0)
    minimum: float = Field(default=0.0, ge=0)


class SpreadConfig(StrictConfig):
    model: Literal["fixed_bps", "liquidity_proxy"] = "fixed_bps"
    fixed_bps: float = Field(default=5.0, ge=0)
    stress_multiplier: float = Field(default=1.0, ge=0)


class SlippageConfig(StrictConfig):
    model: Literal["fixed_bps", "volatility_scaled", "participation_scaled"] = "volatility_scaled"
    fixed_bps: float = Field(default=2.0, ge=0)
    volatility_coefficient: float = Field(default=0.1, ge=0)
    participation_coefficient: float = Field(default=0.1, ge=0)


class ImpactConfig(StrictConfig):
    """Square-root market impact. An estimate, not a law -- see docs/limitations.md."""

    model: Literal["square_root", "linear", "zero"] = "square_root"
    coefficient: float = Field(default=0.1, ge=0)
    exponent: float = Field(default=0.5, gt=0, le=1.0)


class CostConfig(StrictConfig):
    commission: CommissionConfig = Field(default_factory=CommissionConfig)
    spread: SpreadConfig = Field(default_factory=SpreadConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    impact: ImpactConfig = Field(default_factory=ImpactConfig)
    scenario_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"zero": 0.0, "base": 1.0, "double": 2.0, "triple": 3.0}
    )

    @model_validator(mode="after")
    def _check_multipliers(self) -> CostConfig:
        for name, mult in self.scenario_multipliers.items():
            if mult < 0:
                raise ValueError(f"negative cost multiplier for scenario '{name}': {mult}")
        return self


class RiskConfig(StrictConfig):
    covariance_method: Literal["sample", "ledoit_wolf", "oas", "shrinkage"] = "ledoit_wolf"
    covariance_lookback_sessions: int = Field(default=252, gt=1)
    shrinkage_intensity: float | None = Field(default=None, ge=0, le=1)
    beta_lookback_sessions: int = Field(default=252, gt=1)
    min_observations: int = Field(default=60, gt=1)
    risk_aversion: float = Field(default=5.0, ge=0)

    @model_validator(mode="after")
    def _check_lookbacks(self) -> RiskConfig:
        if self.min_observations > self.covariance_lookback_sessions:
            raise ValueError(
                f"min_observations ({self.min_observations}) exceeds covariance_lookback_sessions "
                f"({self.covariance_lookback_sessions})"
            )
        return self


class ValidationConfig(StrictConfig):
    """Walk-forward schedule and leakage controls (spec section 16)."""

    min_train_years: float = Field(default=3.0, gt=0)
    validation_months: int = Field(default=12, gt=0)
    test_months: int = Field(default=12, gt=0)
    step_months: int = Field(default=12, gt=0)
    embargo_sessions: int = Field(
        default=20,
        ge=0,
        description="Sessions blanked after each validation/test boundary. Should be >= the "
        "forecast horizon so adjacent windows cannot share label information.",
    )
    purge_enabled: bool = True
    holdout: DateRange | None = Field(
        default=None, description="Locked final period. Untouchable during any selection."
    )
    allow_holdout_evaluation: bool = Field(
        default=False,
        description="Must be explicitly enabled to touch the holdout. Enabling it burns the "
        "holdout's statistical value: it can only honestly be looked at once.",
    )
    train_range: DateRange | None = None

    @model_validator(mode="after")
    def _check_holdout_isolation(self) -> ValidationConfig:
        if (
            self.holdout is not None
            and self.train_range is not None
            and self.holdout.overlaps(self.train_range)
        ):
            raise ValueError(
                f"locked holdout ({self.holdout.start}..{self.holdout.end}) overlaps the "
                f"training range ({self.train_range.start}..{self.train_range.end}); the "
                f"holdout would no longer be out-of-sample"
            )
        return self


class BacktestConfig(StrictConfig):
    initial_capital: float = Field(default=10_000_000.0, gt=0)
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    cost_scenario: str = "base"
    reinvest_dividends: bool = True
    allow_partial_fills: bool = True


class FeatureConfig(StrictConfig):
    enabled_families: list[str] = Field(
        default=["momentum", "reversal", "risk", "liquidity", "value", "quality"]
    )
    winsorize_quantile: float = Field(default=0.01, ge=0, lt=0.5)
    neutralize_sector: bool = False
    neutralize_size: bool = False
    min_cross_section: int = Field(
        default=10, gt=1, description="Minimum names before cross-sectional stats are meaningful."
    )
    feature_version: str = "1.0.0"


class DataSourceConfig(StrictConfig):
    prices: str = "fixture"
    fundamentals: str = "fixture"
    macro: str = "fixture"
    constituents: str = "fixture"
    corporate_actions: str = "fixture"
    cache_dir: str = "data/interim"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


class PaperTradingConfig(StrictConfig):
    require_manual_confirmation: bool = Field(
        default=True,
        description="Orders stay proposals until a human approves. Disabling this on a live "
        "broker path is deliberately made a conscious act.",
    )
    broker: Literal["simulated", "nautilus"] = "simulated"
    state_dir: str = "paper_state"
    initial_capital: float = Field(default=1_000_000.0, gt=0)


class AcceptanceGateConfig(StrictConfig):
    """Research gates (spec section 25). Nothing is auto-promoted to production."""

    min_holdout_net_information_ratio: float = 0.0
    require_positive_net_excess_at_double_cost: bool = True
    require_beating_factor_baseline: bool = True
    max_single_sector_excess_contribution: float = Field(default=0.5, gt=0, le=1.0)
    max_single_year_excess_contribution: float = Field(default=0.5, gt=0, le=1.0)
    max_drawdown_limit: float = Field(default=0.35, gt=0, le=1.0)
    max_tracking_error: float = Field(default=0.10, gt=0)
    max_annual_turnover: float = Field(default=6.0, gt=0)
    min_deflated_sharpe: float | None = 0.0
    max_pbo: float | None = Field(default=0.5, ge=0, le=1.0)


class ResearchCharter(StrictConfig):
    """The top-level resolved configuration. Snapshotted with every experiment."""

    name: str = "default_charter"
    description: str = ""
    random_seed: int = 42
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
    acceptance_gates: AcceptanceGateConfig = Field(default_factory=AcceptanceGateConfig)

    @model_validator(mode="after")
    def _check_cross_section(self) -> ResearchCharter:
        if self.backtest.cost_scenario not in self.costs.scenario_multipliers:
            raise ValueError(
                f"backtest.cost_scenario '{self.backtest.cost_scenario}' is not defined in "
                f"costs.scenario_multipliers {sorted(self.costs.scenario_multipliers)}"
            )
        # The embargo must cover the forecast horizon, or adjacent folds share label windows
        # and the "out-of-sample" test is contaminated.
        horizon = self.label.forecast_horizon_sessions
        if self.validation.embargo_sessions < horizon:
            raise ValueError(
                f"embargo_sessions ({self.validation.embargo_sessions}) is shorter than "
                f"forecast_horizon_sessions ({horizon}); label windows would straddle the "
                f"train/test boundary and leak"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
