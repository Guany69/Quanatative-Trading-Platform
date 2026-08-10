"""Portfolio, order, execution, and reporting contracts.

The timestamp chain below is the backtest's central honesty mechanism. A naive backtest
computes a signal from today's close and fills at today's close, which is impossible: you
cannot trade at a price you used to decide. These records keep the chain explicit:

    information_date -> signal_date -> order_date -> fill_date -> accounting_date

Each stage may only consume data from the stage before it.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import Field, model_validator

from quant_platform.domain.base import FrozenModel
from quant_platform.domain.enums import (
    CandidateStatus,
    CostScenario,
    EvaluationStage,
    IssueSeverity,
    OrderSide,
    OrderStatus,
    OrderType,
    SolverStatus,
)


class PortfolioTarget(FrozenModel):
    """A desired weight for one security, produced by portfolio construction."""

    security_id: str
    as_of: date = Field(description="Signal date the target was computed from.")
    weight: float = Field(description="Target fraction of portfolio value.")
    previous_weight: float = 0.0
    score: float | None = Field(default=None, description="Signal that drove the target.")
    rank: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def weight_change(self) -> float:
        return self.weight - self.previous_weight


class Position(FrozenModel):
    """A held position at a point in time."""

    security_id: str
    as_of: date
    quantity: float = Field(description="Shares held. Negative denotes short.")
    price: float = Field(gt=0, description="Mark price used for valuation.")
    cost_basis: float | None = None

    @property
    def market_value(self) -> float:
        return self.quantity * self.price


class TransactionCost(FrozenModel):
    """Decomposed trading cost for a single trade.

    Kept decomposed rather than as one number so reports can show *where* cost comes from.
    A strategy killed by market impact needs a different fix than one killed by commissions.
    """

    commission: float = Field(default=0.0, ge=0)
    spread_cost: float = Field(default=0.0, ge=0)
    slippage_cost: float = Field(default=0.0, ge=0)
    impact_cost: float = Field(default=0.0, ge=0)

    @property
    def total(self) -> float:
        return self.commission + self.spread_cost + self.slippage_cost + self.impact_cost

    def scaled(self, multiplier: float) -> TransactionCost:
        """Scale every component -- used for the 2x/3x cost stress scenarios."""
        if multiplier < 0:
            raise ValueError(f"cost multiplier must be non-negative, got {multiplier}")
        return TransactionCost(
            commission=self.commission * multiplier,
            spread_cost=self.spread_cost * multiplier,
            slippage_cost=self.slippage_cost * multiplier,
            impact_cost=self.impact_cost * multiplier,
        )


class Order(FrozenModel):
    """An order proposal.

    ``signal_date`` and ``order_date`` are separate fields, and the validator below enforces
    that an order never precedes the signal that motivated it.
    """

    order_id: str
    security_id: str
    side: OrderSide
    quantity: float = Field(gt=0, description="Shares requested; direction lives in `side`.")
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = Field(default=None, gt=0)
    signal_date: date = Field(description="Date the driving signal was computed.")
    order_date: date = Field(description="Session the order is submitted.")
    status: OrderStatus = OrderStatus.PROPOSED
    target_weight: float | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _check_chain(self) -> Order:
        if self.order_date < self.signal_date:
            raise ValueError(
                f"order_date {self.order_date} precedes signal_date {self.signal_date} for "
                f"order {self.order_id}: cannot trade before the signal exists"
            )
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError(f"limit order {self.order_id} requires a limit_price")
        return self


class Fill(FrozenModel):
    """An execution against an order, including realized costs."""

    order_id: str
    security_id: str
    side: OrderSide
    quantity: float = Field(gt=0, description="Shares actually filled.")
    fill_price: float = Field(gt=0, description="Price paid/received, before cost decomposition.")
    fill_date: date
    reference_price: float | None = Field(
        default=None, description="Unperturbed price, for measuring implementation shortfall."
    )
    costs: TransactionCost = Field(default_factory=TransactionCost)
    is_partial: bool = False

    @property
    def gross_notional(self) -> float:
        return self.quantity * self.fill_price

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == OrderSide.BUY else -self.quantity

    @property
    def cash_impact(self) -> float:
        """Cash delta including costs. Buys consume cash; costs always consume cash."""
        direction = -1.0 if self.side == OrderSide.BUY else 1.0
        return direction * self.gross_notional - self.costs.total


class OptimizationDiagnostics(FrozenModel):
    """Why the optimizer produced what it produced.

    Persisted for every rebalance. Silent constraint drops are the failure mode this exists
    to prevent: if a constraint was relaxed, it is recorded here, not swallowed.
    """

    as_of: date
    solver_status: SolverStatus
    solver_name: str | None = None
    objective_value: float | None = None
    expected_return: float | None = None
    expected_risk: float | None = None
    expected_turnover: float | None = None
    expected_cost: float | None = None
    binding_constraints: list[str] = Field(default_factory=list)
    constraint_violations: dict[str, float] = Field(default_factory=dict)
    relaxations_applied: list[str] = Field(default_factory=list)
    n_holdings: int | None = None
    solve_seconds: float | None = None
    message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.solver_status in {SolverStatus.OPTIMAL, SolverStatus.OPTIMAL_INACCURATE}


class BacktestSnapshot(FrozenModel):
    """One session's state in a backtest.

    Gross and net are tracked separately at every step, so no report can accidentally
    present a cost-free curve as achievable.
    """

    as_of: date
    total_value: float
    cash: float
    positions_value: float
    gross_return: float | None = Field(default=None, description="Return before costs.")
    net_return: float | None = Field(default=None, description="Return after costs.")
    benchmark_return: float | None = None
    n_positions: int = 0
    turnover: float = Field(default=0.0, ge=0, description="One-way turnover this session.")
    costs: TransactionCost = Field(default_factory=TransactionCost)
    dividends_received: float = 0.0
    beta: float | None = None
    sector_weights: dict[str, float] = Field(default_factory=dict)

    @property
    def excess_return(self) -> float | None:
        if self.net_return is None or self.benchmark_return is None:
            return None
        return self.net_return - self.benchmark_return


class DataQualityIssue(FrozenModel):
    """A finding from data validation. CRITICAL issues stop the pipeline."""

    check_name: str
    severity: IssueSeverity
    message: str
    security_id: str | None = None
    observation_date: date | None = None
    dataset: str | None = None
    n_affected: int = 1
    detected_at: datetime | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_critical(self) -> bool:
        return self.severity == IssueSeverity.CRITICAL


class PerformanceReport(FrozenModel):
    """Portfolio metrics plus the provenance needed to interpret them.

    The provenance fields are not decoration. A Sharpe ratio means nothing without knowing
    which data produced it, whether costs were applied, and whether the period was truly
    out-of-sample. ``stage`` and the completeness flags force those disclosures to travel
    with the numbers.
    """

    strategy_name: str
    stage: EvaluationStage
    start_date: date
    end_date: date
    cost_scenario: CostScenario = CostScenario.BASE

    # Provenance / honesty disclosures (spec section 30)
    data_source: str = "unknown"
    benchmark_source: str = "unknown"
    is_synthetic_data: bool = Field(
        default=False, description="True for fixture runs. Must be stated prominently."
    )
    historical_constituents_complete: bool = False
    delisting_returns_complete: bool = False
    point_in_time_fundamentals_complete: bool = False
    execution_delay_sessions: int = 1

    metrics: dict[str, float] = Field(default_factory=dict)
    benchmark_metrics: dict[str, float] = Field(default_factory=dict)
    n_observations: int = 0

    @property
    def disclosure_note(self) -> str:
        """The statement every report must carry."""
        base = (
            "Historical results shown are SIMULATED historical performance. Simulated "
            "performance does not guarantee future results and is not a prediction of "
            "future returns."
        )
        if self.is_synthetic_data:
            return (
                "SYNTHETIC DEMONSTRATION DATA -- these results describe randomly generated "
                "fixture securities and carry no investment meaning whatsoever. " + base
            )
        return base


class ExperimentRecord(FrozenModel):
    """A single tracked experiment, including rejected ones.

    Retaining rejected trials is a statistical requirement, not bookkeeping: the Deflated
    Sharpe Ratio and PBO both need the *number of trials attempted*. Keeping only winners
    makes those corrections impossible and turns selection bias invisible.
    """

    experiment_id: str
    created_at: datetime
    git_commit: str | None = None
    config_hash: str | None = None
    data_snapshot_id: str | None = None
    feature_version: str | None = None
    label_version: str | None = None
    model_class: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    random_seed: int | None = None
    train_start: date | None = None
    train_end: date | None = None
    validation_start: date | None = None
    validation_end: date | None = None
    test_start: date | None = None
    test_end: date | None = None
    universe: str | None = None
    cost_assumptions: dict[str, Any] = Field(default_factory=dict)
    portfolio_settings: dict[str, Any] = Field(default_factory=dict)
    forecast_metrics: dict[str, float] = Field(default_factory=dict)
    portfolio_metrics: dict[str, float] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    status: CandidateStatus = CandidateStatus.RESEARCH_CANDIDATE
    rejection_reason: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None
