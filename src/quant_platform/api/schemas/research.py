from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from quant_platform.api.schemas.common import ApiModel

RunState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]


class CreateRunRequest(ApiModel):
    snapshot_id: str = "auto"
    models: list[str]
    strategies: list[str]
    cost_scenarios: list[str] = Field(default_factory=lambda: ["base"])
    portfolio_model: str = "factor_composite"
    max_workers: int | None = Field(default=None, ge=1, le=64)
    fixture_securities: int = Field(default=120, ge=20, le=2000)

    @model_validator(mode="after")
    def selections_are_nonempty(self) -> CreateRunRequest:
        for name in ("models", "strategies", "cost_scenarios"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        return self


class CreateRunResponse(ApiModel):
    run_id: str
    state: RunState


class RunFailure(ApiModel):
    code: str
    message: str
    stage: str | None = None


class RunDetail(ApiModel):
    run_id: str
    state: RunState
    stage: str | None = None
    completed_stages: int = 0
    total_stages: int = 6
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str
    failure: RunFailure | None = None
    snapshot_id: str | None = None
    charter_hash: str | None = None
    code_version: str | None = None
    code_dirty: bool | None = None
    seed: int | None = None
    models: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    cost_scenarios: list[str] = Field(default_factory=list)
    fold_schedule_hash: str | None = None
    duration_seconds: float | None = None


class RunSummaryResponse(ApiModel):
    run: RunDetail
    result_counts: dict[str, int]
    cache_hit: bool | None = None
    timings: dict[str, float | None] = Field(default_factory=dict)


class FoldRecord(ApiModel):
    fold: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    purge: bool
    embargo: int


class ModelMetricRecord(ApiModel):
    model: str
    fold: int
    metric: str
    value: float | None


class PredictionRecord(ApiModel):
    model: str
    fold: int
    as_of: str
    security_id: str
    score: float | None
    rank: float | None


class StrategyMetricRecord(ApiModel):
    model: str
    strategy: str
    cost_scenario: str
    metric: str
    value: float | None


class EquityRecord(ApiModel):
    model: str
    strategy: str
    cost_scenario: str
    as_of: str
    total_value: float | None
    gross_return: float | None
    net_return: float | None
    benchmark_return: float | None
    turnover: float | None
    costs: float | None


class TradeRecord(ApiModel):
    model: str
    strategy: str
    cost_scenario: str
    information_date: str
    signal_date: str
    order_date: str
    fill_date: str
    accounting_date: str
    security_id: str
    side: str
    shares: float
    reference_price: float
    fill_price: float
    notional: float
    commission: float
    spread_cost: float
    slippage_cost: float
    impact_cost: float


class WeightRecord(ApiModel):
    model: str
    strategy: str
    as_of: str
    security_id: str
    weight: float


class OptimizerDiagnosticRecord(ApiModel):
    model: str
    strategy: str
    as_of: str
    solver_status: str
    solver_name: str | None
    objective_value: float | None
    expected_return: float | None
    expected_risk: float | None
    expected_turnover: float | None
    expected_cost: float | None
    relaxations: list[str]
    message: str | None


class OverfittingRecord(ApiModel):
    metric: str
    value: float | None
    optimistic: bool
    details: dict[str, Any]


class ArtifactRecord(ApiModel):
    id: str
    model: str
    fold: int
    artifact_available: bool
    metadata_available: bool


class HoldoutEvaluationRequest(ApiModel):
    model: str
    acknowledged_by: str = Field(min_length=1, max_length=200)
    acknowledge_one_shot_evaluation: bool


class HoldoutEvaluationResponse(ApiModel):
    run_id: str
    model: str
    metrics: dict[str, float | None]
    consumed: bool = True
