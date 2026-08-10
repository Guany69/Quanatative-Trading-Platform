from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from quant_platform.api.schemas.common import ApiModel


class PaperInitRequest(ApiModel):
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    account_id: str = Field(default="paper-001", min_length=1, max_length=100)


class ProductionModel(ApiModel):
    run_id: str
    model: str
    designated_at: str
    designated_by: str


class ProductionModelRequest(ApiModel):
    run_id: str
    model: str
    designated_by: str = Field(min_length=1, max_length=200)


class PaperSummary(ApiModel):
    account_id: str
    created_at: str
    updated_at: str
    initial_capital: float
    total_value: float
    cash: float
    positions_value: float
    n_positions: int
    total_return: float | None
    realized_pnl: float
    total_costs: float
    n_rebalances: int
    pending_approvals: int
    production_model: ProductionModel | None
    metrics: dict[str, Any]


class PositionRecord(ApiModel):
    security_id: str
    quantity: float
    average_cost: float
    last_price: float
    market_value: float
    unrealized_pnl: float
    opened_at: str


class ProposalRecord(ApiModel):
    order_id: str
    rebalance_id: str | None = None
    security_id: str
    side: str
    quantity: float
    signal_date: str
    order_date: str
    target_weight: float
    current_weight: float
    reference_price: float
    expected_cost: float
    expected_cost_components: dict[str, float] | None = None
    status: str
    approved: bool
    approved_at: str | None = None
    approved_by: str | None = None
    rejection_reason: str | None = None
    rejected_at: str | None = None
    rejected_by: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def stable_uppercase_status(cls, value: str) -> str:
        return value.upper()


class ProposalDecisionRequest(ApiModel):
    decision: Literal["APPROVE", "REJECT"]
    actor: str = Field(min_length=1, max_length=200)
    reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def rejection_has_reason(self) -> ProposalDecisionRequest:
        if self.decision == "REJECT" and not self.reason:
            raise ValueError("reason is required when rejecting a proposal")
        return self


class CreateRebalanceRequest(ApiModel):
    strategy: str = "equal_weight"


class RebalanceResponse(ApiModel):
    rebalance_id: str
    signal_date: str
    order_date: str
    state: str
    proposed: int
    approved: int = 0
    rejected: int = 0
    filled: int = 0
    realized_cost: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class HistoryResponse(ApiModel):
    rebalances: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    fills: list[dict[str, Any]]
    reconciliations: list[dict[str, Any]]
    production_models: list[dict[str, Any]]
