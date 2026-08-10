"""Serialized application service over the existing paper-trading coordinator."""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_platform.api.config import ApiSettings
from quant_platform.api.errors import ApiError
from quant_platform.api.schemas.common import json_safe
from quant_platform.api.schemas.paper import (
    HistoryResponse,
    PaperSummary,
    PositionRecord,
    ProductionModel,
    ProposalDecisionRequest,
    ProposalRecord,
    RebalanceResponse,
)
from quant_platform.config import load_charter
from quant_platform.costs.model import CompositeCostModel
from quant_platform.domain.enums import OrderSide, OrderStatus
from quant_platform.domain.portfolio import TransactionCost
from quant_platform.execution.base import (
    OrderNotApprovedError,
    OrderProposal,
    SimulatedBroker,
    approve,
    reject,
)
from quant_platform.paper.coordinator import production_predictions
from quant_platform.paper.rebalance import (
    execute_approved,
    monitoring_report,
    propose_orders,
    reconcile,
    record_rebalance,
)
from quant_platform.paper.state import PaperTradingState
from quant_platform.portfolio.constructors import get_constructor
from quant_platform.research.runner import TARGET_STRATEGIES
from quant_platform.utilities.narrow import as_date
from quant_platform.utilities.reproducibility import set_global_seeds


class PaperService:
    """Own every paper read-modify-write cycle under a process-local lock."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._lock = threading.RLock()

    @property
    def state_path(self) -> Path:
        if self.settings.paper_state_path is not None:
            return self.settings.paper_state_path
        charter = load_charter(self.settings.charter_path)
        return Path(charter.paper_trading.state_dir) / "state.json"

    def initialize(self, initial_capital: float, account_id: str) -> PaperSummary:
        with self._lock:
            if self.state_path.exists():
                raise ApiError(
                    "PAPER_STATE_ALREADY_INITIALIZED",
                    "The paper account is already initialized.",
                    status_code=409,
                )
            state = PaperTradingState.initialize(initial_capital, account_id)
            state.save(self.state_path)
            return self._summary(state)

    def summary(self) -> PaperSummary:
        with self._lock:
            return self._summary(self._load())

    def production_model(self) -> ProductionModel | None:
        with self._lock:
            reference = self._load().production_model
            return ProductionModel.model_validate(asdict(reference)) if reference else None

    def designate_model(self, run_id: str, model: str, designated_by: str) -> ProductionModel:
        with self._lock:
            state = self._load()
            try:
                reference = state.designate_production_model(
                    run_id, model, designated_by, self.settings.results_db
                )
            except Exception as exc:
                code = (
                    "RUN_NOT_FOUND" if "unknown run_id" in str(exc) else "INVALID_PRODUCTION_MODEL"
                )
                status = 404 if code == "RUN_NOT_FOUND" else 409
                raise ApiError(code, str(exc), status_code=status) from exc
            state.save(self.state_path)
            return ProductionModel.model_validate(asdict(reference))

    def create_rebalance(self, strategy: str) -> RebalanceResponse:
        with self._lock:
            if strategy not in TARGET_STRATEGIES:
                raise ApiError(
                    "INVALID_REQUEST", f"Unknown portfolio strategy '{strategy}'.", status_code=400
                )
            state = self._load()
            if state.pending_proposals:
                raise ApiError(
                    "REBALANCE_ALREADY_ACTIVE",
                    "A paper rebalance is already awaiting decisions or execution.",
                    status_code=409,
                )
            charter = load_charter(self.settings.charter_path)
            set_global_seeds(charter.random_seed)
            try:
                reference, data, predictions, metadata = production_predictions(
                    state,
                    charter,
                    results_db=self.settings.results_db,
                    snapshot_root=self.settings.snapshot_root,
                    cache_root=self.settings.cache_root,
                )
            except Exception as exc:
                code = (
                    "PRODUCTION_MODEL_NOT_SET"
                    if "no production model" in str(exc)
                    else "PAPER_DATA_INVALID"
                )
                raise ApiError(code, str(exc), status_code=409) from exc
            signal_date = as_date(predictions["as_of"].max(), "signal date")
            latest = predictions.filter(pl.col("as_of") == signal_date)
            order_date = data.calendar.shift(
                signal_date, charter.backtest.rebalance.execution_delay_sessions
            )
            if order_date is None:
                raise ApiError(
                    "PAPER_DATA_INVALID",
                    "No later admitted trading session exists for execution.",
                    status_code=409,
                )
            constructor = get_constructor(strategy, charter.portfolio)
            targets, _diagnostics = constructor.build(latest, state.current_weights(), signal_date)
            price_rows = (
                data.prices.filter(pl.col("observation_date") <= order_date)
                .group_by("security_id")
                .agg(pl.col("adjusted_close").last().alias("price"))
            )
            prices = {
                str(security): float(price)
                for security, price in zip(
                    price_rows["security_id"], price_rows["price"], strict=True
                )
            }
            cost_model = CompositeCostModel(charter.costs, charter.backtest.cost_scenario)
            proposals = propose_orders(
                state,
                targets,
                prices,
                cost_model,
                signal_date,
                order_date,
                min_trade_notional=charter.portfolio.min_trade_size,
            )
            rebalance_id = f"RB-{uuid.uuid4().hex[:8].upper()}"
            state.stage_proposals(
                proposals,
                context={
                    "rebalance_id": rebalance_id,
                    "strategy": strategy,
                    "signal_date": str(signal_date),
                    "order_date": str(order_date),
                    "target_weights": targets,
                    "prices": prices,
                    "model_name": reference.model,
                    "model_version": str(metadata.get("model_version", "unknown")),
                    "feature_version": charter.features.feature_version,
                },
            )
            state.save(self.state_path)
            return RebalanceResponse(
                rebalance_id=rebalance_id,
                signal_date=str(signal_date),
                order_date=str(order_date),
                state="AWAITING_DECISIONS",
                proposed=len(proposals),
            )

    def proposals(self, status: str | None = None) -> list[ProposalRecord]:
        with self._lock:
            state = self._load()
            rebalance_id = (state.pending_rebalance or {}).get("rebalance_id")
            records = [
                ProposalRecord.model_validate({**json_safe(row), "rebalance_id": rebalance_id})
                for row in state.pending_proposals
            ]
            return [record for record in records if record.status == status] if status else records

    def decide(self, order_id: str, decision: ProposalDecisionRequest) -> ProposalRecord:
        with self._lock:
            state = self._load()
            proposals = self._proposals_from_state(state)
            proposal = next((item for item in proposals if item.order_id == order_id), None)
            if proposal is None:
                raise ApiError(
                    "ORDER_NOT_FOUND", f"Unknown pending proposal '{order_id}'.", status_code=404
                )
            if proposal.status != OrderStatus.PROPOSED:
                raise ApiError(
                    "ORDER_ALREADY_DECIDED",
                    f"Order '{order_id}' is already {proposal.status.value}.",
                    status_code=409,
                )
            if decision.decision == "APPROVE":
                approve(proposals, approver=decision.actor, order_ids=[order_id])
            else:
                reject(
                    proposals,
                    approver=decision.actor,
                    reason=decision.reason or "rejected",
                    order_ids=[order_id],
                )
            state.record_decisions(proposals)
            state.save(self.state_path)
            updated = next(item for item in state.pending_proposals if item["order_id"] == order_id)
            return ProposalRecord.model_validate(
                {
                    **json_safe(updated),
                    "rebalance_id": (state.pending_rebalance or {}).get("rebalance_id"),
                }
            )

    def execute_rebalance(self, rebalance_id: str) -> RebalanceResponse:
        with self._lock:
            state = self._load()
            context = state.pending_rebalance or {}
            if context.get("rebalance_id") != rebalance_id:
                raise ApiError(
                    "REBALANCE_NOT_FOUND",
                    f"Unknown pending rebalance '{rebalance_id}'.",
                    status_code=404,
                )
            proposals = self._proposals_from_state(state)
            undecided = [item.order_id for item in proposals if item.status == OrderStatus.PROPOSED]
            if undecided:
                error = OrderNotApprovedError(
                    f"{len(undecided)} proposal(s) still require an explicit decision."
                )
                raise ApiError(
                    "ORDER_NOT_APPROVED",
                    str(error),
                    status_code=409,
                    details={"orderIds": undecided},
                ) from error
            charter = load_charter(self.settings.charter_path)
            prices = {str(key): float(value) for key, value in context["prices"].items()}
            targets = {str(key): float(value) for key, value in context["target_weights"].items()}
            broker = SimulatedBroker(
                CompositeCostModel(charter.costs, charter.backtest.cost_scenario),
                cash=state.cash,
            )
            fills, rejected = execute_approved(state, proposals, broker, prices)
            order_date = date.fromisoformat(context["order_date"])
            signal_date = date.fromisoformat(context["signal_date"])
            reconciliation = reconcile(state, targets, proposals, fills, prices, order_date)
            actors = sorted(
                {item.approved_by for item in proposals if item.approved_by is not None}
            )
            record_rebalance(
                state,
                proposals,
                fills,
                rejected,
                targets,
                reconciliation,
                signal_date,
                order_date,
                model_name=context.get("model_name", ""),
                model_version=context.get("model_version", ""),
                feature_version=context.get("feature_version", ""),
                approved_by=", ".join(actors) or None,
                rebalance_id=rebalance_id,
            )
            state.record_equity(order_date, prices)
            state.save(self.state_path)
            return RebalanceResponse(
                rebalance_id=rebalance_id,
                signal_date=str(signal_date),
                order_date=str(order_date),
                state="EXECUTED",
                proposed=len(proposals),
                approved=sum(item.approved for item in proposals),
                rejected=sum(item.rejection_reason is not None for item in proposals),
                filled=len(fills),
                realized_cost=reconciliation.realized_cost,
                warnings=reconciliation.warnings,
            )

    def positions(self) -> list[PositionRecord]:
        with self._lock:
            return [
                PositionRecord(
                    **asdict(position),
                    market_value=position.market_value,
                    unrealized_pnl=position.unrealized_pnl,
                )
                for position in self._load().positions.values()
            ]

    def fills(self) -> list[dict[str, Any]]:
        with self._lock:
            return json_safe(self._load().fill_history)

    def reconciliations(self) -> list[dict[str, Any]]:
        with self._lock:
            return json_safe(self._load().reconciliation_history)

    def rebalances(self) -> list[dict[str, Any]]:
        with self._lock:
            return json_safe([asdict(item) for item in self._load().rebalances])

    def history(self) -> HistoryResponse:
        with self._lock:
            state = self._load()
            return HistoryResponse(
                rebalances=json_safe([asdict(item) for item in state.rebalances]),
                proposals=json_safe(state.proposal_history),
                decisions=json_safe(state.decision_history),
                fills=json_safe(state.fill_history),
                reconciliations=json_safe(state.reconciliation_history),
                production_models=json_safe(state.production_model_history),
            )

    def _load(self) -> PaperTradingState:
        try:
            return PaperTradingState.load(self.state_path)
        except FileNotFoundError as exc:
            raise ApiError("PAPER_STATE_NOT_INITIALIZED", str(exc), status_code=409) from exc

    def _summary(self, state: PaperTradingState) -> PaperSummary:
        report = json_safe(monitoring_report(state))
        report["pending_approvals"] = sum(
            row.get("status") == OrderStatus.PROPOSED.value for row in state.pending_proposals
        )
        reference = state.production_model
        return PaperSummary(
            account_id=state.account_id,
            created_at=state.created_at,
            updated_at=state.updated_at,
            initial_capital=state.initial_capital,
            total_value=report.pop("total_value"),
            cash=report.pop("cash"),
            positions_value=report.pop("positions_value"),
            n_positions=report.pop("n_positions"),
            total_return=report.pop("total_return"),
            realized_pnl=report.pop("realized_pnl"),
            total_costs=report.pop("total_costs"),
            n_rebalances=report.pop("n_rebalances"),
            pending_approvals=report.pop("pending_approvals"),
            production_model=(
                ProductionModel.model_validate(asdict(reference)) if reference else None
            ),
            metrics=report,
        )

    @staticmethod
    def _proposals_from_state(state: PaperTradingState) -> list[OrderProposal]:
        proposals: list[OrderProposal] = []
        for row in state.pending_proposals:
            components = row.get("expected_cost_components") or {
                "commission": row.get("expected_cost", 0.0)
            }
            proposal = OrderProposal(
                order_id=row["order_id"],
                security_id=row["security_id"],
                side=OrderSide(row["side"]),
                quantity=float(row["quantity"]),
                signal_date=date.fromisoformat(row["signal_date"]),
                order_date=date.fromisoformat(row["order_date"]),
                target_weight=float(row["target_weight"]),
                current_weight=float(row["current_weight"]),
                reference_price=float(row["reference_price"]),
                expected_cost=TransactionCost(**components),
                approved=bool(row.get("approved")),
                approved_at=(
                    datetime.fromisoformat(row["approved_at"]) if row.get("approved_at") else None
                ),
                approved_by=row.get("approved_by"),
                rejection_reason=row.get("rejection_reason"),
                rejected_at=(
                    datetime.fromisoformat(row["rejected_at"]) if row.get("rejected_at") else None
                ),
                rejected_by=row.get("rejected_by"),
            )
            proposals.append(proposal)
        return proposals
