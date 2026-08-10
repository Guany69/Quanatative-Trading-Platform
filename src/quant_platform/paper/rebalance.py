"""Paper-trading rebalance and reconciliation (spec section 26).

Turns target weights into reviewable order proposals, executes approved orders through the
simulated broker, and then reconciles: did the positions end up where they should be, did the
cash balance, and did costs match the model's prediction?

Reconciliation is the part that earns its keep. A backtest can never tell you its cost model
is wrong, because the same model generates both the expectation and the outcome. Paper
trading can, by comparing predicted cost against what the fill simulation actually charged --
and later, against real broker fills.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import numpy as np

from quant_platform.costs.model import CompositeCostModel, TradeContext
from quant_platform.domain.enums import OrderSide
from quant_platform.execution.base import (
    OrderProposal,
    SimulatedBroker,
    generate_order_id,
)
from quant_platform.paper.state import PaperTradingState, RebalanceRecord
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("paper.rebalance")


@dataclass
class ReconciliationReport:
    """Post-execution comparison of intent versus outcome."""

    as_of: str
    weight_differences: dict[str, float] = field(default_factory=dict)
    max_weight_difference: float = 0.0
    cash_reconciled: bool = True
    cash_difference: float = 0.0
    expected_cost: float = 0.0
    realized_cost: float = 0.0
    fill_shortfall: float = 0.0
    unfilled_orders: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def cost_surprise_bps(self) -> float:
        """Cost error in basis points of expected cost."""
        if abs(self.expected_cost) < 1e-9:
            return float("nan")
        return (self.realized_cost - self.expected_cost) / self.expected_cost * 1e4

    def is_clean(self, weight_tolerance: float = 0.005) -> bool:
        return (
            self.max_weight_difference <= weight_tolerance
            and self.cash_reconciled
            and not self.unfilled_orders
        )


def propose_orders(
    state: PaperTradingState,
    target_weights: dict[str, float],
    prices: dict[str, float],
    cost_model: CompositeCostModel,
    signal_date: date,
    order_date: date,
    adv: dict[str, float] | None = None,
    volatility: dict[str, float] | None = None,
    min_trade_notional: float = 0.0,
) -> list[OrderProposal]:
    """Build order proposals to move from current holdings to target weights.

    Proposals only -- nothing executes here. Each carries its expected cost so a reviewer can
    see what the rebalance will pay before approving it.
    """
    portfolio_value = state.market_value(prices)
    if portfolio_value <= 0:
        logger.error("portfolio value is non-positive; cannot rebalance")
        return []

    current_weights = state.current_weights(prices)
    proposals: list[OrderProposal] = []
    missing: list[str] = []

    universe = set(target_weights) | set(state.positions)
    for security_id in sorted(universe):
        price = prices.get(security_id, 0.0)
        if price <= 0:
            if security_id in target_weights:
                missing.append(security_id)
            continue

        target = target_weights.get(security_id, 0.0)
        current = current_weights.get(security_id, 0.0)
        delta_value = (target - current) * portfolio_value
        if abs(delta_value) < max(min_trade_notional, 1e-6):
            continue

        quantity = abs(delta_value) / price
        held = state.positions.get(security_id)
        side = OrderSide.BUY if delta_value > 0 else OrderSide.SELL
        if side == OrderSide.SELL:
            # Never propose selling more than is held (long-only).
            quantity = min(quantity, held.quantity if held else 0.0)
            if quantity <= 1e-9:
                continue

        ctx = TradeContext(
            notional=quantity * price,
            price=price,
            shares=quantity,
            adv_dollar=(adv or {}).get(security_id),
            volatility=(volatility or {}).get(security_id),
            is_buy=side == OrderSide.BUY,
        )
        proposals.append(
            OrderProposal(
                order_id=generate_order_id(),
                security_id=security_id,
                side=side,
                quantity=quantity,
                signal_date=signal_date,
                order_date=order_date,
                target_weight=target,
                current_weight=current,
                reference_price=price,
                expected_cost=cost_model.compute(ctx),
            )
        )

    if missing:
        logger.warning(
            "%d target securities have no price on %s and were skipped: %s",
            len(missing),
            order_date,
            missing[:5],
        )
    # Sells first: they release cash that the buys then consume, which mirrors how a real
    # rebalance must be sequenced when not using margin.
    proposals.sort(key=lambda p: (p.side != OrderSide.SELL, -abs(p.notional)))
    logger.info("proposed %d orders for %s", len(proposals), order_date)
    return proposals


def execute_approved(
    state: PaperTradingState,
    proposals: list[OrderProposal],
    broker: SimulatedBroker,
    prices: dict[str, float],
    adv: dict[str, float] | None = None,
    volatility: dict[str, float] | None = None,
) -> tuple[list[Any], list[tuple[OrderProposal, str]]]:
    """Execute approved proposals. Unapproved ones are skipped, never auto-approved."""
    approved = [p for p in proposals if p.approved]
    skipped = [(p, "not approved") for p in proposals if not p.approved]

    if skipped:
        logger.info("%d proposals were not approved and will not execute", len(skipped))

    # Keep the broker's cash in sync with the account before executing.
    broker.cash = state.cash
    broker.positions = {k: v.quantity for k, v in state.positions.items()}

    fills = []
    for proposal in approved:
        price = prices.get(proposal.security_id, 0.0)
        fill = broker.submit(
            proposal,
            price,
            adv_dollar=(adv or {}).get(proposal.security_id),
            volatility=(volatility or {}).get(proposal.security_id),
        )
        if fill is not None:
            state.apply_fill(fill)
            fills.append(fill)

    return fills, skipped + list(broker.rejected)


def reconcile(
    state: PaperTradingState,
    target_weights: dict[str, float],
    proposals: list[OrderProposal],
    fills: list[Any],
    prices: dict[str, float],
    as_of: date,
    weight_tolerance: float = 0.005,
) -> ReconciliationReport:
    """Compare intended targets against what actually happened."""
    report = ReconciliationReport(as_of=str(as_of))

    achieved = state.current_weights(prices)
    all_ids = set(target_weights) | set(achieved)
    for security_id in all_ids:
        diff = achieved.get(security_id, 0.0) - target_weights.get(security_id, 0.0)
        if abs(diff) > 1e-9:
            report.weight_differences[security_id] = diff
    if report.weight_differences:
        report.max_weight_difference = max(abs(v) for v in report.weight_differences.values())

    report.expected_cost = sum(p.expected_cost.total for p in proposals if p.approved)
    report.realized_cost = sum(f.costs.total for f in fills)

    filled_ids = {f.order_id for f in fills}
    report.unfilled_orders = [
        p.order_id for p in proposals if p.approved and p.order_id not in filled_ids
    ]

    expected_shares = {p.security_id: p.quantity for p in proposals if p.approved}
    filled_shares: dict[str, float] = {}
    for f in fills:
        filled_shares[f.security_id] = filled_shares.get(f.security_id, 0.0) + f.quantity
    shortfalls = [expected_shares[s] - filled_shares.get(s, 0.0) for s in expected_shares]
    report.fill_shortfall = float(sum(max(x, 0.0) for x in shortfalls))

    if state.cash < -1e-6:
        report.cash_reconciled = False
        report.cash_difference = state.cash
        report.warnings.append(f"cash is negative ({state.cash:,.2f}); leverage was implied")

    if report.max_weight_difference > weight_tolerance:
        report.warnings.append(
            f"largest weight deviation {report.max_weight_difference:.3%} exceeds tolerance "
            f"{weight_tolerance:.3%}"
        )
    if report.unfilled_orders:
        report.warnings.append(f"{len(report.unfilled_orders)} approved orders did not fill")

    surprise = report.realized_cost - report.expected_cost
    if report.expected_cost > 0 and abs(surprise) / report.expected_cost > 0.25:
        report.warnings.append(
            f"realized cost {report.realized_cost:,.0f} differs from expected "
            f"{report.expected_cost:,.0f} by {surprise / report.expected_cost:+.1%}; "
            f"the cost model may be miscalibrated"
        )
    return report


def record_rebalance(
    state: PaperTradingState,
    proposals: list[OrderProposal],
    fills: list[Any],
    rejected: list[tuple[OrderProposal, str]],
    target_weights: dict[str, float],
    reconciliation: ReconciliationReport,
    signal_date: date,
    order_date: date,
    model_name: str = "",
    model_version: str = "",
    feature_version: str = "",
    approved_by: str | None = None,
    rebalance_id: str | None = None,
) -> RebalanceRecord:
    """Append a complete audit record of the cycle."""
    approved = [p for p in proposals if p.approved]
    turnover = 0.0
    total_value = state.market_value()
    if total_value > 0:
        turnover = sum(f.gross_notional for f in fills) / total_value / 2.0

    record = RebalanceRecord(
        rebalance_id=rebalance_id or f"RB-{uuid.uuid4().hex[:8].upper()}",
        signal_date=str(signal_date),
        order_date=str(order_date),
        created_at=datetime.now(UTC).isoformat(),
        model_name=model_name,
        model_version=model_version,
        feature_version=feature_version,
        n_proposed=len(proposals),
        n_approved=len(approved),
        n_filled=len(fills),
        n_rejected=len(rejected),
        expected_cost=reconciliation.expected_cost,
        realized_cost=reconciliation.realized_cost,
        turnover=turnover,
        target_weights=dict(target_weights),
        missing_securities=[],
        approved_by=approved_by,
        notes="; ".join(reconciliation.warnings),
    )
    state.rebalances.append(record)
    state.record_decisions(proposals)
    state.record_fills(fills)
    state.record_reconciliation(reconciliation)
    state.clear_proposals()
    return record


def monitoring_report(
    state: PaperTradingState, prices: dict[str, float] | None = None
) -> dict[str, Any]:
    """Health metrics for the paper account (spec section 26)."""
    summary = state.summary(prices)

    costs = [r.realized_cost for r in state.rebalances]
    expected = [r.expected_cost for r in state.rebalances]
    surprises = [
        (r.realized_cost - r.expected_cost) / r.expected_cost
        for r in state.rebalances
        if r.expected_cost > 1e-9
    ]

    equity = state.equity_frame()
    live_metrics: dict[str, float] = {}
    if not equity.is_empty() and equity.height > 2:
        from quant_platform.evaluation.metrics import max_drawdown, sharpe_ratio

        values = equity["total_value"].to_numpy()
        returns = np.diff(values) / values[:-1]
        live_metrics = {
            "paper_sharpe": sharpe_ratio(returns),
            "paper_max_drawdown": max_drawdown(returns),
            "paper_total_return": float(values[-1] / values[0] - 1.0)
            if values[0] > 0
            else float("nan"),
            "n_observations": float(len(returns)),
        }

    return {
        **summary,
        "total_expected_cost": float(np.sum(expected)) if expected else 0.0,
        "total_realized_cost": float(np.sum(costs)) if costs else 0.0,
        "mean_cost_surprise_pct": float(np.mean(surprises)) if surprises else float("nan"),
        "avg_turnover": float(np.mean([r.turnover for r in state.rebalances]))
        if state.rebalances
        else float("nan"),
        **live_metrics,
        "disclaimer": (
            "PAPER TRADING: all fills are simulated. Real execution differs in price, "
            "timing, and available liquidity."
        ),
    }
