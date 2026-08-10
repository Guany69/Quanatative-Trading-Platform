"""Simulated broker and paper-trading execution tests (spec sections 26, 31.4).

Two properties are non-negotiable here:

1. **No order executes without human approval.** This is the boundary where software would
   become money, so the gate is enforced in code, not documentation.
2. **A long-only account never goes short of cash.** Costs are paid in cash, so a
   fully-invested target plus costs exceeds 100% of the account unless orders are sized
   against what cash actually covers. Getting this wrong silently implies margin.
"""

from __future__ import annotations

from datetime import date

import pytest

from quant_platform.config.models import CostConfig
from quant_platform.costs.model import CompositeCostModel
from quant_platform.domain.enums import OrderSide
from quant_platform.execution.base import (
    NautilusBrokerAdapter,
    OrderNotApprovedError,
    OrderProposal,
    SimulatedBroker,
    approve,
    generate_order_id,
)
from quant_platform.paper.rebalance import (
    execute_approved,
    propose_orders,
    reconcile,
)
from quant_platform.paper.state import PaperTradingState

SIGNAL_DATE = date(2024, 6, 14)
ORDER_DATE = date(2024, 6, 17)


def _cost_model(scenario: str = "base") -> CompositeCostModel:
    return CompositeCostModel(CostConfig(), scenario)


def _proposal(security_id: str = "A", quantity: float = 100, price: float = 50.0) -> OrderProposal:
    return OrderProposal(
        order_id=generate_order_id(),
        security_id=security_id,
        side=OrderSide.BUY,
        quantity=quantity,
        signal_date=SIGNAL_DATE,
        order_date=ORDER_DATE,
        target_weight=0.05,
        current_weight=0.0,
        reference_price=price,
    )


class TestApprovalGate:
    def test_unapproved_order_is_refused(self):
        """The critical safety property."""
        broker = SimulatedBroker(_cost_model(), cash=1_000_000)
        with pytest.raises(OrderNotApprovedError, match="not been approved"):
            broker.submit(_proposal(), market_price=50.0)

    def test_approved_order_executes(self):
        broker = SimulatedBroker(_cost_model(), cash=1_000_000)
        proposal = _proposal()
        approve([proposal], approver="test")
        fill = broker.submit(proposal, market_price=50.0)
        assert fill is not None
        assert fill.quantity == pytest.approx(100)
        assert broker.positions["A"] == pytest.approx(100)

    def test_partial_approval_executes_only_the_approved_subset(self):
        broker = SimulatedBroker(_cost_model(), cash=1_000_000)
        proposals = [_proposal(f"S{i}") for i in range(5)]
        approve(proposals, approver="test", order_ids=[proposals[0].order_id])

        assert sum(p.approved for p in proposals) == 1
        fill = broker.submit(proposals[0], market_price=50.0)
        assert fill is not None
        with pytest.raises(OrderNotApprovedError):
            broker.submit(proposals[1], market_price=50.0)

    def test_approval_records_who_and_when(self):
        proposal = _proposal()
        approve([proposal], approver="alex")
        assert proposal.approved
        assert proposal.approved_by == "alex"
        assert proposal.approved_at is not None


class TestCashDiscipline:
    def test_cash_never_goes_negative_on_a_fully_invested_rebalance(self):
        """Costs are paid in cash, so a 100% target must be sized against available cash.

        This is a regression test: an earlier version passed the cost-adjusted price as
        fill_price while also subtracting costs.total, double-charging friction and pushing
        the account into an implicit margin loan.
        """
        state = PaperTradingState.initialize(1_000_000.0)
        prices = {f"S{i}": 50.0 + i for i in range(20)}
        targets = {f"S{i}": 0.05 for i in range(20)}  # exactly 100% invested

        cost_model = _cost_model()
        proposals = propose_orders(state, targets, prices, cost_model, SIGNAL_DATE, ORDER_DATE)
        approve(proposals, approver="test")
        broker = SimulatedBroker(cost_model, cash=state.cash)
        execute_approved(state, proposals, broker, prices)

        assert state.cash >= -1e-6, (
            f"cash is {state.cash:,.2f} after a fully-invested rebalance; the account is "
            f"implicitly borrowing to pay transaction costs"
        )

    def test_costs_are_not_double_counted(self):
        """Cash spent must equal notional plus costs exactly -- no more."""
        broker = SimulatedBroker(_cost_model(), cash=1_000_000)
        proposal = _proposal(quantity=100, price=50.0)
        approve([proposal], approver="test")
        starting = broker.cash
        fill = broker.submit(proposal, market_price=50.0)
        assert fill is not None

        spent = starting - broker.cash
        expected = 100 * 50.0 + fill.costs.total
        assert spent == pytest.approx(expected, rel=1e-9), (
            f"spent {spent:,.2f} but notional + costs is {expected:,.2f}; frictions are being "
            f"charged more than once"
        )

    def test_insufficient_cash_produces_a_partial_fill(self):
        broker = SimulatedBroker(_cost_model(), cash=1000.0, allow_partial_fills=True)
        proposal = _proposal(quantity=1000, price=50.0)  # wants $50,000
        approve([proposal], approver="test")
        fill = broker.submit(proposal, market_price=50.0)
        assert fill is not None
        assert fill.is_partial
        assert fill.quantity < 1000
        assert broker.cash >= -1e-6

    def test_insufficient_cash_rejects_when_partials_disabled(self):
        broker = SimulatedBroker(_cost_model(), cash=1000.0, allow_partial_fills=False)
        proposal = _proposal(quantity=1000, price=50.0)
        approve([proposal], approver="test")
        assert broker.submit(proposal, market_price=50.0) is None
        assert broker.rejected


class TestLiquidityLimits:
    def test_adv_participation_caps_fill_size(self):
        broker = SimulatedBroker(_cost_model(), cash=10_000_000, max_participation=0.02)
        proposal = _proposal(quantity=10_000, price=50.0)  # $500k
        approve([proposal], approver="test")
        # ADV of $1m at 2% participation allows only ~$20k.
        fill = broker.submit(proposal, market_price=50.0, adv_dollar=1_000_000)
        assert fill is not None
        assert fill.is_partial
        assert fill.quantity * 50.0 <= 20_000 * 1.01

    def test_cannot_sell_more_than_held(self):
        """Long-only: a sell is bounded by the existing position."""
        broker = SimulatedBroker(_cost_model(), cash=1_000_000)
        broker.positions["A"] = 50.0
        proposal = _proposal(quantity=200, price=50.0)
        proposal.side = OrderSide.SELL
        approve([proposal], approver="test")
        fill = broker.submit(proposal, market_price=50.0)
        assert fill is not None
        assert fill.quantity == pytest.approx(50.0)
        assert "A" not in broker.positions


class TestReconciliation:
    def test_reconciliation_flags_unapproved_shortfall(self):
        """Approving only some orders must show up as a weight deviation."""
        state = PaperTradingState.initialize(1_000_000.0)
        prices = {f"S{i}": 50.0 for i in range(10)}
        targets = {f"S{i}": 0.1 for i in range(10)}
        cost_model = _cost_model()

        proposals = propose_orders(state, targets, prices, cost_model, SIGNAL_DATE, ORDER_DATE)
        approve(proposals, approver="test", order_ids=[p.order_id for p in proposals[:5]])
        broker = SimulatedBroker(cost_model, cash=state.cash)
        fills, _ = execute_approved(state, proposals, broker, prices)

        report = reconcile(state, targets, proposals, fills, prices, ORDER_DATE)
        assert report.max_weight_difference > 0.05
        assert report.warnings

    def test_clean_reconciliation_when_everything_fills(self):
        state = PaperTradingState.initialize(1_000_000.0)
        prices = {f"S{i}": 50.0 for i in range(10)}
        targets = {f"S{i}": 0.09 for i in range(10)}  # 90% invested, leaves a cash buffer
        cost_model = _cost_model()

        proposals = propose_orders(state, targets, prices, cost_model, SIGNAL_DATE, ORDER_DATE)
        approve(proposals, approver="test")
        broker = SimulatedBroker(cost_model, cash=state.cash)
        fills, _ = execute_approved(state, proposals, broker, prices)

        report = reconcile(state, targets, proposals, fills, prices, ORDER_DATE)
        assert not report.unfilled_orders
        assert report.cash_reconciled
        assert report.max_weight_difference < 0.01

    def test_cost_surprise_is_measured(self):
        state = PaperTradingState.initialize(1_000_000.0)
        prices = {"A": 50.0}
        cost_model = _cost_model()
        proposals = propose_orders(state, {"A": 0.5}, prices, cost_model, SIGNAL_DATE, ORDER_DATE)
        approve(proposals, approver="test")
        broker = SimulatedBroker(cost_model, cash=state.cash)
        fills, _ = execute_approved(state, proposals, broker, prices)
        report = reconcile(state, {"A": 0.5}, proposals, fills, prices, ORDER_DATE)
        # The simulator uses the same model as the proposal, so they should agree closely.
        assert report.expected_cost > 0
        assert abs(report.realized_cost - report.expected_cost) / report.expected_cost < 0.05


class TestStatePersistence:
    def test_state_round_trips(self, tmp_path):
        state = PaperTradingState.initialize(500_000.0)
        prices = {"A": 50.0, "B": 25.0}
        cost_model = _cost_model()
        proposals = propose_orders(
            state, {"A": 0.4, "B": 0.4}, prices, cost_model, SIGNAL_DATE, ORDER_DATE
        )
        approve(proposals, approver="test")
        broker = SimulatedBroker(cost_model, cash=state.cash)
        execute_approved(state, proposals, broker, prices)
        state.record_equity(ORDER_DATE, prices)

        path = tmp_path / "state.json"
        state.save(path)
        loaded = PaperTradingState.load(path)

        assert loaded.cash == pytest.approx(state.cash)
        assert set(loaded.positions) == set(state.positions)
        assert len(loaded.equity_history) == len(state.equity_history)

    def test_missing_state_raises_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="paper-init"):
            PaperTradingState.load(tmp_path / "nope.json")


class TestLiveBrokerIsBlocked:
    def test_nautilus_adapter_cannot_be_instantiated(self):
        """No path to a real broker exists in this codebase."""
        with pytest.raises(NotImplementedError, match="interface placeholder"):
            NautilusBrokerAdapter()
