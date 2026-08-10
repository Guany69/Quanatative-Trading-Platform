"""Broker interface and simulated execution (spec section 26).

Broker-neutral by design, with one hard rule enforced in code rather than documentation:
**orders are proposals until a human approves them.** ``SimulatedBroker.submit`` refuses any
order that has not passed through ``approve()``.

That gate exists because this module is the boundary where software becomes money. Everything
before it is a research artifact; a bug here moves capital. Requiring explicit human
acknowledgement makes an automated trading accident require a deliberate act, not an
oversight.

No real broker is implemented. ``NautilusBrokerAdapter`` is an interface stub that raises on
use, so a future integration has a defined shape without any possibility of an accidental
live order today.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from quant_platform.costs.model import CompositeCostModel, TradeContext
from quant_platform.domain.enums import OrderSide, OrderStatus, OrderType
from quant_platform.domain.portfolio import Fill, Order, TransactionCost
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("execution")


class OrderNotApprovedError(RuntimeError):
    """Raised when an unapproved order is submitted for execution."""


class ExecutionTimingError(RuntimeError):
    """Raised when a proposal collapses signal and simulated fill onto one session."""


@dataclass
class OrderProposal:
    """A proposed order awaiting human review.

    Carries the reasoning (target vs current weight, expected cost) so a reviewer can judge
    it rather than rubber-stamping an opaque instruction.
    """

    order_id: str
    security_id: str
    side: OrderSide
    quantity: float
    signal_date: date
    order_date: date
    target_weight: float
    current_weight: float
    reference_price: float
    expected_cost: TransactionCost = field(default_factory=TransactionCost)
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    approved: bool = False
    approved_at: datetime | None = None
    approved_by: str | None = None
    rejection_reason: str | None = None
    rejected_at: datetime | None = None
    rejected_by: str | None = None

    @property
    def status(self) -> OrderStatus:
        if self.approved:
            return OrderStatus.APPROVED
        if self.rejection_reason is not None:
            return OrderStatus.REJECTED
        return OrderStatus.PROPOSED

    @property
    def notional(self) -> float:
        return self.quantity * self.reference_price

    @property
    def weight_change(self) -> float:
        return self.target_weight - self.current_weight

    def to_order(self) -> Order:
        return Order(
            order_id=self.order_id,
            security_id=self.security_id,
            side=self.side,
            quantity=self.quantity,
            order_type=self.order_type,
            limit_price=self.limit_price,
            signal_date=self.signal_date,
            order_date=self.order_date,
            status=OrderStatus.APPROVED if self.approved else OrderStatus.PROPOSED,
            target_weight=self.target_weight,
        )

    def describe(self) -> str:
        arrow = "->"
        return (
            f"{self.side.value.upper():4s} {self.security_id:10s} "
            f"{self.quantity:>12,.0f} sh @ ~{self.reference_price:8.2f} "
            f"= {self.notional:>14,.0f}  "
            f"weight {self.current_weight:.3%} {arrow} {self.target_weight:.3%}  "
            f"est. cost {self.expected_cost.total:,.0f}"
        )


class BrokerAdapter(ABC):
    """Interface every execution venue must implement."""

    name: str = "base"
    is_live: bool = False

    @abstractmethod
    def submit(self, proposal: OrderProposal, market_price: float) -> Fill | None: ...

    @abstractmethod
    def account_value(self) -> float: ...


class SimulatedBroker(BrokerAdapter):
    """Fills orders against supplied market prices, applying the same cost model as the
    backtest.

    Sharing the cost model with the backtest is deliberate: if paper trading used friendlier
    assumptions, paper results would diverge from backtest results for reasons that have
    nothing to do with the market.
    """

    name = "simulated"
    is_live = False

    def __init__(
        self,
        cost_model: CompositeCostModel,
        cash: float = 1_000_000.0,
        allow_partial_fills: bool = True,
        max_participation: float = 0.02,
    ) -> None:
        self.cost_model = cost_model
        self.cash = cash
        self.positions: dict[str, float] = {}
        self.allow_partial_fills = allow_partial_fills
        self.max_participation = max_participation
        self.fills: list[Fill] = []
        self.rejected: list[tuple[OrderProposal, str]] = []

    def submit(
        self,
        proposal: OrderProposal,
        market_price: float,
        adv_dollar: float | None = None,
        volatility: float | None = None,
    ) -> Fill | None:
        """Execute an APPROVED order. Refuses anything else."""
        if not proposal.approved:
            raise OrderNotApprovedError(
                f"order {proposal.order_id} ({proposal.security_id}) has not been approved. "
                f"Paper-trading orders remain proposals until a human approves them; call "
                f"approve() first."
            )
        if proposal.order_date <= proposal.signal_date:
            raise ExecutionTimingError(
                f"order {proposal.order_id} would fill on {proposal.order_date}, not later than "
                f"its signal date {proposal.signal_date}; paper fills require >= 1 session delay"
            )
        if market_price <= 0:
            self.rejected.append((proposal, "no valid market price"))
            return None

        quantity = proposal.quantity

        # Liquidity limit: a fill larger than a plausible share of daily volume is not
        # achievable, so it is truncated (or rejected) rather than granted.
        if adv_dollar and adv_dollar > 0:
            max_notional = adv_dollar * self.max_participation
            requested = quantity * market_price
            if requested > max_notional:
                if not self.allow_partial_fills:
                    self.rejected.append((proposal, "exceeds ADV participation limit"))
                    return None
                quantity = max_notional / market_price
                logger.warning(
                    "%s: partial fill %.0f of %.0f shares (ADV participation limit)",
                    proposal.security_id,
                    quantity,
                    proposal.quantity,
                )

        ctx = TradeContext(
            notional=quantity * market_price,
            price=market_price,
            shares=quantity,
            adv_dollar=adv_dollar,
            volatility=volatility,
            is_buy=proposal.side == OrderSide.BUY,
        )
        costs = self.cost_model.compute(ctx)
        # The Fill contract treats fill_price as the RAW transacted price and accounts for
        # every friction separately in `costs` (see Fill.cash_impact). Passing the
        # cost-adjusted effective price here instead would charge spread, slippage, and impact
        # twice -- once inside the price and once again via costs.total -- which overdraws
        # cash on a fully-invested rebalance. effective_price() remains available to callers
        # that want the achieved per-share price for reporting.
        fill_price = market_price

        if proposal.side == OrderSide.BUY:
            # Reserve the FULL cost, not just commission: all frictions are paid in cash.
            required = quantity * fill_price + costs.total
            if required > self.cash:
                if not self.allow_partial_fills:
                    self.rejected.append((proposal, "insufficient cash"))
                    return None
                # Scale the order down to what cash actually covers, leaving a small buffer
                # so the recomputed cost on the smaller trade still fits.
                affordable = max(0.0, (self.cash - costs.total) / fill_price) * 0.999
                if affordable < 1:
                    self.rejected.append((proposal, "insufficient cash"))
                    return None
                quantity = affordable
                # Costs scale with size, so recompute them for the reduced order.
                ctx = TradeContext(
                    notional=quantity * market_price,
                    price=market_price,
                    shares=quantity,
                    adv_dollar=adv_dollar,
                    volatility=volatility,
                    is_buy=True,
                )
                costs = self.cost_model.compute(ctx)
        else:
            held = self.positions.get(proposal.security_id, 0.0)
            if quantity > held:
                quantity = held  # no shorting in the long-only MVP
                if quantity <= 0:
                    self.rejected.append((proposal, "no position to sell"))
                    return None

        fill = Fill(
            order_id=proposal.order_id,
            security_id=proposal.security_id,
            side=proposal.side,
            quantity=quantity,
            fill_price=fill_price,
            fill_date=proposal.order_date,
            reference_price=market_price,
            costs=costs,
            is_partial=quantity < proposal.quantity - 1e-9,
        )

        self.cash += fill.cash_impact
        delta = quantity if proposal.side == OrderSide.BUY else -quantity
        self.positions[proposal.security_id] = self.positions.get(proposal.security_id, 0.0) + delta
        if abs(self.positions[proposal.security_id]) < 1e-9:
            self.positions.pop(proposal.security_id, None)

        self.fills.append(fill)
        return fill

    def account_value(self, prices: dict[str, float] | None = None) -> float:
        if not prices:
            return self.cash
        holdings = sum(shares * prices.get(sec, 0.0) for sec, shares in self.positions.items())
        return self.cash + holdings

    def snapshot(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "positions": dict(self.positions),
            "n_fills": len(self.fills),
            "n_rejected": len(self.rejected),
        }


class NautilusBrokerAdapter(BrokerAdapter):
    """Interface stub for a future NautilusTrader integration.

    Deliberately non-functional. It defines the shape a live adapter must satisfy while making
    it impossible to route a real order through this codebase today.
    """

    name = "nautilus"
    is_live = True

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "The NautilusTrader adapter is an interface placeholder, not an implementation. "
            "Connecting to a real broker requires: (1) implementing this adapter, "
            "(2) supplying credentials via environment variables, and (3) an explicit "
            "human decision to trade real capital. Nothing in this platform is validated "
            "for live trading."
        )

    def submit(self, proposal: OrderProposal, market_price: float) -> Fill | None:
        raise NotImplementedError

    def account_value(self) -> float:
        raise NotImplementedError


def generate_order_id() -> str:
    return f"ORD-{uuid.uuid4().hex[:10].upper()}"


def approve(
    proposals: list[OrderProposal], approver: str = "manual", order_ids: list[str] | None = None
) -> list[OrderProposal]:
    """Approve proposals for execution -- the explicit human gate.

    Passing ``order_ids`` approves a subset, so a reviewer can accept some orders and leave
    others pending rather than facing an all-or-nothing choice.
    """
    now = datetime.now()
    wanted = set(order_ids) if order_ids else None
    approved: list[OrderProposal] = []
    for p in proposals:
        if wanted is not None and p.order_id not in wanted:
            continue
        p.approved = True
        p.approved_at = now
        p.approved_by = approver
        approved.append(p)
    logger.info(
        "approved %d of %d proposed orders (by %s)", len(approved), len(proposals), approver
    )
    return approved


def reject(
    proposals: list[OrderProposal],
    approver: str,
    reason: str,
    order_ids: list[str] | None = None,
) -> list[OrderProposal]:
    """Record an explicit human rejection decision for selected proposals."""
    now = datetime.now()
    wanted = set(order_ids) if order_ids else None
    rejected: list[OrderProposal] = []
    for proposal in proposals:
        if wanted is not None and proposal.order_id not in wanted:
            continue
        proposal.approved = False
        proposal.rejection_reason = reason
        proposal.rejected_at = now
        proposal.rejected_by = approver
        rejected.append(proposal)
    return rejected
