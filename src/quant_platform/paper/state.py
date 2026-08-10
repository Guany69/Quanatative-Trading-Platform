"""Persistent paper-trading state and workflow (spec section 26).

Paper trading differs from backtesting in one crucial way: it runs forward in real time, one
day at a time, and cannot be re-run. State must therefore survive process restarts, and every
decision must be recorded when it is made rather than reconstructed later.

The workflow is deliberately interrupted in the middle:

    refresh -> validate -> features -> predict -> targets -> PROPOSE
                                                              |
                                                     [human approves]
                                                              |
                                                    execute -> reconcile -> monitor

Nothing crosses the approval boundary automatically. ``PaperTradingState`` records what was
proposed, what a human approved, what filled, and how the fills differed from expectations --
the last of which is the honest measure of whether the cost model is any good.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_platform.domain.portfolio import Fill
from quant_platform.execution.base import OrderProposal
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("paper")


@dataclass
class PaperPosition:
    security_id: str
    quantity: float
    average_cost: float
    last_price: float = 0.0
    opened_at: str = ""

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.last_price - self.average_cost) * self.quantity


@dataclass
class RebalanceRecord:
    """One rebalance cycle, from proposal through reconciliation."""

    rebalance_id: str
    signal_date: str
    order_date: str
    created_at: str
    model_name: str = ""
    model_version: str = ""
    feature_version: str = ""
    n_proposed: int = 0
    n_approved: int = 0
    n_filled: int = 0
    n_rejected: int = 0
    expected_cost: float = 0.0
    realized_cost: float = 0.0
    turnover: float = 0.0
    target_weights: dict[str, float] = field(default_factory=dict)
    missing_securities: list[str] = field(default_factory=list)
    approved_by: str | None = None
    notes: str = ""

    @property
    def cost_surprise(self) -> float:
        """Realized minus expected cost.

        Persistently positive means the cost model is optimistic, which would make every
        backtest built on it optimistic too.
        """
        return self.realized_cost - self.expected_cost


@dataclass
class PaperTradingState:
    """The full persistent state of a paper-trading account."""

    account_id: str = "paper-001"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    initial_capital: float = 1_000_000.0
    cash: float = 1_000_000.0
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    rebalances: list[RebalanceRecord] = field(default_factory=list)
    pending_proposals: list[dict[str, Any]] = field(default_factory=list)
    equity_history: list[dict[str, Any]] = field(default_factory=list)
    benchmark_history: list[dict[str, Any]] = field(default_factory=list)
    realized_pnl: float = 0.0
    total_costs: float = 0.0

    # ------------------------------------------------------------------ valuation
    def market_value(self, prices: dict[str, float] | None = None) -> float:
        if prices:
            self.mark_to_market(prices)
        return self.cash + sum(p.market_value for p in self.positions.values())

    def mark_to_market(self, prices: dict[str, float]) -> None:
        for sec, position in self.positions.items():
            if sec in prices and prices[sec] > 0:
                position.last_price = prices[sec]

    def current_weights(self, prices: dict[str, float] | None = None) -> dict[str, float]:
        total = self.market_value(prices)
        if total <= 0:
            return {}
        return {sec: position.market_value / total for sec, position in self.positions.items()}

    # ------------------------------------------------------------------ mutation
    def apply_fill(self, fill: Fill) -> None:
        """Update cash and positions from an execution, tracking realized P&L."""
        from quant_platform.domain.enums import OrderSide

        self.cash += fill.cash_impact
        self.total_costs += fill.costs.total

        existing = self.positions.get(fill.security_id)
        if fill.side == OrderSide.BUY:
            if existing is None:
                self.positions[fill.security_id] = PaperPosition(
                    security_id=fill.security_id,
                    quantity=fill.quantity,
                    average_cost=fill.fill_price,
                    last_price=fill.fill_price,
                    opened_at=str(fill.fill_date),
                )
            else:
                total_qty = existing.quantity + fill.quantity
                if total_qty > 0:
                    existing.average_cost = (
                        existing.average_cost * existing.quantity + fill.fill_price * fill.quantity
                    ) / total_qty
                existing.quantity = total_qty
                existing.last_price = fill.fill_price
        else:
            if existing is None:
                logger.warning("sell fill for %s with no position on record", fill.security_id)
                return
            self.realized_pnl += (fill.fill_price - existing.average_cost) * fill.quantity
            existing.quantity -= fill.quantity
            existing.last_price = fill.fill_price
            if existing.quantity <= 1e-9:
                self.positions.pop(fill.security_id, None)

        self.updated_at = datetime.now(UTC).isoformat()

    def record_equity(
        self, as_of: date, prices: dict[str, float], benchmark_value: float | None = None
    ) -> None:
        value = self.market_value(prices)
        self.equity_history.append(
            {
                "as_of": str(as_of),
                "total_value": value,
                "cash": self.cash,
                "positions_value": value - self.cash,
                "n_positions": len(self.positions),
                "realized_pnl": self.realized_pnl,
                "total_costs": self.total_costs,
            }
        )
        if benchmark_value is not None:
            self.benchmark_history.append({"as_of": str(as_of), "benchmark_value": benchmark_value})

    def stage_proposals(self, proposals: list[OrderProposal]) -> None:
        """Persist proposals awaiting approval.

        Written to disk before any human sees them, so an approval decision always refers to
        an immutable, recorded set of orders.
        """
        self.pending_proposals = [
            {
                "order_id": p.order_id,
                "security_id": p.security_id,
                "side": p.side.value,
                "quantity": p.quantity,
                "signal_date": str(p.signal_date),
                "order_date": str(p.order_date),
                "target_weight": p.target_weight,
                "current_weight": p.current_weight,
                "reference_price": p.reference_price,
                "expected_cost": p.expected_cost.total,
                "approved": p.approved,
            }
            for p in proposals
        ]
        self.updated_at = datetime.now(UTC).isoformat()

    def clear_proposals(self) -> None:
        self.pending_proposals = []

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "account_id": self.account_id,
            "created_at": self.created_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "rebalances": [asdict(r) for r in self.rebalances],
            "pending_proposals": self.pending_proposals,
            "equity_history": self.equity_history,
            "benchmark_history": self.benchmark_history,
            "realized_pnl": self.realized_pnl,
            "total_costs": self.total_costs,
        }
        # Write via a temp file and rename: an interrupted save must not leave a truncated
        # state file, which would lose the account's history.
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> PaperTradingState:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"no paper-trading state at {p}. Initialize with 'paper-init' first."
            )
        with p.open() as fh:
            data = json.load(fh)
        state = cls(
            account_id=data.get("account_id", "paper-001"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            initial_capital=data.get("initial_capital", 1_000_000.0),
            cash=data.get("cash", 0.0),
            realized_pnl=data.get("realized_pnl", 0.0),
            total_costs=data.get("total_costs", 0.0),
        )
        state.positions = {k: PaperPosition(**v) for k, v in (data.get("positions") or {}).items()}
        state.rebalances = [RebalanceRecord(**r) for r in (data.get("rebalances") or [])]
        state.pending_proposals = data.get("pending_proposals") or []
        state.equity_history = data.get("equity_history") or []
        state.benchmark_history = data.get("benchmark_history") or []
        return state

    @classmethod
    def initialize(
        cls, initial_capital: float = 1_000_000.0, account_id: str = "paper-001"
    ) -> PaperTradingState:
        return cls(account_id=account_id, initial_capital=initial_capital, cash=initial_capital)

    # ------------------------------------------------------------------ reporting
    def equity_frame(self) -> pl.DataFrame:
        if not self.equity_history:
            return pl.DataFrame()
        return pl.DataFrame(self.equity_history).with_columns(
            pl.col("as_of").str.to_date(strict=False)
        )

    def summary(self, prices: dict[str, float] | None = None) -> dict[str, Any]:
        value = self.market_value(prices)
        total_return = (value / self.initial_capital - 1.0) if self.initial_capital else 0.0
        return {
            "account_id": self.account_id,
            "total_value": value,
            "cash": self.cash,
            "positions_value": value - self.cash,
            "n_positions": len(self.positions),
            "total_return": total_return,
            "realized_pnl": self.realized_pnl,
            "total_costs": self.total_costs,
            "n_rebalances": len(self.rebalances),
            "pending_approvals": len(self.pending_proposals),
            "updated_at": self.updated_at,
        }
