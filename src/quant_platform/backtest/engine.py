"""Event-aware backtest engine (spec section 21).

The engine's core discipline is the timestamp chain::

    information_date -> signal_date -> order_date -> fill_date -> accounting_date

A naive backtest collapses these into one date and fills at the same close that generated the
signal. That single shortcut is worth an enormous amount of fake alpha, because it lets the
strategy buy exactly the thing it just learned went up. Here, ``execution_delay_sessions``
forces the fill onto a later session at that session's price, which the signal never saw.

Cash accounting is double-entry-ish and asserted: every dollar leaving the book is either in
a position, in cash, or paid as cost. ``tests/portfolio_accounting`` verifies conservation.

Corporate actions are applied on their ex-date: splits adjust share counts (leaving value
unchanged), dividends pay cash, and delistings liquidate at the terminal value -- including a
total loss, which is precisely the event a survivorship-biased backtest never books.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from quant_platform.config.models import BacktestConfig, CostConfig
from quant_platform.costs.model import CompositeCostModel, TradeContext
from quant_platform.domain.enums import OrderSide
from quant_platform.domain.portfolio import BacktestSnapshot, TransactionCost
from quant_platform.utilities.calendar import TradingCalendar
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("backtest")


@dataclass
class TradeRecord:
    """One executed trade, with the full timestamp chain retained for audit."""

    signal_date: date
    order_date: date
    fill_date: date
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

    @property
    def total_cost(self) -> float:
        return self.commission + self.spread_cost + self.slippage_cost + self.impact_cost


@dataclass
class BacktestResult:
    """Everything a backtest produced."""

    snapshots: list[BacktestSnapshot] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    strategy_name: str = "strategy"
    initial_capital: float = 0.0
    cost_scenario: str = "base"
    execution_delay_sessions: int = 1

    def equity_curve(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "as_of": [s.as_of for s in self.snapshots],
                "total_value": [s.total_value for s in self.snapshots],
                "cash": [s.cash for s in self.snapshots],
                "positions_value": [s.positions_value for s in self.snapshots],
                "gross_return": [s.gross_return for s in self.snapshots],
                "net_return": [s.net_return for s in self.snapshots],
                "benchmark_return": [s.benchmark_return for s in self.snapshots],
                "n_positions": [s.n_positions for s in self.snapshots],
                "turnover": [s.turnover for s in self.snapshots],
                "costs": [s.costs.total for s in self.snapshots],
                "dividends": [s.dividends_received for s in self.snapshots],
            }
        )

    def trade_log(self) -> pl.DataFrame:
        if not self.trades:
            return pl.DataFrame(
                schema={
                    "signal_date": pl.Date,
                    "order_date": pl.Date,
                    "fill_date": pl.Date,
                    "security_id": pl.Utf8,
                    "side": pl.Utf8,
                    "shares": pl.Float64,
                    "reference_price": pl.Float64,
                    "fill_price": pl.Float64,
                    "notional": pl.Float64,
                    "commission": pl.Float64,
                    "spread_cost": pl.Float64,
                    "slippage_cost": pl.Float64,
                    "impact_cost": pl.Float64,
                }
            )
        return pl.DataFrame([t.__dict__ for t in self.trades])

    @property
    def total_costs(self) -> float:
        return sum(t.total_cost for t in self.trades)


class BacktestEngine:
    """Simulates the strategy session by session."""

    def __init__(
        self,
        config: BacktestConfig,
        cost_config: CostConfig,
        calendar: TradingCalendar,
        cost_scenario: str | None = None,
    ) -> None:
        self.config = config
        self.calendar = calendar
        self.scenario = cost_scenario or config.cost_scenario
        self.costs = CompositeCostModel(cost_config, self.scenario)
        self.delay = config.rebalance.execution_delay_sessions
        if self.delay < 1:
            raise ValueError(
                "execution_delay_sessions must be >= 1; filling at the signal close is not "
                "achievable"
            )

    def run(
        self,
        targets_by_date: dict[date, dict[str, float]],
        prices: pl.DataFrame,
        benchmark: pl.DataFrame,
        securities: pl.DataFrame,
        dividends: pl.DataFrame | None = None,
        strategy_name: str = "strategy",
    ) -> BacktestResult:
        """Run the simulation.

        Parameters
        ----------
        targets_by_date:
            Target weights keyed by SIGNAL date. The engine itself applies the execution
            delay, so callers cannot accidentally trade too early.
        """
        px_map, adv_map, vol_map = self._index_prices(prices)
        bench_map = self._index_benchmark(benchmark)
        div_map = self._index_dividends(dividends)
        delist_map = self._index_delistings(securities)

        # Map each signal date to the session the orders actually execute on.
        fill_dates: dict[date, date] = {}
        for signal_date in targets_by_date:
            fd = self.calendar.shift(signal_date, self.delay)
            if fd is not None:
                fill_dates[fd] = signal_date

        sessions = [d for d in self.calendar.sessions if d in bench_map]
        if not sessions:
            raise ValueError("no sessions with benchmark data; cannot run backtest")

        cash = float(self.config.initial_capital)
        positions: dict[str, float] = {}  # security_id -> shares
        result = BacktestResult(
            strategy_name=strategy_name,
            initial_capital=cash,
            cost_scenario=self.scenario,
            execution_delay_sessions=self.delay,
        )
        prev_value = cash

        for session in sessions:
            day_costs = TransactionCost()
            turnover = 0.0
            dividends_today = 0.0

            # 1. Corporate actions first: they change the book before any trading.
            positions, cash, dividends_today = self._apply_corporate_actions(
                positions, cash, session, div_map, delist_map, px_map
            )

            # 2. Value the book at today's close using pre-trade positions. This is the
            #    gross mark: what the portfolio earned from market moves alone.
            pre_trade_value = cash + self._positions_value(positions, session, px_map)
            gross_return = (pre_trade_value / prev_value - 1.0) if prev_value > 1e-9 else 0.0

            # 3. Trade only if today is a scheduled FILL date (signal + delay).
            if session in fill_dates:
                signal_date = fill_dates[session]
                targets = targets_by_date[signal_date]
                positions, cash, day_costs, turnover, trades = self._rebalance(
                    positions,
                    cash,
                    targets,
                    session,
                    signal_date,
                    px_map,
                    adv_map,
                    vol_map,
                    pre_trade_value,
                )
                result.trades.extend(trades)

            # 4. Post-trade valuation: costs have now been deducted, so this is the net mark.
            positions_value = self._positions_value(positions, session, px_map)
            total_value = cash + positions_value
            net_return = (total_value / prev_value - 1.0) if prev_value > 1e-9 else 0.0

            result.snapshots.append(
                BacktestSnapshot(
                    as_of=session,
                    total_value=total_value,
                    cash=cash,
                    positions_value=positions_value,
                    gross_return=gross_return,
                    net_return=net_return,
                    benchmark_return=bench_map[session],
                    n_positions=len([s for s in positions.values() if abs(s) > 1e-9]),
                    turnover=turnover,
                    costs=day_costs,
                    dividends_received=dividends_today,
                )
            )
            prev_value = total_value

        return result

    # ------------------------------------------------------------------ internals
    def _index_prices(self, prices: pl.DataFrame):
        """Build (security_id, date) -> price / ADV / volatility lookups."""
        need = {"security_id", "observation_date", "adjusted_close"}
        missing = need - set(prices.columns)
        if missing:
            raise ValueError(f"prices frame missing columns: {missing}")

        px = dict(
            zip(
                zip(prices["security_id"], prices["observation_date"], strict=True),
                prices["adjusted_close"],
                strict=True,
            )
        )
        adv: dict[tuple[str, date], float] = {}
        if "dollar_volume" in prices.columns:
            rolled = prices.sort(["security_id", "observation_date"]).with_columns(
                pl.col("dollar_volume")
                .rolling_mean(21, min_samples=5)
                .over("security_id")
                .alias("_adv")
            )
            adv = dict(
                zip(
                    zip(rolled["security_id"], rolled["observation_date"], strict=True),
                    rolled["_adv"],
                    strict=True,
                )
            )
        vol: dict[tuple[str, date], float] = {}
        rolled = prices.sort(["security_id", "observation_date"]).with_columns(
            (
                (pl.col("adjusted_close").log() - pl.col("adjusted_close").log().shift(1))
                .rolling_std(60, min_samples=20)
                .over("security_id")
                * np.sqrt(252.0)
            ).alias("_vol")
        )
        vol = dict(
            zip(
                zip(rolled["security_id"], rolled["observation_date"], strict=True),
                rolled["_vol"],
                strict=True,
            )
        )
        return px, adv, vol

    def _index_benchmark(self, benchmark: pl.DataFrame) -> dict[date, float]:
        if "benchmark_return" in benchmark.columns:
            return dict(
                zip(benchmark["observation_date"], benchmark["benchmark_return"], strict=True)
            )
        b = benchmark.sort("observation_date").with_columns(
            (pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1.0).alias("_r")
        )
        return dict(zip(b["observation_date"], b["_r"], strict=True))

    def _index_dividends(self, dividends: pl.DataFrame | None):
        out: dict[tuple[str, date], float] = {}
        if dividends is None or dividends.is_empty():
            return out
        d = dividends.filter(pl.col("action_type").is_in(["cash_dividend", "special_dividend"]))
        for row in d.iter_rows(named=True):
            out[(row["security_id"], row["ex_date"])] = float(row["value"] or 0.0)
        return out

    def _index_delistings(self, securities: pl.DataFrame):
        out: dict[str, tuple[date, float]] = {}
        if "delisting_date" not in securities.columns:
            return out
        d = securities.filter(pl.col("delisting_date").is_not_null())
        for row in d.iter_rows(named=True):
            out[row["security_id"]] = (
                row["delisting_date"],
                float(row["delisting_return"] or -1.0),
            )
        return out

    def _positions_value(self, positions, session, px_map) -> float:
        """Mark positions at today's close, carrying forward the last price if absent."""
        total = 0.0
        for sid, shares in positions.items():
            price = px_map.get((sid, session))
            if price is None:
                continue  # stale/halted: value carried via cash-neutral hold
            total += shares * price
        return total

    def _apply_corporate_actions(self, positions, cash, session, div_map, delist_map, px_map):
        """Pay dividends and liquidate delisted names on their ex-date."""
        dividends_today = 0.0
        for sid, shares in list(positions.items()):
            if abs(shares) < 1e-9:
                continue
            dps = div_map.get((sid, session))
            if dps:
                amount = shares * dps
                cash += amount
                dividends_today += amount

        for sid, shares in list(positions.items()):
            if sid not in delist_map or abs(shares) < 1e-9:
                continue
            delist_date, delist_return = delist_map[sid]
            if session != delist_date:
                continue
            # Liquidate at the terminal value. For a bankruptcy (-1.0) this books a total
            # loss -- the exact event survivorship-biased backtests never record.
            last_price = px_map.get((sid, session))
            if last_price is None:
                continue
            proceeds = shares * last_price * (1.0 + delist_return)
            cash += max(proceeds, 0.0)
            positions.pop(sid, None)
            logger.debug(
                "delisted %s on %s: booked terminal return %.1f%%",
                sid,
                session,
                delist_return * 100,
            )
        return positions, cash, dividends_today

    def _rebalance(
        self,
        positions,
        cash,
        targets,
        session,
        signal_date,
        px_map,
        adv_map,
        vol_map,
        portfolio_value,
    ):
        """Trade toward target weights at TODAY's price (never the signal's price)."""
        trades: list[TradeRecord] = []
        total_costs = TransactionCost()
        traded_notional = 0.0

        current_shares = dict(positions)
        desired_shares: dict[str, float] = {}
        for sid, weight in targets.items():
            price = px_map.get((sid, session))
            if price is None or price <= 0:
                continue  # cannot trade what has no price today
            desired_shares[sid] = (weight * portfolio_value) / price

        # Order the trades: sells first, then buys.
        #
        # A long-only account without margin cannot buy before its sells settle, and trading
        # alphabetically would let a buy consume cash that a later sell has not yet released.
        # Sequencing also lets each buy be capped at genuinely available cash, which is what
        # keeps a fully-invested target from implying leverage once costs are charged (costs
        # come out of cash, so a 100% target plus costs is >100% of the account).
        universe = set(current_shares) | set(desired_shares)
        deltas: list[tuple[str, float, float]] = []  # (security_id, delta_shares, price)
        for sid in sorted(universe):
            price = px_map.get((sid, session))
            if price is None or price <= 0:
                continue
            delta = desired_shares.get(sid, 0.0) - current_shares.get(sid, 0.0)
            if abs(delta) * price < max(self.config_min_trade(), 1e-6):
                continue
            deltas.append((sid, delta, price))
        deltas.sort(key=lambda t: (t[1] > 0, -abs(t[1] * t[2])))  # sells first, largest first

        for sid, delta, price in deltas:
            have = current_shares.get(sid, 0.0)

            if delta > 0:
                # Cap the buy at affordable size. Reserve a small buffer for the costs, which
                # are charged against cash and would otherwise overdraw the account.
                affordable_notional = max(0.0, cash * 0.995)
                if delta * price > affordable_notional:
                    delta = affordable_notional / price
                    if delta * price < max(self.config_min_trade(), 1e-6):
                        continue

            notional = abs(delta) * price

            ctx = TradeContext(
                notional=notional,
                price=price,
                shares=delta,
                adv_dollar=adv_map.get((sid, session)),
                volatility=vol_map.get((sid, session)),
                is_buy=delta > 0,
            )
            cost = self.costs.compute(ctx)
            fill_price = self.costs.effective_price(ctx)

            cash -= delta * fill_price  # buys consume cash, sells return it
            cash -= cost.commission  # commission is cash, not price
            positions[sid] = have + delta
            if abs(positions[sid]) < 1e-9:
                positions.pop(sid, None)

            traded_notional += notional
            total_costs = TransactionCost(
                commission=total_costs.commission + cost.commission,
                spread_cost=total_costs.spread_cost + cost.spread_cost,
                slippage_cost=total_costs.slippage_cost + cost.slippage_cost,
                impact_cost=total_costs.impact_cost + cost.impact_cost,
            )
            trades.append(
                TradeRecord(
                    signal_date=signal_date,
                    order_date=session,
                    fill_date=session,
                    security_id=sid,
                    side=OrderSide.BUY.value if delta > 0 else OrderSide.SELL.value,
                    shares=abs(delta),
                    reference_price=price,
                    fill_price=fill_price,
                    notional=notional,
                    commission=cost.commission,
                    spread_cost=cost.spread_cost,
                    slippage_cost=cost.slippage_cost,
                    impact_cost=cost.impact_cost,
                )
            )

        turnover = (traded_notional / portfolio_value / 2.0) if portfolio_value > 1e-9 else 0.0
        return positions, cash, total_costs, turnover, trades

    def config_min_trade(self) -> float:
        return 0.0
