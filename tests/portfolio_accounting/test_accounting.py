"""Portfolio accounting tests (spec section 31.4).

Accounting bugs are the most dangerous class of backtest error because they are invisible in
the output: the equity curve still looks like an equity curve, it is just wrong. A backtest
that loses track of cash can manufacture returns from nothing.

The central invariant tested here is **conservation**: every dollar is either in cash, in a
position, or was paid as a cost. Nothing appears from nowhere and nothing silently vanishes.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from quant_platform.backtest.engine import BacktestEngine
from quant_platform.config.models import BacktestConfig, CostConfig, RebalanceConfig
from quant_platform.domain.enums import OrderSide
from quant_platform.domain.portfolio import Fill, TransactionCost
from quant_platform.utilities.calendar import TradingCalendar

START = date(2020, 1, 1)
END = date(2020, 12, 31)


def _calendar() -> TradingCalendar:
    return TradingCalendar(START, END)


def _prices(security_ids: list[str], price: float = 100.0, drift: float = 0.0) -> pl.DataFrame:
    sessions = _calendar().sessions
    rows = []
    for sid in security_ids:
        for i, s in enumerate(sessions):
            rows.append(
                {
                    "security_id": sid,
                    "observation_date": s,
                    "adjusted_close": price * (1 + drift) ** i,
                    "close": price * (1 + drift) ** i,
                    "dollar_volume": 5e8,
                    "volume": 5e6,
                }
            )
    return pl.DataFrame(rows)


def _benchmark(drift: float = 0.0) -> pl.DataFrame:
    sessions = _calendar().sessions
    level = [100.0 * (1 + drift) ** i for i in range(len(sessions))]
    return pl.DataFrame(
        {
            "observation_date": sessions,
            "adjusted_close": level,
            "close": level,
            "benchmark_return": [0.0] + [drift] * (len(sessions) - 1),
        }
    )


def _securities(security_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": security_ids,
            "delisting_date": [None] * len(security_ids),
            "delisting_return": [None] * len(security_ids),
        },
        schema_overrides={"delisting_date": pl.Date, "delisting_return": pl.Float64},
    )


def _engine(cost_scenario: str = "base", delay: int = 1) -> BacktestEngine:
    return BacktestEngine(
        BacktestConfig(
            initial_capital=1_000_000.0,
            rebalance=RebalanceConfig(execution_delay_sessions=delay),
        ),
        CostConfig(),
        _calendar(),
        cost_scenario=cost_scenario,
    )


class TestCashConservation:
    def test_no_trading_preserves_capital_exactly(self):
        """With no targets, the account must hold its starting cash to the cent."""
        engine = _engine()
        result = engine.run({}, _prices(["A"]), _benchmark(), _securities(["A"]))
        final = result.snapshots[-1]
        assert final.total_value == pytest.approx(1_000_000.0, abs=1e-6)
        assert final.cash == pytest.approx(1_000_000.0, abs=1e-6)
        assert not result.trades

    def test_zero_cost_flat_market_conserves_value(self):
        """Flat prices and zero costs: value must be unchanged after a full rebalance."""
        engine = _engine(cost_scenario="zero")
        sessions = _calendar().sessions
        targets = {sessions[10]: {"A": 0.5, "B": 0.5}}
        result = engine.run(targets, _prices(["A", "B"]), _benchmark(), _securities(["A", "B"]))
        final = result.snapshots[-1]
        assert result.trades, "expected trades to be generated"
        assert final.total_value == pytest.approx(1_000_000.0, rel=1e-9)

    def test_costs_reduce_value_by_exactly_the_costs_charged(self):
        """The conservation identity: final value == initial - total costs (flat market)."""
        engine = _engine(cost_scenario="base")
        sessions = _calendar().sessions
        targets = {sessions[10]: {"A": 0.5, "B": 0.5}}
        result = engine.run(targets, _prices(["A", "B"]), _benchmark(), _securities(["A", "B"]))
        final = result.snapshots[-1]
        assert result.total_costs > 0
        assert final.total_value == pytest.approx(1_000_000.0 - result.total_costs, rel=1e-6), (
            "value change does not equal the costs charged; cash is leaking"
        )

    def test_position_value_plus_cash_equals_total(self):
        """Every snapshot must satisfy total = cash + positions."""
        engine = _engine()
        sessions = _calendar().sessions
        targets = {sessions[10]: {"A": 0.4, "B": 0.4}}
        result = engine.run(targets, _prices(["A", "B"]), _benchmark(), _securities(["A", "B"]))
        for snap in result.snapshots:
            assert snap.total_value == pytest.approx(snap.cash + snap.positions_value, abs=1e-6)

    def test_cash_never_goes_negative_in_long_only(self):
        engine = _engine()
        sessions = _calendar().sessions
        targets = {sessions[10]: {"A": 0.5, "B": 0.5}}
        result = engine.run(targets, _prices(["A", "B"]), _benchmark(), _securities(["A", "B"]))
        worst = min(s.cash for s in result.snapshots)
        assert worst >= -1.0, f"cash went to {worst:,.2f}; leverage was implied"


class TestExecutionDelay:
    def test_fill_happens_after_the_signal_never_on_it(self):
        """The core anti-look-ahead property of the engine."""
        engine = _engine(delay=1)
        sessions = _calendar().sessions
        signal_day = sessions[10]
        result = engine.run(
            {signal_day: {"A": 1.0}}, _prices(["A"]), _benchmark(), _securities(["A"])
        )
        assert result.trades
        for trade in result.trades:
            assert trade.fill_date > trade.signal_date, (
                f"trade filled on {trade.fill_date} from a signal dated "
                f"{trade.signal_date}: this is trading at a price the signal already saw"
            )
            assert trade.order_date > trade.signal_date
            assert trade.fill_date >= trade.order_date
            assert trade.accounting_date >= trade.fill_date
            assert trade.fill_date == sessions[11]

    def test_longer_delay_pushes_fills_further_out(self):
        sessions = _calendar().sessions
        signal_day = sessions[10]
        for delay in (1, 2, 3):
            engine = _engine(delay=delay)
            result = engine.run(
                {signal_day: {"A": 1.0}}, _prices(["A"]), _benchmark(), _securities(["A"])
            )
            assert result.trades[0].fill_date == sessions[10 + delay]

    def test_zero_delay_is_rejected(self):
        with pytest.raises(Exception):
            BacktestEngine(
                BacktestConfig(rebalance=RebalanceConfig(execution_delay_sessions=0)),
                CostConfig(),
                _calendar(),
            )


class TestCostsAndGrossNet:
    def test_gross_and_net_differ_when_costs_apply(self):
        engine = _engine(cost_scenario="base")
        sessions = _calendar().sessions
        targets = {s: {"A": 0.5, "B": 0.5} for s in sessions[10:60:5]}
        result = engine.run(
            targets, _prices(["A", "B"], drift=0.0003), _benchmark(0.0002), _securities(["A", "B"])
        )
        curve = result.equity_curve()
        gross = np.nansum(curve["gross_return"].to_numpy())
        net = np.nansum(curve["net_return"].to_numpy())
        assert gross > net, "gross must exceed net when costs are charged"

    def test_higher_cost_scenarios_reduce_returns_monotonically(self):
        sessions = _calendar().sessions
        targets = {s: {"A": 0.5, "B": 0.5} for s in sessions[10:80:5]}
        finals = {}
        for scenario in ("zero", "base", "double", "triple"):
            engine = _engine(cost_scenario=scenario)
            result = engine.run(targets, _prices(["A", "B"]), _benchmark(), _securities(["A", "B"]))
            finals[scenario] = result.snapshots[-1].total_value
        assert finals["zero"] > finals["base"] > finals["double"] > finals["triple"], (
            f"cost scenarios are not monotonic: {finals}"
        )

    def test_zero_cost_scenario_charges_nothing(self):
        engine = _engine(cost_scenario="zero")
        sessions = _calendar().sessions
        result = engine.run(
            {sessions[10]: {"A": 1.0}}, _prices(["A"]), _benchmark(), _securities(["A"])
        )
        assert result.total_costs == pytest.approx(0.0, abs=1e-9)


class TestDividendsAndCorporateActions:
    def test_dividends_are_credited_to_cash(self):
        sessions = _calendar().sessions
        ex_date = sessions[30]
        actions = pl.DataFrame(
            {
                "security_id": ["A"],
                "action_type": ["cash_dividend"],
                "ex_date": [ex_date],
                "value": [2.0],
            }
        )
        engine = _engine(cost_scenario="zero")
        result = engine.run(
            {sessions[10]: {"A": 1.0}},
            _prices(["A"]),
            _benchmark(),
            _securities(["A"]),
            dividends=actions,
        )
        received = sum(s.dividends_received for s in result.snapshots)
        # ~1,000,000 / 100 = 10,000 shares * $2 = ~$20,000.
        assert received > 15_000, f"dividends of only {received:,.0f} were credited"

    def test_delisting_books_a_total_loss(self):
        """A bankruptcy must destroy the position's value, not silently drop it."""
        sessions = _calendar().sessions
        delist_day = sessions[60]
        securities = pl.DataFrame(
            {
                "security_id": ["A", "DEAD"],
                "delisting_date": [None, delist_day],
                "delisting_return": [None, -1.0],
            },
            schema_overrides={"delisting_date": pl.Date, "delisting_return": pl.Float64},
        )
        engine = _engine(cost_scenario="zero")
        result = engine.run(
            {sessions[10]: {"A": 0.5, "DEAD": 0.5}},
            _prices(["A", "DEAD"]),
            _benchmark(),
            securities,
        )
        final = result.snapshots[-1]
        # Half the book was wiped out, so the account must be worth roughly half.
        assert final.total_value < 600_000, (
            f"final value {final.total_value:,.0f}: the -100% delisting was not booked"
        )
        assert final.total_value > 400_000


class TestFillAccounting:
    def test_buy_cash_impact_includes_costs(self):
        fill = Fill(
            order_id="O1",
            security_id="A",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=50.0,
            fill_date=date(2020, 6, 15),
            costs=TransactionCost(commission=5.0, spread_cost=10.0),
        )
        assert fill.cash_impact == pytest.approx(-(100 * 50.0) - 15.0)

    def test_sell_cash_impact_subtracts_costs(self):
        fill = Fill(
            order_id="O1",
            security_id="A",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=50.0,
            fill_date=date(2020, 6, 15),
            costs=TransactionCost(commission=5.0, spread_cost=10.0),
        )
        assert fill.cash_impact == pytest.approx(100 * 50.0 - 15.0)

    def test_turnover_is_bounded(self):
        engine = _engine()
        sessions = _calendar().sessions
        targets = {s: {"A": 0.5, "B": 0.5} for s in sessions[10:40:5]}
        result = engine.run(targets, _prices(["A", "B"]), _benchmark(), _securities(["A", "B"]))
        for snap in result.snapshots:
            assert 0.0 <= snap.turnover <= 2.0, f"implausible turnover {snap.turnover}"


class TestBenchmarkAlignment:
    def test_every_snapshot_has_a_benchmark_return(self):
        engine = _engine()
        result = engine.run({}, _prices(["A"]), _benchmark(0.0002), _securities(["A"]))
        assert all(s.benchmark_return is not None for s in result.snapshots)

    def test_snapshot_dates_are_ordered_and_unique(self):
        engine = _engine()
        result = engine.run({}, _prices(["A"]), _benchmark(), _securities(["A"]))
        dates = [s.as_of for s in result.snapshots]
        assert dates == sorted(dates)
        assert len(dates) == len(set(dates))
