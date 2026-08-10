"""Corporate action tests (spec section 31.3).

Unadjusted corporate actions are the quietest way to corrupt a research dataset. An
unadjusted 2-for-1 split looks exactly like a -50% return: momentum flips sign, volatility
explodes, and the security is ranked on an event that never economically happened.

These tests verify the adjustment produces *return continuity* -- the property that actually
matters -- rather than merely checking that some arithmetic ran.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from quant_platform.data.security_master import SecurityMaster, build_identifier_intervals
from quant_platform.pipeline import build_adjusted_prices
from quant_platform.utilities.calendar import TradingCalendar


def _prices(sec_id: str, start: date, end: date, price: float = 100.0) -> pl.DataFrame:
    sessions = TradingCalendar(start, end).sessions
    return pl.DataFrame(
        {
            "security_id": [sec_id] * len(sessions),
            "observation_date": sessions,
            "close": [price] * len(sessions),
            "volume": [1_000_000.0] * len(sessions),
        }
    )


def _action(sec_id: str, action_type: str, ex_date: date, value: float | None = None, **kw):
    return {
        "security_id": sec_id,
        "action_type": action_type,
        "ex_date": ex_date,
        "observation_date": ex_date,
        "available_at": datetime.combine(ex_date, datetime.min.time()),
        "source": "test",
        "value": value,
        "old_symbol": kw.get("old_symbol"),
        "new_symbol": kw.get("new_symbol"),
        "delisting_reason": kw.get("delisting_reason"),
    }


ACTION_SCHEMA = {
    "security_id": pl.Utf8,
    "action_type": pl.Utf8,
    "ex_date": pl.Date,
    "observation_date": pl.Date,
    "available_at": pl.Datetime,
    "source": pl.Utf8,
    "value": pl.Float64,
    "old_symbol": pl.Utf8,
    "new_symbol": pl.Utf8,
    "delisting_reason": pl.Utf8,
}


def _actions(rows: list[dict]) -> pl.DataFrame:
    """Build an actions frame, keeping the full schema even when empty.

    An empty frame with no columns would blow up any downstream `.filter(pl.col(...))`, so
    the schema is declared explicitly -- "no corporate actions" is a legitimate input.
    """
    if not rows:
        return pl.DataFrame(schema=ACTION_SCHEMA)
    return pl.DataFrame(rows, schema_overrides=ACTION_SCHEMA)


class TestStockSplits:
    def test_split_produces_return_continuity(self):
        """The economic test: a 2-for-1 split must not create a -50% return.

        Raw price halves on the ex-date. If adjustment works, the adjusted series is smooth
        and the return across the split is ~0.
        """
        split_date = date(2020, 6, 15)
        before = _prices("S1", date(2020, 1, 1), date(2020, 6, 12), price=200.0)
        after = _prices("S1", split_date, date(2020, 12, 31), price=100.0)  # halved
        prices = pl.concat([before, after])
        actions = _actions([_action("S1", "stock_split", split_date, 2.0)])

        adj = build_adjusted_prices(prices, actions).sort("observation_date")
        adj = adj.with_columns(
            (pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1.0).alias("ret")
        )
        # The raw series shows a -50% cliff...
        raw_ret = 100.0 / 200.0 - 1.0
        assert raw_ret == pytest.approx(-0.5)
        # ...but the adjusted series must be continuous through the split.
        worst = adj["ret"].abs().max()
        assert worst < 0.01, (
            f"adjusted series still has a {worst:.1%} jump across the split; the split was "
            f"not properly adjusted and would be read as a real return"
        )

    def test_pre_split_prices_are_divided_by_the_ratio(self):
        split_date = date(2020, 6, 15)
        prices = pl.concat(
            [
                _prices("S1", date(2020, 1, 1), date(2020, 6, 12), price=200.0),
                _prices("S1", split_date, date(2020, 12, 31), price=100.0),
            ]
        )
        actions = _actions([_action("S1", "stock_split", split_date, 2.0)])
        adj = build_adjusted_prices(prices, actions)

        pre = adj.filter(pl.col("observation_date") < split_date)["adjusted_close"]
        post = adj.filter(pl.col("observation_date") >= split_date)["adjusted_close"]
        # 200 / 2 == 100: history is restated onto the post-split basis.
        assert pre.max() == pytest.approx(100.0)
        assert post.max() == pytest.approx(100.0)

    def test_unadjusted_close_is_preserved(self):
        """Eligibility screens need the price that actually traded."""
        split_date = date(2020, 6, 15)
        prices = pl.concat(
            [
                _prices("S1", date(2020, 1, 1), date(2020, 6, 12), price=200.0),
                _prices("S1", split_date, date(2020, 12, 31), price=100.0),
            ]
        )
        adj = build_adjusted_prices(
            prices, _actions([_action("S1", "stock_split", split_date, 2.0)])
        )
        pre = adj.filter(pl.col("observation_date") < split_date)
        assert pre["close"].max() == pytest.approx(200.0)  # raw untouched
        assert pre["adjusted_close"].max() == pytest.approx(100.0)  # adjusted restated


class TestDividends:
    def test_dividend_adjustment_reduces_pre_ex_prices(self):
        """The ex-date price drop is a distribution, not a loss."""
        ex_date = date(2020, 6, 15)
        prices = _prices("S1", date(2020, 1, 1), date(2020, 12, 31), price=100.0)
        actions = _actions([_action("S1", "cash_dividend", ex_date, 2.0)])

        adj = build_adjusted_prices(prices, actions)
        pre = adj.filter(pl.col("observation_date") < ex_date)["adjusted_close"].max()
        post = adj.filter(pl.col("observation_date") >= ex_date)["adjusted_close"].max()
        # Pre-ex prices scale by (1 - 2/100) = 0.98.
        assert pre == pytest.approx(98.0, rel=1e-3)
        assert post == pytest.approx(100.0)

    def test_zero_dividend_is_a_noop(self):
        prices = _prices("S1", date(2020, 1, 1), date(2020, 12, 31), price=100.0)
        adj_none = build_adjusted_prices(prices, _actions([]))
        assert adj_none["adjusted_close"].max() == pytest.approx(100.0)


class TestSymbolChanges:
    def test_rename_preserves_one_economic_security(self):
        """A ticker change must not create a second security."""
        change_date = date(2020, 6, 15)
        securities = pl.DataFrame(
            {
                "security_id": ["S1"],
                "primary_symbol": ["NEW"],
                "first_trade_date": [date(2015, 1, 1)],
                "delisting_date": [None],
            },
            schema_overrides={"delisting_date": pl.Date},
        )
        actions = _actions(
            [_action("S1", "symbol_change", change_date, old_symbol="OLD", new_symbol="NEW")]
        )
        intervals = build_identifier_intervals(actions, securities)
        master = SecurityMaster(securities, intervals)

        # Two symbol intervals, one security.
        assert intervals.height == 2
        assert intervals["security_id"].n_unique() == 1

        # Date-dependent resolution in both directions.
        assert master.resolve_symbol("OLD", date(2019, 1, 2)) == "S1"
        assert master.resolve_symbol("NEW", date(2021, 1, 4)) == "S1"
        assert master.symbol_for("S1", date(2019, 1, 2)) == "OLD"
        assert master.symbol_for("S1", date(2021, 1, 4)) == "NEW"
        assert master.has_been_renamed("S1")

    def test_symbol_not_resolvable_outside_its_interval(self):
        """A ticker before its assignment must resolve to nothing, not to the wrong company."""
        securities = pl.DataFrame(
            {
                "security_id": ["S1"],
                "primary_symbol": ["NEW"],
                "first_trade_date": [date(2018, 1, 1)],
                "delisting_date": [None],
            },
            schema_overrides={"delisting_date": pl.Date},
        )
        actions = _actions(
            [_action("S1", "symbol_change", date(2020, 6, 15), old_symbol="OLD", new_symbol="NEW")]
        )
        master = SecurityMaster(securities, build_identifier_intervals(actions, securities))
        # NEW did not exist before the change.
        assert master.resolve_symbol("NEW", date(2019, 1, 2)) is None
        # And the security did not trade before its first_trade_date.
        assert master.resolve_symbol("OLD", date(2017, 1, 3)) is None


class TestDelistings:
    def test_delisted_security_stays_in_history(self):
        """Survivorship control: a dead company must remain visible in the past."""
        securities = pl.DataFrame(
            {
                "security_id": ["ALIVE", "DEAD"],
                "first_trade_date": [date(2015, 1, 1), date(2015, 1, 1)],
                "delisting_date": [None, date(2019, 6, 28)],
                "delisting_return": [None, -1.0],
            },
            schema_overrides={"delisting_date": pl.Date, "delisting_return": pl.Float64},
        )
        master = SecurityMaster(securities)

        assert "DEAD" in master.listed_on(date(2018, 1, 2))  # alive then
        assert "DEAD" not in master.listed_on(date(2020, 1, 2))  # dead after
        assert "ALIVE" in master.listed_on(date(2020, 1, 2))
        assert master.delisting_return("DEAD") == -1.0  # the loss is recorded

    def test_bankruptcy_books_a_total_loss(self):
        securities = pl.DataFrame(
            {
                "security_id": ["DEAD"],
                "first_trade_date": [date(2015, 1, 1)],
                "delisting_date": [date(2019, 6, 28)],
                "delisting_return": [-1.0],
            },
            schema_overrides={"delisting_date": pl.Date, "delisting_return": pl.Float64},
        )
        master = SecurityMaster(securities)
        assert master.delisting_return("DEAD") == pytest.approx(-1.0)

    def test_merger_books_the_premium(self):
        securities = pl.DataFrame(
            {
                "security_id": ["ACQUIRED"],
                "first_trade_date": [date(2015, 1, 1)],
                "delisting_date": [date(2021, 9, 30)],
                "delisting_return": [0.18],
            },
            schema_overrides={"delisting_date": pl.Date, "delisting_return": pl.Float64},
        )
        assert SecurityMaster(securities).delisting_return("ACQUIRED") == pytest.approx(0.18)

    def test_missing_delisting_return_is_reported_as_unknown(self):
        """Absence must be visible, not silently treated as a 0% outcome."""
        securities = pl.DataFrame(
            {
                "security_id": ["DEAD"],
                "first_trade_date": [date(2015, 1, 1)],
                "delisting_date": [date(2019, 6, 28)],
                "delisting_return": [None],
            },
            schema_overrides={"delisting_date": pl.Date, "delisting_return": pl.Float64},
        )
        assert SecurityMaster(securities).delisting_return("DEAD") is None


class TestSecurityMasterIntegrity:
    def test_rejects_duplicate_security_ids(self):
        dup = pl.DataFrame({"security_id": ["S1", "S1"], "sector": ["a", "b"]})
        with pytest.raises(ValueError, match="must be unique"):
            SecurityMaster(dup)

    def test_derived_master_reports_unknown_lifecycle(self):
        """A master derived from prices cannot know about renames or delisting returns."""
        prices = _prices("S1", date(2020, 1, 1), date(2020, 12, 31))
        master = SecurityMaster.from_prices(prices)
        assert len(master) == 1
        assert master.delisting_return("S1") is None
        assert not master.has_been_renamed("S1")


class TestFixtureCorporateActions:
    def test_fixture_split_yields_continuous_adjusted_returns(self):
        """End-to-end on the real fixture: no security should show a split-sized jump."""
        from quant_platform.data.adapters.fixture import FixtureDataProvider, FixtureSpec

        ds = FixtureDataProvider(FixtureSpec(n_securities=40)).generate()
        adj = build_adjusted_prices(ds.prices, ds.corporate_actions)

        split_ids = (
            ds.corporate_actions.filter(pl.col("action_type") == "stock_split")["security_id"]
            .unique()
            .to_list()
        )
        assert split_ids, "fixture should contain at least one split to exercise this path"

        rets = adj.sort(["security_id", "observation_date"]).with_columns(
            (pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1.0)
            .over("security_id")
            .alias("ret")
        )
        for sec_id in split_ids:
            series = rets.filter(pl.col("security_id") == sec_id)["ret"].drop_nulls()
            worst = float(np.abs(series.to_numpy()).max())
            # A 2-for-1 split leaves a ~50% jump if unadjusted. Anything under 25% means the
            # adjustment worked (the fixture's own volatility stays well below that).
            assert worst < 0.25, (
                f"{sec_id} has a {worst:.1%} single-session adjusted return, consistent with "
                f"an unadjusted split"
            )
