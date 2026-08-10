"""Domain contract tests.

These assert that the contracts *reject* the specific mistakes that create look-ahead,
survivorship, and accounting bias. A validator that never fires is not protection, so each
test constructs the bad record and demands a failure.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from quant_platform.domain import (
    CorporateAction,
    CorporateActionType,
    DateInterval,
    Fill,
    FundamentalObservation,
    LabelObservation,
    MacroObservation,
    Order,
    OrderSide,
    OrderType,
    PriceBar,
    Security,
    TargetType,
    TransactionCost,
)


class TestPointInTimeAvailability:
    def test_rejects_information_available_before_it_existed(self):
        """available_at < observation_date is impossible and must not be constructible."""
        with pytest.raises(ValidationError, match="precedes observation_date"):
            PriceBar(
                security_id="SEC001",
                observation_date=date(2020, 6, 15),
                available_at=datetime(2020, 6, 1),  # known 2 weeks before it happened
                source="test",
                close=100.0,
                adjusted_close=100.0,
            )

    def test_is_available_at_gates_on_knowledge_date(self):
        bar = PriceBar(
            security_id="SEC001",
            observation_date=date(2020, 6, 15),
            available_at=datetime(2020, 6, 15, 21, 0),  # after the close
            source="test",
            close=100.0,
            adjusted_close=100.0,
        )
        assert not bar.is_available_at(datetime(2020, 6, 15, 9, 30))  # at the open: unknown
        assert bar.is_available_at(datetime(2020, 6, 16, 9, 30))  # next morning: known


class TestFundamentalPointInTime:
    def test_filing_cannot_predate_the_period_it_reports(self):
        with pytest.raises(ValidationError, match="precedes fiscal_period_end"):
            FundamentalObservation(
                security_id="SEC001",
                observation_date=date(2020, 12, 31),
                fiscal_period_end=date(2020, 12, 31),
                filing_date=date(2020, 11, 1),  # filed before the quarter ended
                available_at=datetime(2021, 2, 15),
                source="test",
            )

    def test_realistic_reporting_lag_is_preserved(self):
        """A Q4 filed in mid-February must not be visible in January."""
        obs = FundamentalObservation(
            security_id="SEC001",
            observation_date=date(2020, 12, 31),
            fiscal_period_end=date(2020, 12, 31),
            filing_date=date(2021, 2, 15),
            available_at=datetime(2021, 2, 15, 17, 0),
            source="test",
            net_income=1_000_000.0,
        )
        assert obs.reporting_lag_days == 46
        assert not obs.is_available_at(date(2021, 1, 20))
        assert obs.is_available_at(date(2021, 3, 1))


class TestMacroRevisions:
    def test_original_and_revision_share_period_but_differ_in_availability(self):
        """The revision is unknowable until published, even though it describes the same period."""
        original = MacroObservation(
            series_id="GDP",
            observation_date=date(2020, 3, 31),
            available_at=datetime(2020, 4, 29),
            source="test",
            revision_id=0,
            value=21_000.0,
        )
        revised = MacroObservation(
            series_id="GDP",
            observation_date=date(2020, 3, 31),
            available_at=datetime(2020, 6, 25),
            source="test",
            revision_id=1,
            is_revision=True,
            value=20_800.0,
        )
        as_of = date(2020, 5, 1)
        assert original.is_available_at(as_of)
        assert not revised.is_available_at(as_of)  # using this on 2020-05-01 would be a leak

    def test_revision_flag_requires_nonzero_revision_id(self):
        with pytest.raises(ValidationError, match="revision_id=0"):
            MacroObservation(
                series_id="GDP",
                observation_date=date(2020, 3, 31),
                available_at=datetime(2020, 4, 29),
                source="test",
                revision_id=0,
                is_revision=True,
            )


class TestPriceBarIntegrity:
    def test_rejects_high_below_low(self):
        with pytest.raises(ValidationError, match=r"high .* < low"):
            PriceBar(
                security_id="SEC001",
                observation_date=date(2020, 6, 15),
                available_at=datetime(2020, 6, 15, 21),
                source="test",
                high=95.0,
                low=105.0,
                close=100.0,
                adjusted_close=100.0,
            )

    def test_rejects_close_outside_high_low_range(self):
        with pytest.raises(ValidationError, match="exceeds high"):
            PriceBar(
                security_id="SEC001",
                observation_date=date(2020, 6, 15),
                available_at=datetime(2020, 6, 15, 21),
                source="test",
                high=105.0,
                low=95.0,
                close=120.0,
                adjusted_close=120.0,
            )

    def test_rejects_nonpositive_price(self):
        with pytest.raises(ValidationError):
            PriceBar(
                security_id="SEC001",
                observation_date=date(2020, 6, 15),
                available_at=datetime(2020, 6, 15, 21),
                source="test",
                close=0.0,
                adjusted_close=0.0,
            )


class TestSecurityIdentity:
    def test_delisted_security_remains_in_history(self):
        """Delisting must not erase the past; it bounds the tradeable window."""
        sec = Security(
            security_id="SEC_DEAD",
            first_trade_date=date(2015, 1, 2),
            delisting_date=date(2019, 6, 28),
            delisting_reason="bankruptcy",
            delisting_return=-1.0,
        )
        assert sec.is_listed_on(date(2018, 1, 2))  # tradeable while alive
        assert not sec.is_listed_on(date(2020, 1, 2))  # not after death
        assert sec.delisting_return == -1.0  # the loss is booked, not vanished

    def test_delisting_reason_requires_a_date(self):
        with pytest.raises(ValidationError, match="delisting_reason but no delisting_date"):
            Security(security_id="SEC001", delisting_reason="merger")

    def test_rejects_delisting_before_first_trade(self):
        with pytest.raises(ValidationError, match="precedes first_trade_date"):
            Security(
                security_id="SEC001",
                first_trade_date=date(2020, 1, 2),
                delisting_date=date(2019, 1, 2),
            )


class TestDateInterval:
    def test_half_open_boundary_excludes_end(self):
        iv = DateInterval(start_date=date(2020, 1, 1), end_date=date(2020, 6, 1))
        assert iv.contains(date(2020, 1, 1))  # start included
        assert not iv.contains(date(2020, 6, 1))  # end excluded
        assert not iv.contains(date(2019, 12, 31))

    def test_adjacent_intervals_do_not_overlap(self):
        """Half-open semantics let consecutive memberships abut without double-counting."""
        a = DateInterval(start_date=date(2020, 1, 1), end_date=date(2020, 6, 1))
        b = DateInterval(start_date=date(2020, 6, 1), end_date=date(2021, 1, 1))
        assert not a.overlaps(b)

    def test_rejects_inverted_interval(self):
        with pytest.raises(ValidationError, match="precedes start_date"):
            DateInterval(start_date=date(2020, 6, 1), end_date=date(2020, 1, 1))


class TestLabelWindows:
    def test_forward_window_cannot_start_in_the_past(self):
        with pytest.raises(ValidationError, match="cannot start in the past"):
            LabelObservation(
                security_id="SEC001",
                as_of=date(2020, 6, 15),
                target_type=TargetType.EXCESS_RETURN,
                value=0.02,
                horizon_sessions=20,
                window_start=date(2020, 6, 1),  # before as_of
                window_end=date(2020, 7, 15),
            )

    def test_overlap_detection_drives_purging(self):
        """A sample whose forward window reaches into the test period must be purgeable."""
        label = LabelObservation(
            security_id="SEC001",
            as_of=date(2020, 6, 15),
            target_type=TargetType.EXCESS_RETURN,
            value=0.02,
            horizon_sessions=20,
            window_start=date(2020, 6, 16),
            window_end=date(2020, 7, 14),
        )
        # Test period starts inside the label's forward window -> overlapping -> must purge.
        assert label.overlaps_window(date(2020, 7, 1), date(2020, 12, 31))
        # Test period well after the window closes -> safe to train on.
        assert not label.overlaps_window(date(2020, 8, 1), date(2020, 12, 31))


class TestCorporateActions:
    def test_split_requires_a_ratio(self):
        with pytest.raises(ValidationError, match="requires a positive value"):
            CorporateAction(
                security_id="SEC001",
                observation_date=date(2020, 6, 15),
                available_at=datetime(2020, 6, 15),
                source="test",
                action_type=CorporateActionType.STOCK_SPLIT,
                ex_date=date(2020, 6, 15),
                value=None,
            )

    def test_symbol_change_requires_new_symbol(self):
        with pytest.raises(ValidationError, match="requires new_symbol"):
            CorporateAction(
                security_id="SEC001",
                observation_date=date(2020, 6, 15),
                available_at=datetime(2020, 6, 15),
                source="test",
                action_type=CorporateActionType.SYMBOL_CHANGE,
                ex_date=date(2020, 6, 15),
                old_symbol="OLD",
            )


class TestExecutionChain:
    def test_order_cannot_precede_its_signal(self):
        with pytest.raises(ValidationError, match="cannot trade before the signal exists"):
            Order(
                order_id="O1",
                security_id="SEC001",
                side=OrderSide.BUY,
                quantity=100,
                signal_date=date(2020, 6, 15),
                order_date=date(2020, 6, 12),  # trading before deciding
            )

    def test_limit_order_requires_price(self):
        with pytest.raises(ValidationError, match="requires a limit_price"):
            Order(
                order_id="O1",
                security_id="SEC001",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
                signal_date=date(2020, 6, 15),
                order_date=date(2020, 6, 16),
            )

    def test_buy_consumes_cash_including_costs(self):
        fill = Fill(
            order_id="O1",
            security_id="SEC001",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=50.0,
            fill_date=date(2020, 6, 16),
            costs=TransactionCost(commission=1.0, spread_cost=2.0),
        )
        # 100 * 50 = 5000 out, plus 3 of costs -> -5003
        assert fill.cash_impact == pytest.approx(-5003.0)
        assert fill.signed_quantity == 100

    def test_sell_returns_cash_net_of_costs(self):
        fill = Fill(
            order_id="O2",
            security_id="SEC001",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=50.0,
            fill_date=date(2020, 6, 16),
            costs=TransactionCost(commission=1.0, spread_cost=2.0),
        )
        # 5000 in, minus 3 of costs -> +4997. Costs always subtract, regardless of side.
        assert fill.cash_impact == pytest.approx(4997.0)
        assert fill.signed_quantity == -100


class TestTransactionCostScaling:
    def test_total_sums_components(self):
        c = TransactionCost(commission=1.0, spread_cost=2.0, slippage_cost=3.0, impact_cost=4.0)
        assert c.total == pytest.approx(10.0)

    def test_scaling_supports_stress_scenarios(self):
        c = TransactionCost(commission=1.0, spread_cost=2.0, slippage_cost=3.0, impact_cost=4.0)
        assert c.scaled(3.0).total == pytest.approx(30.0)
        assert c.scaled(0.0).total == pytest.approx(0.0)  # zero-cost diagnostic

    def test_rejects_negative_multiplier(self):
        with pytest.raises(ValueError, match="non-negative"):
            TransactionCost(commission=1.0).scaled(-1.0)
