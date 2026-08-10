"""Point-in-time universe leakage tests (spec section 11).

Each test encodes a specific way a backtest lies to itself about the past.
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from quant_platform.config.models import UniverseConfig
from quant_platform.universe.builder import UniverseBuilder


def _prices(security_ids: list[str], start: date, end: date, close: float = 100.0) -> pl.DataFrame:
    """Dense daily bars, liquid enough to pass screens."""
    from quant_platform.utilities.calendar import TradingCalendar

    sessions = TradingCalendar(start, end).sessions
    rows = []
    for sid in security_ids:
        for s in sessions:
            rows.append(
                {
                    "security_id": sid,
                    "observation_date": s,
                    "close": close,
                    "dollar_volume": 50_000_000.0,
                }
            )
    return pl.DataFrame(rows)


def _securities(ids: list[str], **overrides) -> pl.DataFrame:
    rows = []
    for sid in ids:
        row = {
            "security_id": sid,
            "security_type": "common_stock",
            "sector": "financials",
            "first_trade_date": date(2015, 1, 1),
            "delisting_date": None,
        }
        row.update(overrides.get(sid, {}))
        rows.append(row)
    return pl.DataFrame(
        rows, schema_overrides={"delisting_date": pl.Date, "first_trade_date": pl.Date}
    )


def _membership(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema_overrides={
            "start_date": pl.Date,
            "end_date": pl.Date,
            "available_at": pl.Datetime,
        },
    )


CFG = UniverseConfig(
    min_price=5.0,
    min_dollar_volume=1_000_000.0,
    min_history_sessions=20,
    dollar_volume_lookback_sessions=21,
)


class TestFutureMembersExcluded:
    def test_member_joining_later_is_absent_today(self):
        """A security that joins the index in 2020 must not appear in the 2018 universe."""
        secs = _securities(["A", "B"])
        mem = _membership(
            [
                {
                    "security_id": "A",
                    "universe": "sp500",
                    "start_date": date(2015, 1, 1),
                    "end_date": date(9999, 12, 31),
                    "available_at": datetime(2015, 1, 1),
                    "is_survivorship_biased": False,
                },
                {
                    "security_id": "B",
                    "universe": "sp500",
                    "start_date": date(2020, 1, 2),  # joins later
                    "end_date": date(9999, 12, 31),
                    "available_at": datetime(2019, 12, 20),
                    "is_survivorship_biased": False,
                },
            ]
        )
        px = _prices(["A", "B"], date(2015, 1, 1), date(2021, 12, 31))
        b = UniverseBuilder(CFG, secs, mem, px)

        assert b.build(date(2018, 6, 15)).security_ids == ["A"]  # B not yet a member
        assert set(b.build(date(2021, 6, 15)).security_ids) == {"A", "B"}

    def test_announcement_date_gates_membership(self):
        """A pending addition is not tradeable until the change is public."""
        secs = _securities(["A"])
        mem = _membership(
            [
                {
                    "security_id": "A",
                    "universe": "sp500",
                    "start_date": date(2020, 6, 22),  # effective
                    "end_date": date(9999, 12, 31),
                    "available_at": datetime(2020, 6, 15),  # announced a week earlier
                    "is_survivorship_biased": False,
                }
            ]
        )
        px = _prices(["A"], date(2019, 1, 1), date(2020, 12, 31))
        b = UniverseBuilder(CFG, secs, mem, px)

        # Before announcement AND before effective: absent.
        assert b.members_on(date(2020, 6, 10)) == set()
        # After announcement but before effective: still not a member (interval governs).
        assert b.members_on(date(2020, 6, 18)) == set()
        # On/after effective: present.
        assert b.members_on(date(2020, 6, 23)) == {"A"}


class TestRemovedMembersDisappear:
    def test_removed_member_gone_after_effective_date(self):
        secs = _securities(["A"])
        mem = _membership(
            [
                {
                    "security_id": "A",
                    "universe": "sp500",
                    "start_date": date(2015, 1, 1),
                    "end_date": date(2020, 3, 20),  # removed
                    "available_at": datetime(2015, 1, 1),
                    "is_survivorship_biased": False,
                }
            ]
        )
        px = _prices(["A"], date(2015, 1, 1), date(2021, 12, 31))
        b = UniverseBuilder(CFG, secs, mem, px)

        assert b.members_on(date(2020, 3, 19)) == {"A"}  # day before removal: in
        assert b.members_on(date(2020, 3, 20)) == set()  # effective day: out (half-open)
        assert b.members_on(date(2021, 1, 4)) == set()

    def test_rejoining_member_has_a_gap(self):
        """Leave-and-rejoin must produce a genuine hole, not continuous membership."""
        secs = _securities(["A"])
        mem = _membership(
            [
                {
                    "security_id": "A",
                    "universe": "sp500",
                    "start_date": date(2015, 1, 1),
                    "end_date": date(2019, 3, 15),
                    "available_at": datetime(2015, 1, 1),
                    "is_survivorship_biased": False,
                },
                {
                    "security_id": "A",
                    "universe": "sp500",
                    "start_date": date(2021, 6, 21),
                    "end_date": date(9999, 12, 31),
                    "available_at": datetime(2021, 6, 16),
                    "is_survivorship_biased": False,
                },
            ]
        )
        px = _prices(["A"], date(2015, 1, 1), date(2022, 12, 31))
        b = UniverseBuilder(CFG, secs, mem, px)

        assert b.members_on(date(2018, 1, 3)) == {"A"}
        assert b.members_on(date(2020, 1, 3)) == set()  # the gap
        assert b.members_on(date(2022, 1, 3)) == {"A"}


class TestDelistedSecurities:
    def test_delisted_security_tradeable_before_death_absent_after(self):
        """The bankrupt name must exist in history -- that is the point of avoiding
        survivorship bias -- while disappearing from the universe after it dies."""
        secs = _securities(["A", "DEAD"], DEAD={"delisting_date": date(2019, 6, 28)})
        mem = _membership(
            [
                {
                    "security_id": sid,
                    "universe": "sp500",
                    "start_date": date(2015, 1, 1),
                    "end_date": date(2019, 6, 28) if sid == "DEAD" else date(9999, 12, 31),
                    "available_at": datetime(2015, 1, 1),
                    "is_survivorship_biased": False,
                }
                for sid in ("A", "DEAD")
            ]
        )
        px = _prices(["A"], date(2015, 1, 1), date(2021, 12, 31))
        px_dead = _prices(["DEAD"], date(2015, 1, 1), date(2019, 6, 28))  # bars stop at death
        b = UniverseBuilder(CFG, secs, mem, pl.concat([px, px_dead]))

        # Alive: present in the historical universe.
        assert "DEAD" in b.build(date(2018, 6, 15)).security_ids
        # Dead: gone afterwards, but its history was still traded above.
        assert "DEAD" not in b.build(date(2020, 6, 15)).security_ids


class TestSurvivorshipBiasRefusal:
    def test_refuses_biased_membership_by_default(self):
        """Silently accepting a current-membership snapshot as history is the single most
        common way backtests inflate returns. It must be an explicit opt-in."""
        secs = _securities(["A"])
        mem = _membership(
            [
                {
                    "security_id": "A",
                    "universe": "sp500",
                    "start_date": date(2015, 1, 1),
                    "end_date": date(9999, 12, 31),
                    "available_at": datetime(2015, 1, 1),
                    "is_survivorship_biased": True,  # reconstructed from today's members
                }
            ]
        )
        px = _prices(["A"], date(2015, 1, 1), date(2021, 12, 31))
        with pytest.raises(ValueError, match="survivorship-biased"):
            UniverseBuilder(CFG, secs, mem, px)

    def test_opt_in_marks_snapshot_as_biased(self):
        secs = _securities(["A"])
        mem = _membership(
            [
                {
                    "security_id": "A",
                    "universe": "sp500",
                    "start_date": date(2015, 1, 1),
                    "end_date": date(9999, 12, 31),
                    "available_at": datetime(2015, 1, 1),
                    "is_survivorship_biased": True,
                }
            ]
        )
        px = _prices(["A"], date(2015, 1, 1), date(2021, 12, 31))
        cfg = UniverseConfig(
            min_price=5.0,
            min_dollar_volume=1_000_000.0,
            min_history_sessions=20,
            allow_survivorship_biased_fallback=True,
        )
        b = UniverseBuilder(cfg, secs, mem, px)
        snap = b.build(date(2018, 6, 15))
        # It builds, but the taint travels with the result.
        assert snap.is_survivorship_biased is True


class TestEligibilityScreens:
    def test_price_screen_uses_unadjusted_close(self):
        """A $3 stock is ineligible under the $5 rule."""
        secs = _securities(["CHEAP"])
        mem = _membership(
            [
                {
                    "security_id": "CHEAP",
                    "universe": "sp500",
                    "start_date": date(2015, 1, 1),
                    "end_date": date(9999, 12, 31),
                    "available_at": datetime(2015, 1, 1),
                    "is_survivorship_biased": False,
                }
            ]
        )
        px = _prices(["CHEAP"], date(2015, 1, 1), date(2016, 12, 31), close=3.0)
        b = UniverseBuilder(CFG, secs, mem, px)
        snap = b.build(date(2016, 6, 15))
        assert snap.security_ids == []
        assert snap.exclusion_counts.get("below_min_price") == 1

    def test_insufficient_history_excluded(self):
        secs = _securities(["NEW"])
        mem = _membership(
            [
                {
                    "security_id": "NEW",
                    "universe": "sp500",
                    "start_date": date(2015, 1, 1),
                    "end_date": date(9999, 12, 31),
                    "available_at": datetime(2015, 1, 1),
                    "is_survivorship_biased": False,
                }
            ]
        )
        # Only ~10 sessions of history, below the 20-session minimum.
        px = _prices(["NEW"], date(2015, 1, 1), date(2015, 1, 15))
        b = UniverseBuilder(CFG, secs, mem, px)
        snap = b.build(date(2015, 1, 15))
        assert snap.security_ids == []
        assert snap.exclusion_counts.get("insufficient_history") == 1

    def test_liquidity_screen_uses_only_trailing_data(self):
        """A name that becomes liquid only *later* must fail the screen today.

        Using a full-sample average would let future liquidity qualify a security that was
        untradeable at the time.
        """
        from quant_platform.utilities.calendar import TradingCalendar

        sessions = TradingCalendar(date(2015, 1, 1), date(2016, 12, 31)).sessions
        rows = []
        for s in sessions:
            # Illiquid until 2016, very liquid after.
            dv = 100.0 if s < date(2016, 1, 1) else 500_000_000.0
            rows.append(
                {
                    "security_id": "X",
                    "observation_date": s,
                    "close": 100.0,
                    "dollar_volume": dv,
                }
            )
        px = pl.DataFrame(rows)
        secs = _securities(["X"])
        mem = _membership(
            [
                {
                    "security_id": "X",
                    "universe": "sp500",
                    "start_date": date(2015, 1, 1),
                    "end_date": date(9999, 12, 31),
                    "available_at": datetime(2015, 1, 1),
                    "is_survivorship_biased": False,
                }
            ]
        )
        b = UniverseBuilder(CFG, secs, mem, px)
        # In 2015 the trailing ADV is tiny -> excluded, despite later liquidity.
        assert b.build(date(2015, 6, 15)).security_ids == []
        # In 2016 trailing ADV is large -> eligible.
        assert b.build(date(2016, 6, 15)).security_ids == ["X"]

    def test_non_common_equity_excluded(self):
        secs = _securities(["ETF1"], ETF1={"security_type": "etf"})
        mem = _membership(
            [
                {
                    "security_id": "ETF1",
                    "universe": "sp500",
                    "start_date": date(2015, 1, 1),
                    "end_date": date(9999, 12, 31),
                    "available_at": datetime(2015, 1, 1),
                    "is_survivorship_biased": False,
                }
            ]
        )
        px = _prices(["ETF1"], date(2015, 1, 1), date(2016, 12, 31))
        b = UniverseBuilder(CFG, secs, mem, px)
        assert b.build(date(2016, 6, 15)).security_ids == []
