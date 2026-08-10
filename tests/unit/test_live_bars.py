"""Tests for the live-quote overlay.

The two properties that must hold:

1. Today's PRICE enters the panel (so momentum/vol/beta include the current session) while
   today's VOLUME does not (a partial session's volume would corrupt ADV and Amihud, which
   assume complete sessions).
2. When the market is closed, the "live" bar is just the last completed session, which
   history already contains -- it must be skipped, not duplicated.
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl

from quant_platform.analysis import LiveQuote, append_provisional_bars


def _history(ticker: str = "AAPL", last_day: date = date(2026, 7, 22)) -> pl.DataFrame:
    days = [date(2026, 7, 20), date(2026, 7, 21), last_day]
    return pl.DataFrame(
        {
            "security_id": [ticker] * len(days),
            "observation_date": days,
            "close": [100.0, 101.0, 102.0],
            "adjusted_close": [100.0, 101.0, 102.0],
            "volume": [1e6, 1e6, 1e6],
            "dollar_volume": [1e8, 1e8, 1e8],
        }
    )


def _quote(
    ticker: str = "AAPL", session: date = date(2026, 7, 23), price: float = 99.0
) -> LiveQuote:
    return LiveQuote(
        ticker=ticker,
        price=price,
        prev_close=102.0,
        session_date=session,
        fetched_at=datetime(2026, 7, 23, 11, 5),
    )


class TestProvisionalBars:
    def test_todays_price_is_appended(self):
        merged, appended = append_provisional_bars(_history(), {"AAPL": _quote()})
        assert appended
        today = merged.filter(pl.col("observation_date") == date(2026, 7, 23))
        assert today.height == 1
        assert today["close"][0] == 99.0
        assert today["adjusted_close"][0] == 99.0

    def test_partial_volume_is_excluded(self):
        """The critical property: price yes, volume no."""
        merged, _ = append_provisional_bars(_history(), {"AAPL": _quote()})
        today = merged.filter(pl.col("observation_date") == date(2026, 7, 23))
        assert today["volume"][0] is None, "partial-session volume must not enter the panel"
        assert today["dollar_volume"][0] is None

    def test_market_closed_bar_is_not_duplicated(self):
        """A 'live' bar for a date history already has must be skipped entirely."""
        closed_quote = _quote(session=date(2026, 7, 22), price=102.0)
        merged, appended = append_provisional_bars(_history(), {"AAPL": closed_quote})
        assert not appended
        assert merged.height == _history().height
        july22 = merged.filter(pl.col("observation_date") == date(2026, 7, 22))
        assert july22.height == 1  # not duplicated

    def test_empty_quotes_are_a_noop(self):
        merged, appended = append_provisional_bars(_history(), {})
        assert not appended
        assert merged.height == _history().height

    def test_unknown_ticker_still_appends_for_known_ones(self):
        quotes = {"AAPL": _quote(), "ZZZZ": _quote(ticker="ZZZZ")}
        merged, appended = append_provisional_bars(_history(), quotes)
        assert appended
        # ZZZZ has no history; its bar appends too (it becomes that ticker's first bar).
        assert merged.filter(pl.col("security_id") == "ZZZZ").height == 1

    def test_change_pct_computed_from_prev_close(self):
        quote = _quote(price=99.0)  # prev close 102
        assert abs(quote.change_pct - (99.0 / 102.0 - 1.0)) < 1e-12
