"""Trading-session calendar.

Horizons in this platform are expressed in *trading sessions*, never calendar days. A
"20-day" forward return measured in calendar days silently varies between 13 and 15 sessions
depending on weekends and holidays, which makes labels inconsistent and quietly misaligns
them with the benchmark.

This implements a US equity calendar covering weekends and the NYSE holiday rules (including
observed-date shifting and Good Friday). It is self-contained rather than depending on
``pandas_market_calendars`` so the core platform keeps working without optional packages.
Known gap: historical one-off closures (e.g. 9/11, hurricanes, national days of mourning)
are not encoded; see docs/limitations.md.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth ``weekday`` (Mon=0) of a month, e.g. 3rd Monday of January."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last ``weekday`` of a month, e.g. last Monday of May."""
    next_month = date(year + (month == 12), (month % 12) + 1, 1)
    d = next_month - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """Apply the US observed-holiday rule: Saturday -> Friday, Sunday -> Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _easter(year: int) -> date:
    """Gregorian Easter (Anonymous computus). Needed only to derive Good Friday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    li = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * li) // 451
    month, day = divmod(h + li - 7 * m + 114, 31)
    return date(year, month, day + 1)


@lru_cache(maxsize=64)
def nyse_holidays(year: int) -> frozenset[date]:
    """NYSE market holidays for a year (observed dates)."""
    holidays = {
        _observed(date(year, 1, 1)),  # New Year's Day
        _nth_weekday(year, 1, 0, 3),  # MLK Day: 3rd Monday of January
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday: 3rd Monday of February
        _easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day: last Monday of May
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day: 1st Monday of September
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving: 4th Thursday of November
        _observed(date(year, 12, 25)),  # Christmas
    }
    # Juneteenth became a market holiday in 2022.
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))
    return frozenset(holidays)


def is_trading_session(d: date) -> bool:
    """True if ``d`` is a weekday and not an NYSE holiday."""
    if d.weekday() >= 5:
        return False
    return d not in nyse_holidays(d.year)


class TradingCalendar:
    """An ordered list of trading sessions with index-based navigation.

    Sessions are materialized once and indexed, so "20 sessions forward" is an O(1) lookup
    rather than a date-arithmetic loop that could drift.
    """

    def __init__(self, start: date, end: date) -> None:
        if end < start:
            raise ValueError(f"calendar end {end} precedes start {start}")
        self.start = start
        self.end = end
        self._sessions: list[date] = []
        d = start
        while d <= end:
            if is_trading_session(d):
                self._sessions.append(d)
            d += timedelta(days=1)
        self._index: dict[date, int] = {s: i for i, s in enumerate(self._sessions)}

    @property
    def sessions(self) -> list[date]:
        return list(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, d: date) -> bool:
        return d in self._index

    def index_of(self, d: date) -> int | None:
        return self._index.get(d)

    def session_at(self, i: int) -> date | None:
        if 0 <= i < len(self._sessions):
            return self._sessions[i]
        return None

    def shift(self, d: date, n: int) -> date | None:
        """The session ``n`` sessions from ``d`` (negative shifts backward).

        Returns None when the shift runs off the end of the calendar -- the caller must then
        drop the observation rather than clamp, since clamping would fabricate a return.
        """
        i = self._index.get(d)
        if i is None:
            i = self.next_index(d)
            if i is None:
                return None
        return self.session_at(i + n)

    def next_index(self, d: date) -> int | None:
        """Index of the first session on or after ``d``."""
        import bisect

        i = bisect.bisect_left(self._sessions, d)
        return i if i < len(self._sessions) else None

    def next_session(self, d: date, inclusive: bool = False) -> date | None:
        """First session on (optionally) or after ``d``."""
        if inclusive and d in self._index:
            return d
        import bisect

        i = bisect.bisect_right(self._sessions, d)
        return self.session_at(i)

    def previous_session(self, d: date, inclusive: bool = False) -> date | None:
        """Last session on (optionally) or before ``d``."""
        if inclusive and d in self._index:
            return d
        import bisect

        i = bisect.bisect_left(self._sessions, d) - 1
        return self.session_at(i)

    def sessions_between(self, start: date, end: date) -> list[date]:
        """All sessions in the inclusive range ``[start, end]``."""
        import bisect

        lo = bisect.bisect_left(self._sessions, start)
        hi = bisect.bisect_right(self._sessions, end)
        return self._sessions[lo:hi]

    def count_between(self, start: date, end: date) -> int:
        return len(self.sessions_between(start, end))

    def week_end_sessions(self) -> list[date]:
        """Last session of each week -- the default signal day.

        Derived from actual sessions, so a holiday-shortened week yields Thursday rather than
        an assumed Friday.
        """
        result: list[date] = []
        for i, s in enumerate(self._sessions):
            is_last = i == len(self._sessions) - 1
            if is_last or self._sessions[i + 1].isocalendar()[:2] != s.isocalendar()[:2]:
                result.append(s)
        return result

    def month_end_sessions(self) -> list[date]:
        """Last session of each month."""
        result: list[date] = []
        for i, s in enumerate(self._sessions):
            is_last = i == len(self._sessions) - 1
            if is_last or self._sessions[i + 1].month != s.month:
                result.append(s)
        return result

    def rebalance_sessions(self, frequency: str, signal_day: str = "week_end") -> list[date]:
        """Sessions on which signals are formed, per the rebalance config."""
        if frequency == "daily":
            return self.sessions
        if frequency == "weekly":
            if signal_day == "week_end":
                return self.week_end_sessions()
            # week_start: first session of each week
            result: list[date] = []
            for i, s in enumerate(self._sessions):
                if i == 0 or self._sessions[i - 1].isocalendar()[:2] != s.isocalendar()[:2]:
                    result.append(s)
            return result
        if frequency == "monthly":
            return self.month_end_sessions()
        raise ValueError(f"unsupported rebalance frequency: {frequency}")
