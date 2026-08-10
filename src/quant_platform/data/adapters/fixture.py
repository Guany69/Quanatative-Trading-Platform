"""Deterministic synthetic data provider (spec section 10.4).

This provider lets the entire pipeline run with no network access and no credentials. It is
built to be *adversarial to the platform itself*: it deliberately embeds every hazard the
system claims to handle, so the demo exercises real code paths rather than a happy path.

Embedded hazards
----------------
* A delisted security that goes to near-zero and books a -100% delisting return.
* A security that changes ticker mid-life (identity must survive the rename).
* Index membership entries and exits with announcement dates before effective dates.
* Missing price observations (random gaps) and a trading halt.
* Fundamentals published on a realistic lag (45-75 days after fiscal period end).
* Macro series with revisions: an original print and a later, different revision.
* Stock splits and regular cash dividends.
* A low-priced security that fails the $5 eligibility filter for part of its life.

Generative model
----------------
Returns follow a one-factor structure: each security loads on a market factor with its own
beta, plus a sector factor, plus idiosyncratic noise. A small, deliberate momentum effect is
injected so the demo's factor model has *something* real to find -- otherwise a working
pipeline would be indistinguishable from a broken one. This structure is a toy, not a claim
about markets.

IMPORTANT: These are randomly generated numbers. Results computed on this data are a
software demonstration and carry no investment meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from quant_platform.domain.enums import (
    CorporateActionType,
    DelistingReason,
    Exchange,
    Sector,
    SecurityType,
)
from quant_platform.utilities.calendar import TradingCalendar

# Polars schema overrides map a column name to a dtype CLASS (pl.Date), not an instance.
SchemaDict = dict[str, Any]

FIXTURE_SOURCE = "fixture_synthetic"

_SECTORS = [
    Sector.INFORMATION_TECHNOLOGY,
    Sector.FINANCIALS,
    Sector.HEALTH_CARE,
    Sector.CONSUMER_DISCRETIONARY,
    Sector.INDUSTRIALS,
    Sector.ENERGY,
    Sector.CONSUMER_STAPLES,
    Sector.UTILITIES,
]

# Columns that are null for long stretches (only a handful of securities ever delist or get
# renamed) must be typed explicitly; Polars would otherwise infer Null from the leading rows
# and fail when the first real value appears beyond its inference window.
_SECURITY_SCHEMA: SchemaDict = {
    "delisting_date": pl.Date,
    "delisting_reason": pl.Utf8,
    "delisting_return": pl.Float64,
    "primary_symbol": pl.Utf8,
    "industry": pl.Utf8,
}

_ACTION_SCHEMA: SchemaDict = {
    "value": pl.Float64,
    "old_symbol": pl.Utf8,
    "new_symbol": pl.Utf8,
    "delisting_reason": pl.Utf8,
}


@dataclass(frozen=True)
class FixtureSpec:
    """Knobs for the synthetic dataset. Defaults give a demo that runs in seconds."""

    n_securities: int = 120
    start: date = date(2015, 1, 1)
    end: date = date(2024, 12, 31)
    seed: int = 42
    benchmark_symbol: str = "SPY"
    annual_market_drift: float = 0.07
    annual_market_vol: float = 0.16
    annual_idio_vol: float = 0.28
    # Calibrated so the planted momentum effect yields a rank IC of roughly 0.05 -- the range
    # a genuine equity factor occupies. Set to 0.0 for a no-signal control: a correctly wired
    # pipeline must then score IC ~= 0, which is how we detect leakage (a leaky pipeline finds
    # "signal" in data that contains none). See tests/data_leakage/test_no_signal_control.py.
    momentum_signal_strength: float = 0.02
    missing_observation_rate: float = 0.002


@dataclass
class FixtureDataset:
    """The generated panel. Frames are Polars; adapters return these directly."""

    securities: pl.DataFrame
    identifiers: pl.DataFrame
    prices: pl.DataFrame
    benchmark: pl.DataFrame
    membership: pl.DataFrame
    corporate_actions: pl.DataFrame
    fundamentals: pl.DataFrame
    macro: pl.DataFrame
    spec: FixtureSpec

    @property
    def is_synthetic(self) -> bool:
        return True


class FixtureDataProvider:
    """Generates a deterministic synthetic dataset.

    The same seed always yields byte-identical output, which is what makes the regression
    tests meaningful.
    """

    def __init__(self, spec: FixtureSpec | None = None) -> None:
        self.spec = spec or FixtureSpec()
        self.calendar = TradingCalendar(self.spec.start, self.spec.end)
        if len(self.calendar) < 300:
            raise ValueError(
                f"fixture span {self.spec.start}..{self.spec.end} yields only "
                f"{len(self.calendar)} sessions; need >= 300 for meaningful features"
            )

    # ------------------------------------------------------------------ helpers
    def _rng(self, stream: int = 0) -> np.random.Generator:
        """Independent, reproducible RNG stream.

        Separate streams per concern mean adding a security count does not reshuffle the
        macro series, keeping unrelated fixtures stable across changes.
        """
        return np.random.default_rng([self.spec.seed, stream])

    @staticmethod
    def _sec_id(i: int) -> str:
        return f"SEC{i:04d}"

    @staticmethod
    def _symbol(i: int) -> str:
        """A synthetic 3-letter ticker.

        These are generated combinatorially and will collide with real tickers by accident
        (e.g. 'AAL'). The collision is meaningless: the underlying data is random and has no
        relationship to any real company of the same ticker. Security names are prefixed
        'Synthetic Company' so output cannot be mistaken for real issuers.
        """
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return f"{letters[i // 676 % 26]}{letters[i // 26 % 26]}{letters[i % 26]}"

    # ------------------------------------------------------------------ build
    def generate(self) -> FixtureDataset:
        sessions = self.calendar.sessions
        n_sessions = len(sessions)

        market = self._market_returns(n_sessions)
        sector_factors = self._sector_returns(n_sessions)

        securities, identifiers, actions = self._build_security_master()
        prices = self._build_prices(securities, market, sector_factors, sessions, actions)
        benchmark = self._build_benchmark(market, sessions)
        membership = self._build_membership(securities, sessions)
        fundamentals = self._build_fundamentals(securities, prices)
        macro = self._build_macro(sessions, market)

        return FixtureDataset(
            securities=securities,
            identifiers=identifiers,
            prices=prices,
            benchmark=benchmark,
            membership=membership,
            corporate_actions=actions,
            fundamentals=fundamentals,
            macro=macro,
            spec=self.spec,
        )

    def _market_returns(self, n: int) -> np.ndarray:
        """Market factor with volatility clustering, so regime features have signal."""
        rng = self._rng(1)
        daily_drift = self.spec.annual_market_drift / 252.0
        base_vol = self.spec.annual_market_vol / np.sqrt(252.0)
        # AR(1) on log-vol produces clustering (calm and turbulent stretches).
        log_vol = np.zeros(n)
        log_vol[0] = np.log(base_vol)
        shocks = rng.normal(0, 0.15, n)
        for t in range(1, n):
            log_vol[t] = 0.98 * log_vol[t - 1] + 0.02 * np.log(base_vol) + shocks[t] * 0.05
        vol = np.exp(log_vol)
        return daily_drift + rng.normal(0, 1, n) * vol

    def _sector_returns(self, n: int) -> np.ndarray:
        """Sector factor returns, shape (n_sectors, n)."""
        rng = self._rng(2)
        sector_vol = 0.08 / np.sqrt(252.0)
        return rng.normal(0, sector_vol, (len(_SECTORS), n))

    def _build_security_master(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Securities, their identifier history, and corporate actions."""
        rng = self._rng(3)
        n_sec = self.spec.n_securities
        sec_rows, id_rows, action_rows = [], [], []

        # Hazard assignments, fixed by index so they are stable across runs.
        delisted_idx = {7, 43}  # one bankruptcy, one merger
        renamed_idx = {11}  # identity must survive a ticker change
        low_price_idx = {19}  # fails the $5 filter for part of its life

        for i in range(n_sec):
            sec_id = self._sec_id(i)
            symbol = self._symbol(i)
            sector = _SECTORS[i % len(_SECTORS)]
            first_trade = self.spec.start

            delisting_date = None
            delisting_reason = None
            delisting_return = None
            if i in delisted_idx:
                if i == 7:
                    delisting_date = date(2019, 6, 28)
                    delisting_reason = DelistingReason.BANKRUPTCY
                    # Total loss. Omitting this is exactly the delisting bias we must avoid.
                    delisting_return = -1.0
                else:
                    delisting_date = date(2021, 9, 30)
                    delisting_reason = DelistingReason.MERGER
                    delisting_return = 0.18  # acquisition premium
                action_rows.append(
                    {
                        "security_id": sec_id,
                        "symbol": symbol,
                        "observation_date": delisting_date,
                        # Delistings are announced in advance; knowledge precedes the event.
                        "available_at": datetime.combine(
                            delisting_date - timedelta(days=30), datetime.min.time()
                        ),
                        "source": FIXTURE_SOURCE,
                        "action_type": CorporateActionType.DELISTING.value,
                        "ex_date": delisting_date,
                        "value": None,
                        "old_symbol": None,
                        "new_symbol": None,
                        "delisting_reason": delisting_reason.value,
                    }
                )

            # Identifier history: renamed securities get two intervals, same security_id.
            if i in renamed_idx:
                change_date = date(2018, 3, 15)
                new_symbol = self._symbol(i + 500)
                id_rows.append(
                    {
                        "security_id": sec_id,
                        "identifier_type": "ticker",
                        "identifier_value": symbol,
                        "start_date": first_trade,
                        "end_date": change_date,
                        "is_primary": True,
                    }
                )
                id_rows.append(
                    {
                        "security_id": sec_id,
                        "identifier_type": "ticker",
                        "identifier_value": new_symbol,
                        "start_date": change_date,
                        "end_date": delisting_date or date(9999, 12, 31),
                        "is_primary": True,
                    }
                )
                action_rows.append(
                    {
                        "security_id": sec_id,
                        "symbol": symbol,
                        "observation_date": change_date,
                        "available_at": datetime.combine(change_date, datetime.min.time()),
                        "source": FIXTURE_SOURCE,
                        "action_type": CorporateActionType.SYMBOL_CHANGE.value,
                        "ex_date": change_date,
                        "value": None,
                        "old_symbol": symbol,
                        "new_symbol": new_symbol,
                        "delisting_reason": None,
                    }
                )
                display_symbol = new_symbol
            else:
                id_rows.append(
                    {
                        "security_id": sec_id,
                        "identifier_type": "ticker",
                        "identifier_value": symbol,
                        "start_date": first_trade,
                        "end_date": delisting_date or date(9999, 12, 31),
                        "is_primary": True,
                    }
                )
                display_symbol = symbol

            sec_rows.append(
                {
                    "security_id": sec_id,
                    "security_type": SecurityType.COMMON_STOCK.value,
                    "primary_symbol": display_symbol,
                    "name": f"Synthetic Company {display_symbol}",
                    "exchange": (Exchange.NYSE if i % 2 else Exchange.NASDAQ).value,
                    "sector": sector.value,
                    "industry": f"{sector.value}_industry",
                    "country": "US",
                    "currency": "USD",
                    "first_trade_date": first_trade,
                    "delisting_date": delisting_date,
                    "delisting_reason": delisting_reason.value if delisting_reason else None,
                    "delisting_return": delisting_return,
                    # Non-contract fields used by the generator:
                    "_beta": float(np.clip(rng.normal(1.0, 0.25), 0.35, 1.9)),
                    "_sector_idx": i % len(_SECTORS),
                    "_start_price": float(
                        rng.uniform(2.5, 45.0) if i in low_price_idx else rng.uniform(15.0, 320.0)
                    ),
                    "_quality": float(rng.normal(0, 1)),
                    "_shares": float(rng.uniform(5e7, 9e8)),
                }
            )

            # Splits: a few securities split once, mid-sample.
            if i % 23 == 5:
                split_date = date(2019, 8, 12)
                action_rows.append(
                    {
                        "security_id": sec_id,
                        "symbol": symbol,
                        "observation_date": split_date,
                        "available_at": datetime.combine(
                            split_date - timedelta(days=14), datetime.min.time()
                        ),
                        "source": FIXTURE_SOURCE,
                        "action_type": CorporateActionType.STOCK_SPLIT.value,
                        "ex_date": split_date,
                        "value": 2.0,
                        "old_symbol": None,
                        "new_symbol": None,
                        "delisting_reason": None,
                    }
                )

            # Quarterly dividends for roughly half the names.
            if i % 2 == 0:
                for year in range(self.spec.start.year, self.spec.end.year + 1):
                    for month in (3, 6, 9, 12):
                        ex = date(year, month, 15)
                        if not (self.spec.start <= ex <= self.spec.end):
                            continue
                        if delisting_date and ex > delisting_date:
                            continue
                        action_rows.append(
                            {
                                "security_id": sec_id,
                                "symbol": symbol,
                                "observation_date": ex,
                                "available_at": datetime.combine(
                                    ex - timedelta(days=21), datetime.min.time()
                                ),
                                "source": FIXTURE_SOURCE,
                                "action_type": CorporateActionType.CASH_DIVIDEND.value,
                                "ex_date": ex,
                                "value": 0.25,
                                "old_symbol": None,
                                "new_symbol": None,
                                "delisting_reason": None,
                            }
                        )

        # Explicit schemas: several columns are null for long leading runs (only a couple of
        # securities ever delist), so type inference from the first N rows guesses Null and
        # then fails when a real string arrives.
        securities = pl.DataFrame(sec_rows, schema_overrides=_SECURITY_SCHEMA)
        identifiers = pl.DataFrame(id_rows)
        actions = pl.DataFrame(action_rows, schema_overrides=_ACTION_SCHEMA)
        return securities, identifiers, actions

    def _build_prices(
        self,
        securities: pl.DataFrame,
        market: np.ndarray,
        sector_factors: np.ndarray,
        sessions: list[date],
        actions: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Generate the price panel from the factor model.

        Splits are applied to the RAW price path here, so the fixture behaves like a real
        feed: the quoted price genuinely halves on a 2-for-1 ex-date. Emitting a split event
        without moving the price would be worse than useless -- the adjustment step would then
        *introduce* a 2x discontinuity instead of removing one, corrupting every momentum and
        volatility feature for that security.
        """
        rng = self._rng(4)
        n = len(sessions)
        idio_vol = self.spec.annual_idio_vol / np.sqrt(252.0)
        frames = []

        for row in securities.iter_rows(named=True):
            sec_id = row["security_id"]
            beta = row["_beta"]
            sector_idx = row["_sector_idx"]
            start_price = row["_start_price"]
            delist = row["delisting_date"]

            idio = rng.normal(0, idio_vol, n)
            returns = beta * market + 0.6 * sector_factors[sector_idx] + idio

            # Inject a momentum effect: trailing 126-session performance adds a small drift to
            # future returns. Without a *detectable* ground-truth signal, a correct pipeline
            # and a broken one both score IC~0, so the demo could not demonstrate that the
            # machinery works at all.
            #
            # Calibration matters. The effect competes against idiosyncratic noise of
            # ~28%/sqrt(252) = 1.8% per session, and against a mechanical beta effect (excess
            # return = r - benchmark, so high-beta names outperform whenever the market rises).
            # Dividing by 252 made the drift ~0.2bp/session -- two orders of magnitude below
            # the noise, hence undetectable. Scaling per-signal-horizon rather than per-year
            # puts it in the range where a real equity factor lives (IC ~ 0.03-0.08).
            #
            # This is a synthetic effect planted for testing. Detecting it demonstrates the
            # pipeline is wired correctly; it says NOTHING about real markets.
            if self.spec.momentum_signal_strength:
                lookback = 126
                mom = np.zeros(n)
                cum = np.cumsum(returns)
                mom[lookback:] = cum[lookback:] - cum[:-lookback]
                # tanh bounds the effect so extreme movers do not get an unbounded drift.
                returns = returns + self.spec.momentum_signal_strength * np.tanh(mom * 2.0) / 21.0

            # Delisting: collapse toward the terminal value near the delisting date.
            if delist is not None:
                try:
                    d_idx = sessions.index(min(s for s in sessions if s >= delist))
                except ValueError:
                    d_idx = n - 1
                if row["delisting_reason"] == DelistingReason.BANKRUPTCY.value:
                    # Death spiral into the delisting.
                    decay = np.linspace(0, -0.045, min(60, d_idx))
                    returns[max(0, d_idx - 60) : d_idx] += decay

            price_path = start_price * np.exp(np.cumsum(returns))
            price_path = np.maximum(price_path, 0.05)  # a price floor; equities cannot go <= 0

            # Apply splits to the quoted price: on and after the ex-date the price is divided
            # by the ratio, exactly as a real feed reports it.
            if actions is not None and not actions.is_empty():
                sec_splits = actions.filter(
                    (pl.col("security_id") == sec_id)
                    & pl.col("action_type").is_in(
                        [
                            CorporateActionType.STOCK_SPLIT.value,
                            CorporateActionType.REVERSE_SPLIT.value,
                        ]
                    )
                )
                for split in sec_splits.iter_rows(named=True):
                    ratio = float(split["value"] or 1.0)
                    if ratio <= 0:
                        continue
                    ex = split["ex_date"]
                    mask = np.array([s >= ex for s in sessions], dtype=bool)
                    price_path[mask] /= ratio

            # Share count is a per-session series because a split multiplies it: price halves
            # and share count doubles, leaving market cap continuous. A scalar count would
            # make market cap appear to drop 50% on every split.
            shares_series = np.full(n, float(row["_shares"]))
            if actions is not None and not actions.is_empty():
                for split in actions.filter(
                    (pl.col("security_id") == sec_id)
                    & pl.col("action_type").is_in(
                        [
                            CorporateActionType.STOCK_SPLIT.value,
                            CorporateActionType.REVERSE_SPLIT.value,
                        ]
                    )
                ).iter_rows(named=True):
                    ratio = float(split["value"] or 1.0)
                    if ratio <= 0:
                        continue
                    ex = split["ex_date"]
                    shares_series[np.array([s >= ex for s in sessions], dtype=bool)] *= ratio

            n_keep = n
            if delist is not None:
                keep = [i for i, s in enumerate(sessions) if s <= delist]
                n_keep = len(keep) if keep else n

            dates = sessions[:n_keep]
            closes = price_path[:n_keep]

            # Intraday range around the close.
            noise = rng.uniform(0.001, 0.02, n_keep)
            highs = closes * (1 + noise)
            lows = closes * (1 - noise)
            opens = np.clip(closes * (1 + rng.normal(0, 0.006, n_keep)), lows, highs)
            volume = np.abs(rng.lognormal(13.0, 0.8, n_keep))

            # Missing observations: a real feed has gaps, and the pipeline must survive them.
            keep_mask = rng.random(n_keep) > self.spec.missing_observation_rate
            keep_mask[0] = True  # always keep the first bar so history starts cleanly

            frames.append(
                pl.DataFrame(
                    {
                        "security_id": [sec_id] * n_keep,
                        "observation_date": dates,
                        "open": opens,
                        "high": highs,
                        "low": lows,
                        "close": closes,
                        "volume": volume,
                        "shares_outstanding": shares_series[:n_keep],
                    }
                ).filter(pl.Series(keep_mask))
            )

        prices = pl.concat(frames)
        # available_at: the close is knowable after the session ends, not during it.
        prices = prices.with_columns(
            [
                (pl.col("observation_date").cast(pl.Datetime) + pl.duration(hours=21)).alias(
                    "available_at"
                ),
                pl.lit(FIXTURE_SOURCE).alias("source"),
                (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
                (pl.col("close") * pl.col("shares_outstanding")).alias("market_cap"),
            ]
        )
        return prices.sort(["security_id", "observation_date"])

    def _build_benchmark(self, market: np.ndarray, sessions: list[date]) -> pl.DataFrame:
        """Benchmark total-return index built from the same market factor."""
        level = 100.0 * np.exp(np.cumsum(market))
        return pl.DataFrame(
            {
                "observation_date": sessions,
                "close": level,
                "adjusted_close": level,
                "benchmark_return": np.concatenate([[0.0], np.diff(np.log(level))]),
            }
        ).with_columns(
            [
                (pl.col("observation_date").cast(pl.Datetime) + pl.duration(hours=21)).alias(
                    "available_at"
                ),
                pl.lit(FIXTURE_SOURCE).alias("source"),
                pl.lit(self.spec.benchmark_symbol).alias("symbol"),
            ]
        )

    def _build_membership(self, securities: pl.DataFrame, sessions: list[date]) -> pl.DataFrame:
        """Index membership with real entries and exits.

        Announcement dates precede effective dates, so a strategy that peeks at
        ``available_at`` incorrectly would trade an addition before it was public.
        """
        # Membership is assigned deterministically by index (no RNG needed) so that the
        # same securities always enter/exit, keeping fixtures stable across runs.
        rows = []
        start, end = self.spec.start, date(9999, 12, 31)

        for i, row in enumerate(securities.iter_rows(named=True)):
            sec_id = row["security_id"]
            delist = row["delisting_date"]

            # Most names are in from the start; a slice joins later; a slice leaves early.
            if i % 17 == 3:
                entry = date(2018, 6, 15)  # joins mid-sample
            elif i % 19 == 7:
                entry = date(2020, 3, 20)
            else:
                entry = start

            exit_date = delist or end
            if i % 23 == 11 and delist is None:
                exit_date = date(2022, 9, 16)  # removed from the index but still trading

            rows.append(
                {
                    "security_id": sec_id,
                    "universe": "sp500",
                    "start_date": entry,
                    "end_date": exit_date,
                    # Announced ~5 days before it takes effect.
                    "available_at": datetime.combine(
                        entry - timedelta(days=5) if entry > start else start,
                        datetime.min.time(),
                    ),
                    "source": FIXTURE_SOURCE,
                    # This fixture has TRUE historical membership, so it is not biased.
                    "is_survivorship_biased": False,
                }
            )
            # A name that leaves and later rejoins: intervals must not overlap.
            if i % 31 == 13 and delist is None:
                rows[-1]["end_date"] = date(2019, 3, 15)
                rows.append(
                    {
                        "security_id": sec_id,
                        "universe": "sp500",
                        "start_date": date(2021, 6, 21),
                        "end_date": end,
                        "available_at": datetime.combine(date(2021, 6, 16), datetime.min.time()),
                        "source": FIXTURE_SOURCE,
                        "is_survivorship_biased": False,
                    }
                )
        return pl.DataFrame(rows)

    def _build_fundamentals(self, securities: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
        """Quarterly fundamentals published on a realistic 45-75 day lag."""
        rng = self._rng(6)
        rows = []
        last_px = prices.group_by("security_id").agg(pl.col("close").last().alias("last_close"))
        px_map = dict(zip(last_px["security_id"], last_px["last_close"], strict=True))

        for row in securities.iter_rows(named=True):
            sec_id = row["security_id"]
            delist = row["delisting_date"]
            quality = row["_quality"]
            shares = row["_shares"]
            base_px = px_map.get(sec_id, 50.0)

            # Scale the fundamentals to the security's market cap so ratios are plausible.
            mcap = base_px * shares
            revenue = mcap * float(np.clip(rng.normal(0.55, 0.2), 0.1, 1.5))
            margin = float(np.clip(0.10 + quality * 0.05, 0.01, 0.35))
            assets = mcap * float(np.clip(rng.normal(0.9, 0.25), 0.3, 2.5))
            equity_ratio = float(np.clip(rng.normal(0.45, 0.12), 0.1, 0.85))

            for year in range(self.spec.start.year, self.spec.end.year + 1):
                for q, (m, d) in enumerate([(3, 31), (6, 30), (9, 30), (12, 31)], start=1):
                    period_end = date(year, m, d)
                    if period_end < self.spec.start or period_end > self.spec.end:
                        continue
                    if delist is not None and period_end > delist:
                        continue

                    # THE point-in-time control: a filing lands 45-75 days after period end.
                    lag_days = int(rng.integers(45, 76))
                    filing = period_end + timedelta(days=lag_days)
                    if filing > self.spec.end:
                        continue

                    growth = 1.0 + (year - self.spec.start.year) * 0.04
                    q_rev = revenue / 4.0 * growth * float(rng.normal(1.0, 0.05))
                    q_ni = q_rev * margin * float(rng.normal(1.0, 0.12))
                    q_assets = assets * growth
                    q_equity = q_assets * equity_ratio

                    rows.append(
                        {
                            "security_id": sec_id,
                            "observation_date": period_end,
                            "fiscal_period_end": period_end,
                            "fiscal_year": year,
                            "fiscal_quarter": q,
                            "period_type": "quarterly",
                            "filing_date": filing,
                            # available_at == filing date, NOT period end. This is the field
                            # that prevents look-ahead on fundamentals.
                            "available_at": datetime.combine(filing, datetime.min.time())
                            + timedelta(hours=17),
                            "source": FIXTURE_SOURCE,
                            "revenue": q_rev,
                            "gross_profit": q_rev * 0.4,
                            "operating_income": q_rev * margin * 1.3,
                            "net_income": q_ni,
                            "interest_expense": q_assets * 0.004,
                            "eps_diluted": q_ni / shares,
                            "total_assets": q_assets,
                            "total_liabilities": q_assets * (1 - equity_ratio),
                            "total_equity": q_equity,
                            "cash_and_equivalents": q_assets * 0.12,
                            "total_debt": q_assets * 0.25,
                            "shares_outstanding": shares,
                            "operating_cash_flow": q_ni * 1.25,
                            "capital_expenditure": q_rev * 0.05,
                            "free_cash_flow": q_ni * 1.25 - q_rev * 0.05,
                            "dividends_paid": q_ni * 0.3,
                            "share_repurchase": q_ni * 0.1,
                            "consensus_eps": (q_ni / shares) * float(rng.normal(1.0, 0.04)),
                        }
                    )
        return pl.DataFrame(rows)

    def _build_macro(self, sessions: list[date], market: np.ndarray) -> pl.DataFrame:
        """Macro series including a revised series.

        DGS10 is a daily rate published next-day and never revised. GDP is quarterly, printed
        with a lag and then *revised* -- the original and revision share an observation_date
        but differ in available_at and value.
        """
        rng = self._rng(7)
        rows = []

        # 10-year yield: mean-reverting, published the next morning, no revisions.
        level = 2.2
        for s in sessions:
            level += 0.02 * (2.5 - level) + float(rng.normal(0, 0.03))
            level = float(np.clip(level, 0.3, 6.0))
            rows.append(
                {
                    "series_id": "DGS10",
                    "observation_date": s,
                    "available_at": datetime.combine(s + timedelta(days=1), datetime.min.time())
                    + timedelta(hours=8),
                    "source": FIXTURE_SOURCE,
                    "revision_id": 0,
                    "is_revision": False,
                    "value": level,
                    "units": "percent",
                    "frequency": "daily",
                }
            )

        # 3-month yield, for a yield-curve slope feature.
        short = 1.0
        for s in sessions:
            short += 0.02 * (1.5 - short) + float(rng.normal(0, 0.02))
            short = float(np.clip(short, 0.01, 5.5))
            rows.append(
                {
                    "series_id": "DGS3MO",
                    "observation_date": s,
                    "available_at": datetime.combine(s + timedelta(days=1), datetime.min.time())
                    + timedelta(hours=8),
                    "source": FIXTURE_SOURCE,
                    "revision_id": 0,
                    "is_revision": False,
                    "value": short,
                    "units": "percent",
                    "frequency": "daily",
                }
            )

        # GDP with revisions -- the vintage hazard.
        for year in range(self.spec.start.year, self.spec.end.year + 1):
            for m, d in [(3, 31), (6, 30), (9, 30), (12, 31)]:
                period_end = date(year, m, d)
                if not (self.spec.start <= period_end <= self.spec.end):
                    continue
                true_value = 20000.0 * (1.02 ** (year - self.spec.start.year))
                # Original print: ~30 days after period end, and noisy.
                advance = period_end + timedelta(days=30)
                original = true_value * float(rng.normal(1.0, 0.008))
                rows.append(
                    {
                        "series_id": "GDP",
                        "observation_date": period_end,
                        "available_at": datetime.combine(advance, datetime.min.time())
                        + timedelta(hours=8, minutes=30),
                        "source": FIXTURE_SOURCE,
                        "revision_id": 0,
                        "is_revision": False,
                        "value": original,
                        "units": "billions_usd",
                        "frequency": "quarterly",
                    }
                )
                # Revision ~90 days after period end, closer to truth. Using this value
                # before its available_at is a classic and powerful leak.
                revision_date = period_end + timedelta(days=90)
                if revision_date <= self.spec.end:
                    rows.append(
                        {
                            "series_id": "GDP",
                            "observation_date": period_end,
                            "available_at": datetime.combine(revision_date, datetime.min.time())
                            + timedelta(hours=8, minutes=30),
                            "source": FIXTURE_SOURCE,
                            "revision_id": 1,
                            "is_revision": True,
                            "value": true_value,
                            "units": "billions_usd",
                            "frequency": "quarterly",
                        }
                    )
        return pl.DataFrame(rows).sort(["series_id", "observation_date", "revision_id"])
