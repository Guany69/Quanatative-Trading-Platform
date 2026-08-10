"""Single-stock factor scorecard against a peer universe.

This platform ranks securities *cross-sectionally*; it does not score a stock in isolation.
"AAPL's 12-month momentum is 0.23" is close to meaningless on its own -- the number only
becomes a signal when compared against a peer group ("87th percentile"). Every function here
therefore requires a peer universe, and the scorecard reports percentiles rather than raw
values wherever a percentile is the honest unit.

What this module can and cannot say
-----------------------------------
It CAN say: where a stock currently sits, relative to its peers, on momentum, risk, and
liquidity factors, and what the transparent factor composite makes of that combination.

It CANNOT say whether the stock is a good investment. The composite is a descriptive summary
of the factor characteristics a stock exhibits today. It is not a forecast, it carries no
statistical guarantee, and the pillars it can evaluate here are limited (see below).

Known gaps in this path, stated plainly:

* **Only price-derived factors are available.** yfinance supplies prices, not point-in-time
  fundamentals, so the value and quality pillars are inert. The composite renormalizes across
  the pillars it can actually evaluate and reports which those were -- it does not quietly
  treat missing pillars as neutral.
* **The peer universe is survivorship-biased.** It is a list of currently-listed large caps.
  Companies that failed are absent by construction, which flatters any historical comparison.
* **The peer list is an approximation**, not the real S&P 500. Index membership is not
  available from free sources (see docs/limitations.md).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_platform.features.compute import compute_price_features
from quant_platform.utilities.narrow import as_date
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("analysis")

BENCHMARK_TICKER = "SPY"

# A diversified set of large, liquid US listings used as the default peer group.
#
# This is an APPROXIMATION of a large-cap universe, not the S&P 500: it is a fixed list of
# names that are listed *today*, so it is survivorship-biased by construction and its
# composition does not change with history. It exists to make cross-sectional ranking
# possible at all; supply your own list when the comparison matters.
DEFAULT_PEER_UNIVERSE: tuple[str, ...] = (
    # Information technology
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    "ORCL",
    "CRM",
    "AMD",
    "ADBE",
    "CSCO",
    "ACN",
    "INTC",
    "IBM",
    "QCOM",
    "TXN",
    "INTU",
    "NOW",
    "AMAT",
    "MU",
    "ADI",
    "LRCX",
    # Communication services
    "GOOGL",
    "META",
    "NFLX",
    "DIS",
    "CMCSA",
    "VZ",
    "T",
    "TMUS",
    "EA",
    "OMC",
    # Consumer discretionary
    "AMZN",
    "TSLA",
    "HD",
    "MCD",
    "NKE",
    "LOW",
    "SBUX",
    "TJX",
    "BKNG",
    "GM",
    # Consumer staples
    "WMT",
    "PG",
    "KO",
    "PEP",
    "COST",
    "PM",
    "MO",
    "MDLZ",
    "CL",
    "KMB",
    # Health care
    "UNH",
    "JNJ",
    "LLY",
    "ABBV",
    "MRK",
    "PFE",
    "TMO",
    "ABT",
    "DHR",
    "BMY",
    "AMGN",
    "GILD",
    "CVS",
    "MDT",
    "ISRG",
    # Financials
    "BRK-B",
    "JPM",
    "V",
    "MA",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "AXP",
    "BLK",
    "SCHW",
    "C",
    "SPGI",
    "CB",
    "PGR",
    # Industrials
    "CAT",
    "BA",
    "HON",
    "UNP",
    "GE",
    "RTX",
    "LMT",
    "DE",
    "UPS",
    "MMM",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "EOG",
    "PSX",
    "MPC",
    "VLO",
    # Utilities / real estate / materials
    "NEE",
    "DUK",
    "SO",
    "D",
    "AMT",
    "PLD",
    "LIN",
    "APD",
    "SHW",
    "NEM",
)

# Factor families, with the sign that makes "higher = more of this characteristic".
_SCORECARD_FACTORS: dict[str, list[tuple[str, float, str]]] = {
    "momentum": [
        ("momentum_252d", 1.0, "12-month price momentum"),
        ("momentum_126d", 1.0, "6-month price momentum"),
        ("momentum_12m_ex1m", 1.0, "12-month momentum excluding last month"),
        ("dist_52w_high", 1.0, "proximity to 52-week high"),
        ("trend_r2_63d", 1.0, "trend consistency (R^2)"),
    ],
    "reversal": [
        ("reversal_5d", 1.0, "1-week reversal (recent losers score higher)"),
        ("reversal_21d", 1.0, "1-month reversal"),
    ],
    "defensive": [
        ("volatility_252d", -1.0, "1-year realized volatility (lower is more defensive)"),
        ("beta_252d", -1.0, "market beta (lower is more defensive)"),
        ("idio_vol_252d", -1.0, "idiosyncratic volatility"),
        ("max_drawdown_252d", 1.0, "1-year max drawdown (shallower is better)"),
        ("downside_vol_60d", -1.0, "downside volatility"),
    ],
    "liquidity": [
        ("adv_21d", 1.0, "average daily dollar volume"),
        ("amihud_illiquidity_21d", -1.0, "Amihud illiquidity (lower is more liquid)"),
        ("spread_proxy_21d", -1.0, "estimated bid-ask spread"),
    ],
}

# Pillars that cannot be evaluated without point-in-time fundamentals.
UNAVAILABLE_PILLARS = {
    "value": "needs point-in-time fundamentals (earnings, book value, cash flow)",
    "quality": "needs point-in-time fundamentals (margins, ROE, leverage)",
}


@dataclass
class FactorScore:
    """One factor's standing for a stock relative to its peers."""

    name: str
    description: str
    raw_value: float
    percentile: float  # 0-100, where 100 = strongest on this characteristic
    peer_median: float

    @property
    def label(self) -> str:
        """Plain-language bucket. Deliberately coarse: percentile precision implies more
        resolution than a ~100-name cross-section actually supports."""
        if self.percentile >= 80:
            return "very high"
        if self.percentile >= 60:
            return "high"
        if self.percentile >= 40:
            return "average"
        if self.percentile >= 20:
            return "low"
        return "very low"


# --------------------------------------------------------------------------- glossary
@dataclass(frozen=True)
class FactorExplanation:
    """Plain-language help for one factor.

    Every scorecard number should be readable by someone who does not already know the
    jargon. ``measures`` says what the raw number is, ``high_means``/``low_means`` say what
    the percentile implies, and ``read_value`` turns the raw number into a sentence with
    units attached ("moved about 2.1x the market", not "2.1286").

    All wording stays DESCRIPTIVE. A high momentum percentile means the stock has risen more
    than its peers -- not that it is a good buy.
    """

    measures: str
    high_means: str
    low_means: str
    read_value: Callable[[float], str] | None = None


FAMILY_EXPLANATIONS: dict[str, str] = {
    "momentum": (
        "How strongly the stock has trended UP relative to peers over months. Momentum is a "
        "long-documented anomaly: past winners have historically tended to keep winning over "
        "3-12 month horizons. It says nothing about whether a company is cheap or sound."
    ),
    "reversal": (
        "The opposite effect at SHORT horizons. Over days-to-weeks, sharp losers have "
        "historically tended to bounce, so a HIGH reversal score means the stock has recently "
        "fallen relative to peers. High momentum and high reversal together mean a long-term "
        "winner that has pulled back recently."
    ),
    "defensive": (
        "How calm the stock is. Combines volatility, market sensitivity (beta), and drawdown. "
        "A HIGH score means it moves less than peers. The low-volatility anomaly is the "
        "observation that calmer stocks have historically delivered better risk-adjusted "
        "returns than their raw returns suggest -- not that they return more."
    ),
    "liquidity": (
        "How easily the stock can be traded without moving its own price. Matters because a "
        "signal you cannot trade cheaply is not a signal: transaction costs consume the edge. "
        "A HIGH score means large orders are absorbed easily."
    ),
}

COMPOSITE_EXPLANATION = (
    "The average of the factor families above, each equally weighted. It summarizes what "
    "KIND of stock this is right now -- trending or reverting, calm or volatile, liquid or "
    "thin. It is NOT a prediction, NOT a rating, and NOT a buy/sell signal. Two stocks with "
    "the same composite can behave completely differently."
)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


FACTOR_GLOSSARY: dict[str, FactorExplanation] = {
    "momentum_252d": FactorExplanation(
        "Total price change over the past 12 months (as a log return).",
        "has risen much more than peers over the past year",
        "has risen less, or fallen, versus peers over the past year",
        lambda v: f"about {_pct(np.expm1(v))} price change over 12 months",
    ),
    "momentum_126d": FactorExplanation(
        "Total price change over the past 6 months.",
        "has gained strongly over 6 months versus peers",
        "has performed weakly over 6 months versus peers",
        lambda v: f"about {_pct(np.expm1(v))} price change over 6 months",
    ),
    "momentum_12m_ex1m": FactorExplanation(
        "12-month change EXCLUDING the most recent month. The academic standard: the last "
        "month is dropped because short-term reversal contaminates it.",
        "shows a strong longer-run trend, ignoring last month's noise",
        "shows a weak longer-run trend",
        lambda v: f"about {_pct(np.expm1(v))} over months 2-12",
    ),
    "dist_52w_high": FactorExplanation(
        "How far below its own 52-week high the stock is trading.",
        "is trading near its 1-year peak",
        "is trading well below its 1-year peak",
        lambda v: f"about {_pct(abs(np.expm1(v)))} below its 52-week high",
    ),
    "trend_r2_63d": FactorExplanation(
        "How well a straight line fits the last 3 months of price (R-squared, 0 to 1). "
        "Measures the SMOOTHNESS of the move, not its direction.",
        "has moved in a steady, consistent line",
        "has moved erratically -- any gain came in jumps rather than a steady trend",
        lambda v: (
            f"R2 = {v:.2f} ("
            + ("very smooth" if v > 0.7 else "moderately smooth" if v > 0.3 else "erratic, choppy")
            + ")"
        ),
    ),
    "reversal_5d": FactorExplanation(
        "Last week's return, NEGATED, so recent losers score higher.",
        "has fallen over the past week (the reversal effect favours recent losers)",
        "has risen over the past week",
        lambda v: f"about {_pct(np.expm1(-v))} over the past week",
    ),
    "reversal_21d": FactorExplanation(
        "Last month's return, negated.",
        "has fallen over the past month",
        "has risen over the past month",
        lambda v: f"about {_pct(np.expm1(-v))} over the past month",
    ),
    "volatility_252d": FactorExplanation(
        "Annualized standard deviation of daily returns over 1 year -- how much the price "
        "swings around.",
        "is calmer than its peers",
        "swings much more than its peers",
        lambda v: f"{_pct(v)} annualized; a typical year moves roughly +/-{_pct(v)}",
    ),
    "beta_252d": FactorExplanation(
        "Sensitivity to the S&P 500 (SPY). Beta 1.0 moves with the market; 2.0 moves twice "
        "as much; negative moves opposite.",
        "is less market-sensitive than its peers",
        "amplifies market moves far more than its peers",
        lambda v: (
            f"moves about {abs(v):.2f}x the market"
            + (", in the OPPOSITE direction" if v < 0 else "")
            + f" -- a 1% market move implies roughly {v:+.2f}%"
        ),
    ),
    "idio_vol_252d": FactorExplanation(
        "Volatility of the part of the return the market does NOT explain -- company-specific "
        "risk.",
        "is driven mostly by the market, with little company-specific noise",
        "has large company-specific swings independent of the market",
        lambda v: f"{_pct(v)} annualized stock-specific volatility",
    ),
    "max_drawdown_252d": FactorExplanation(
        "Worst peak-to-trough fall over the past year.",
        "had a shallower worst-case fall than its peers",
        "suffered a much deeper fall than its peers",
        lambda v: f"fell {_pct(abs(v))} from its peak at the worst point",
    ),
    "downside_vol_60d": FactorExplanation(
        "Volatility counting DOWN days only. Separates painful moves from pleasant ones.",
        "has smaller downside swings than its peers",
        "has larger downside swings than its peers",
        lambda v: f"{_pct(v)} annualized downside volatility",
    ),
    "adv_21d": FactorExplanation(
        "Average daily dollar volume over 21 sessions (log scale). How much money changes "
        "hands each day.",
        "is very heavily traded -- large orders are absorbed easily",
        "is thinly traded -- large orders would move the price",
        lambda v: f"roughly ${np.expm1(v) / 1e6:,.0f}M traded per day",
    ),
    "amihud_illiquidity_21d": FactorExplanation(
        "Amihud illiquidity: how much the price moves per dollar traded. Low = liquid.",
        "has a price that barely reacts to trading volume (very liquid)",
        "has a price that moves sharply on modest volume (illiquid)",
        lambda v: (
            ("very liquid" if v < 0.005 else "liquid" if v < 0.05 else "less liquid")
            + f" (raw {v:.4f})"
        ),
    ),
    "spread_proxy_21d": FactorExplanation(
        "Estimated bid-ask spread from the daily high-low range -- the cost of a round trip.",
        "has tight spreads and is cheap to trade",
        "has wide spreads, so each trade costs more",
        lambda v: f"roughly {_pct(v)} of price as daily range",
    ),
}


def explain_factor(name: str) -> FactorExplanation | None:
    return FACTOR_GLOSSARY.get(name)


def interpret_score(score: FactorScore) -> str:
    """One-sentence reading of a specific factor score for a specific stock."""
    explanation = FACTOR_GLOSSARY.get(score.name)
    if explanation is None:
        return ""
    if score.percentile >= 60:
        stance = explanation.high_means
    elif score.percentile <= 40:
        stance = explanation.low_means
    else:
        stance = "sits roughly in line with peers on this measure"
    return f"This stock {stance}."


@dataclass
class Scorecard:
    """A stock's factor standing versus its peer group."""

    ticker: str
    as_of: date
    price: float
    peer_count: int
    factors: dict[str, list[FactorScore]] = field(default_factory=dict)
    family_percentiles: dict[str, float] = field(default_factory=dict)
    composite_percentile: float = float("nan")
    composite_rank: int = 0
    unavailable_pillars: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of": str(self.as_of),
            "price": self.price,
            "peer_count": self.peer_count,
            "composite_percentile": self.composite_percentile,
            "composite_rank": self.composite_rank,
            "family_percentiles": self.family_percentiles,
            "factors": {
                family: [
                    {
                        "name": f.name,
                        "description": f.description,
                        "raw_value": f.raw_value,
                        "percentile": f.percentile,
                        "peer_median": f.peer_median,
                        "label": f.label,
                    }
                    for f in scores
                ]
                for family, scores in self.factors.items()
            },
            "unavailable_pillars": self.unavailable_pillars,
            "warnings": self.warnings,
        }


def fetch_universe_prices(
    tickers: list[str],
    lookback_days: int = 600,
    end: date | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Download prices for the tickers plus the benchmark.

    ``lookback_days`` defaults to ~600 calendar days because the longest feature needs 252
    *trading* sessions, and weekends/holidays mean that requires roughly 350+ calendar days;
    600 leaves headroom for the warm-up plus gaps.

    Returns (prices, benchmark, missing_tickers). Tickers that return nothing are reported
    rather than silently dropped -- a missing name usually means a typo or a delisting, and
    both are worth knowing about.
    """
    from quant_platform.data.adapters.open_data import YFinancePriceSource

    end = end or date.today()
    start = end - timedelta(days=lookback_days)

    wanted = list(dict.fromkeys([*tickers, BENCHMARK_TICKER]))  # de-dupe, keep order
    source = YFinancePriceSource()
    raw = source.load_prices(wanted, start=start, end=end)

    returned = set(raw["security_id"].unique().to_list())
    missing = [t for t in wanted if t not in returned]
    if missing:
        logger.warning("no data returned for: %s", ", ".join(missing))

    benchmark = raw.filter(pl.col("security_id") == BENCHMARK_TICKER).sort("observation_date")
    if benchmark.is_empty():
        raise RuntimeError(
            f"benchmark {BENCHMARK_TICKER} returned no data; cannot compute beta or "
            f"benchmark-relative features"
        )
    benchmark = benchmark.with_columns(
        (pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1.0).alias(
            "benchmark_return"
        )
    )

    prices = raw.filter(pl.col("security_id") != BENCHMARK_TICKER)
    return prices, benchmark, missing


# --------------------------------------------------------------------------- price cache
CACHE_DIR = Path("data/interim/price_cache")


def _cache_path(tickers: list[str], lookback_days: int, end: date) -> Path:
    """Cache key covering everything that changes the downloaded content."""
    digest = hashlib.sha256(
        f"{sorted(tickers)}|{lookback_days}|{end.isoformat()}".encode()
    ).hexdigest()[:16]
    return CACHE_DIR / f"{end.isoformat()}_{digest}.parquet"


def _with_benchmark_return(benchmark: pl.DataFrame) -> pl.DataFrame:
    """Attach the benchmark return series.

    Computed in exactly one place so the cached and freshly-downloaded paths cannot diverge:
    the cache stores raw price columns only, and this derived column is added on the way out
    of both branches.
    """
    return benchmark.sort("observation_date").with_columns(
        (pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1.0).alias(
            "benchmark_return"
        )
    )


def fetch_universe_prices_cached(
    tickers: list[str],
    lookback_days: int = 600,
    end: date | None = None,
    use_cache: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Cached wrapper around :func:`fetch_universe_prices`.

    Downloading ~110 tickers takes 30-60 seconds, which is fine for a one-off CLI call and
    unusable for a web UI where every page load would repeat it. The cache is keyed on the
    ticker set, lookback, and end date, so a different request still fetches fresh data.

    Entries are keyed by end date, so a new trading day naturally produces a new file rather
    than serving stale prices. Older files are pruned.
    """
    end = end or date.today()
    if not use_cache:
        prices, benchmark, missing = fetch_universe_prices(tickers, lookback_days, end)
        return prices, benchmark, missing

    path = _cache_path(tickers, lookback_days, end)
    if path.exists():
        try:
            cached = pl.read_parquet(path)
            prices = cached.filter(pl.col("security_id") != BENCHMARK_TICKER)
            benchmark = cached.filter(pl.col("security_id") == BENCHMARK_TICKER)
            if not prices.is_empty() and not benchmark.is_empty():
                returned = set(cached["security_id"].unique().to_list())
                missing = [t for t in tickers if t not in returned]
                logger.info("using cached prices (%s)", path.name)
                return prices, _with_benchmark_return(benchmark), missing
        except Exception as exc:  # a corrupt cache must never block a real request
            logger.warning("ignoring unreadable cache %s: %s", path, exc)

    prices, benchmark, missing = fetch_universe_prices(tickers, lookback_days, end)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Store raw columns only; derived series are recomputed on read.
        raw_benchmark = benchmark.drop("benchmark_return", strict=False)
        pl.concat([prices, raw_benchmark], how="diagonal").write_parquet(path)
        for old in CACHE_DIR.glob("*.parquet"):
            if not old.name.startswith(end.isoformat()):
                old.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("could not write price cache: %s", exc)

    return prices, benchmark, missing


def build_peer_features(prices: pl.DataFrame, benchmark: pl.DataFrame) -> pl.DataFrame:
    """Compute the price-derived feature panel for the peer universe."""
    if "shares_outstanding" not in prices.columns:
        # turnover needs a share count; without it that single feature is null rather than
        # silently wrong.
        prices = prices.with_columns(pl.lit(None, dtype=pl.Float64).alias("shares_outstanding"))
    return compute_price_features(prices, benchmark)


def _percentile_of(values: np.ndarray, target: float, higher_is_better: bool) -> float:
    """Percentile rank of ``target`` within ``values`` (0-100)."""
    finite = values[np.isfinite(values)]
    if finite.size < 2 or not np.isfinite(target):
        return float("nan")
    below = float((finite < target).sum())
    pct = below / finite.size * 100.0
    return pct if higher_is_better else 100.0 - pct


def build_scorecard(
    ticker: str,
    features: pl.DataFrame,
    prices: pl.DataFrame,
    as_of: date | None = None,
    min_peers: int = 20,
) -> Scorecard:
    """Rank one ticker against the peer cross-section on the latest available date."""
    ticker = ticker.upper()

    # Use the last date where this ticker actually has features, then take that whole
    # cross-section. Using the panel's global max date would compare the stock against peers
    # on a day it may not have traded.
    ticker_rows = features.filter(pl.col("security_id") == ticker).drop_nulls("momentum_252d")
    if ticker_rows.is_empty():
        raise ValueError(
            f"{ticker} has no computable features. It may be missing from the download, or "
            f"have less than ~1 year of price history (the longest factor needs 252 sessions)."
        )

    target_date = as_of or as_date(ticker_rows["as_of"].max(), f"{ticker} latest feature date")
    cross_section = features.filter(pl.col("as_of") == target_date)
    if cross_section.height < min_peers:
        raise ValueError(
            f"only {cross_section.height} peers have features on {target_date}; need at least "
            f"{min_peers} for a percentile to be meaningful"
        )

    row = cross_section.filter(pl.col("security_id") == ticker)
    if row.is_empty():
        raise ValueError(f"{ticker} has no features on {target_date}")

    price_row = prices.filter(
        (pl.col("security_id") == ticker) & (pl.col("observation_date") <= target_date)
    ).sort("observation_date")
    last_price = float(price_row["close"][-1]) if not price_row.is_empty() else float("nan")

    card = Scorecard(
        ticker=ticker,
        as_of=target_date,
        price=last_price,
        peer_count=cross_section.height - 1,  # exclude the stock itself
        unavailable_pillars=dict(UNAVAILABLE_PILLARS),
    )

    family_scores: dict[str, list[float]] = {}
    for family, definitions in _SCORECARD_FACTORS.items():
        scores: list[FactorScore] = []
        for column, sign, description in definitions:
            if column not in cross_section.columns:
                continue
            peers = cross_section[column].to_numpy().astype(float)
            value = float(row[column][0]) if row[column][0] is not None else float("nan")
            if not np.isfinite(value):
                continue
            pct = _percentile_of(peers, value, higher_is_better=sign > 0)
            if not np.isfinite(pct):
                continue
            finite_peers = peers[np.isfinite(peers)]
            scores.append(
                FactorScore(
                    name=column,
                    description=description,
                    raw_value=value,
                    percentile=pct,
                    peer_median=float(np.median(finite_peers))
                    if finite_peers.size
                    else float("nan"),
                )
            )
        if scores:
            card.factors[family] = scores
            family_scores[family] = [s.percentile for s in scores]
            card.family_percentiles[family] = float(np.mean([s.percentile for s in scores]))

    if not card.family_percentiles:
        raise ValueError(f"no factors could be evaluated for {ticker} on {target_date}")

    # Composite: average the family percentiles this data can actually support, then rank the
    # whole cross-section the same way so the stock's standing is comparable.
    composite_by_security = _composite_across_cross_section(cross_section)
    if ticker in composite_by_security:
        own = composite_by_security[ticker]
        all_values = np.array(list(composite_by_security.values()))
        card.composite_percentile = _percentile_of(all_values, own, higher_is_better=True)
        ordering = sorted(composite_by_security.items(), key=lambda kv: kv[1], reverse=True)
        card.composite_rank = [t for t, _ in ordering].index(ticker) + 1

    card.warnings = [
        "Peer universe is survivorship-biased: it contains only currently-listed companies.",
        "Value and quality pillars are NOT evaluated (no point-in-time fundamentals via this "
        "data source), so the composite reflects momentum, reversal, defensive, and liquidity "
        "characteristics only.",
        "Percentiles describe what the stock looks like today. They are not a forecast and "
        "not a recommendation.",
    ]
    return card


def _composite_across_cross_section(cross_section: pl.DataFrame) -> dict[str, float]:
    """Composite score for every security in the cross-section.

    Computed by ranking each factor within the cross-section, applying its sign, and
    averaging by family then across families. Equal family weighting is used because the
    weights in the research charter assume value and quality are present; silently reusing
    them here would over-weight whatever remains.
    """
    scores: dict[str, list[float]] = {}
    n = cross_section.height
    ids = cross_section["security_id"].to_list()

    for _family, definitions in _SCORECARD_FACTORS.items():
        family_matrix: list[np.ndarray] = []
        for column, sign, _ in definitions:
            if column not in cross_section.columns:
                continue
            values = cross_section[column].to_numpy().astype(float)
            if not np.isfinite(values).any():
                continue
            order = values * sign
            ranks = np.full(n, np.nan)
            finite = np.isfinite(order)
            if finite.sum() < 2:
                continue
            # Percentile rank within the finite subset.
            sub = order[finite]
            sub_ranks = sub.argsort().argsort().astype(float) / max(sub.size - 1, 1)
            ranks[finite] = sub_ranks
            family_matrix.append(ranks)
        if family_matrix:
            stacked = np.vstack(family_matrix)
            with np.errstate(invalid="ignore"):
                family_mean = np.nanmean(stacked, axis=0)
            scores.setdefault("_families", [])
            for i, sec in enumerate(ids):
                scores.setdefault(sec, []).append(float(family_mean[i]))

    scores.pop("_families", None)
    out: dict[str, float] = {}
    for sec, family_values in scores.items():
        finite_values = [v for v in family_values if np.isfinite(v)]
        if finite_values:
            out[sec] = float(np.mean(finite_values))
    return out


def analyze_tickers(
    tickers: list[str],
    peers: list[str] | None = None,
    lookback_days: int = 600,
    end: date | None = None,
    use_cache: bool = True,
) -> tuple[list[Scorecard], list[str]]:
    """Fetch data and build scorecards for one or more tickers.

    The requested tickers are added to the peer universe automatically, so a stock outside
    the default list can still be ranked.
    """
    tickers = [t.upper() for t in tickers]
    universe = list(dict.fromkeys([*(peers or DEFAULT_PEER_UNIVERSE), *tickers]))

    logger.info("downloading %d tickers (%d peers + benchmark)", len(universe), len(universe))
    prices, benchmark, missing = fetch_universe_prices_cached(
        universe, lookback_days, end, use_cache=use_cache
    )

    logger.info("computing features for %d securities", prices["security_id"].n_unique())
    features = build_peer_features(prices, benchmark)

    cards: list[Scorecard] = []
    failures: list[str] = []

    # A requested ticker that returned no data is a real problem for the user (typo, or the
    # company is delisted and yfinance cannot serve it). Report it rather than letting it
    # vanish into a generic "could not score" later.
    for ticker in tickers:
        if ticker in missing:
            failures.append(
                f"{ticker}: no data returned. Check the symbol; note that delisted companies "
                f"cannot be retrieved from this source."
            )

    for ticker in [t for t in tickers if t not in missing]:
        try:
            cards.append(build_scorecard(ticker, features, prices))
        except ValueError as exc:
            logger.warning("could not score %s: %s", ticker, exc)
            failures.append(f"{ticker}: {exc}")
    return cards, failures


def _wrap(text: str, width: int) -> list[str]:
    """Wrap text for fixed-width terminal output."""
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def format_scorecard(card: Scorecard, explain: bool = False) -> str:
    """Render a scorecard as readable text.

    ``explain=True`` adds a plain-language gloss to every value: what the metric measures,
    the number restated with units, and what this stock's reading indicates.
    """
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append(f"  {card.ticker}  --  factor scorecard as of {card.as_of}")
    lines.append(f"  last close ${card.price:,.2f}   |   ranked against {card.peer_count} peers")
    lines.append("=" * 74)

    if np.isfinite(card.composite_percentile):
        lines.append("")
        lines.append(
            f"  COMPOSITE STANDING: {card.composite_percentile:.0f}th percentile "
            f"(#{card.composite_rank} of {card.peer_count + 1})"
        )
        lines.append("")
        for chunk in _wrap(COMPOSITE_EXPLANATION, 70):
            lines.append(f"  {chunk}")

    for family, scores in card.factors.items():
        family_pct = card.family_percentiles.get(family, float("nan"))
        lines.append("")
        lines.append(f"  {family.upper()}  ({family_pct:.0f}th percentile overall)")
        lines.append("  " + "-" * 70)
        if explain and family in FAMILY_EXPLANATIONS:
            for chunk in _wrap(FAMILY_EXPLANATIONS[family], 68):
                lines.append(f"    {chunk}")
            lines.append("")
        for score in scores:
            bar_len = round(score.percentile / 5)
            bar = "#" * bar_len + "." * (20 - bar_len)
            lines.append(
                f"    {score.description:<44s} {bar} {score.percentile:5.0f}%  ({score.label})"
            )
            explanation = FACTOR_GLOSSARY.get(score.name)
            plain = ""
            if explanation and explanation.read_value:
                try:
                    plain = explanation.read_value(score.raw_value)
                except Exception:
                    plain = ""
            if plain:
                lines.append(f"      -> {plain}")
            lines.append(
                f"      {'raw: ' + f'{score.raw_value:+.4f}':<20s} peer median: "
                f"{score.peer_median:+.4f}"
            )
            if explain and explanation:
                for chunk in _wrap(f"MEASURES: {explanation.measures}", 64):
                    lines.append(f"        {chunk}")
                for chunk in _wrap(f"READING: {interpret_score(score)}", 64):
                    lines.append(f"        {chunk}")
                lines.append("")

    if card.unavailable_pillars:
        lines.append("")
        lines.append("  NOT EVALUATED")
        lines.append("  " + "-" * 70)
        for pillar, reason in card.unavailable_pillars.items():
            lines.append(f"    {pillar:<12s} {reason}")

    lines.append("")
    lines.append("  CAVEATS")
    lines.append("  " + "-" * 70)
    for warning in card.warnings:
        lines.append(f"    - {warning}")
    lines.append("")
    return "\n".join(lines)


def compare_scorecards(cards: list[Scorecard]) -> pl.DataFrame:
    """Side-by-side comparison of several scorecards."""
    if not cards:
        return pl.DataFrame()
    families = sorted({f for c in cards for f in c.family_percentiles})
    return pl.DataFrame(
        [
            {
                "ticker": c.ticker,
                "price": c.price,
                "composite_pct": c.composite_percentile,
                "rank": c.composite_rank,
                **{f: c.family_percentiles.get(f, float("nan")) for f in families},
            }
            for c in cards
        ]
    ).sort("composite_pct", descending=True, nulls_last=True)
