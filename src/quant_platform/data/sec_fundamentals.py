"""Point-in-time fundamentals from SEC EDGAR companyfacts.

This is the one open data source that gets point-in-time correctness *right*: every XBRL
fact carries a ``filed`` date, which is the authoritative moment the number became public.
That lets value and quality factors be computed without the look-ahead bias that plagues
fundamental data pulled from a screener.

Three practical realities shape this module:

**XBRL tags vary by filer, and drift over time.** Apple stopped reporting the ``Revenues``
tag in 2018 and moved to ``RevenueFromContractWithCustomerExcludingAssessedTax``. A single
hardcoded tag silently yields stale or missing data, so every concept has an ordered list of
fallbacks and the first one with usable data wins.

**Each companyfacts document is multi-megabyte.** Apple's is ~3.7 MB. Fetching a hundred
peers uncached would move ~400 MB and hammer SEC's servers, so extracted results are cached
on disk and requests are rate-limited.

**SEC requires attribution.** Their fair-access policy asks for a descriptive User-Agent
containing a reachable contact address, and rate-limits to roughly 10 requests/second. Both
are respected here; requests without a contact are refused locally rather than sent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("data.sec")

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

CACHE_DIR = Path("data/interim/sec_cache")
# Fundamentals change when a company files, i.e. quarterly. A week-long cache is far shorter
# than that cadence while still sparing SEC a re-download on every run.
CACHE_TTL_DAYS = 7

# SEC asks for <= 10 requests/second. We stay comfortably under.
_MIN_REQUEST_INTERVAL = 0.15
_last_request_at = 0.0

# Ordered tag fallbacks. First tag with usable data wins.
#
# `is_flow` marks income/cash-flow items measured OVER a period (summable to trailing twelve
# months). Balance-sheet items are instantaneous and must never be summed -- adding four
# quarters of total assets would quadruple the balance sheet.
CONCEPTS: dict[str, dict[str, Any]] = {
    "revenue": {
        "tags": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ],
        "is_flow": True,
    },
    "net_income": {"tags": ["NetIncomeLoss", "ProfitLoss"], "is_flow": True},
    "gross_profit": {"tags": ["GrossProfit"], "is_flow": True},
    "operating_income": {"tags": ["OperatingIncomeLoss"], "is_flow": True},
    "operating_cash_flow": {
        "tags": [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
        "is_flow": True,
    },
    "capital_expenditure": {
        "tags": [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        ],
        "is_flow": True,
    },
    "total_assets": {"tags": ["Assets"], "is_flow": False},
    "total_equity": {
        "tags": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "is_flow": False,
    },
    "total_liabilities": {"tags": ["Liabilities"], "is_flow": False},
    "total_debt": {
        "tags": [
            "LongTermDebtNoncurrent",
            "LongTermDebt",
            "DebtLongtermAndShorttermCombinedAmount",
        ],
        "is_flow": False,
    },
    "shares_outstanding": {
        "tags": [
            "CommonStockSharesOutstanding",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
        ],
        "is_flow": False,
    },
}


class SecUnavailable(RuntimeError):
    """Raised when SEC data cannot be fetched."""


@dataclass(frozen=True)
class Fact:
    """One reported value with its publication date."""

    concept: str
    value: float
    period_end: date
    filed: date
    form: str
    fiscal_period: str
    is_flow: bool


def _user_agent() -> str:
    """The User-Agent SEC requires. Refuses to proceed without a contact address."""
    import os

    agent = os.environ.get("QP_SEC_USER_AGENT", "").strip()
    if not agent or "@" not in agent:
        raise SecUnavailable(
            "SEC EDGAR requires a descriptive User-Agent containing a reachable contact "
            "email (their fair-access policy). Set it in .env:\n"
            "    QP_SEC_USER_AGENT='Your Name your@email.com'"
        )
    return agent


def _throttled_get(url: str, timeout: int = 30) -> Any:
    """GET with rate limiting, so we stay inside SEC's request budget."""
    global _last_request_at
    import requests

    elapsed = time.monotonic() - _last_request_at
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_at = time.monotonic()

    response = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=timeout)
    if response.status_code == 403:
        raise SecUnavailable(
            "SEC returned 403 Forbidden. This usually means the User-Agent lacks a valid "
            "contact address, or requests were sent too quickly."
        )
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise SecUnavailable(f"SEC returned HTTP {response.status_code} for {url}")
    return response.json()


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _read_cache(name: str, ttl_days: int = CACHE_TTL_DAYS) -> Any | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    age_days = (time.time() - path.stat().st_mtime) / 86400
    if age_days > ttl_days:
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception as exc:  # a corrupt cache must never block a real request
        logger.warning("ignoring unreadable cache %s: %s", path, exc)
        return None


def _write_cache(name: str, payload: Any) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _cache_path(name).with_suffix(".tmp")
        with tmp.open("w") as fh:
            json.dump(payload, fh)
        tmp.replace(_cache_path(name))
    except Exception as exc:
        logger.warning("could not write cache %s: %s", name, exc)


def ticker_to_cik_map(refresh: bool = False) -> dict[str, str]:
    """Map ticker -> zero-padded CIK, from SEC's official list."""
    if not refresh:
        cached = _read_cache("ticker_cik_map", ttl_days=30)
        if cached:
            return cached

    payload = _throttled_get(TICKER_MAP_URL)
    if not payload:
        raise SecUnavailable("could not download SEC's ticker/CIK map")

    mapping = {
        str(entry["ticker"]).upper(): str(entry["cik_str"]).zfill(10) for entry in payload.values()
    }
    _write_cache("ticker_cik_map", mapping)
    logger.info("SEC ticker map: %d companies", len(mapping))
    return mapping


def _extract_facts(payload: dict[str, Any]) -> list[Fact]:
    """Pull the concepts we need out of a companyfacts document.

    Only USD (or share-count) units are kept, and only facts carrying both a period end and a
    filing date -- a fact without ``filed`` cannot be placed in time and is therefore unusable
    for point-in-time work.
    """
    gaap = payload.get("facts", {}).get("us-gaap", {})
    dei = payload.get("facts", {}).get("dei", {})
    facts: list[Fact] = []

    for concept, spec in CONCEPTS.items():
        chosen: list[Fact] = []
        for tag in spec["tags"]:
            source = gaap.get(tag) or dei.get(tag)
            if not source:
                continue
            units = source.get("units", {})
            unit_key = next(
                (k for k in ("USD", "shares", "USD/shares") if k in units),
                next(iter(units), None),
            )
            if unit_key is None:
                continue

            for item in units[unit_key]:
                if "end" not in item or "filed" not in item or item.get("val") is None:
                    continue
                try:
                    chosen.append(
                        Fact(
                            concept=concept,
                            value=float(item["val"]),
                            period_end=date.fromisoformat(item["end"]),
                            filed=date.fromisoformat(item["filed"]),
                            form=str(item.get("form", "")),
                            fiscal_period=str(item.get("fp", "")),
                            is_flow=bool(spec["is_flow"]),
                        )
                    )
                except (ValueError, TypeError):
                    continue
            if chosen:
                break  # first tag with usable data wins
        facts.extend(chosen)
    return facts


def fetch_company_facts(cik: str, ticker: str = "") -> list[Fact]:
    """Fetch and extract one company's facts, using the disk cache when fresh."""
    cached = _read_cache(f"facts_{cik}")
    if cached is not None:
        return [
            Fact(
                concept=f["concept"],
                value=f["value"],
                period_end=date.fromisoformat(f["period_end"]),
                filed=date.fromisoformat(f["filed"]),
                form=f["form"],
                fiscal_period=f["fiscal_period"],
                is_flow=f["is_flow"],
            )
            for f in cached
        ]

    payload = _throttled_get(COMPANYFACTS_URL.format(cik=cik))
    if payload is None:
        logger.debug("no companyfacts for %s (%s)", ticker or cik, cik)
        _write_cache(f"facts_{cik}", [])
        return []

    facts = _extract_facts(payload)
    _write_cache(
        f"facts_{cik}",
        [
            {
                "concept": f.concept,
                "value": f.value,
                "period_end": f.period_end.isoformat(),
                "filed": f.filed.isoformat(),
                "form": f.form,
                "fiscal_period": f.fiscal_period,
                "is_flow": f.is_flow,
            }
            for f in facts
        ],
    )
    return facts


def latest_values_as_of(facts: list[Fact], as_of: date) -> dict[str, float]:
    """Reduce facts to the values a reader could have known on ``as_of``.

    The point-in-time filter is the first line: only facts with ``filed <= as_of`` are
    considered, so a filing published tomorrow is invisible today.

    Flow items (revenue, earnings, cash flow) are summed over the four most recent distinct
    quarterly periods to give a trailing-twelve-month figure, falling back to the latest
    annual report when quarterly coverage is incomplete. Stock items (assets, equity) take
    the single most recent reported value -- summing a balance sheet would be nonsense.
    """
    visible = [f for f in facts if f.filed <= as_of]
    if not visible:
        return {}

    out: dict[str, float] = {}
    by_concept: dict[str, list[Fact]] = {}
    for fact in visible:
        by_concept.setdefault(fact.concept, []).append(fact)

    for concept, items in by_concept.items():
        if not items:
            continue
        if not items[0].is_flow:
            # Instantaneous: take the most recently filed, breaking ties on period end.
            newest = max(items, key=lambda f: (f.filed, f.period_end))
            out[concept] = newest.value
            continue

        # Flow: prefer a TTM sum of four distinct recent quarters.
        quarterly = [
            f for f in items if f.form.startswith("10-Q") or f.fiscal_period.startswith("Q")
        ]
        by_period: dict[date, Fact] = {}
        for fact in sorted(quarterly, key=lambda f: f.filed):
            by_period[fact.period_end] = fact  # later filing supersedes (restatements)
        recent = sorted(by_period.values(), key=lambda f: f.period_end, reverse=True)[:4]

        if len(recent) == 4:
            span_days = (recent[0].period_end - recent[-1].period_end).days
            if 250 <= span_days <= 400:  # four quarters really do span about a year
                out[concept] = sum(f.value for f in recent)
                continue

        annual = [f for f in items if f.form.startswith("10-K") or f.fiscal_period == "FY"]
        if annual:
            newest = max(annual, key=lambda f: (f.period_end, f.filed))
            out[concept] = newest.value
        elif recent:
            # Neither clean quarterly coverage nor an annual report: scale what we have
            # rather than silently reporting a partial year as if it were annual.
            out[concept] = sum(f.value for f in recent) * (4.0 / len(recent))
    return out


def fetch_fundamentals_panel(
    tickers: list[str], as_of: date | None = None, show_progress: bool = True
) -> pl.DataFrame:
    """Build a point-in-time fundamentals frame for a list of tickers.

    Returns one row per ticker with the values knowable on ``as_of``, plus the latest filing
    date behind them so callers can see how stale each row is.
    """
    as_of = as_of or date.today()
    mapping = ticker_to_cik_map()

    rows: list[dict[str, Any]] = []
    unmapped: list[str] = []

    for i, ticker in enumerate(tickers, start=1):
        cik = mapping.get(ticker.upper())
        if cik is None:
            unmapped.append(ticker)
            continue
        try:
            facts = fetch_company_facts(cik, ticker)
        except SecUnavailable:
            raise
        except Exception as exc:
            logger.warning("could not fetch %s: %s", ticker, exc)
            continue

        values = latest_values_as_of(facts, as_of)
        if not values:
            continue

        visible = [f for f in facts if f.filed <= as_of]
        rows.append(
            {
                "security_id": ticker.upper(),
                "as_of": as_of,
                "latest_filing": max(f.filed for f in visible) if visible else None,
                **values,
            }
        )
        if show_progress and i % 25 == 0:
            logger.info("  SEC: %d/%d tickers", i, len(tickers))

    if unmapped:
        logger.warning(
            "%d ticker(s) have no SEC CIK (likely foreign issuers or ETFs): %s",
            len(unmapped),
            ", ".join(unmapped[:8]),
        )
    if not rows:
        return pl.DataFrame()

    frame = pl.DataFrame(rows, infer_schema_length=None)
    logger.info("SEC fundamentals: %d/%d tickers with data", frame.height, len(tickers))
    return frame


def compute_value_quality_features(
    fundamentals: pl.DataFrame, market_caps: dict[str, float]
) -> pl.DataFrame:
    """Derive value and quality ratios from fundamentals plus market capitalization.

    Denominators are guarded: negative equity and zero interest expense are common in real
    filings and would otherwise produce infinities that dominate any cross-sectional ranking.
    Ratios that cannot be computed are left null rather than defaulted, so a missing value
    stays visibly missing.
    """
    if fundamentals.is_empty():
        return fundamentals

    df = fundamentals.with_columns(
        pl.col("security_id")
        .replace_strict(market_caps, default=None, return_dtype=pl.Float64)
        .alias("market_cap")
    )

    def ratio(numerator: str, denominator: str, name: str) -> pl.Expr:
        if numerator not in df.columns or denominator not in df.columns:
            return pl.lit(None, dtype=pl.Float64).alias(name)
        return (
            pl.when(pl.col(denominator).abs() > 1.0)
            .then(pl.col(numerator) / pl.col(denominator))
            .otherwise(None)
            .alias(name)
        )

    exprs = [
        # Value: what you get per dollar of market price.
        ratio("net_income", "market_cap", "earnings_yield"),
        ratio("total_equity", "market_cap", "book_to_market"),
        ratio("revenue", "market_cap", "sales_to_price"),
        # Quality: how productive and how leveraged the business is.
        ratio("gross_profit", "total_assets", "gross_profitability"),
        ratio("net_income", "total_equity", "return_on_equity"),
        ratio("net_income", "total_assets", "return_on_assets"),
        ratio("operating_income", "revenue", "operating_margin"),
        ratio("total_debt", "total_assets", "debt_to_assets"),
    ]
    df = df.with_columns(exprs)

    if {"operating_cash_flow", "capital_expenditure", "market_cap"} <= set(df.columns):
        df = df.with_columns(
            pl.when(pl.col("market_cap") > 1.0)
            .then(
                (pl.col("operating_cash_flow") - pl.col("capital_expenditure").abs())
                / pl.col("market_cap")
            )
            .otherwise(None)
            .alias("fcf_yield")
        )

    # Clip implausible ratios. Tiny denominators in real filings produce values in the
    # thousands, which would swamp every percentile in the cross-section.
    ratio_columns = [
        c
        for c in (
            "earnings_yield",
            "book_to_market",
            "sales_to_price",
            "fcf_yield",
            "gross_profitability",
            "return_on_equity",
            "return_on_assets",
            "operating_margin",
            "debt_to_assets",
        )
        if c in df.columns
    ]
    return df.with_columns([pl.col(c).clip(-10.0, 10.0).alias(c) for c in ratio_columns])


VALUE_FEATURES = ("earnings_yield", "book_to_market", "sales_to_price", "fcf_yield")
QUALITY_FEATURES = (
    "gross_profitability",
    "return_on_equity",
    "return_on_assets",
    "operating_margin",
    "debt_to_assets",
)
