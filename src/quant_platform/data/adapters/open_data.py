"""Open-data adapters (spec section 10.2).

These pull from free sources. Each one is deliberately explicit about what it *cannot* do,
because the gap between free data and research-grade data is exactly where backtests go wrong:

* **yfinance** returns only currently-listed tickers. Ask for a company that went bankrupt and
  you get nothing -- the dataset is survivorship-biased by construction, and no amount of
  careful coding downstream repairs it.
* **SEC companyfacts** has genuine filing dates (excellent for PIT), but mapping XBRL tags to
  a clean fundamental panel is lossy and coverage varies by filer.
* **FRED** serves latest-revised values by default. ALFRED carries true vintages; without it,
  macro history is contaminated by revisions that were unknowable at the time.
* **Kenneth French** factor data is research-grade but monthly/daily aggregates only.

None of these provide reliable historical S&P 500 constituents. That remains a
bring-your-own-file problem, which is why ``LocalConstituentSource`` exists.

All network access is lazy and optional: importing this module never touches the network, and
missing packages/credentials produce a clear message rather than a traceback.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta

import polars as pl

from quant_platform.data.base import SourceProvenance
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("data.open")


class DataSourceUnavailable(RuntimeError):
    """Raised when a source needs a package, credential, or network access it lacks."""


def _require(package: str, extra: str) -> object:
    import importlib

    try:
        return importlib.import_module(package)
    except ImportError as exc:
        raise DataSourceUnavailable(
            f"'{package}' is required for this adapter but is not installed. "
            f"Install it with: uv sync --extra {extra}"
        ) from exc


class YFinancePriceSource:
    """Daily bars from Yahoo Finance.

    Suitable for exploration and for wiring up a real-data run. NOT suitable for research
    conclusions: the survivorship problem below is fundamental, not a configuration issue.
    """

    PROVENANCE = SourceProvenance(
        name="yfinance",
        description="Free daily OHLCV from Yahoo Finance.",
        historical_constituents_complete=False,
        delisting_returns_complete=False,
        corporate_actions_complete=True,  # adjusted prices are provided
        requires_network=True,
        known_limitations=[
            "SURVIVORSHIP BIAS: only currently-listed tickers are queryable. Securities that "
            "were delisted, acquired, or went bankrupt are absent, so any universe built from "
            "this source silently excludes failures and overstates historical returns.",
            "No delisting returns: a security that stops trading simply has no further bars, "
            "so terminal losses are never booked.",
            "Adjustment methodology is undocumented and occasionally revised without notice.",
            "Ticker reuse is not handled: a recycled symbol may splice two unrelated companies "
            "into one price history.",
            "Rate-limited and unversioned; results are not reproducible over time.",
        ],
    )

    def __init__(self, symbol_to_security_id: dict[str, str] | None = None) -> None:
        self.symbol_map = symbol_to_security_id or {}

    def load_prices(
        self,
        security_ids: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame:
        yf = _require("yfinance", "data")

        symbols = security_ids or list(self.symbol_map)
        if not symbols:
            raise ValueError("YFinancePriceSource needs tickers to download")

        logger.warning(
            "Downloading from Yahoo Finance. This data is SURVIVORSHIP-BIASED: delisted "
            "securities cannot be retrieved. See docs/limitations.md."
        )
        raw = yf.download(  # type: ignore[attr-defined]
            tickers=" ".join(symbols),
            start=start.isoformat() if start else None,
            end=end.isoformat() if end else None,
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="ticker",
        )
        if raw is None or len(raw) == 0:
            raise DataSourceUnavailable(
                f"Yahoo Finance returned no data for {symbols[:5]}... "
                f"(network issue, or all tickers are delisted/invalid)"
            )

        frames: list[pl.DataFrame] = []
        for symbol in symbols:
            try:
                sub = raw[symbol] if len(symbols) > 1 else raw
            except (KeyError, TypeError):
                logger.warning("no data returned for %s (possibly delisted)", symbol)
                continue
            sub = sub.dropna(how="all")
            if sub.empty:
                continue
            frames.append(
                pl.DataFrame(
                    {
                        "security_id": [self.symbol_map.get(symbol, symbol)] * len(sub),
                        "symbol": [symbol] * len(sub),
                        "observation_date": [d.date() for d in sub.index],
                        "open": sub["Open"].to_numpy(),
                        "high": sub["High"].to_numpy(),
                        "low": sub["Low"].to_numpy(),
                        "close": sub["Close"].to_numpy(),
                        "adjusted_close": sub["Adj Close"].to_numpy(),
                        "volume": sub["Volume"].to_numpy().astype(float),
                    }
                )
            )

        if not frames:
            raise DataSourceUnavailable("no usable price data returned")

        df = pl.concat(frames).drop_nulls("close").filter(pl.col("close") > 0)
        return df.with_columns(
            [
                (pl.col("observation_date").cast(pl.Datetime) + pl.duration(hours=21)).alias(
                    "available_at"
                ),
                pl.lit(self.PROVENANCE.name).alias("source"),
                (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
            ]
        ).sort(["security_id", "observation_date"])


class FredMacroSource:
    """Macroeconomic series from FRED.

    Standard FRED endpoints return the *latest revised* value for each period. Using those in
    a backtest is a real look-ahead leak: the revised number did not exist at the time. This
    adapter therefore marks vintages as incomplete and, where ALFRED vintage data is
    unavailable, applies a conservative publication lag rather than pretending same-day
    availability.
    """

    PROVENANCE = SourceProvenance(
        name="fred",
        description="Federal Reserve Economic Data.",
        macro_vintages_complete=False,
        requires_credentials=True,
        requires_network=True,
        known_limitations=[
            "REVISION CONTAMINATION: the standard API returns latest-revised values, not the "
            "original prints. Series like GDP and payrolls are revised substantially, so "
            "backtests using them may consume numbers that were unknowable at the time.",
            "Publication lags are approximated per-frequency unless ALFRED vintages are used.",
            "Requires a free API key in QP_FRED_API_KEY.",
        ],
    )

    # Conservative publication lags by frequency, in days.
    DEFAULT_LAGS = {"daily": 1, "weekly": 5, "monthly": 30, "quarterly": 60}

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("QP_FRED_API_KEY", "")

    def load_series(
        self, series_ids: list[str], start: date | None = None, end: date | None = None
    ) -> pl.DataFrame:
        requests = _require("requests", "data")
        if not self.api_key:
            raise DataSourceUnavailable(
                "FRED requires an API key. Get a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html and set QP_FRED_API_KEY "
                "in your .env (see .env.example)."
            )

        rows: list[dict[str, object]] = []
        for series_id in series_ids:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
            }
            if start:
                params["observation_start"] = start.isoformat()
            if end:
                params["observation_end"] = end.isoformat()

            resp = requests.get(  # type: ignore[attr-defined]
                "https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30
            )
            if resp.status_code != 200:
                raise DataSourceUnavailable(
                    f"FRED returned HTTP {resp.status_code} for {series_id}. "
                    f"(Credentials are never logged.)"
                )
            payload = resp.json()
            freq = "daily" if series_id.startswith("DGS") else "monthly"
            lag = self.DEFAULT_LAGS.get(freq, 30)

            for obs in payload.get("observations", []):
                if obs.get("value") in (".", "", None):
                    continue
                obs_date = date.fromisoformat(obs["date"])
                rows.append(
                    {
                        "series_id": series_id,
                        "observation_date": obs_date,
                        "value": float(obs["value"]),
                        # Approximated: the true first-print date needs ALFRED vintages.
                        "available_at": datetime.combine(
                            obs_date + timedelta(days=lag), datetime.min.time()
                        )
                        + timedelta(hours=8, minutes=30),
                        "source": self.PROVENANCE.name,
                        "revision_id": 0,
                        "is_revision": False,
                        "frequency": freq,
                    }
                )

        if not rows:
            raise DataSourceUnavailable(f"FRED returned no observations for {series_ids}")
        logger.warning(
            "FRED values are latest-revised, not original prints. Publication lag was "
            "APPROXIMATED. See docs/limitations.md."
        )
        return pl.DataFrame(rows).sort(["series_id", "observation_date"])


class SecCompanyFactsSource:
    """Fundamentals from SEC EDGAR companyfacts.

    The genuine strength here is ``filed`` -- a real, authoritative publication date, which is
    exactly what point-in-time correctness requires. The weakness is that XBRL tags vary by
    filer, so mapping them into a uniform panel loses coverage.
    """

    PROVENANCE = SourceProvenance(
        name="sec_companyfacts",
        description="SEC EDGAR XBRL company facts.",
        point_in_time_fundamentals_complete=True,  # real filing dates
        requires_network=True,
        known_limitations=[
            "XBRL tag coverage varies by filer; some concepts are missing or use uncommon "
            "tags, so the derived panel is incomplete rather than uniformly populated.",
            "Restatements appear as additional facts for the same period; the adapter keeps "
            "all of them with distinct available_at values so PIT selection stays correct.",
            "Requires a descriptive User-Agent with a contact email per SEC fair-access policy "
            "(QP_SEC_USER_AGENT).",
            "Covers SEC filers only: no foreign issuers that do not file XBRL.",
        ],
    )

    TAG_MAP = {
        "Revenues": "revenue",
        "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
        "NetIncomeLoss": "net_income",
        "Assets": "total_assets",
        "Liabilities": "total_liabilities",
        "StockholdersEquity": "total_equity",
        "CashAndCashEquivalentsAtCarryingValue": "cash_and_equivalents",
        "GrossProfit": "gross_profit",
        "OperatingIncomeLoss": "operating_income",
        "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    }

    def __init__(self, user_agent: str | None = None) -> None:
        self.user_agent = user_agent or os.environ.get("QP_SEC_USER_AGENT", "")

    def load_fundamentals(
        self,
        security_ids: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        cik_map: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        requests = _require("requests", "data")
        if not self.user_agent or "@" not in self.user_agent:
            raise DataSourceUnavailable(
                "SEC EDGAR requires a descriptive User-Agent containing a contact email "
                "(fair-access policy). Set QP_SEC_USER_AGENT='Your Name your@email.com'."
            )
        if not cik_map:
            raise DataSourceUnavailable(
                "SEC companyfacts is keyed by CIK. Supply cik_map={security_id: cik}."
            )

        rows: list[dict[str, object]] = []
        for security_id, cik in cik_map.items():
            if security_ids and security_id not in security_ids:
                continue
            padded = str(cik).zfill(10)
            resp = requests.get(  # type: ignore[attr-defined]
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded}.json",
                headers={"User-Agent": self.user_agent},
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning("SEC returned HTTP %s for CIK %s", resp.status_code, padded)
                continue

            facts = resp.json().get("facts", {}).get("us-gaap", {})
            by_period: dict[tuple[date, date], dict[str, object]] = {}
            for tag, field_name in self.TAG_MAP.items():
                for unit_rows in facts.get(tag, {}).get("units", {}).values():
                    for item in unit_rows:
                        if "end" not in item or "filed" not in item:
                            continue
                        period_end = date.fromisoformat(item["end"])
                        filed = date.fromisoformat(item["filed"])
                        if start and period_end < start:
                            continue
                        if end and period_end > end:
                            continue
                        key = (period_end, filed)
                        entry = by_period.setdefault(
                            key,
                            {
                                "security_id": security_id,
                                "observation_date": period_end,
                                "fiscal_period_end": period_end,
                                "filing_date": filed,
                                # The authoritative PIT timestamp.
                                "available_at": datetime.combine(filed, datetime.min.time())
                                + timedelta(hours=17),
                                "source": self.PROVENANCE.name,
                                "period_type": "quarterly"
                                if item.get("fp", "").startswith("Q")
                                else "annual",
                            },
                        )
                        entry[field_name] = float(item["val"])
            rows.extend(by_period.values())

        if not rows:
            raise DataSourceUnavailable("SEC returned no usable facts for the requested CIKs")
        return pl.DataFrame(rows).sort(["security_id", "observation_date"])


class KenFrenchFactorSource:
    """Fama-French factor returns from the Dartmouth data library.

    Research-grade and revision-free, used for multifactor alpha attribution.
    """

    PROVENANCE = SourceProvenance(
        name="ken_french",
        description="Fama-French research factor returns.",
        requires_network=True,
        known_limitations=[
            "Aggregate factor returns only: no security-level data.",
            "Factor construction follows the library's own universe rules, which will not "
            "match this platform's universe exactly.",
        ],
    )

    URL = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_daily_CSV.zip"
    )

    def load_factors(self, start: date | None = None, end: date | None = None) -> pl.DataFrame:
        import io
        import zipfile

        requests = _require("requests", "data")
        resp = requests.get(self.URL, timeout=60)  # type: ignore[attr-defined]
        if resp.status_code != 200:
            raise DataSourceUnavailable(f"Ken French library returned HTTP {resp.status_code}")

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = zf.namelist()[0]
            text = zf.read(name).decode("latin-1")

        rows: list[dict[str, object]] = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5 or not parts[0].isdigit() or len(parts[0]) != 8:
                continue
            d = date(int(parts[0][:4]), int(parts[0][4:6]), int(parts[0][6:8]))
            if (start and d < start) or (end and d > end):
                continue
            try:
                rows.append(
                    {
                        "observation_date": d,
                        "mkt_rf": float(parts[1]) / 100.0,
                        "smb": float(parts[2]) / 100.0,
                        "hml": float(parts[3]) / 100.0,
                        "rf": float(parts[4]) / 100.0,
                        "source": self.PROVENANCE.name,
                    }
                )
            except ValueError:
                continue

        if not rows:
            raise DataSourceUnavailable("no factor rows parsed from Ken French archive")
        return pl.DataFrame(rows).sort("observation_date")


OPEN_DATA_SOURCES = {
    "yfinance": YFinancePriceSource,
    "fred": FredMacroSource,
    "sec": SecCompanyFactsSource,
    "ken_french": KenFrenchFactorSource,
}


def describe_open_data_limitations() -> list[str]:
    """Every open-data limitation, for the CLI and reports."""
    out: list[str] = []
    for cls in (YFinancePriceSource, FredMacroSource, SecCompanyFactsSource, KenFrenchFactorSource):
        prov: SourceProvenance = cls.PROVENANCE
        out.append(f"[{prov.name}] {prov.description}")
        out.extend(f"    - {limitation}" for limitation in prov.limitation_summary())
    out.append(
        "[constituents] No open source provides reliable historical S&P 500 membership. "
        "Supply a vendor file via LocalConstituentSource, or accept explicitly-marked "
        "survivorship bias."
    )
    return out
