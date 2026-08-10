"""Factor diagnostics (spec section 17).

Before a feature earns a place in a model it should survive these questions: does it have
coverage, does it rank future returns, is that ranking *stable*, does it survive trading
costs, and is it merely a disguised bet on sector, size, or beta?

The last question matters most. A "quality" factor that is really a low-beta bet is not
adding information -- it is repackaging a risk premium that could be bought far more cheaply.
``exposure_analysis`` exists to catch that.

Alphalens is the conventional tool here, but every metric below is computed internally so the
platform's critical diagnostics never depend on an optional package that may not install.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from quant_platform.evaluation.metrics import ic_information_ratio, spearman_ic


@dataclass
class FactorReport:
    """Diagnostics for one factor."""

    name: str
    coverage: float = float("nan")
    missing_rate: float = float("nan")
    mean_ic: float = float("nan")
    ic_std: float = float("nan")
    ic_ir: float = float("nan")
    positive_ic_pct: float = float("nan")
    n_periods: int = 0
    quantile_returns: dict[int, float] = field(default_factory=dict)
    top_minus_bottom: float = float("nan")
    turnover: float = float("nan")
    exposures: dict[str, float] = field(default_factory=dict)
    subperiod_ic: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, object]:
        return {
            "factor": self.name,
            "coverage": self.coverage,
            "missing_rate": self.missing_rate,
            "mean_ic": self.mean_ic,
            "ic_ir": self.ic_ir,
            "positive_ic_pct": self.positive_ic_pct,
            "top_minus_bottom": self.top_minus_bottom,
            "turnover": self.turnover,
            "n_periods": self.n_periods,
            **{f"exposure_{k}": v for k, v in self.exposures.items()},
        }


def coverage_stats(panel: pl.DataFrame, factor: str) -> tuple[float, float]:
    """Fraction of rows with a usable value, and the missing rate."""
    if panel.is_empty() or factor not in panel.columns:
        return float("nan"), float("nan")
    total = panel.height
    present = panel[factor].drop_nulls().drop_nans().len()
    return present / total, 1.0 - present / total


def information_coefficients(
    panel: pl.DataFrame, factor: str, target: str, min_cross_section: int = 5
) -> pl.DataFrame:
    """Per-date rank IC between a factor and the realized target."""
    if panel.is_empty() or factor not in panel.columns or target not in panel.columns:
        return pl.DataFrame(schema={"as_of": pl.Date, "ic": pl.Float64, "n": pl.Int64})

    rows: list[dict[str, object]] = []
    for as_of, group in panel.group_by("as_of", maintain_order=True):
        d = as_of[0] if isinstance(as_of, tuple) else as_of
        if group.height < min_cross_section:
            continue
        ic = spearman_ic(group[factor].to_numpy(), group[target].to_numpy())
        if np.isfinite(ic):
            rows.append({"as_of": d, "ic": ic, "n": group.height})
    if not rows:
        return pl.DataFrame(schema={"as_of": pl.Date, "ic": pl.Float64, "n": pl.Int64})
    return pl.DataFrame(rows).sort("as_of")


def quantile_returns(
    panel: pl.DataFrame, factor: str, target: str, n_quantiles: int = 5
) -> dict[int, float]:
    """Mean forward return by factor quantile.

    A monotone progression across quantiles is far more convincing than a single IC number:
    it shows the factor ranks consistently rather than being driven by one extreme tail.
    """
    if panel.is_empty() or factor not in panel.columns:
        return {}
    ranked = panel.filter(pl.col(factor).is_not_null() & pl.col(target).is_not_null())
    if ranked.is_empty():
        return {}
    ranked = ranked.with_columns(
        ((pl.col(factor).rank("average").over("as_of") - 1) * n_quantiles // pl.len().over("as_of"))
        .clip(0, n_quantiles - 1)
        .cast(pl.Int64)
        .alias("_q")
    )
    agg = ranked.group_by("_q").agg(pl.col(target).mean().alias("mean_target")).sort("_q")
    return {int(r["_q"]) + 1: float(r["mean_target"]) for r in agg.iter_rows(named=True)}


def factor_turnover(panel: pl.DataFrame, factor: str, top_quantile: float = 0.2) -> float:
    """How much the top bucket changes between periods.

    High turnover means the factor's signal decays fast, so trading costs will consume more
    of whatever edge the IC suggests.
    """
    if panel.is_empty() or factor not in panel.columns:
        return float("nan")
    dates = sorted(panel["as_of"].unique().to_list())
    if len(dates) < 2:
        return float("nan")

    previous: set[str] = set()
    overlaps: list[float] = []
    for d in dates:
        group = panel.filter(pl.col("as_of") == d).drop_nulls(factor)
        if group.height < 5:
            continue
        k = max(1, int(group.height * top_quantile))
        top = set(group.sort(factor, descending=True).head(k)["security_id"].to_list())
        if previous:
            retained = len(top & previous) / max(len(top), 1)
            overlaps.append(1.0 - retained)
        previous = top
    return float(np.mean(overlaps)) if overlaps else float("nan")


def exposure_analysis(
    panel: pl.DataFrame, factor: str, exposure_columns: list[str] | None = None
) -> dict[str, float]:
    """Correlation of a factor with common risk characteristics.

    A high |correlation| means the factor is substantially a bet on that characteristic. For
    example a 'quality' score correlating -0.8 with beta is largely a low-beta bet, and its
    apparent alpha may be a known risk premium rather than new information.
    """
    defaults = ["beta_252d", "volatility_252d", "adv_21d", "market_cap", "momentum_252d"]
    columns = exposure_columns or [c for c in defaults if c in panel.columns]
    out: dict[str, float] = {}
    if factor not in panel.columns:
        return out

    for col in columns:
        if col == factor or col not in panel.columns:
            continue
        ics: list[float] = []
        for _as_of, group in panel.group_by("as_of"):
            if group.height < 5:
                continue
            ic = spearman_ic(group[factor].to_numpy(), group[col].to_numpy())
            if np.isfinite(ic):
                ics.append(ic)
        if ics:
            out[col] = float(np.mean(ics))
    return out


def subperiod_ic(panel: pl.DataFrame, factor: str, target: str) -> dict[str, float]:
    """IC by calendar year.

    A factor whose entire IC comes from one year is a curiosity, not a strategy; this makes
    that visible instead of averaging it away.
    """
    ics = information_coefficients(panel, factor, target)
    if ics.is_empty():
        return {}
    ics = ics.with_columns(pl.col("as_of").dt.year().alias("year"))
    agg = ics.group_by("year").agg(pl.col("ic").mean().alias("mean_ic")).sort("year")
    return {str(r["year"]): float(r["mean_ic"]) for r in agg.iter_rows(named=True)}


def analyze_factor(
    panel: pl.DataFrame,
    factor: str,
    target: str,
    n_quantiles: int = 5,
    exposure_columns: list[str] | None = None,
) -> FactorReport:
    """Full diagnostic bundle for one factor."""
    cov, missing = coverage_stats(panel, factor)
    ics = information_coefficients(panel, factor, target)
    ic_values = ics["ic"].to_numpy() if not ics.is_empty() else np.array([])

    q_returns = quantile_returns(panel, factor, target, n_quantiles)
    tmb = float("nan")
    if len(q_returns) >= 2:
        tmb = q_returns[max(q_returns)] - q_returns[min(q_returns)]

    return FactorReport(
        name=factor,
        coverage=cov,
        missing_rate=missing,
        mean_ic=float(np.mean(ic_values)) if ic_values.size else float("nan"),
        ic_std=float(np.std(ic_values, ddof=1)) if ic_values.size > 1 else float("nan"),
        ic_ir=ic_information_ratio(ic_values),
        positive_ic_pct=float((ic_values > 0).mean()) if ic_values.size else float("nan"),
        n_periods=int(ic_values.size),
        quantile_returns=q_returns,
        top_minus_bottom=tmb,
        turnover=factor_turnover(panel, factor),
        exposures=exposure_analysis(panel, factor, exposure_columns),
        subperiod_ic=subperiod_ic(panel, factor, target),
    )


def analyze_factors(
    panel: pl.DataFrame,
    factors: list[str],
    target: str,
    n_quantiles: int = 5,
) -> pl.DataFrame:
    """Diagnostics for many factors, ranked by IC information ratio."""
    reports = [analyze_factor(panel, f, target, n_quantiles) for f in factors if f in panel.columns]
    if not reports:
        return pl.DataFrame()
    frame = pl.DataFrame([r.to_row() for r in reports])
    return frame.sort("ic_ir", descending=True, nulls_last=True)


def quantile_monotonicity(report: FactorReport) -> float:
    """Spearman correlation between quantile index and its mean return.

    +1.0 means returns rise cleanly from the bottom bucket to the top. Near 0 means the
    factor's ordering carries no consistent information even if its headline IC looks fine.
    """
    if len(report.quantile_returns) < 3:
        return float("nan")
    keys = sorted(report.quantile_returns)
    values = [report.quantile_returns[k] for k in keys]
    return spearman_ic(np.array(keys, dtype=float), np.array(values))
