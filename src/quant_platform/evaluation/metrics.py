"""Portfolio and forecast metrics (spec section 22).

Annualization conventions used throughout, stated explicitly because they are the most common
source of silently incomparable numbers:

* 252 trading sessions per year.
* Volatility annualizes by sqrt(252); mean return by *252.
* Sharpe uses arithmetic mean excess return over the risk-free rate (default 0) / stdev.
* CAGR is geometric, derived from the actual elapsed session count -- not from the arithmetic
  mean, which would overstate a volatile series.
* Information ratio = mean(active return) / stdev(active return), annualized, where active
  return is strategy minus benchmark, session by session.

Every function is null-safe and returns NaN rather than raising on degenerate input, because
a stress scenario that produces two data points should not crash the whole sweep.
"""

from __future__ import annotations

import numpy as np

SESSIONS_PER_YEAR = 252


def _clean(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[np.isfinite(r)]


def cumulative_return(returns: np.ndarray) -> float:
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    return float(np.prod(1.0 + r) - 1.0)


def cagr(returns: np.ndarray, sessions_per_year: int = SESSIONS_PER_YEAR) -> float:
    """Geometric annual growth rate implied by the return series."""
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    total = np.prod(1.0 + r)
    if total <= 0:
        return -1.0  # wiped out
    years = r.size / sessions_per_year
    if years <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def annualized_volatility(returns: np.ndarray, sessions_per_year: int = SESSIONS_PER_YEAR) -> float:
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(sessions_per_year))


def sharpe_ratio(
    returns: np.ndarray, risk_free: float = 0.0, sessions_per_year: int = SESSIONS_PER_YEAR
) -> float:
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    excess = r - risk_free / sessions_per_year
    sd = np.std(excess, ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(excess) / sd * np.sqrt(sessions_per_year))


def sortino_ratio(
    returns: np.ndarray, risk_free: float = 0.0, sessions_per_year: int = SESSIONS_PER_YEAR
) -> float:
    """Like Sharpe but penalizes only downside deviation."""
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    excess = r - risk_free / sessions_per_year
    downside = excess[excess < 0]
    if downside.size < 2:
        return float("nan")
    dd = np.sqrt(np.mean(downside**2))
    if dd < 1e-12:
        return float("nan")
    return float(np.mean(excess) / dd * np.sqrt(sessions_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough decline of the cumulative curve (negative)."""
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve / peak - 1.0))


def calmar_ratio(returns: np.ndarray, sessions_per_year: int = SESSIONS_PER_YEAR) -> float:
    dd = max_drawdown(returns)
    if not np.isfinite(dd) or abs(dd) < 1e-12:
        return float("nan")
    return float(cagr(returns, sessions_per_year) / abs(dd))


def tracking_error(
    returns: np.ndarray, benchmark: np.ndarray, sessions_per_year: int = SESSIONS_PER_YEAR
) -> float:
    active = _active(returns, benchmark)
    if active.size < 2:
        return float("nan")
    return float(np.std(active, ddof=1) * np.sqrt(sessions_per_year))


def information_ratio(
    returns: np.ndarray, benchmark: np.ndarray, sessions_per_year: int = SESSIONS_PER_YEAR
) -> float:
    """Annualized mean active return / annualized tracking error."""
    active = _active(returns, benchmark)
    if active.size < 2:
        return float("nan")
    sd = np.std(active, ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(active) / sd * np.sqrt(sessions_per_year))


def _active(returns: np.ndarray, benchmark: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    n = min(r.size, b.size)
    r, b = r[:n], b[:n]
    mask = np.isfinite(r) & np.isfinite(b)
    return r[mask] - b[mask]


def beta(returns: np.ndarray, benchmark: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    n = min(r.size, b.size)
    r, b = r[:n], b[:n]
    mask = np.isfinite(r) & np.isfinite(b)
    r, b = r[mask], b[mask]
    if r.size < 2:
        return float("nan")
    var = np.var(b, ddof=1)
    if var < 1e-16:
        return float("nan")
    return float(np.cov(r, b, ddof=1)[0, 1] / var)


def capm_alpha(
    returns: np.ndarray,
    benchmark: np.ndarray,
    risk_free: float = 0.0,
    sessions_per_year: int = SESSIONS_PER_YEAR,
) -> float:
    """Annualized CAPM intercept: return not explained by benchmark exposure."""
    r = np.asarray(returns, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    n = min(r.size, b.size)
    r, b = r[:n], b[:n]
    mask = np.isfinite(r) & np.isfinite(b)
    r, b = r[mask], b[mask]
    if r.size < 2:
        return float("nan")
    rf = risk_free / sessions_per_year
    bt = beta(r, b)
    if not np.isfinite(bt):
        return float("nan")
    return float((np.mean(r - rf) - bt * np.mean(b - rf)) * sessions_per_year)


def value_at_risk(returns: np.ndarray, level: float = 0.95) -> float:
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    return float(np.quantile(r, 1.0 - level))


def expected_shortfall(returns: np.ndarray, level: float = 0.95) -> float:
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    var = np.quantile(r, 1.0 - level)
    tail = r[r <= var]
    return float(np.mean(tail)) if tail.size else float("nan")


def win_rate(returns: np.ndarray) -> float:
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    return float(np.mean(r > 0))


def profit_factor(returns: np.ndarray) -> float:
    r = _clean(returns)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses < 1e-12:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def spearman_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Rank information coefficient: Spearman correlation of prediction vs outcome.

    The core forecast metric for a cross-sectional ranker. Implemented via Pearson on ranks
    so it needs no SciPy.
    """
    p = np.asarray(predictions, dtype=float)
    t = np.asarray(targets, dtype=float)
    mask = np.isfinite(p) & np.isfinite(t)
    p, t = p[mask], t[mask]
    if p.size < 3:
        return float("nan")
    pr = _rankdata(p)
    tr = _rankdata(t)
    if np.std(pr) < 1e-12 or np.std(tr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(pr, tr)[0, 1])


def pearson_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    p = np.asarray(predictions, dtype=float)
    t = np.asarray(targets, dtype=float)
    mask = np.isfinite(p) & np.isfinite(t)
    p, t = p[mask], t[mask]
    if p.size < 3 or np.std(p) < 1e-12 or np.std(t) < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (matches scipy.stats.rankdata's 'average')."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # Average tied groups.
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def ic_information_ratio(ics: np.ndarray) -> float:
    """Stability of the IC over time: mean(IC)/std(IC). The 'is it reliable' number."""
    x = _clean(ics)
    if x.size < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(x) / sd)


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    p, t = np.asarray(predictions, float), np.asarray(targets, float)
    mask = np.isfinite(p) & np.isfinite(t)
    if not mask.any():
        return float("nan")
    return float(np.mean((p[mask] - t[mask]) ** 2))


def mean_absolute_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    p, t = np.asarray(predictions, float), np.asarray(targets, float)
    mask = np.isfinite(p) & np.isfinite(t)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(p[mask] - t[mask])))


def portfolio_metrics(
    returns: np.ndarray,
    benchmark: np.ndarray | None = None,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """The standard metric bundle for one return series."""
    m: dict[str, float] = {
        "cumulative_return": cumulative_return(returns),
        "cagr": cagr(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free),
        "sortino_ratio": sortino_ratio(returns, risk_free),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns),
        "var_95": value_at_risk(returns),
        "expected_shortfall_95": expected_shortfall(returns),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "n_observations": float(_clean(returns).size),
    }
    if benchmark is not None:
        m.update(
            {
                "tracking_error": tracking_error(returns, benchmark),
                "information_ratio": information_ratio(returns, benchmark),
                "beta": beta(returns, benchmark),
                "capm_alpha": capm_alpha(returns, benchmark, risk_free),
                "benchmark_cagr": cagr(benchmark),
                "excess_cagr": cagr(returns) - cagr(benchmark),
            }
        )
    return m
