"""Multiple-testing and backtest-overfitting statistics (spec section 23).

The central problem: try enough strategies and one will look excellent by chance alone. The
best of 100 coin-flipping strategies has an impressive backtest and zero expected future
return. A Sharpe ratio reported without reference to how many configurations were tried is
close to meaningless.

Implemented here, following Bailey & Lopez de Prado:

* **Probabilistic Sharpe Ratio (PSR)** -- probability the true Sharpe exceeds a benchmark,
  correcting for track-record length plus the skew and kurtosis of returns. Financial returns
  are negatively skewed and fat-tailed, which makes the naive Sharpe overstate confidence.
* **Deflated Sharpe Ratio (DSR)** -- PSR against a benchmark raised to reflect the *number of
  trials*. This is the number that answers "is this better than the best of N random tries?"
* **Probability of Backtest Overfitting (PBO)** -- via combinatorially symmetric
  cross-validation: how often the in-sample winner underperforms out of sample.

Every function returns NaN rather than a fabricated value when its assumptions are not met
(too few observations, degenerate variance). The spec is explicit that a fake approximation
is worse than an honest gap.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

SESSIONS_PER_YEAR = 252


def _clean(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[np.isfinite(r)]


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no SciPy dependency)."""
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, ~1e-9 accurate)."""
    if not 0.0 < p < 1.0:
        return float("nan")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return float(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > p_high:
        q = np.sqrt(-2 * np.log(1 - p))
        return float(
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return float(
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


@dataclass
class SharpeStatistics:
    """Sharpe with the corrections that make it interpretable."""

    sharpe_ratio: float
    n_observations: int
    skewness: float
    kurtosis: float
    probabilistic_sharpe: float
    deflated_sharpe: float | None = None
    n_trials: int | None = None
    benchmark_sharpe: float = 0.0

    def verdict(self) -> str:
        """Plain-language reading of the statistics."""
        if not np.isfinite(self.sharpe_ratio):
            return "insufficient data to evaluate"
        parts = [f"Sharpe {self.sharpe_ratio:.2f} over {self.n_observations} observations"]
        if np.isfinite(self.probabilistic_sharpe):
            parts.append(
                f"PSR {self.probabilistic_sharpe:.1%} (probability true Sharpe > "
                f"{self.benchmark_sharpe:.2f})"
            )
        if self.deflated_sharpe is not None and np.isfinite(self.deflated_sharpe):
            parts.append(
                f"DSR {self.deflated_sharpe:.1%} after correcting for {self.n_trials} trials"
            )
            if self.deflated_sharpe < 0.95:
                parts.append("NOT significant once selection bias is accounted for")
        return "; ".join(parts)


def sample_skewness(returns: np.ndarray) -> float:
    r = _clean(returns)
    if r.size < 3:
        return float("nan")
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(((r - r.mean()) / sd) ** 3))


def sample_kurtosis(returns: np.ndarray) -> float:
    """Non-excess kurtosis (3.0 for a normal distribution)."""
    r = _clean(returns)
    if r.size < 4:
        return float("nan")
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(((r - r.mean()) / sd) ** 4))


def probabilistic_sharpe_ratio(
    returns: np.ndarray, benchmark_sharpe: float = 0.0, annualized: bool = True
) -> float:
    """Probability that the true Sharpe ratio exceeds ``benchmark_sharpe``.

    PSR = Phi( (SR - SR*) * sqrt(n-1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )

    The denominator is the point: negative skew and fat tails *inflate* the estimator's
    variance, so a strategy with occasional large losses needs a higher raw Sharpe to reach
    the same confidence. Computation is done on per-period (not annualized) Sharpe, since the
    correction is defined in those units.
    """
    r = _clean(returns)
    n = r.size
    if n < 4:
        return float("nan")
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return float("nan")

    sr_period = r.mean() / sd
    sr_star = benchmark_sharpe / np.sqrt(SESSIONS_PER_YEAR) if annualized else benchmark_sharpe

    skew = sample_skewness(r)
    kurt = sample_kurtosis(r)
    if not (np.isfinite(skew) and np.isfinite(kurt)):
        return float("nan")

    variance_term = 1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * sr_period**2
    if variance_term <= 0:
        return float("nan")

    z = (sr_period - sr_star) * np.sqrt(n - 1) / np.sqrt(variance_term)
    return float(_norm_cdf(float(z)))


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """Expected maximum Sharpe from ``n_trials`` strategies with NO real skill.

    Uses the standard extreme-value approximation for the maximum of N normals. This is the
    bar a genuinely skilful strategy has to clear: with enough attempts, a high Sharpe is the
    *expected* outcome of luck alone.
    """
    if n_trials < 2 or trial_sharpe_std <= 0:
        return 0.0
    euler = 0.5772156649015329
    e = np.e
    z = (1 - euler) * _norm_ppf(1 - 1.0 / n_trials) + euler * _norm_ppf(1 - 1.0 / (n_trials * e))
    return float(trial_sharpe_std * z)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    trial_sharpe_std: float | None = None,
    all_trial_sharpes: list[float] | None = None,
    annualized: bool = True,
) -> float:
    """PSR measured against the Sharpe expected from selection alone.

    Answers the question that matters after a search: "would the best of N random attempts
    have looked this good anyway?" A DSR below ~0.95 means the result is not distinguishable
    from the product of the search itself.

    ``trial_sharpe_std`` is the dispersion of Sharpe ratios across attempted configurations;
    it is estimated from ``all_trial_sharpes`` when supplied. Passing neither makes the
    correction impossible, and NaN is returned rather than a fabricated number.
    """
    if n_trials < 1:
        return float("nan")
    if trial_sharpe_std is None:
        if all_trial_sharpes and len(all_trial_sharpes) > 1:
            trial_sharpe_std = float(np.std(np.asarray(all_trial_sharpes, float), ddof=1))
        else:
            return float("nan")
    if not np.isfinite(trial_sharpe_std) or trial_sharpe_std <= 0:
        return float("nan")

    sr_star_annual = expected_max_sharpe(n_trials, trial_sharpe_std)
    return probabilistic_sharpe_ratio(returns, sr_star_annual, annualized=annualized)


def analyze_sharpe(
    returns: np.ndarray,
    n_trials: int | None = None,
    all_trial_sharpes: list[float] | None = None,
    benchmark_sharpe: float = 0.0,
) -> SharpeStatistics:
    """Full Sharpe diagnostics including the multiple-testing correction."""
    from quant_platform.evaluation.metrics import sharpe_ratio

    r = _clean(returns)
    dsr = None
    if n_trials and n_trials > 1:
        dsr = deflated_sharpe_ratio(r, n_trials, all_trial_sharpes=all_trial_sharpes)

    return SharpeStatistics(
        sharpe_ratio=sharpe_ratio(r),
        n_observations=int(r.size),
        skewness=sample_skewness(r),
        kurtosis=sample_kurtosis(r),
        probabilistic_sharpe=probabilistic_sharpe_ratio(r, benchmark_sharpe),
        deflated_sharpe=dsr,
        n_trials=n_trials,
        benchmark_sharpe=benchmark_sharpe,
    )


def probability_of_backtest_overfitting(
    performance_matrix: np.ndarray, n_splits: int = 8
) -> dict[str, float]:
    """PBO via combinatorially symmetric cross-validation (Bailey et al.).

    ``performance_matrix`` is (T x N): per-period returns for N candidate configurations.

    Method: split the timeline into ``n_splits`` blocks, and for every balanced split into
    in-sample and out-of-sample halves, pick the IS winner and record its OOS rank. If
    selection carried real information, the IS winner should land in the upper half OOS. PBO
    is the fraction of splits where it lands in the *lower* half -- i.e. selection was noise.

    A PBO above ~0.5 means the selection process is worse than random.
    """
    M = np.asarray(performance_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return {"pbo": float("nan"), "n_combinations": 0.0}

    T, _n_strategies = M.shape
    if n_splits % 2 != 0:
        n_splits += 1
    if n_splits * 2 > T:
        return {"pbo": float("nan"), "n_combinations": 0.0}

    block_size = T // n_splits
    blocks = [M[i * block_size : (i + 1) * block_size] for i in range(n_splits)]

    half = n_splits // 2
    logits: list[float] = []
    underperformed = 0
    total = 0

    for combo in itertools.combinations(range(n_splits), half):
        is_idx = list(combo)
        oos_idx = [i for i in range(n_splits) if i not in combo]

        is_data = np.vstack([blocks[i] for i in is_idx])
        oos_data = np.vstack([blocks[i] for i in oos_idx])

        is_perf = _sharpe_columns(is_data)
        oos_perf = _sharpe_columns(oos_data)
        if not np.isfinite(is_perf).any() or not np.isfinite(oos_perf).any():
            continue

        best = int(np.nanargmax(is_perf))
        # Relative rank of the IS winner among OOS results, in (0, 1).
        finite = np.isfinite(oos_perf)
        if finite.sum() < 2:
            continue
        rank = float((oos_perf[finite] < oos_perf[best]).sum()) / float(finite.sum())
        rank = min(max(rank, 1e-6), 1 - 1e-6)

        total += 1
        if rank < 0.5:
            underperformed += 1
        logits.append(float(np.log(rank / (1 - rank))))

    if total == 0:
        return {"pbo": float("nan"), "n_combinations": 0.0}

    return {
        "pbo": underperformed / total,
        "n_combinations": float(total),
        "mean_logit": float(np.mean(logits)) if logits else float("nan"),
    }


def _sharpe_columns(data: np.ndarray) -> np.ndarray:
    """Per-column Sharpe of a (T x N) matrix."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0, ddof=1)
        out = np.where(std > 1e-12, mean / std, np.nan)
    return out


def bootstrap_confidence_interval(
    returns: np.ndarray,
    statistic=None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    block_size: int | None = None,
    random_seed: int = 42,
) -> dict[str, float]:
    """Block-bootstrap confidence interval for a statistic.

    Uses a *moving block* bootstrap rather than an iid one. Financial returns exhibit
    volatility clustering, and resampling individual observations destroys that dependence,
    producing intervals that are far too narrow and overstate confidence.
    """
    from quant_platform.evaluation.metrics import sharpe_ratio

    statistic = statistic or sharpe_ratio
    r = _clean(returns)
    n = r.size
    if n < 30:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan")}

    # Default block length ~ n^(1/3), the standard rule for dependent data.
    block = block_size or max(2, int(n ** (1 / 3)))
    rng = np.random.default_rng(random_seed)
    n_blocks = int(np.ceil(n / block))

    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        sample = np.concatenate([r[s : s + block] for s in starts])[:n]
        estimates[i] = statistic(sample)

    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan")}

    alpha = 1.0 - confidence
    return {
        "point": float(statistic(r)),
        "lower": float(np.quantile(estimates, alpha / 2)),
        "upper": float(np.quantile(estimates, 1 - alpha / 2)),
        "bootstrap_mean": float(estimates.mean()),
        "bootstrap_std": float(estimates.std(ddof=1)),
        "n_valid": float(estimates.size),
    }


def minimum_track_record_length(
    returns: np.ndarray, benchmark_sharpe: float = 0.0, confidence: float = 0.95
) -> float:
    """Observations needed to establish the Sharpe exceeds a benchmark at ``confidence``.

    Frequently sobering: a Sharpe of 0.5 typically needs many years of data to distinguish
    from zero, which is why short backtests cannot settle the question.
    """
    r = _clean(returns)
    if r.size < 4:
        return float("nan")
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return float("nan")

    sr = r.mean() / sd
    sr_star = benchmark_sharpe / np.sqrt(SESSIONS_PER_YEAR)
    if sr <= sr_star:
        return float("inf")  # never, at this performance level

    skew = sample_skewness(r)
    kurt = sample_kurtosis(r)
    if not (np.isfinite(skew) and np.isfinite(kurt)):
        return float("nan")

    z = _norm_ppf(confidence)
    variance_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if variance_term <= 0:
        return float("nan")
    return float(1.0 + variance_term * (z / (sr - sr_star)) ** 2)
