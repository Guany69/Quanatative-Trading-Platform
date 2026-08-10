"""Covariance and beta estimation (spec section 19).

Sample covariance is nearly unusable for portfolio optimization at realistic dimensions. With
N securities and T observations, the sample estimate has O(N^2) parameters fitted from N*T
numbers; when N approaches T the matrix becomes ill-conditioned or outright singular. The
optimizer then discovers "arbitrage" in the estimation error itself and concentrates the
portfolio into whichever noise looked most favourable.

Shrinkage (Ledoit-Wolf) pulls the estimate toward a structured target, trading a little bias
for a large variance reduction. It is the default here for that reason, and every path
returns a matrix that is symmetric and positive definite -- checked, not assumed, because a
solver handed an indefinite matrix fails in confusing ways.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from quant_platform.config.models import RiskConfig
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("risk.covariance")

TRADING_DAYS = 252


@dataclass
class RiskModel:
    """An estimated covariance matrix plus the metadata needed to use it safely."""

    security_ids: list[str]
    covariance: np.ndarray
    volatilities: np.ndarray
    betas: np.ndarray
    method: str
    n_observations: int
    shrinkage_intensity: float | None = None
    condition_number: float | None = None

    def index_of(self, security_id: str) -> int | None:
        try:
            return self.security_ids.index(security_id)
        except ValueError:
            return None

    def subset(self, security_ids: list[str]) -> RiskModel:
        """Restrict the model to a subset, preserving positive definiteness."""
        idx = [i for i, s in enumerate(self.security_ids) if s in set(security_ids)]
        if not idx:
            raise ValueError("no requested securities are present in the risk model")
        ix = np.ix_(idx, idx)
        return RiskModel(
            security_ids=[self.security_ids[i] for i in idx],
            covariance=self.covariance[ix],
            volatilities=self.volatilities[idx],
            betas=self.betas[idx],
            method=self.method,
            n_observations=self.n_observations,
            shrinkage_intensity=self.shrinkage_intensity,
        )

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Ex-ante annualized portfolio volatility."""
        variance = float(weights @ self.covariance @ weights)
        return float(np.sqrt(max(variance, 0.0)))

    def portfolio_beta(self, weights: np.ndarray) -> float:
        return float(weights @ self.betas)

    def marginal_risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """Each position's marginal contribution to portfolio volatility."""
        vol = self.portfolio_volatility(weights)
        if vol < 1e-12:
            return np.zeros_like(weights)
        return (self.covariance @ weights) / vol

    def risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """Total risk contribution per position; sums to portfolio volatility."""
        return weights * self.marginal_risk_contribution(weights)

    def tracking_error(self, weights: np.ndarray, benchmark_weights: np.ndarray) -> float:
        active = weights - benchmark_weights
        return float(np.sqrt(max(float(active @ self.covariance @ active), 0.0)))


def _returns_matrix(
    prices: pl.DataFrame, security_ids: list[str], as_of, lookback: int
) -> tuple[np.ndarray, list[str]]:
    """Build a (T x N) return matrix from trailing data only."""
    window = prices.filter(
        (pl.col("observation_date") <= as_of) & pl.col("security_id").is_in(security_ids)
    ).sort(["security_id", "observation_date"])

    if window.is_empty():
        return np.zeros((0, 0)), []

    window = window.with_columns(
        (pl.col("adjusted_close").log() - pl.col("adjusted_close").log().shift(1))
        .over("security_id")
        .alias("ret")
    )
    recent = window.group_by("security_id", maintain_order=True).tail(lookback).drop_nulls("ret")
    wide = recent.pivot(values="ret", index="observation_date", on="security_id").sort(
        "observation_date"
    )
    kept = [c for c in wide.columns if c != "observation_date"]
    if not kept:
        return np.zeros((0, 0)), []
    matrix = wide.select(kept).to_numpy().astype(float)
    return matrix, kept


def _make_positive_definite(cov: np.ndarray, epsilon: float = 1e-10) -> tuple[np.ndarray, float]:
    """Force symmetry and positive definiteness.

    Floating-point asymmetry and tiny negative eigenvalues are routine after shrinkage and
    NaN handling; solvers reject such matrices, so eigenvalues are floored rather than left
    to fail deep inside CVXPY.
    """
    cov = 0.5 * (cov + cov.T)  # exact symmetry
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    smallest = float(eigenvalues.min())
    if smallest < epsilon:
        eigenvalues = np.maximum(eigenvalues, epsilon)
        cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        cov = 0.5 * (cov + cov.T)
        logger.debug("covariance repaired: min eigenvalue %.2e -> %.2e", smallest, epsilon)
    condition = float(eigenvalues.max() / max(eigenvalues.min(), epsilon))
    return cov, condition


def estimate_covariance(
    prices: pl.DataFrame,
    security_ids: list[str],
    as_of,
    config: RiskConfig,
    benchmark: pl.DataFrame | None = None,
) -> RiskModel:
    """Estimate the covariance matrix, volatilities, and betas as of a date.

    Uses only data on or before ``as_of``. Securities with too little history are dropped
    rather than imputed: a fabricated covariance row is worse than a smaller universe.
    """
    matrix, kept = _returns_matrix(prices, security_ids, as_of, config.covariance_lookback_sessions)
    if matrix.size == 0 or matrix.shape[0] < config.min_observations:
        raise ValueError(
            f"insufficient return history at {as_of}: {matrix.shape[0] if matrix.size else 0} "
            f"observations, need {config.min_observations}"
        )

    # Drop columns with too many gaps, then fill the remainder with 0 (a no-move day).
    valid_counts = np.sum(~np.isnan(matrix), axis=0)
    keep_mask = valid_counts >= config.min_observations
    if not keep_mask.any():
        raise ValueError(f"no security has {config.min_observations} observations at {as_of}")
    matrix = matrix[:, keep_mask]
    kept = [s for s, k in zip(kept, keep_mask, strict=True) if k]
    matrix = np.nan_to_num(matrix, nan=0.0)

    n_obs, n_sec = matrix.shape
    # Plain str, not the config Literal: the estimator may fall back and relabel itself.
    method: str = config.covariance_method
    shrinkage: float | None = config.shrinkage_intensity

    if method in {"ledoit_wolf", "oas", "shrinkage"} and n_sec > 1:
        try:
            from sklearn.covariance import OAS, LedoitWolf

            estimator = OAS() if method == "oas" else LedoitWolf()
            estimator.fit(matrix)
            cov = np.asarray(estimator.covariance_, dtype=float)
            shrinkage = float(getattr(estimator, "shrinkage_", np.nan))
        except Exception as exc:
            logger.warning("shrinkage estimator failed (%s); falling back to sample", exc)
            cov = np.cov(matrix, rowvar=False, ddof=1)
            method = "sample_fallback"
    else:
        cov = (
            np.cov(matrix, rowvar=False, ddof=1) if n_sec > 1 else np.array([[matrix.var(ddof=1)]])
        )

    cov = np.atleast_2d(cov)
    cov, condition = _make_positive_definite(cov)
    cov_annual = cov * TRADING_DAYS

    volatilities = np.sqrt(np.diag(cov_annual))
    betas = _estimate_betas(prices, kept, as_of, config, benchmark)

    if condition > 1e12:
        logger.warning(
            "covariance is poorly conditioned (condition number %.2e); optimizer output may "
            "be unstable and driven by estimation error",
            condition,
        )

    return RiskModel(
        security_ids=kept,
        covariance=cov_annual,
        volatilities=volatilities,
        betas=betas,
        method=method,
        n_observations=n_obs,
        shrinkage_intensity=shrinkage,
        condition_number=condition,
    )


def _estimate_betas(
    prices: pl.DataFrame,
    security_ids: list[str],
    as_of,
    config: RiskConfig,
    benchmark: pl.DataFrame | None,
) -> np.ndarray:
    """Rolling OLS beta of each security against the benchmark."""
    if benchmark is None or benchmark.is_empty():
        # Beta 1.0 is the neutral assumption; flagged so callers know it is not estimated.
        logger.debug("no benchmark supplied; assuming beta = 1.0")
        return np.ones(len(security_ids))

    price_col = "adjusted_close" if "adjusted_close" in benchmark.columns else "close"
    bench = (
        benchmark.filter(pl.col("observation_date") <= as_of)
        .sort("observation_date")
        .with_columns((pl.col(price_col).log() - pl.col(price_col).log().shift(1)).alias("_bret"))
        .tail(config.beta_lookback_sessions)
        .select(["observation_date", "_bret"])
        .drop_nulls()
    )
    if bench.height < config.min_observations:
        return np.ones(len(security_ids))

    betas = np.ones(len(security_ids))
    window = (
        prices.filter(
            (pl.col("observation_date") <= as_of) & pl.col("security_id").is_in(security_ids)
        )
        .sort(["security_id", "observation_date"])
        .with_columns(
            (pl.col("adjusted_close").log() - pl.col("adjusted_close").log().shift(1))
            .over("security_id")
            .alias("ret")
        )
    )
    joined = window.join(bench, on="observation_date", how="inner").drop_nulls(["ret", "_bret"])

    bench_var = float(np.var(bench["_bret"].to_numpy(), ddof=1))
    if bench_var < 1e-16:
        return betas

    for i, sec in enumerate(security_ids):
        sub = joined.filter(pl.col("security_id") == sec)
        if sub.height < config.min_observations:
            continue
        r = sub["ret"].to_numpy()
        b = sub["_bret"].to_numpy()
        cov = float(np.cov(r, b, ddof=1)[0, 1])
        betas[i] = cov / float(np.var(b, ddof=1))
    # Extreme betas are almost always estimation artifacts on thin history.
    return np.clip(betas, -3.0, 5.0)


def ex_post_volatility(returns: np.ndarray) -> float:
    """Realized annualized volatility -- for comparing against the ex-ante forecast."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))


def concentration_metrics(weights: np.ndarray) -> dict[str, float]:
    """Herfindahl index, effective breadth, and top-position share."""
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return {"herfindahl": float("nan"), "effective_n": float("nan"), "top_weight": float("nan")}
    total = w.sum()
    if total <= 0:
        return {"herfindahl": float("nan"), "effective_n": float("nan"), "top_weight": float("nan")}
    normalized = w / total
    hhi = float(np.sum(normalized**2))
    return {
        "herfindahl": hhi,
        # Effective number of independent bets: 1/HHI. 100 equal positions -> 100.
        "effective_n": float(1.0 / hhi) if hhi > 0 else float("nan"),
        "top_weight": float(normalized.max()),
        "top10_weight": float(np.sort(normalized)[-10:].sum()) if normalized.size >= 10 else 1.0,
    }
