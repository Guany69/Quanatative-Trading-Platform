"""Feature computation on the price panel (spec sections 12.1-12.3).

All features here are built from strictly trailing windows. The critical rule is that a
feature stamped ``as_of = t`` may only use bars with ``observation_date <= t``. Polars'
rolling operations are trailing by default (they look back, never forward), so the main
hazards are (a) accidentally centering a window and (b) shifting in the wrong direction.
Both are covered by tests in ``tests/data_leakage/test_feature_availability.py``.

Returns are computed from ``adjusted_close`` (splits/dividends removed) while eligibility
screens use raw ``close`` -- see ``universe.builder`` for why.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_platform.features.registry import (
    REGISTRY,
    FeatureFamily,
    MissingPolicy,
    register,
)

# --------------------------------------------------------------------------- definitions
# Registered up front so the pipeline knows each feature's warm-up needs before computing.

for _n in (5, 21):
    register(
        name=f"reversal_{_n}d",
        family=FeatureFamily.REVERSAL,
        required_inputs=("adjusted_close",),
        lookback_sessions=_n + 1,
        min_observations=_n,
        description=f"Negated {_n}-session return; short-horizon losers tend to bounce.",
    )

for _n in (21, 63, 126, 252):
    register(
        name=f"momentum_{_n}d",
        family=FeatureFamily.MOMENTUM,
        required_inputs=("adjusted_close",),
        lookback_sessions=_n + 1,
        min_observations=int(_n * 0.8),
        description=f"{_n}-session cumulative return.",
    )

register(
    name="momentum_12m_ex1m",
    family=FeatureFamily.MOMENTUM,
    required_inputs=("adjusted_close",),
    lookback_sessions=253,
    min_observations=200,
    description="12-month momentum excluding the most recent month; the classic academic "
    "specification, which drops the last month because short-horizon reversal contaminates it.",
)
register(
    name="dist_52w_high",
    family=FeatureFamily.MOMENTUM,
    required_inputs=("adjusted_close",),
    lookback_sessions=252,
    min_observations=200,
    description="Proximity to the trailing 52-week high (0 = at the high, negative = below).",
)
register(
    name="ma_distance_50d",
    family=FeatureFamily.MOMENTUM,
    required_inputs=("adjusted_close",),
    lookback_sessions=50,
    min_observations=40,
    description="Log distance of price from its 50-session moving average.",
)
register(
    name="momentum_consistency_63d",
    family=FeatureFamily.MOMENTUM,
    required_inputs=("adjusted_close",),
    lookback_sessions=64,
    min_observations=50,
    description="Fraction of the last 63 sessions with positive returns; steady trends score "
    "higher than trends driven by one jump.",
)
register(
    name="trend_slope_63d",
    family=FeatureFamily.MOMENTUM,
    required_inputs=("adjusted_close",),
    lookback_sessions=63,
    min_observations=50,
    description="OLS slope of log price over 63 sessions, normalized by price level.",
)
register(
    name="trend_r2_63d",
    family=FeatureFamily.MOMENTUM,
    required_inputs=("adjusted_close",),
    lookback_sessions=63,
    min_observations=50,
    description="R-squared of the 63-session log-price trend; how well a line explains it.",
)

for _n in (20, 60, 252):
    register(
        name=f"volatility_{_n}d",
        family=FeatureFamily.RISK,
        required_inputs=("adjusted_close",),
        lookback_sessions=_n + 1,
        min_observations=int(_n * 0.8),
        description=f"Annualized realized volatility over {_n} sessions.",
    )

register(
    name="downside_vol_60d",
    family=FeatureFamily.RISK,
    required_inputs=("adjusted_close",),
    lookback_sessions=61,
    min_observations=45,
    description="Annualized volatility of negative returns only (semi-deviation).",
)
register(
    name="beta_252d",
    family=FeatureFamily.RISK,
    required_inputs=("adjusted_close", "benchmark_return"),
    lookback_sessions=252,
    min_observations=120,
    description="Rolling OLS beta to the benchmark.",
)
register(
    name="idio_vol_252d",
    family=FeatureFamily.RISK,
    required_inputs=("adjusted_close", "benchmark_return"),
    lookback_sessions=252,
    min_observations=120,
    description="Annualized volatility of residuals from the benchmark regression.",
)
register(
    name="corr_benchmark_252d",
    family=FeatureFamily.RISK,
    required_inputs=("adjusted_close", "benchmark_return"),
    lookback_sessions=252,
    min_observations=120,
    description="Rolling correlation to the benchmark.",
)
register(
    name="max_drawdown_252d",
    family=FeatureFamily.RISK,
    required_inputs=("adjusted_close",),
    lookback_sessions=252,
    min_observations=120,
    description="Worst peak-to-trough decline within the trailing year (negative).",
)
register(
    name="var_95_252d",
    family=FeatureFamily.RISK,
    required_inputs=("adjusted_close",),
    lookback_sessions=252,
    min_observations=120,
    description="Historical 95% one-session Value-at-Risk (5th percentile of returns).",
)
register(
    name="expected_shortfall_95_252d",
    family=FeatureFamily.RISK,
    required_inputs=("adjusted_close",),
    lookback_sessions=252,
    min_observations=120,
    description="Mean return in the worst 5% of sessions (conditional VaR).",
)

register(
    name="adv_21d",
    family=FeatureFamily.LIQUIDITY,
    required_inputs=("dollar_volume",),
    lookback_sessions=21,
    min_observations=15,
    description="Average daily dollar volume over 21 sessions (log-scaled).",
)
register(
    name="turnover_21d",
    family=FeatureFamily.LIQUIDITY,
    required_inputs=("volume", "shares_outstanding"),
    lookback_sessions=21,
    min_observations=15,
    description="Share turnover: mean volume / shares outstanding.",
)
register(
    name="amihud_illiquidity_21d",
    family=FeatureFamily.LIQUIDITY,
    required_inputs=("adjusted_close", "dollar_volume"),
    lookback_sessions=22,
    min_observations=15,
    description="Amihud illiquidity: mean(|return| / dollar volume); price impact per dollar.",
)
register(
    name="zero_return_freq_63d",
    family=FeatureFamily.LIQUIDITY,
    required_inputs=("adjusted_close",),
    lookback_sessions=63,
    min_observations=40,
    description="Fraction of sessions with zero return; a classic staleness/illiquidity proxy.",
)
register(
    name="volume_surprise_21d",
    family=FeatureFamily.LIQUIDITY,
    required_inputs=("volume",),
    lookback_sessions=63,
    min_observations=40,
    description="Recent 21-session volume relative to its 63-session norm (log ratio).",
)
register(
    name="spread_proxy_21d",
    family=FeatureFamily.LIQUIDITY,
    required_inputs=("high", "low"),
    lookback_sessions=21,
    min_observations=15,
    description="High-low range proxy for the bid-ask spread (Corwin-Schultz style, simplified).",
    missing_policy=MissingPolicy.CROSS_SECTIONAL_MEDIAN,
)

ANNUALIZATION = np.sqrt(252.0)


def _rolling_ols_beta(y: pl.Expr, x: pl.Expr, window: int, min_periods: int) -> pl.Expr:
    """Rolling OLS slope of y on x: cov(x,y)/var(x).

    Expressed via rolling moments rather than a python loop, so it stays vectorized.
    """
    cov = (y * x).rolling_mean(window, min_samples=min_periods) - (
        y.rolling_mean(window, min_samples=min_periods)
        * x.rolling_mean(window, min_samples=min_periods)
    )
    var = (x * x).rolling_mean(window, min_samples=min_periods) - (
        x.rolling_mean(window, min_samples=min_periods) ** 2
    )
    return cov / var


def compute_price_features(
    prices: pl.DataFrame,
    benchmark: pl.DataFrame,
    families: list[str] | None = None,
) -> pl.DataFrame:
    """Compute the price-derived feature panel.

    Parameters
    ----------
    prices:
        Long panel with security_id, observation_date, adjusted_close, and volume columns.
    benchmark:
        Benchmark frame with observation_date and benchmark_return.
    families:
        Restrict to these families (used by the ablation stress tests).

    Returns a long panel of (security_id, as_of, <feature columns>).
    """
    wanted = set(families) if families else None

    def enabled(fam: FeatureFamily) -> bool:
        return wanted is None or fam.value in wanted

    df = prices.sort(["security_id", "observation_date"])

    # Attach benchmark returns for beta/correlation features.
    bench = benchmark.select(["observation_date", "benchmark_return"])
    df = df.join(bench, on="observation_date", how="left")

    # Log return from ADJUSTED close: splits/dividends must not masquerade as performance.
    df = df.with_columns(
        [
            (pl.col("adjusted_close").log() - pl.col("adjusted_close").log().shift(1))
            .over("security_id")
            .alias("_ret"),
        ]
    )

    exprs: list[pl.Expr] = []

    if enabled(FeatureFamily.REVERSAL):
        for n in (5, 21):
            # Negated: recent losers are the attractive side of short-horizon reversal.
            exprs.append(
                (-(pl.col("adjusted_close").log() - pl.col("adjusted_close").log().shift(n)))
                .over("security_id")
                .alias(f"reversal_{n}d")
            )

    if enabled(FeatureFamily.MOMENTUM):
        for n in (21, 63, 126, 252):
            exprs.append(
                (pl.col("adjusted_close").log() - pl.col("adjusted_close").log().shift(n))
                .over("security_id")
                .alias(f"momentum_{n}d")
            )
        # 12m ex 1m: return from t-252 to t-21. Excludes the last month by construction.
        exprs.append(
            (pl.col("adjusted_close").log().shift(21) - pl.col("adjusted_close").log().shift(252))
            .over("security_id")
            .alias("momentum_12m_ex1m")
        )
        exprs.append(
            (
                pl.col("adjusted_close").log()
                - pl.col("adjusted_close").rolling_max(252, min_samples=200).log()
            )
            .over("security_id")
            .alias("dist_52w_high")
        )
        exprs.append(
            (
                pl.col("adjusted_close").log()
                - pl.col("adjusted_close").rolling_mean(50, min_samples=40).log()
            )
            .over("security_id")
            .alias("ma_distance_50d")
        )
        exprs.append(
            (pl.col("_ret") > 0)
            .cast(pl.Float64)
            .rolling_mean(63, min_samples=50)
            .over("security_id")
            .alias("momentum_consistency_63d")
        )

    if enabled(FeatureFamily.RISK):
        for n in (20, 60, 252):
            exprs.append(
                (pl.col("_ret").rolling_std(n, min_samples=int(n * 0.8)) * ANNUALIZATION)
                .over("security_id")
                .alias(f"volatility_{n}d")
            )
        exprs.append(
            (
                pl.when(pl.col("_ret") < 0)
                .then(pl.col("_ret"))
                .otherwise(None)
                .rolling_std(60, min_samples=20)
                * ANNUALIZATION
            )
            .over("security_id")
            .alias("downside_vol_60d")
        )
        exprs.append(
            _rolling_ols_beta(pl.col("_ret"), pl.col("benchmark_return"), 252, 120)
            .over("security_id")
            .alias("beta_252d")
        )
        # Correlation to benchmark.
        exprs.append(
            pl.rolling_corr(
                pl.col("_ret"), pl.col("benchmark_return"), window_size=252, min_samples=120
            )
            .over("security_id")
            .alias("corr_benchmark_252d")
        )
        # VaR / ES from the empirical return distribution.
        exprs.append(
            pl.col("_ret")
            .rolling_quantile(0.05, window_size=252, min_samples=120)
            .over("security_id")
            .alias("var_95_252d")
        )

    if enabled(FeatureFamily.LIQUIDITY):
        exprs.append(
            pl.col("dollar_volume")
            .rolling_mean(21, min_samples=15)
            .log1p()
            .over("security_id")
            .alias("adv_21d")
        )
        exprs.append(
            (pl.col("volume").rolling_mean(21, min_samples=15) / pl.col("shares_outstanding"))
            .over("security_id")
            .alias("turnover_21d")
        )
        exprs.append(
            (
                (pl.col("_ret").abs() / (pl.col("dollar_volume") + 1.0)).rolling_mean(
                    21, min_samples=15
                )
                * 1e9
            )
            .over("security_id")
            .alias("amihud_illiquidity_21d")
        )
        exprs.append(
            (pl.col("_ret").abs() < 1e-8)
            .cast(pl.Float64)
            .rolling_mean(63, min_samples=40)
            .over("security_id")
            .alias("zero_return_freq_63d")
        )
        exprs.append(
            (
                pl.col("volume").rolling_mean(21, min_samples=15).log1p()
                - pl.col("volume").rolling_mean(63, min_samples=40).log1p()
            )
            .over("security_id")
            .alias("volume_surprise_21d")
        )
        exprs.append(
            ((pl.col("high") - pl.col("low")) / pl.col("adjusted_close"))
            .rolling_mean(21, min_samples=15)
            .over("security_id")
            .alias("spread_proxy_21d")
        )

    df = df.with_columns(exprs)

    # Features needing a two-pass computation (residual vol, drawdown, ES, trend fit).
    df = _add_multipass_features(df, enabled)

    feature_cols = [
        c
        for c in df.columns
        if not c.startswith("_") and c not in prices.columns and c != "benchmark_return"
    ]
    out = df.select(["security_id", pl.col("observation_date").alias("as_of"), *feature_cols])
    return out


def _add_multipass_features(df: pl.DataFrame, enabled) -> pl.DataFrame:
    """Features that need a value computed in an earlier pass (e.g. beta -> residual)."""
    if enabled(FeatureFamily.RISK):
        if "beta_252d" in df.columns:
            # Idiosyncratic vol: volatility of the part of the return the benchmark cannot
            # explain. Uses the beta computed above, so it must run in a second pass.
            df = df.with_columns(
                (pl.col("_ret") - pl.col("beta_252d") * pl.col("benchmark_return")).alias("_resid")
            )
            df = df.with_columns(
                (pl.col("_resid").rolling_std(252, min_samples=120) * ANNUALIZATION)
                .over("security_id")
                .alias("idio_vol_252d")
            )
        if "var_95_252d" in df.columns:
            # Expected shortfall: mean of returns at or below the 5th percentile.
            df = df.with_columns(
                pl.when(pl.col("_ret") <= pl.col("var_95_252d"))
                .then(pl.col("_ret"))
                .otherwise(None)
                .alias("_tail")
            )
            df = df.with_columns(
                pl.col("_tail")
                .rolling_mean(252, min_samples=6)
                .over("security_id")
                .alias("expected_shortfall_95_252d")
            )
        # Max drawdown over the trailing year.
        df = df.with_columns(
            (
                pl.col("adjusted_close")
                / pl.col("adjusted_close").rolling_max(252, min_samples=120)
                - 1.0
            )
            .over("security_id")
            .rolling_min(252, min_samples=120)
            .over("security_id")
            .alias("max_drawdown_252d")
        )

    if enabled(FeatureFamily.MOMENTUM):
        df = _add_trend_features(df)
    return df


def _add_trend_features(df: pl.DataFrame, window: int = 63) -> pl.DataFrame:
    """Rolling trend slope and R^2 of log price.

    Regressing log-price on a fixed 0..n-1 time index makes the design matrix constant, so
    the slope reduces to a covariance ratio with a known variance -- no per-window solve.
    """
    t = np.arange(window, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    logp = pl.col("adjusted_close").log()
    # cov(t, logp) over the window via rolling moments against a constant index.
    # sum((t - t_mean) * (y - y_mean)) == sum(t*y) - n*t_mean*y_mean
    weighted = pl.sum_horizontal(
        [logp.shift(window - 1 - i) * float(t[i] - t_mean) for i in range(window)]
    )
    slope = (weighted / t_var).over("security_id")

    y_mean = logp.rolling_mean(window, min_samples=window).over("security_id")
    tss = pl.sum_horizontal([(logp.shift(window - 1 - i) - y_mean) ** 2 for i in range(window)])
    # explained sum of squares = slope^2 * sum((t - t_mean)^2)
    ess = (slope**2) * t_var
    r2 = (ess / tss).clip(0.0, 1.0)

    return df.with_columns(
        [
            (slope / pl.col("adjusted_close").log().abs().clip(0.1, None)).alias("trend_slope_63d"),
            r2.alias("trend_r2_63d"),
        ]
    )


def feature_columns(families: list[str] | None = None) -> list[str]:
    """Names of the features produced for the given families."""
    defs = REGISTRY.for_families(families) if families else REGISTRY.all()
    return [d.name for d in defs]
