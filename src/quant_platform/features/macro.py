"""Macro and regime features with vintage-correct selection (spec section 12.5).

Macro data carries a hazard that price data does not: **revisions**. GDP for a quarter is
printed, then revised, sometimes substantially. The final revised value is the one every
public database serves by default -- and it was unknowable at the time. Training on it is a
subtle, powerful leak, because revisions systematically move toward what actually happened.

``select_macro_vintage`` therefore picks, for each date, the latest *vintage* whose
``available_at <= date`` -- meaning an early-2020 backtest sees the original noisy print of
2019Q4 GDP, exactly as a contemporary observer would have.

Regime features here deliberately *modulate* rather than gate: they describe market state so
a model can condition on it, rather than switching the strategy off entirely. Hard on/off
regime rules are a well-known overfitting trap, since a handful of regime flips can be tuned
to sidestep every historical drawdown.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_platform.features.registry import FeatureFamily, MissingPolicy, register

_MACRO_FEATURES = [
    ("benchmark_volatility_21d", "Annualized 21-session benchmark volatility."),
    ("benchmark_volatility_63d", "Annualized 63-session benchmark volatility."),
    ("benchmark_trend_126d", "126-session benchmark return; the broad market trend."),
    ("benchmark_drawdown", "Benchmark drawdown from its trailing 252-session peak."),
    ("market_breadth", "Fraction of securities above their own 50-session moving average."),
    (
        "cross_sectional_dispersion",
        "Cross-sectional stdev of 21-session returns; the spread "
        "of opportunity available to a stock picker.",
    ),
    (
        "average_correlation",
        "Mean pairwise correlation proxy; high values mean the market is "
        "moving as one and selection adds little.",
    ),
    ("yield_curve_slope", "10-year minus 3-month yield."),
    ("real_rate_proxy", "10-year yield minus trailing inflation proxy."),
    ("inflation_change", "Year-over-year change in the inflation proxy."),
    ("growth_change", "Year-over-year change in the growth proxy."),
]

for _name, _desc in _MACRO_FEATURES:
    register(
        name=_name,
        family=FeatureFamily.MACRO,
        required_inputs=("benchmark", "macro"),
        lookback_sessions=252,
        min_observations=60,
        description=_desc,
        # Macro values are identical across securities on a given date, so cross-sectional
        # normalization would annihilate them (every name gets the same rank).
        requires_cross_sectional_normalization=False,
        missing_policy=MissingPolicy.KEEP_NULL,
    )

MACRO_FEATURE_NAMES = [n for n, _ in _MACRO_FEATURES]
ANNUALIZATION = np.sqrt(252.0)


def select_macro_vintage(macro: pl.DataFrame, series_id: str) -> pl.DataFrame:
    """Reduce a series to one row per (observation_date, vintage) ready for as-of joining.

    Keeps every vintage rather than collapsing to the latest, because which vintage is
    correct depends on the date the feature is being computed for. Collapsing here would
    destroy exactly the information that prevents the leak.
    """
    df = macro.filter(pl.col("series_id") == series_id)
    if df.is_empty():
        return df
    return df.select(
        [
            "observation_date",
            "available_at",
            pl.col("value").alias(series_id.lower()),
            pl.col("revision_id"),
        ]
    ).sort("available_at")


def attach_macro_series(
    panel: pl.DataFrame,
    macro: pl.DataFrame,
    series_ids: list[str],
    as_of_col: str = "as_of",
) -> pl.DataFrame:
    """As-of join macro series onto a panel, honouring publication and revision timing."""
    if macro.is_empty() or not series_ids:
        return panel

    out = panel.with_columns(pl.col(as_of_col).cast(pl.Datetime).alias("_as_of_dt")).sort(
        "_as_of_dt"
    )
    for series_id in series_ids:
        vintages = select_macro_vintage(macro, series_id)
        if vintages.is_empty():
            continue
        # backward = latest vintage published at or before this date.
        out = out.join_asof(
            vintages.select(["available_at", series_id.lower()]).sort("available_at"),
            left_on="_as_of_dt",
            right_on="available_at",
            strategy="backward",
        )
    return out.drop("_as_of_dt")


def compute_benchmark_regime_features(benchmark: pl.DataFrame) -> pl.DataFrame:
    """Volatility, trend, and drawdown of the benchmark, on trailing windows only."""
    bm = benchmark.sort("observation_date")
    price_col = "adjusted_close" if "adjusted_close" in bm.columns else "close"

    bm = bm.with_columns((pl.col(price_col).log() - pl.col(price_col).log().shift(1)).alias("_ret"))
    return bm.select(
        [
            pl.col("observation_date").alias("as_of"),
            (pl.col("_ret").rolling_std(21, min_samples=15) * ANNUALIZATION).alias(
                "benchmark_volatility_21d"
            ),
            (pl.col("_ret").rolling_std(63, min_samples=40) * ANNUALIZATION).alias(
                "benchmark_volatility_63d"
            ),
            (pl.col(price_col).log() - pl.col(price_col).log().shift(126)).alias(
                "benchmark_trend_126d"
            ),
            (pl.col(price_col) / pl.col(price_col).rolling_max(252, min_samples=60) - 1.0).alias(
                "benchmark_drawdown"
            ),
        ]
    )


def compute_market_state_features(prices: pl.DataFrame) -> pl.DataFrame:
    """Breadth, dispersion, and correlation, computed across the cross-section per date.

    These describe how much idiosyncratic opportunity exists. When dispersion is low and
    correlation is high, even a good ranking earns little, because the names barely differ.
    """
    df = prices.sort(["security_id", "observation_date"]).with_columns(
        [
            (pl.col("adjusted_close").log() - pl.col("adjusted_close").log().shift(1))
            .over("security_id")
            .alias("_ret"),
            pl.col("adjusted_close")
            .rolling_mean(50, min_samples=30)
            .over("security_id")
            .alias("_ma50"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("adjusted_close") > pl.col("_ma50")).cast(pl.Float64).alias("_above_ma"),
            (pl.col("adjusted_close").log() - pl.col("adjusted_close").log().shift(21))
            .over("security_id")
            .alias("_ret21"),
        ]
    )

    per_date = df.group_by("observation_date").agg(
        [
            pl.col("_above_ma").mean().alias("market_breadth"),
            pl.col("_ret21").std().alias("cross_sectional_dispersion"),
            pl.col("_ret").std().alias("_daily_dispersion"),
            pl.col("_ret").mean().alias("_mean_ret"),
        ]
    )

    # Correlation proxy: when the average stock's move is dominated by the common component,
    # the ratio of |mean return| to cross-sectional dispersion rises. Cheap, and monotone in
    # true average correlation, which is what a regime feature needs.
    per_date = per_date.with_columns(
        (pl.col("_mean_ret").abs() / (pl.col("_daily_dispersion") + 1e-9))
        .clip(0.0, 10.0)
        .alias("_corr_raw")
    )
    per_date = per_date.sort("observation_date").with_columns(
        pl.col("_corr_raw").rolling_mean(21, min_samples=10).alias("average_correlation")
    )

    return per_date.select(
        [
            pl.col("observation_date").alias("as_of"),
            "market_breadth",
            "cross_sectional_dispersion",
            "average_correlation",
        ]
    )


def compute_macro_features(
    panel: pl.DataFrame,
    benchmark: pl.DataFrame,
    prices: pl.DataFrame,
    macro: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach all macro and regime features to a panel."""
    out = panel

    regime = compute_benchmark_regime_features(benchmark)
    out = out.join(regime, on="as_of", how="left")

    state = compute_market_state_features(prices)
    out = out.join(state, on="as_of", how="left")

    if macro is not None and not macro.is_empty():
        available = macro["series_id"].unique().to_list()
        wanted = [s for s in ("DGS10", "DGS3MO", "CPIAUCSL", "GDP") if s in available]
        out = attach_macro_series(out, macro, wanted)

        if {"dgs10", "dgs3mo"} <= set(out.columns):
            out = out.with_columns((pl.col("dgs10") - pl.col("dgs3mo")).alias("yield_curve_slope"))
        # Growth proxy from GDP vintages: YoY change of whatever vintage was visible.
        if "gdp" in out.columns:
            out = out.sort("as_of").with_columns(
                pl.col("gdp").pct_change(252).over("security_id").alias("growth_change")
                if "security_id" in out.columns
                else pl.col("gdp").pct_change(252).alias("growth_change")
            )
        if "cpiaucsl" in out.columns:
            out = out.sort("as_of").with_columns(
                (
                    pl.col("cpiaucsl").pct_change(252).over("security_id")
                    if "security_id" in out.columns
                    else pl.col("cpiaucsl").pct_change(252)
                ).alias("inflation_change")
            )
            if "dgs10" in out.columns:
                out = out.with_columns(
                    (pl.col("dgs10") / 100.0 - pl.col("inflation_change")).alias("real_rate_proxy")
                )

    return out


def available_macro_features(panel: pl.DataFrame) -> list[str]:
    return [c for c in MACRO_FEATURE_NAMES if c in panel.columns]
