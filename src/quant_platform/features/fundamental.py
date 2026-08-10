"""Fundamental features with strict point-in-time gating (spec section 12.4).

This module contains the single most important join in the platform. For each (security,
date) the correct fundamental record is:

    the LATEST record whose available_at <= that date

Not the latest record. Not the record for the most recent fiscal period. Getting this wrong
is the classic look-ahead bug: a Q4 that ended 31 December is not public until mid-February,
so a model trading in January that "knows" Q4 earnings has read the future and every metric
downstream is fiction.

``asof_join`` with ``strategy="backward"`` implements exactly this semantic, and the tests in
``tests/feature_availability/`` assert the boundary empirically rather than trusting it.
"""

from __future__ import annotations

import polars as pl

from quant_platform.features.registry import (
    FeatureFamily,
    MissingPolicy,
    register,
)

# --------------------------------------------------------------------------- definitions
_VALUE_FEATURES = [
    ("earnings_yield", "Trailing 12-month net income / market cap."),
    ("book_to_market", "Total equity / market cap."),
    ("sales_to_price", "Trailing 12-month revenue / market cap."),
    ("fcf_yield", "Trailing 12-month free cash flow / market cap."),
]
_QUALITY_FEATURES = [
    ("gross_profitability", "Gross profit / total assets (Novy-Marx)."),
    ("return_on_equity", "Trailing 12-month net income / total equity."),
    ("return_on_assets", "Trailing 12-month net income / total assets."),
    ("operating_margin", "Operating income / revenue."),
    ("margin_stability", "Negated volatility of operating margin; steadier is better."),
    ("debt_to_assets", "Total debt / total assets."),
    ("interest_coverage", "Operating income / interest expense."),
]
_INVESTMENT_FEATURES = [
    (
        "accruals",
        "(Net income - operating cash flow) / total assets; high accruals are a "
        "known negative signal.",
    ),
    ("asset_growth", "Year-over-year growth in total assets; fast growers tend to underperform."),
    ("capex_growth", "Year-over-year growth in capital expenditure."),
    ("share_issuance", "Year-over-year growth in share count; dilution is a negative signal."),
    ("repurchase_activity", "Buybacks / market cap."),
    ("earnings_growth", "Year-over-year growth in trailing 12-month net income."),
    ("revenue_growth", "Year-over-year growth in trailing 12-month revenue."),
    ("earnings_surprise", "(Actual EPS - consensus) / |consensus|."),
    ("pead_proxy", "Post-earnings-announcement drift proxy: surprise decayed by staleness."),
]

for _name, _desc in _VALUE_FEATURES:
    register(
        name=_name,
        family=FeatureFamily.VALUE,
        required_inputs=("fundamentals", "market_cap"),
        lookback_sessions=0,  # point-in-time snapshot, not a rolling window
        min_observations=0,
        description=_desc,
        missing_policy=MissingPolicy.CROSS_SECTIONAL_MEDIAN,
    )

for _name, _desc in _QUALITY_FEATURES:
    register(
        name=_name,
        family=FeatureFamily.QUALITY,
        required_inputs=("fundamentals",),
        lookback_sessions=0,
        min_observations=0,
        description=_desc,
    )

for _name, _desc in _INVESTMENT_FEATURES:
    register(
        name=_name,
        family=FeatureFamily.INVESTMENT,
        required_inputs=("fundamentals",),
        lookback_sessions=0,
        min_observations=0,
        description=_desc,
    )

FUNDAMENTAL_FEATURE_NAMES = (
    [n for n, _ in _VALUE_FEATURES]
    + [n for n, _ in _QUALITY_FEATURES]
    + [n for n, _ in _INVESTMENT_FEATURES]
)


def build_trailing_fundamentals(fundamentals: pl.DataFrame) -> pl.DataFrame:
    """Add trailing-twelve-month aggregates and year-over-year growth.

    TTM sums four quarters so the ratios are annual and comparable across filers with
    different fiscal calendars. All rolling operations run over records ordered by
    ``available_at``, so a restatement arriving later never retroactively changes an earlier
    vintage's TTM.
    """
    df = fundamentals.sort(["security_id", "available_at", "fiscal_period_end"])

    flow_columns = [
        c
        for c in (
            "revenue",
            "net_income",
            "gross_profit",
            "operating_income",
            "operating_cash_flow",
            "free_cash_flow",
            "capital_expenditure",
            "share_repurchase",
            "interest_expense",
        )
        if c in df.columns
    ]

    # TTM sums for flow items (income/cash-flow); stock items (balance sheet) are point-in-time.
    df = df.with_columns(
        [
            pl.col(c).rolling_sum(4, min_samples=4).over("security_id").alias(f"{c}_ttm")
            for c in flow_columns
        ]
    )

    # Year-over-year comparisons use a 4-quarter lag.
    yoy_sources = [(f"{c}_ttm", c) for c in ("revenue", "net_income") if c in df.columns]
    for ttm_col, base in yoy_sources:
        df = df.with_columns(
            (
                (pl.col(ttm_col) - pl.col(ttm_col).shift(4).over("security_id"))
                / pl.col(ttm_col).shift(4).over("security_id").abs()
            ).alias(f"{base}_growth")
        )

    for col, out in (
        ("total_assets", "asset_growth"),
        ("shares_outstanding", "share_issuance"),
        ("capital_expenditure_ttm", "capex_growth"),
    ):
        if col in df.columns:
            df = df.with_columns(
                (
                    (pl.col(col) - pl.col(col).shift(4).over("security_id"))
                    / pl.col(col).shift(4).over("security_id").abs()
                ).alias(out)
            )

    # Margin stability: negated rolling volatility of the operating margin, so that "high is
    # good" matches every other feature's orientation.
    if {"operating_income", "revenue"}.issubset(df.columns):
        df = df.with_columns((pl.col("operating_income") / pl.col("revenue")).alias("_op_margin"))
        df = df.with_columns(
            (-pl.col("_op_margin").rolling_std(8, min_samples=4).over("security_id")).alias(
                "margin_stability"
            )
        )

    if {"eps_diluted", "consensus_eps"}.issubset(df.columns):
        df = df.with_columns(
            (
                (pl.col("eps_diluted") - pl.col("consensus_eps"))
                / pl.col("consensus_eps").abs().clip(1e-6, None)
            )
            .clip(-5.0, 5.0)
            .alias("earnings_surprise")
        )

    return df


def attach_point_in_time_fundamentals(
    panel: pl.DataFrame,
    fundamentals: pl.DataFrame,
    as_of_col: str = "as_of",
) -> pl.DataFrame:
    """Join each panel row to the latest fundamental record KNOWN at that date.

    Uses an as-of (backward) join on ``available_at``, which selects the most recent record
    whose publication timestamp does not exceed the panel date. Two properties follow:

    * a filing published after ``as_of`` is invisible, regardless of its fiscal period;
    * a restatement of an old period only becomes visible from its own publication date, so
      earlier rows keep seeing the original print.

    Both are required for point-in-time correctness and neither is achievable with a plain
    join on fiscal period.
    """
    if fundamentals.is_empty():
        return panel

    enriched = build_trailing_fundamentals(fundamentals)

    # as-of joins require both sides sorted on the join key.
    left = panel.with_columns(pl.col(as_of_col).cast(pl.Datetime).alias("_as_of_dt")).sort(
        "_as_of_dt"
    )
    right = enriched.sort("available_at")

    keep = [
        c
        for c in right.columns
        if c
        not in {
            "observation_date",
            "source",
            "source_record_id",
            "revision_id",
            "ingested_at",
            "is_adjusted",
            "data_quality_status",
            "symbol",
        }
    ]

    joined = left.join_asof(
        right.select(keep),
        left_on="_as_of_dt",
        right_on="available_at",
        by="security_id",
        strategy="backward",  # latest record at or before the panel date -- never after
    )
    return joined.drop("_as_of_dt")


def compute_fundamental_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Compute fundamental ratios from an already PIT-joined panel.

    Expects the output of ``attach_point_in_time_fundamentals`` plus a ``market_cap`` column.
    Ratios are guarded against zero/negative denominators, which are common in real data
    (negative equity, zero interest expense) and would otherwise produce infinities that
    poison downstream normalization.
    """
    df = panel
    have = set(df.columns)

    def safe_ratio(num: str, den: str, out: str, den_floor: float = 1e-6) -> pl.Expr:
        return (
            pl.when(pl.col(den).abs() > den_floor)
            .then(pl.col(num) / pl.col(den))
            .otherwise(None)
            .alias(out)
        )

    exprs: list[pl.Expr] = []

    if {"net_income_ttm", "market_cap"} <= have:
        exprs.append(safe_ratio("net_income_ttm", "market_cap", "earnings_yield"))
    if {"total_equity", "market_cap"} <= have:
        exprs.append(safe_ratio("total_equity", "market_cap", "book_to_market"))
    if {"revenue_ttm", "market_cap"} <= have:
        exprs.append(safe_ratio("revenue_ttm", "market_cap", "sales_to_price"))
    if {"free_cash_flow_ttm", "market_cap"} <= have:
        exprs.append(safe_ratio("free_cash_flow_ttm", "market_cap", "fcf_yield"))
    if {"share_repurchase_ttm", "market_cap"} <= have:
        exprs.append(safe_ratio("share_repurchase_ttm", "market_cap", "repurchase_activity"))

    if {"gross_profit_ttm", "total_assets"} <= have:
        exprs.append(safe_ratio("gross_profit_ttm", "total_assets", "gross_profitability"))
    if {"net_income_ttm", "total_equity"} <= have:
        exprs.append(safe_ratio("net_income_ttm", "total_equity", "return_on_equity"))
    if {"net_income_ttm", "total_assets"} <= have:
        exprs.append(safe_ratio("net_income_ttm", "total_assets", "return_on_assets"))
    if {"operating_income_ttm", "revenue_ttm"} <= have:
        exprs.append(safe_ratio("operating_income_ttm", "revenue_ttm", "operating_margin"))
    if {"total_debt", "total_assets"} <= have:
        exprs.append(safe_ratio("total_debt", "total_assets", "debt_to_assets"))
    if {"operating_income_ttm", "interest_expense_ttm"} <= have:
        exprs.append(
            safe_ratio("operating_income_ttm", "interest_expense_ttm", "interest_coverage")
        )

    if {"net_income_ttm", "operating_cash_flow_ttm", "total_assets"} <= have:
        exprs.append(
            pl.when(pl.col("total_assets").abs() > 1e-6)
            .then(
                (pl.col("net_income_ttm") - pl.col("operating_cash_flow_ttm"))
                / pl.col("total_assets")
            )
            .otherwise(None)
            .alias("accruals")
        )

    df = df.with_columns(exprs) if exprs else df

    # PEAD proxy: an earnings surprise matters most right after publication and decays as it
    # becomes stale news. Staleness is measured from the filing that is currently visible.
    if {"earnings_surprise", "filing_date"} <= set(df.columns):
        df = df.with_columns(
            (
                pl.col("earnings_surprise")
                * (
                    1.0
                    - (
                        (pl.col("as_of") - pl.col("filing_date")).dt.total_days().cast(pl.Float64)
                        / 90.0
                    ).clip(0.0, 1.0)
                )
            ).alias("pead_proxy")
        )

    # Clip extreme ratios: real filings produce wild values (tiny denominators) that would
    # dominate cross-sectional statistics.
    ratio_cols = [c for c in FUNDAMENTAL_FEATURE_NAMES if c in df.columns]
    if ratio_cols:
        df = df.with_columns([pl.col(c).clip(-100.0, 100.0).alias(c) for c in ratio_cols])
    return df


def available_fundamental_features(panel: pl.DataFrame) -> list[str]:
    """Which fundamental features actually got computed for this panel."""
    return [c for c in FUNDAMENTAL_FEATURE_NAMES if c in panel.columns]
