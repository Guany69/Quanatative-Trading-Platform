"""Cross-sectional preprocessing, fitted on the training window only (spec section 12.6).

This module is where scaler leakage normally enters a quant pipeline. The usual bug is
innocuous-looking::

    scaler.fit(all_data)          # <-- sees test-period means and quantiles
    train = scaler.transform(train_data)
    test  = scaler.transform(test_data)

The fitted statistics then encode the future, and every downstream metric is optimistic.
``CrossSectionalPreprocessor`` mirrors the scikit-learn fit/transform split and holds its
learned statistics in ``fitted_``, so a test can assert that fitting on a subset produces
different statistics than fitting on everything -- proving the boundary is real.

Two distinct normalizations happen here, and they are not interchangeable:

* **Cross-sectional** (within a date): ranking and z-scoring across securities. This needs no
  fitted state -- it only uses that date's own cross-section, which is fully knowable at the
  time. It is safe by construction.
* **Panel-level** (across dates): winsorization bounds and imputation values. These DO need
  fitted state, and that state must come from training data only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from quant_platform.utilities.narrow import as_float


@dataclass
class FittedStats:
    """Statistics learned from the training window. Never recomputed on test data."""

    lower_bounds: dict[str, float] = field(default_factory=dict)
    upper_bounds: dict[str, float] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    centers: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    n_train_rows: int = 0


class NotFittedError(RuntimeError):
    """Raised when transform is called before fit."""


class CrossSectionalPreprocessor:
    """Winsorize, impute, and normalize features.

    Parameters
    ----------
    feature_columns:
        Columns to process.
    winsorize_quantile:
        Tail fraction clipped at each end (0.01 = 1st/99th percentile).
    method:
        ``rank`` maps each date's cross-section to [-0.5, 0.5]; ``robust_z`` uses a
        median/MAD z-score. Rank is the default because it is immune to outliers and to
        regime shifts in feature scale.
    add_missing_indicators:
        Emit an ``<name>_was_missing`` column. Missingness is often informative (a stock with
        no fundamentals is a different animal), so discarding it loses signal.
    """

    def __init__(
        self,
        feature_columns: list[str],
        winsorize_quantile: float = 0.01,
        method: str = "rank",
        add_missing_indicators: bool = True,
        min_cross_section: int = 10,
    ) -> None:
        if not 0.0 <= winsorize_quantile < 0.5:
            raise ValueError(f"winsorize_quantile must be in [0, 0.5), got {winsorize_quantile}")
        if method not in {"rank", "robust_z"}:
            raise ValueError(f"unknown method: {method}")
        self.feature_columns = list(feature_columns)
        self.winsorize_quantile = winsorize_quantile
        self.method = method
        self.add_missing_indicators = add_missing_indicators
        self.min_cross_section = min_cross_section
        self.fitted_: FittedStats | None = None

    @property
    def is_fitted(self) -> bool:
        return self.fitted_ is not None

    def fit(self, train: pl.DataFrame) -> CrossSectionalPreprocessor:
        """Learn winsorization bounds and imputation values from TRAINING DATA ONLY."""
        stats = FittedStats(n_train_rows=train.height)
        q_lo, q_hi = self.winsorize_quantile, 1.0 - self.winsorize_quantile

        for col in self.feature_columns:
            if col not in train.columns:
                continue
            s = (
                train[col].drop_nulls().drop_nans()
                if train[col].dtype.is_float()
                else train[col].drop_nulls()
            )
            if s.len() == 0:
                stats.lower_bounds[col] = 0.0
                stats.upper_bounds[col] = 0.0
                stats.medians[col] = 0.0
                stats.centers[col] = 0.0
                stats.scales[col] = 1.0
                continue

            lo = as_float(
                s.quantile(q_lo) if self.winsorize_quantile > 0 else s.min(),
                context=f"{col}.lower_bound",
            )
            hi = as_float(
                s.quantile(q_hi) if self.winsorize_quantile > 0 else s.max(),
                context=f"{col}.upper_bound",
            )
            med = as_float(s.median(), context=f"{col}.median")
            # MAD-based scale: robust to the fat tails that equity features always have.
            mad = as_float((s - med).abs().median(), default=0.0, context=f"{col}.mad")
            # 1.4826 makes MAD a consistent estimator of sigma under normality.
            scale = (
                mad * 1.4826
                if mad > 1e-12
                else (as_float(s.std(), default=1.0, context=f"{col}.std") or 1.0)
            )

            stats.lower_bounds[col] = lo
            stats.upper_bounds[col] = hi
            stats.medians[col] = med
            stats.centers[col] = med
            stats.scales[col] = scale if scale > 1e-12 else 1.0

        self.fitted_ = stats
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the learned statistics. Requires fit() first."""
        if self.fitted_ is None:
            raise NotFittedError(
                "transform() called before fit(). Fitting on the data being transformed "
                "would leak test-period statistics into preprocessing."
            )
        stats = self.fitted_
        out = df

        if self.add_missing_indicators:
            out = out.with_columns(
                [
                    pl.col(c).is_null().cast(pl.Float64).alias(f"{c}_was_missing")
                    for c in self.feature_columns
                    if c in out.columns
                ]
            )

        # 1. Winsorize using TRAIN bounds. A test-period outlier is clipped to the training
        #    range rather than moving the bound, which is what fitting on test would do.
        out = out.with_columns(
            [
                pl.col(c).clip(stats.lower_bounds[c], stats.upper_bounds[c]).alias(c)
                for c in self.feature_columns
                if c in out.columns and c in stats.lower_bounds
            ]
        )

        # 2. Impute. Cross-sectional median (same date) is preferred because it is knowable
        #    at that date; the train median is the fallback when a whole date is empty.
        out = out.with_columns(
            [
                pl.col(c)
                .fill_null(pl.col(c).median().over("as_of"))
                .fill_null(stats.medians.get(c, 0.0))
                .fill_nan(stats.medians.get(c, 0.0))
                .alias(c)
                for c in self.feature_columns
                if c in out.columns
            ]
        )

        # 3. Normalize within each date's cross-section. Safe by construction: only uses
        #    contemporaneous data.
        if self.method == "rank":
            out = out.with_columns(
                [
                    (
                        (pl.col(c).rank("average").over("as_of") - 0.5) / pl.len().over("as_of")
                        - 0.5
                    ).alias(c)
                    for c in self.feature_columns
                    if c in out.columns
                ]
            )
        else:  # robust_z, centered on the date's own median
            out = out.with_columns(
                [
                    (
                        (pl.col(c) - pl.col(c).median().over("as_of"))
                        / (
                            (pl.col(c) - pl.col(c).median().over("as_of"))
                            .abs()
                            .median()
                            .over("as_of")
                            * 1.4826
                            + 1e-12
                        )
                    )
                    .clip(-5.0, 5.0)
                    .alias(c)
                    for c in self.feature_columns
                    if c in out.columns
                ]
            )

        # Dates with too few names produce meaningless cross-sectional statistics.
        counts = out.group_by("as_of").agg(pl.len().alias("_n"))
        out = (
            out.join(counts, on="as_of", how="left")
            .filter(pl.col("_n") >= self.min_cross_section)
            .drop("_n")
        )

        return out

    def fit_transform(self, train: pl.DataFrame) -> pl.DataFrame:
        """Fit on train and transform it. Only valid for the training split."""
        return self.fit(train).transform(train)

    def output_columns(self) -> list[str]:
        cols = list(self.feature_columns)
        if self.add_missing_indicators:
            cols += [f"{c}_was_missing" for c in self.feature_columns]
        return cols


def neutralize(
    df: pl.DataFrame, feature_columns: list[str], by: str, as_of_col: str = "as_of"
) -> pl.DataFrame:
    """Demean features within groups (e.g. sector) on each date.

    Sector neutralization removes the part of a feature that is just a sector bet. Done
    within each date only, so it introduces no look-ahead.
    """
    return df.with_columns(
        [
            (pl.col(c) - pl.col(c).mean().over([as_of_col, by])).alias(c)
            for c in feature_columns
            if c in df.columns
        ]
    )


def winsorize_series(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Clip an array to bounds; kept separate so unit tests can exercise it directly."""
    return np.clip(values, lower, upper)
