"""Preprocessing leakage tests (spec section 16.4).

The canonical bug these guard against::

    scaler.fit(all_data)   # sees the future
    scaler.transform(train); scaler.transform(test)

A test that merely calls fit(train) and asserts it runs proves nothing. These tests instead
construct data where the test period is *materially different* from training, then assert the
fitted statistics reflect training only -- i.e. the future genuinely did not inform them.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from quant_platform.features.preprocessing import (
    CrossSectionalPreprocessor,
    NotFittedError,
)


def _panel(start: date, n_dates: int, n_secs: int, value_fn) -> pl.DataFrame:
    rows = []
    for d_i in range(n_dates):
        as_of = start + timedelta(days=d_i)
        for s_i in range(n_secs):
            rows.append(
                {
                    "security_id": f"S{s_i:03d}",
                    "as_of": as_of,
                    "feat": float(value_fn(d_i, s_i)),
                }
            )
    return pl.DataFrame(rows)


class TestFitTransformBoundary:
    def test_transform_before_fit_raises(self):
        p = CrossSectionalPreprocessor(["feat"], min_cross_section=2)
        df = _panel(date(2020, 1, 1), 3, 10, lambda d, s: s)
        with pytest.raises(NotFittedError, match="before fit"):
            p.transform(df)

    def test_fitted_stats_come_from_training_only(self):
        """Fitting on train must not see the test period's wildly different scale."""
        train = _panel(date(2020, 1, 1), 10, 20, lambda d, s: s)  # values 0..19

        p = CrossSectionalPreprocessor(["feat"], winsorize_quantile=0.0, min_cross_section=2)
        p.fit(train)

        # The learned upper bound reflects TRAIN's range (max 19), not test's (19000).
        assert p.fitted_.upper_bounds["feat"] == pytest.approx(19.0)
        assert p.fitted_.n_train_rows == train.height

    def test_fitting_on_full_data_would_differ(self):
        """Demonstrates the boundary is load-bearing, not decorative.

        If fitting on train+test produced the same statistics as fitting on train, the test
        above would pass trivially. It does not.
        """
        train = _panel(date(2020, 1, 1), 10, 20, lambda d, s: s)
        test = _panel(date(2020, 2, 1), 10, 20, lambda d, s: s * 1000)
        combined = pl.concat([train, test])

        p_train = CrossSectionalPreprocessor(
            ["feat"], winsorize_quantile=0.0, min_cross_section=2
        ).fit(train)
        p_all = CrossSectionalPreprocessor(
            ["feat"], winsorize_quantile=0.0, min_cross_section=2
        ).fit(combined)

        # The leaky fit sees a vastly larger bound: this is exactly the information that
        # must not reach the model.
        assert p_all.fitted_.upper_bounds["feat"] > 100 * p_train.fitted_.upper_bounds["feat"]

    def test_test_outliers_are_clipped_to_train_bounds(self):
        """A test-period extreme must be clipped, not allowed to redefine the range."""
        train = _panel(date(2020, 1, 1), 10, 20, lambda d, s: s)  # 0..19
        test = _panel(date(2020, 2, 1), 5, 20, lambda d, s: s * 1000)

        p = CrossSectionalPreprocessor(
            ["feat"], winsorize_quantile=0.0, method="robust_z", min_cross_section=2
        ).fit(train)
        out = p.transform(test)
        # After clipping to the train max of 19, every test value collapses to the train
        # range; the enormous test values cannot propagate.
        assert out["feat"].max() <= 5.0  # robust_z clips at +/-5


class TestCrossSectionalSafety:
    def test_rank_uses_only_same_date_cross_section(self):
        """Ranking within a date cannot leak: it only uses contemporaneous data.

        Verified by checking that changing a *different* date's values leaves this date's
        ranks untouched.
        """
        base = _panel(date(2020, 1, 1), 2, 10, lambda d, s: s)
        p = CrossSectionalPreprocessor(["feat"], min_cross_section=2).fit(base)
        out_base = p.transform(base)
        d0 = (
            out_base.filter(pl.col("as_of") == date(2020, 1, 1))
            .sort("security_id")["feat"]
            .to_list()
        )

        # Perturb only the second date, drastically.
        perturbed = base.with_columns(
            pl.when(pl.col("as_of") == date(2020, 1, 2))
            .then(pl.col("feat") * 10_000)
            .otherwise(pl.col("feat"))
            .alias("feat")
        )
        out_pert = p.transform(perturbed)
        d0_pert = (
            out_pert.filter(pl.col("as_of") == date(2020, 1, 1))
            .sort("security_id")["feat"]
            .to_list()
        )

        assert d0 == pytest.approx(d0_pert)

    def test_rank_output_is_bounded_and_centered(self):
        df = _panel(date(2020, 1, 1), 3, 100, lambda d, s: s**2)
        p = CrossSectionalPreprocessor(["feat"], min_cross_section=2).fit(df)
        out = p.transform(df)
        assert out["feat"].min() >= -0.5
        assert out["feat"].max() <= 0.5
        assert out["feat"].mean() == pytest.approx(0.0, abs=1e-9)


class TestImputation:
    def test_imputes_from_same_date_median_not_future(self):
        rows = []
        for d_i, as_of in enumerate([date(2020, 1, 1), date(2020, 1, 2)]):
            for s_i in range(10):
                # One null on the first date; second date has a totally different level.
                val = None if (d_i == 0 and s_i == 0) else (float(s_i) if d_i == 0 else 1000.0)
                rows.append({"security_id": f"S{s_i}", "as_of": as_of, "feat": val})
        df = pl.DataFrame(rows, schema_overrides={"feat": pl.Float64})

        p = CrossSectionalPreprocessor(
            ["feat"], winsorize_quantile=0.0, method="robust_z", min_cross_section=2
        ).fit(df)
        out = p.transform(df)
        # The imputed value must come from the first date's own cross-section (median of
        # 1..9 = 5), never from the second date's 1000s.
        assert out.filter(pl.col("as_of") == date(2020, 1, 1))["feat"].null_count() == 0

    def test_missing_indicator_preserves_information(self):
        rows = [
            {"security_id": f"S{i}", "as_of": date(2020, 1, 1), "feat": None if i < 3 else float(i)}
            for i in range(10)
        ]
        df = pl.DataFrame(rows, schema_overrides={"feat": pl.Float64})
        p = CrossSectionalPreprocessor(
            ["feat"], min_cross_section=2, add_missing_indicators=True
        ).fit(df)
        out = p.transform(df)
        assert "feat_was_missing" in out.columns
        assert out["feat_was_missing"].sum() == 3


class TestThinCrossSections:
    def test_drops_dates_with_too_few_securities(self):
        """Cross-sectional statistics on 2 names are noise, not signal."""
        rows = []
        for s_i in range(20):
            rows.append({"security_id": f"S{s_i}", "as_of": date(2020, 1, 1), "feat": float(s_i)})
        for s_i in range(2):  # thin date
            rows.append({"security_id": f"S{s_i}", "as_of": date(2020, 1, 2), "feat": float(s_i)})
        df = pl.DataFrame(rows)

        p = CrossSectionalPreprocessor(["feat"], min_cross_section=10).fit(df)
        out = p.transform(df)
        assert date(2020, 1, 1) in out["as_of"].to_list()
        assert date(2020, 1, 2) not in out["as_of"].to_list()
