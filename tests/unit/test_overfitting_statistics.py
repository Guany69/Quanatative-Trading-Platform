"""Tests for the multiple-testing and overfitting statistics (spec section 23).

These are checked against analytically known values and known qualitative behaviour, because
a subtly wrong statistical correction is worse than none: it produces false confidence with a
respectable-looking number attached.

The headline test is `test_dsr_unmasks_a_lucky_winner`: the best of 200 zero-alpha strategies
has a genuinely impressive naive Sharpe, and the Deflated Sharpe Ratio must see through it.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_platform.validation.overfitting import (
    _norm_cdf,
    _norm_ppf,
    analyze_sharpe,
    bootstrap_confidence_interval,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sample_kurtosis,
    sample_skewness,
)


class TestNormalDistributionHelpers:
    @pytest.mark.parametrize(
        ("p", "expected"),
        [(0.5, 0.0), (0.975, 1.959964), (0.95, 1.644854), (0.025, -1.959964)],
    )
    def test_ppf_matches_known_quantiles(self, p, expected):
        assert _norm_ppf(p) == pytest.approx(expected, abs=1e-5)

    def test_cdf_matches_known_values(self):
        assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-9)
        assert _norm_cdf(1.959964) == pytest.approx(0.975, abs=1e-5)

    def test_ppf_and_cdf_are_inverses(self):
        for p in (0.01, 0.1, 0.3, 0.7, 0.9, 0.99):
            assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-6)

    def test_ppf_rejects_out_of_range(self):
        assert np.isnan(_norm_ppf(0.0))
        assert np.isnan(_norm_ppf(1.0))


class TestMoments:
    def test_normal_sample_moments(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, 200_000)
        assert sample_skewness(z) == pytest.approx(0.0, abs=0.05)
        assert sample_kurtosis(z) == pytest.approx(3.0, abs=0.1)

    def test_left_skewed_sample_has_negative_skew(self):
        rng = np.random.default_rng(1)
        x = -rng.gamma(2.0, 1.0, 50_000)
        assert sample_skewness(x) < -0.5

    def test_too_few_observations_returns_nan(self):
        assert np.isnan(sample_skewness(np.array([1.0, 2.0])))
        assert np.isnan(sample_kurtosis(np.array([1.0, 2.0, 3.0])))


class TestProbabilisticSharpe:
    def test_psr_increases_with_track_record_length(self):
        """Same underlying Sharpe, more evidence -> more confidence."""
        rng = np.random.default_rng(3)
        base = rng.normal(0.0005, 0.01, 100_000)

        def standardize(x):
            return (x - x.mean()) / x.std() * 0.01 + 0.0005

        short = probabilistic_sharpe_ratio(standardize(base[:100]))
        long = probabilistic_sharpe_ratio(standardize(base[:5000]))
        assert long > short

    def test_negative_skew_reduces_confidence(self):
        """Controlled: identical mean and sd, only skewness differs.

        Negative skew inflates the Sharpe estimator's variance, so the same raw Sharpe should
        yield LESS confidence. This is the correction that makes PSR worth computing.
        """
        rng = np.random.default_rng(7)
        n = 5000

        def standardize(x, mu=0.0005, sd=0.01):
            return (x - x.mean()) / x.std() * sd + mu

        left = standardize(-rng.gamma(1.5, 1, n))
        symmetric = standardize(rng.normal(0, 1, n))
        right = standardize(rng.gamma(1.5, 1, n))

        psr_left = probabilistic_sharpe_ratio(left)
        psr_sym = probabilistic_sharpe_ratio(symmetric)
        psr_right = probabilistic_sharpe_ratio(right)

        assert sample_skewness(left) < 0 < sample_skewness(right)
        assert psr_left < psr_sym < psr_right, (
            f"PSR should fall as skew turns negative: "
            f"left={psr_left:.4f} sym={psr_sym:.4f} right={psr_right:.4f}"
        )

    def test_zero_mean_returns_give_psr_near_half(self):
        rng = np.random.default_rng(11)
        x = rng.normal(0, 0.01, 5000)
        x = x - x.mean()  # exactly zero mean
        assert probabilistic_sharpe_ratio(x) == pytest.approx(0.5, abs=0.05)

    def test_insufficient_data_returns_nan(self):
        assert np.isnan(probabilistic_sharpe_ratio(np.array([0.01, 0.02])))


class TestDeflatedSharpe:
    def test_expected_max_sharpe_grows_with_trials(self):
        """More attempts -> a higher Sharpe is expected from luck alone."""
        values = [expected_max_sharpe(n, 0.5) for n in (2, 10, 100, 1000)]
        assert values == sorted(values)
        assert values[0] > 0

    def test_single_trial_needs_no_deflation(self):
        assert expected_max_sharpe(1, 0.5) == 0.0

    def test_dsr_unmasks_a_lucky_winner(self):
        """The best of 200 zero-alpha strategies must not look significant.

        This is the entire point of the Deflated Sharpe Ratio: a naive Sharpe (and even PSR)
        will look excellent for the winner of a large search, because selecting the maximum of
        many noisy estimates is itself a source of apparent skill.
        """
        rng = np.random.default_rng(42)
        trials = [rng.normal(0, 0.01, 500) for _ in range(200)]  # zero expected return
        sharpes = [float(x.mean() / x.std() * np.sqrt(252)) for x in trials]
        best = max(trials, key=lambda x: x.mean() / x.std())

        naive = float(best.mean() / best.std() * np.sqrt(252))
        psr = probabilistic_sharpe_ratio(best)
        dsr = deflated_sharpe_ratio(best, n_trials=200, all_trial_sharpes=sharpes)

        assert naive > 1.0, "the lucky winner should have an impressive naive Sharpe"
        assert psr > 0.9, "PSR alone does not account for selection"
        assert dsr < psr, "DSR must be more conservative than PSR after a 200-trial search"
        assert dsr < 0.95, f"DSR of {dsr:.3f} treats a zero-alpha lucky winner as significant"

    def test_dsr_returns_nan_without_trial_dispersion(self):
        """No fabricated correction when the required input is missing."""
        rng = np.random.default_rng(5)
        x = rng.normal(0.0005, 0.01, 1000)
        assert np.isnan(deflated_sharpe_ratio(x, n_trials=100))

    def test_analyze_sharpe_bundles_everything(self):
        rng = np.random.default_rng(9)
        x = rng.normal(0.0005, 0.01, 2000)
        sharpes = [float(rng.normal(0.5, 0.4)) for _ in range(50)]
        stats = analyze_sharpe(x, n_trials=50, all_trial_sharpes=sharpes)
        assert stats.n_observations == 2000
        assert np.isfinite(stats.sharpe_ratio)
        assert np.isfinite(stats.probabilistic_sharpe)
        assert stats.n_trials == 50
        assert "Sharpe" in stats.verdict()


class TestPBO:
    def test_random_strategies_give_pbo_near_half(self):
        """With no real skill, in-sample selection should be a coin flip out of sample."""
        rng = np.random.default_rng(13)
        matrix = rng.normal(0, 0.01, (1000, 20))
        result = probability_of_backtest_overfitting(matrix, n_splits=8)
        assert np.isfinite(result["pbo"])
        assert 0.25 <= result["pbo"] <= 0.75, f"PBO {result['pbo']:.3f} is far from chance"
        assert result["n_combinations"] > 0

    def test_genuinely_superior_strategy_lowers_pbo(self):
        """One strategy with real edge should be picked consistently, lowering PBO."""
        rng = np.random.default_rng(17)
        matrix = rng.normal(0, 0.01, (1500, 10))
        matrix[:, 0] += 0.002  # a genuinely better strategy, present in every period
        result = probability_of_backtest_overfitting(matrix, n_splits=8)
        assert result["pbo"] < 0.35, (
            f"PBO {result['pbo']:.3f}: a consistently superior strategy should be selected "
            f"reliably out of sample"
        )

    def test_insufficient_data_returns_nan(self):
        rng = np.random.default_rng(19)
        assert np.isnan(probability_of_backtest_overfitting(rng.normal(0, 1, (10, 5)))["pbo"])

    def test_single_strategy_returns_nan(self):
        rng = np.random.default_rng(23)
        assert np.isnan(probability_of_backtest_overfitting(rng.normal(0, 1, (500, 1)))["pbo"])


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(29)
        x = rng.normal(0.0005, 0.01, 2000)
        ci = bootstrap_confidence_interval(x, n_bootstrap=300, random_seed=1)
        assert ci["lower"] <= ci["point"] <= ci["upper"]

    def test_wider_confidence_gives_wider_interval(self):
        rng = np.random.default_rng(31)
        x = rng.normal(0.0005, 0.01, 2000)
        narrow = bootstrap_confidence_interval(x, n_bootstrap=300, confidence=0.80, random_seed=1)
        wide = bootstrap_confidence_interval(x, n_bootstrap=300, confidence=0.99, random_seed=1)
        assert (wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])

    def test_is_reproducible_with_a_seed(self):
        rng = np.random.default_rng(37)
        x = rng.normal(0.0005, 0.01, 1000)
        a = bootstrap_confidence_interval(x, n_bootstrap=200, random_seed=7)
        b = bootstrap_confidence_interval(x, n_bootstrap=200, random_seed=7)
        assert a["lower"] == pytest.approx(b["lower"])

    def test_short_series_returns_nan(self):
        assert np.isnan(bootstrap_confidence_interval(np.array([0.01] * 10))["point"])


class TestMinimumTrackRecord:
    def test_stronger_performance_needs_less_data(self):
        rng = np.random.default_rng(41)

        def standardize(x, mu, sd=0.01):
            return (x - x.mean()) / x.std() * sd + mu

        base = rng.normal(0, 1, 3000)
        weak = minimum_track_record_length(standardize(base, 0.0002))
        strong = minimum_track_record_length(standardize(base, 0.0015))
        assert strong < weak

    def test_non_positive_edge_is_never_establishable(self):
        rng = np.random.default_rng(43)
        x = rng.normal(0, 0.01, 1000)
        x = x - x.mean() - 0.0001  # negative mean
        assert minimum_track_record_length(x) == float("inf")
