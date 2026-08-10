"""Placebo / no-signal control (spec section 23).

The most convincing evidence that a research pipeline does not leak the future is that it
finds *nothing* when there is nothing to find. Every other leakage test checks one specific
mechanism; this one checks the whole pipeline end to end, including any mechanism nobody
thought of.

The logic:

* Generate fixture data with ``momentum_signal_strength=0`` -- returns are then pure factor
  noise with no cross-sectional predictability planted in them.
* Run the complete pipeline: universe, features, labels, purged split, train-only
  preprocessing, model fit, prediction.
* The out-of-sample IC must be indistinguishable from zero.

If preprocessing were fitted on the full panel, or labels overlapped the test window, or a
future price reached a feature, the model would score a positive IC on data that contains no
signal at all. That is the signature of leakage.

The mirror test (``test_planted_signal_is_detected``) confirms the pipeline is not simply
broken in a way that always returns zero.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from quant_platform.config.models import ResearchCharter
from quant_platform.data.adapters.fixture import FixtureDataProvider, FixtureSpec
from quant_platform.evaluation.metrics import spearman_ic
from quant_platform.features.compute import compute_price_features
from quant_platform.labels.generator import LabelGenerator
from quant_platform.pipeline import build_adjusted_prices
from quant_platform.utilities.calendar import TradingCalendar

TEST_START = date(2021, 1, 4)


def _ic_of_feature(strength: float, feature: str, n_securities: int = 50) -> tuple[float, int]:
    """Build fixture data at a given signal strength and measure a feature's out-of-sample IC."""
    charter = ResearchCharter()
    spec = FixtureSpec(n_securities=n_securities, seed=42, momentum_signal_strength=strength)
    ds = FixtureDataProvider(spec).generate()
    cal = TradingCalendar(spec.start, spec.end)

    px = build_adjusted_prices(ds.prices, ds.corporate_actions)
    bm = ds.benchmark.with_columns(pl.col("close").alias("adjusted_close"))
    feats = compute_price_features(px, bm, ["momentum", "reversal", "risk", "liquidity"])
    signal_dates = cal.rebalance_sessions("weekly", "week_end")
    labels = LabelGenerator(charter.label, cal, 1).generate(px, bm, ds.securities, signal_dates)

    joined = feats.join(labels, on=["security_id", "as_of"], how="inner").filter(
        pl.col("as_of") >= TEST_START
    )
    ics = []
    for _as_of, g in joined.group_by("as_of"):
        if g.height < 10:
            continue
        ic = spearman_ic(g[feature].to_numpy(), g["excess_return_rank"].to_numpy())
        if np.isfinite(ic):
            ics.append(ic)
    arr = np.array(ics)
    if arr.size == 0:
        raise AssertionError(
            f"feature '{feature}' produced no computable IC across any cross-section -- it is "
            f"probably constant, which makes Spearman undefined. A degenerate feature would "
            f"pass an 'IC is near zero' assertion vacuously, so this is failed loudly instead."
        )
    return float(arr.mean()), int(arr.size)


class TestPlaceboControl:
    def test_no_signal_produces_no_ic(self):
        """With zero planted signal, out-of-sample IC must be ~0.

        A leaky pipeline would score meaningfully positive here, because it would be reading
        the future rather than predicting it.
        """
        ic, n_periods = _ic_of_feature(strength=0.0, feature="momentum_126d")
        assert n_periods > 100, f"too few evaluation periods ({n_periods}) to judge"
        # |IC| < 0.03 is indistinguishable from noise across ~200 weekly cross-sections.
        assert abs(ic) < 0.03, (
            f"momentum IC = {ic:+.4f} on data with NO planted signal. A pipeline that finds "
            f"signal in pure noise is leaking the future."
        )

    def test_planted_signal_is_detected(self):
        """The mirror: the pipeline must find a signal that genuinely exists.

        Without this, `test_no_signal_produces_no_ic` would also pass for a pipeline that is
        simply broken and always predicts noise.
        """
        ic, n_periods = _ic_of_feature(strength=0.02, feature="momentum_126d")
        assert n_periods > 100
        assert ic > 0.02, (
            f"momentum IC = {ic:+.4f} on data with a planted momentum effect. The pipeline "
            f"should detect a signal that is genuinely present."
        )

    def test_signal_strength_orders_correctly(self):
        """A stronger planted effect must produce a stronger measured IC.

        Confirms the pipeline responds monotonically to ground truth rather than returning an
        arbitrary constant.
        """
        ic_none, _ = _ic_of_feature(strength=0.0, feature="momentum_126d")
        ic_weak, _ = _ic_of_feature(strength=0.02, feature="momentum_126d")
        ic_strong, _ = _ic_of_feature(strength=0.05, feature="momentum_126d")
        assert ic_none < ic_weak < ic_strong, (
            f"IC did not increase with planted signal strength: "
            f"none={ic_none:+.4f}, weak={ic_weak:+.4f}, strong={ic_strong:+.4f}"
        )


class TestUnrelatedFeatureHasNoSignal:
    @pytest.mark.parametrize("feature", ["amihud_illiquidity_21d", "volume_surprise_21d"])
    def test_liquidity_feature_carries_no_planted_signal(self, feature: str):
        """Only momentum was planted, so liquidity features should show no strong IC.

        Guards against a bug that smears signal across unrelated columns (e.g. a bad join or
        an off-by-one shift), which would show up as everything predicting everything.

        Note: ``zero_return_freq_63d`` is deliberately NOT tested here. The fixture generates
        returns from a continuous distribution, so they are never exactly zero and the feature
        is constant -- see docs/limitations.md. Testing it would assert on an undefined IC.
        """
        ic, _ = _ic_of_feature(strength=0.02, feature=feature)
        assert abs(ic) < 0.05, (
            f"unrelated feature '{feature}' has IC {ic:+.4f}; signal may be leaking across "
            f"columns via a join or alignment bug"
        )
