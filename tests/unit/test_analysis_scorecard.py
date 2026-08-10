"""Scorecard tests (single-stock analysis against a peer universe).

Percentile logic is easy to get subtly backwards -- an inverted sign silently reports the
most volatile stock as the most defensive. These tests use synthetic panels with a *known*
ordering so the direction of every factor is pinned down.

Network-dependent behaviour is not tested here; these run offline against constructed frames.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from quant_platform.analysis import (
    DEFAULT_PEER_UNIVERSE,
    _percentile_of,
    build_scorecard,
    compare_scorecards,
    format_scorecard,
)


def _feature_panel(n: int = 40, as_of: date = date(2024, 6, 14)) -> pl.DataFrame:
    """A cross-section ordered so that risk rises monotonically with the security index.

    S000 is the safest name and S039 the riskiest, on every risk measure *consistently*.
    That consistency matters: max_drawdown is signed the other way round (a shallower, i.e.
    less negative, drawdown is better), so it must run DOWN as the index rises while
    volatility runs UP. An earlier version of this fixture had them moving together, which
    made S000 both the calmest and the most drawn-down stock -- a contradiction that produced
    a middling defensive score for a name the test expected to be the most defensive.
    """
    rows = []
    for i in range(n):
        rows.append(
            {
                "security_id": f"S{i:03d}",
                "as_of": as_of,
                "momentum_252d": i / n,
                "momentum_126d": i / n,
                "momentum_12m_ex1m": i / n,
                "dist_52w_high": -1.0 + i / n,
                "trend_r2_63d": i / n,
                "reversal_5d": i / n,
                "reversal_21d": i / n,
                "volatility_252d": 0.10 + i / n,
                "beta_252d": 0.5 + i / n,
                "idio_vol_252d": 0.10 + i / n,
                "max_drawdown_252d": -(i / n),  # shallower (better) at low index
                "downside_vol_60d": 0.10 + i / n,
                "adv_21d": 15.0 + i / n,
                "amihud_illiquidity_21d": 0.001 + i / n,
                "spread_proxy_21d": 0.001 + i / n,
            }
        )
    return pl.DataFrame(rows)


def _price_panel(n: int = 40, as_of: date = date(2024, 6, 14)) -> pl.DataFrame:
    rows = []
    for i in range(n):
        for back in range(3):
            rows.append(
                {
                    "security_id": f"S{i:03d}",
                    "observation_date": as_of - timedelta(days=back),
                    "close": 100.0 + i,
                    "adjusted_close": 100.0 + i,
                }
            )
    return pl.DataFrame(rows)


class TestPercentileHelper:
    def test_higher_is_better_orders_ascending(self):
        values = np.arange(0.0, 100.0)
        assert _percentile_of(values, 99.0, higher_is_better=True) == pytest.approx(99.0)
        assert _percentile_of(values, 0.0, higher_is_better=True) == pytest.approx(0.0)

    def test_lower_is_better_inverts(self):
        """A low raw value must score a HIGH percentile when low is desirable."""
        values = np.arange(0.0, 100.0)
        assert _percentile_of(values, 0.0, higher_is_better=False) == pytest.approx(100.0)
        assert _percentile_of(values, 99.0, higher_is_better=False) == pytest.approx(1.0)

    def test_nan_target_returns_nan(self):
        assert np.isnan(_percentile_of(np.arange(10.0), float("nan"), True))

    def test_ignores_non_finite_peers(self):
        values = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
        assert np.isfinite(_percentile_of(values, 2.0, higher_is_better=True))

    def test_too_few_peers_returns_nan(self):
        assert np.isnan(_percentile_of(np.array([1.0]), 1.0, True))


class TestScorecardDirection:
    def test_high_momentum_stock_scores_high_on_momentum(self):
        card = build_scorecard("S039", _feature_panel(), _price_panel())
        assert card.family_percentiles["momentum"] > 90

    def test_low_momentum_stock_scores_low(self):
        card = build_scorecard("S000", _feature_panel(), _price_panel())
        assert card.family_percentiles["momentum"] < 10

    def test_low_volatility_stock_scores_HIGH_on_defensive(self):
        """The sign test that matters: defensive means low vol and low beta.

        An inverted sign here would report the riskiest stock as the safest.
        """
        card = build_scorecard("S000", _feature_panel(), _price_panel())
        assert card.family_percentiles["defensive"] > 90, (
            "the lowest-volatility, lowest-beta stock must rank as MOST defensive"
        )

    def test_high_volatility_stock_scores_low_on_defensive(self):
        card = build_scorecard("S039", _feature_panel(), _price_panel())
        assert card.family_percentiles["defensive"] < 10

    def test_liquid_stock_scores_high_on_liquidity(self):
        """High ADV but also high Amihud in this synthetic panel -- they offset.

        Verified instead on the individual factor, where direction is unambiguous.
        """
        card = build_scorecard("S039", _feature_panel(), _price_panel())
        adv = next(f for f in card.factors["liquidity"] if f.name == "adv_21d")
        illiq = next(f for f in card.factors["liquidity"] if f.name == "amihud_illiquidity_21d")
        assert adv.percentile > 90, "highest dollar volume must rank as most liquid"
        assert illiq.percentile < 10, "highest Amihud illiquidity must rank as least liquid"


class TestScorecardStructure:
    def test_reports_peer_count_excluding_self(self):
        card = build_scorecard("S010", _feature_panel(n=40), _price_panel(n=40))
        assert card.peer_count == 39

    def test_composite_rank_is_within_universe(self):
        card = build_scorecard("S020", _feature_panel(), _price_panel())
        assert 1 <= card.composite_rank <= 40
        assert 0 <= card.composite_percentile <= 100

    def test_unavailable_pillars_are_declared(self):
        """Value and quality must be reported as NOT evaluated, never silently omitted."""
        card = build_scorecard("S010", _feature_panel(), _price_panel())
        assert "value" in card.unavailable_pillars
        assert "quality" in card.unavailable_pillars

    def test_caveats_always_present(self):
        card = build_scorecard("S010", _feature_panel(), _price_panel())
        assert card.warnings
        joined = " ".join(card.warnings).lower()
        assert "survivorship" in joined
        assert "not a forecast" in joined or "not a recommendation" in joined

    def test_unknown_ticker_raises_clearly(self):
        with pytest.raises(ValueError, match="no computable features"):
            build_scorecard("NOPE", _feature_panel(), _price_panel())

    def test_thin_cross_section_is_refused(self):
        """A percentile against 3 peers is noise; it must fail rather than mislead."""
        with pytest.raises(ValueError, match="need at least"):
            build_scorecard("S001", _feature_panel(n=5), _price_panel(n=5), min_peers=20)

    def test_price_is_reported(self):
        card = build_scorecard("S005", _feature_panel(), _price_panel())
        assert card.price == pytest.approx(105.0)


class TestRendering:
    def test_format_includes_key_sections(self):
        text = format_scorecard(build_scorecard("S030", _feature_panel(), _price_panel()))
        for expected in ("COMPOSITE STANDING", "MOMENTUM", "DEFENSIVE", "NOT EVALUATED", "CAVEATS"):
            assert expected in text
        assert "not a recommendation" in text.lower()

    def test_comparison_frame_sorts_by_composite(self):
        features, prices = _feature_panel(), _price_panel()
        cards = [build_scorecard(t, features, prices) for t in ("S000", "S020", "S039")]
        frame = compare_scorecards(cards)
        assert frame.height == 3
        percentiles = frame["composite_pct"].to_list()
        assert percentiles == sorted(percentiles, reverse=True)

    def test_to_dict_round_trips(self):
        import json

        card = build_scorecard("S010", _feature_panel(), _price_panel())
        payload = json.loads(json.dumps(card.to_dict(), default=str))
        assert payload["ticker"] == "S010"
        assert payload["warnings"]


class TestPeerUniverse:
    def test_default_universe_is_diversified_and_deduped(self):
        assert len(DEFAULT_PEER_UNIVERSE) >= 80
        assert len(set(DEFAULT_PEER_UNIVERSE)) == len(DEFAULT_PEER_UNIVERSE)

    def test_benchmark_is_not_in_the_peer_list(self):
        """SPY is the benchmark; including it as a peer would rank a stock against an ETF."""
        assert "SPY" not in DEFAULT_PEER_UNIVERSE
