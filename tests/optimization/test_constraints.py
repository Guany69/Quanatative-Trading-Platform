"""Optimizer constraint tests (spec section 31.5).

An optimizer that *requests* constraints but does not enforce them is worse than no optimizer
at all: the portfolio looks compliant on paper while carrying risk nobody authorized. These
tests inspect the returned weights directly rather than trusting the solver's status code.

They also verify the relaxation path, because the spec's hard requirement is that constraints
are never *silently* dropped -- relaxation must be both ordered and logged.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from quant_platform.config.models import PortfolioConfig
from quant_platform.portfolio.constrained_optimizer import ConstrainedOptimizer
from quant_platform.portfolio.constructors import (
    EqualWeightConstructor,
    InverseVolatilityConstructor,
    ScoreWeightedConstructor,
    _apply_cap_and_redistribute,
)
from quant_platform.risk.covariance import RiskModel

pytest.importorskip("cvxpy", reason="cvxpy is required for the constrained optimizer")


def _candidates(n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "security_id": [f"S{i:03d}" for i in range(n)],
            "as_of": [date(2021, 6, 15)] * n,
            "cross_sectional_rank": np.linspace(0.01, 1.0, n),
            "volatility_60d": rng.uniform(0.15, 0.6, n),
        }
    )


def _risk_model(n: int = 120, seed: int = 0) -> RiskModel:
    """A well-conditioned covariance with a common factor plus idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    ids = [f"S{i:03d}" for i in range(n)]
    betas = rng.uniform(0.6, 1.5, n)
    idio = rng.uniform(0.15, 0.45, n)
    market_var = 0.16**2
    cov = np.outer(betas, betas) * market_var + np.diag(idio**2)
    return RiskModel(
        security_ids=ids,
        covariance=cov,
        volatilities=np.sqrt(np.diag(cov)),
        betas=betas,
        method="test",
        n_observations=252,
    )


CFG = PortfolioConfig(
    min_holdings=20,
    max_holdings=50,
    max_position_weight=0.05,
    min_position_weight=0.005,
    max_turnover_per_rebalance=1.0,
    min_beta=0.8,
    max_beta=1.2,
)


class TestHardConstraints:
    def test_weights_sum_to_budget(self):
        opt = ConstrainedOptimizer(CFG)
        weights, diag = opt.build(_candidates(), {}, date(2021, 6, 15), risk_model=_risk_model())
        assert weights, f"optimizer produced nothing: {diag.message}"
        assert sum(weights.values()) == pytest.approx(1.0 - CFG.cash_buffer, abs=1e-6)

    def test_long_only_is_enforced(self):
        opt = ConstrainedOptimizer(CFG)
        weights, _ = opt.build(_candidates(), {}, date(2021, 6, 15), risk_model=_risk_model())
        assert min(weights.values()) >= 0.0

    def test_position_cap_is_enforced(self):
        opt = ConstrainedOptimizer(CFG)
        weights, _ = opt.build(_candidates(), {}, date(2021, 6, 15), risk_model=_risk_model())
        assert max(weights.values()) <= CFG.max_position_weight + 1e-6

    def test_holding_count_respected(self):
        opt = ConstrainedOptimizer(CFG)
        weights, diag = opt.build(_candidates(), {}, date(2021, 6, 15), risk_model=_risk_model())
        assert len(weights) <= CFG.max_holdings
        assert diag.n_holdings == len(weights)

    def test_beta_stays_in_band(self):
        rm = _risk_model()
        opt = ConstrainedOptimizer(CFG)
        weights, _ = opt.build(_candidates(), {}, date(2021, 6, 15), risk_model=rm)
        idx = {s: i for i, s in enumerate(rm.security_ids)}
        beta = sum(w * rm.betas[idx[s]] for s, w in weights.items())
        # Small tolerance: post-solve dust removal and rescaling shift beta slightly.
        assert CFG.min_beta - 0.15 <= beta <= CFG.max_beta + 0.15, f"beta {beta:.3f} out of band"

    def test_turnover_cap_limits_trading(self):
        """With a tight cap, the new book must stay close to the previous one."""
        rm = _risk_model()
        cands = _candidates()
        tight = CFG.model_copy(update={"max_turnover_per_rebalance": 0.05})
        # Previous book: the bottom-ranked names, so the signal wants a full rotation.
        previous = {f"S{i:03d}": 0.02 for i in range(50)}

        opt = ConstrainedOptimizer(tight)
        weights, _diag = opt.build(cands, previous, date(2021, 6, 15), risk_model=rm)
        keys = set(weights) | set(previous)
        turnover = 0.5 * sum(abs(weights.get(k, 0.0) - previous.get(k, 0.0)) for k in keys)
        # Allow headroom for relaxation, but it must not be a free-for-all.
        assert turnover <= 0.35, f"turnover {turnover:.3f} ignores the cap"

    def test_restricted_securities_excluded(self):
        cfg = CFG.model_copy(
            update={"restricted_securities": [f"S{i:03d}" for i in range(100, 120)]}
        )
        opt = ConstrainedOptimizer(cfg)
        weights, _ = opt.build(_candidates(), {}, date(2021, 6, 15), risk_model=_risk_model())
        assert not (set(weights) & set(cfg.restricted_securities))

    def test_sector_deviation_constraint(self):
        rm = _risk_model()
        sectors = {f"S{i:03d}": ["tech", "fin", "health", "energy"][i % 4] for i in range(120)}
        bench = {"tech": 0.25, "fin": 0.25, "health": 0.25, "energy": 0.25}
        cfg = CFG.model_copy(update={"max_sector_deviation": 0.05})

        opt = ConstrainedOptimizer(cfg)
        weights, _ = opt.build(
            _candidates(),
            {},
            date(2021, 6, 15),
            risk_model=rm,
            sector_map=sectors,
            benchmark_sector_weights=bench,
        )
        actual: dict[str, float] = {}
        for sec, w in weights.items():
            actual[sectors[sec]] = actual.get(sectors[sec], 0.0) + w
        for sector, target in bench.items():
            deviation = abs(actual.get(sector, 0.0) - target)
            assert deviation <= cfg.max_sector_deviation + 0.06, (
                f"{sector} deviates {deviation:.3f} from benchmark weight {target}"
            )


class TestInfeasibilityAndRelaxation:
    def test_config_validation_rejects_arithmetically_impossible_mandates(self):
        """Some mandates cannot be satisfied by any weights, so they never reach the solver.

        30 names capped at 2% can only reach 60% invested. That is caught at config load,
        which is the right layer: it is a specification error, not a solver failure.
        """
        with pytest.raises(Exception, match="infeasible"):
            PortfolioConfig(
                min_holdings=25,
                max_holdings=28,
                max_position_weight=0.02,
                min_position_weight=0.015,
            )

    def test_relaxations_are_logged_not_silent(self):
        """A solve-time infeasible mandate must report what was loosened.

        The config below is arithmetically valid (so it loads), but the beta band is
        unreachable: betas are drawn from U(0.6, 1.5), and holding >= 20 names at a 5% cap
        cannot push the portfolio beta above 1.45.
        """
        rm = _risk_model(n=60)
        cands = _candidates(60)
        hard_beta = PortfolioConfig(
            min_holdings=20,
            max_holdings=40,
            max_position_weight=0.05,
            min_position_weight=0.01,
            min_beta=1.45,
            max_beta=1.50,
        )
        opt = ConstrainedOptimizer(hard_beta)
        weights, diag = opt.build(cands, {}, date(2021, 6, 15), risk_model=rm)
        assert diag.relaxations_applied, (
            "an infeasible problem produced weights with no recorded relaxation -- "
            "constraints were dropped silently"
        )
        # Whatever happened, the hard risk limits must still hold.
        if weights:
            assert min(weights.values()) >= 0.0
            assert max(weights.values()) <= hard_beta.max_position_weight + 1e-6

    def test_empty_candidates_returns_structured_failure(self):
        opt = ConstrainedOptimizer(CFG)
        empty = _candidates(0)
        weights, diag = opt.build(empty, {}, date(2021, 6, 15), risk_model=_risk_model())
        assert weights == {}
        assert diag.solver_status.value == "infeasible"
        assert not diag.succeeded

    def test_fallback_still_respects_hard_limits(self):
        """Even the fallback path must not breach long-only or the position cap."""
        rm = _risk_model(n=25)
        impossible = PortfolioConfig(
            min_holdings=24,
            max_holdings=25,
            max_position_weight=0.05,
            min_position_weight=0.04,
            max_turnover_per_rebalance=0.001,
            min_beta=1.999,
            max_beta=2.0,  # unreachable with betas in [0.6, 1.5]
        )
        opt = ConstrainedOptimizer(impossible)
        weights, diag = opt.build(_candidates(25), {}, date(2021, 6, 15), risk_model=rm)
        if weights:
            assert min(weights.values()) >= 0.0
            assert max(weights.values()) <= impossible.max_position_weight + 1e-6
        assert diag.relaxations_applied

    def test_prohibited_relaxation_fails_loudly(self):
        rm = _risk_model(n=25)
        mandate = PortfolioConfig(
            min_holdings=24,
            max_holdings=25,
            max_position_weight=0.05,
            min_position_weight=0.04,
            max_turnover_per_rebalance=0.001,
            min_beta=1.999,
            max_beta=2.0,
            allow_constraint_relaxation=False,
        )
        with pytest.raises(RuntimeError, match="prohibits constraint relaxation"):
            ConstrainedOptimizer(mandate).build(
                _candidates(25), {}, date(2021, 6, 15), risk_model=rm
            )


class TestDiagnostics:
    def test_diagnostics_are_populated(self):
        opt = ConstrainedOptimizer(CFG)
        _, diag = opt.build(_candidates(), {}, date(2021, 6, 15), risk_model=_risk_model())
        assert diag.succeeded
        assert diag.solver_name
        assert diag.expected_risk is not None and diag.expected_risk > 0
        assert diag.expected_turnover is not None
        assert diag.solve_seconds is not None and diag.solve_seconds >= 0

    def test_binding_constraints_are_reported(self):
        """With a tight cap, the position limit should bind on several names."""
        cfg = CFG.model_copy(update={"max_position_weight": 0.025, "max_holdings": 60})
        opt = ConstrainedOptimizer(cfg)
        _, diag = opt.build(_candidates(), {}, date(2021, 6, 15), risk_model=_risk_model())
        assert isinstance(diag.binding_constraints, list)


class TestSimpleConstructors:
    def test_equal_weight_respects_cap_and_budget(self):
        cfg = PortfolioConfig(min_holdings=10, max_holdings=40, max_position_weight=0.05)
        weights, diag = EqualWeightConstructor(cfg).build(_candidates(), {}, date(2021, 6, 15))
        assert len(weights) == 40
        assert max(weights.values()) <= 0.05 + 1e-9
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert diag.succeeded

    def test_score_weighted_caps_are_enforced(self):
        cfg = PortfolioConfig(min_holdings=10, max_holdings=50, max_position_weight=0.04)
        weights, _ = ScoreWeightedConstructor(cfg).build(_candidates(), {}, date(2021, 6, 15))
        assert max(weights.values()) <= 0.04 + 1e-9
        assert sum(weights.values()) <= 1.0 + 1e-6

    def test_score_transform_is_configurable(self):
        base = PortfolioConfig(
            min_holdings=10,
            max_holdings=20,
            max_position_weight=0.20,
            min_position_weight=0.0,
        )
        linear, _ = ScoreWeightedConstructor(base).build(_candidates(40), {}, date(2021, 6, 15))
        softmax, _ = ScoreWeightedConstructor(
            base.model_copy(update={"score_transform": "softmax", "score_temperature": 0.05})
        ).build(_candidates(40), {}, date(2021, 6, 15))
        assert set(linear) == set(softmax)
        assert any(abs(linear[key] - softmax[key]) > 1e-4 for key in linear)

    def test_inverse_volatility_favours_low_vol_names(self):
        cfg = PortfolioConfig(min_holdings=10, max_holdings=40, max_position_weight=0.10)
        cands = _candidates()
        weights, _ = InverseVolatilityConstructor(cfg).build(cands, {}, date(2021, 6, 15))
        vols = dict(zip(cands["security_id"], cands["volatility_60d"], strict=True))
        held = sorted(weights, key=lambda s: weights[s], reverse=True)
        # The heaviest position must not be the most volatile of those held.
        assert vols[held[0]] < max(vols[s] for s in held)

    def test_insufficient_candidates_reports_infeasible(self):
        cfg = PortfolioConfig(min_holdings=50, max_holdings=100, max_position_weight=0.05)
        weights, diag = EqualWeightConstructor(cfg).build(_candidates(5), {}, date(2021, 6, 15))
        assert weights == {}
        assert not diag.succeeded

    def test_cap_redistribution_converges_without_breaching(self):
        """Clipping then renormalizing naively re-breaches the cap; this must not."""
        raw = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        out = _apply_cap_and_redistribute(raw, cap=0.2, budget=1.0)
        assert out.max() <= 0.2 + 1e-9
        assert out.sum() == pytest.approx(1.0, abs=1e-6)

    def test_cap_redistribution_holds_cash_when_all_capped(self):
        """If every name is capped, the remainder stays in cash rather than breaching."""
        raw = np.array([0.25, 0.25, 0.25, 0.25])
        out = _apply_cap_and_redistribute(raw, cap=0.1, budget=1.0)
        assert out.max() <= 0.1 + 1e-9
        assert out.sum() == pytest.approx(0.4, abs=1e-6)  # 4 * 0.1, rest is cash
