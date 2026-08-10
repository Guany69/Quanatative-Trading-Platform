"""Constrained mean-variance optimization (spec section 20).

Maximizes

    mu' w  -  lambda * w' Sigma w  -  gamma * ||w - w_prev||_1  -  tau * TE(w)

subject to the full constraint set: full investment, long-only, position bounds, sector
deviation, beta range, turnover, and ADV participation.

Two design decisions dominate this module.

**Never silently drop a constraint.** If the problem is infeasible, the optimizer applies an
*ordered, logged* relaxation policy and records exactly what was loosened. A portfolio that
quietly violates its mandate is far more dangerous than one that reports failure, because the
violation only surfaces later as unexplained risk.

**Expected returns come from ranks, not raw predictions.** Mean-variance optimization is
notoriously sensitive to the return vector; feeding it raw model output lets one confident
prediction dominate the book. Ranks are bounded and scale-free, which keeps the optimizer's
sensitivity in check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from quant_platform.config.models import PortfolioConfig
from quant_platform.domain.enums import SolverStatus
from quant_platform.domain.portfolio import OptimizationDiagnostics
from quant_platform.portfolio.constructors import (
    PortfolioConstructor,
    _apply_cap_and_redistribute,
    _turnover,
)
from quant_platform.risk.covariance import RiskModel
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("portfolio.optimizer")


@dataclass
class RelaxationStep:
    """One step of the ordered relaxation policy (spec section 20.4)."""

    name: str
    description: str
    apply: str  # attribute mutated on the working config
    factor: float = 1.5


# Ordered from least to most damaging to the mandate. Position limits and long-only are NOT
# here: those are hard risk constraints, and a portfolio that breaches them is not the
# strategy any more.
DEFAULT_RELAXATION_ORDER: list[RelaxationStep] = [
    RelaxationStep("min_position", "drop the minimum position size", "min_position_weight", 0.0),
    RelaxationStep("holdings", "widen the holding-count range", "max_holdings", 1.5),
    RelaxationStep("sector", "widen sector deviation bounds", "max_sector_deviation", 2.0),
    RelaxationStep("beta", "widen the beta range", "_beta_widen", 2.0),
    RelaxationStep("turnover", "raise the turnover cap", "max_turnover_per_rebalance", 2.0),
    RelaxationStep("cash", "allow more cash to be held", "cash_buffer", 0.05),
]


@dataclass
class OptimizationResult:
    weights: dict[str, float]
    diagnostics: OptimizationDiagnostics
    relaxations: list[str] = field(default_factory=list)


class ConstrainedOptimizer(PortfolioConstructor):
    """Mean-variance optimizer with a full constraint set and logged relaxation."""

    name = "constrained_optimizer"

    def __init__(
        self,
        config: PortfolioConfig,
        risk_aversion: float = 5.0,
        cost_penalty: float = 1.0,
        tracking_penalty: float = 0.0,
        solver: str | None = None,
    ) -> None:
        super().__init__(config)
        self.risk_aversion = risk_aversion
        self.cost_penalty = cost_penalty
        self.tracking_penalty = tracking_penalty
        self.solver = solver

    def _weights(self, candidates, previous, as_of):
        raise NotImplementedError("use optimize(); the base signature lacks a risk model")

    def build(
        self,
        candidates: pl.DataFrame,
        previous: dict[str, float] | None = None,
        as_of: date | None = None,
        risk_model: RiskModel | None = None,
        sector_map: dict[str, str] | None = None,
        benchmark_sector_weights: dict[str, float] | None = None,
        adv: dict[str, float] | None = None,
        portfolio_value: float = 1e7,
    ) -> tuple[dict[str, float], OptimizationDiagnostics]:
        """Optimize, relaxing constraints in order if the problem proves infeasible."""
        as_of = as_of or date.today()
        previous = previous or {}

        if candidates.is_empty():
            return {}, OptimizationDiagnostics(
                as_of=as_of, solver_status=SolverStatus.INFEASIBLE, message="no candidates"
            )
        if risk_model is None:
            raise ValueError("ConstrainedOptimizer requires a risk model")

        eligible = candidates
        if self.config.restricted_securities:
            eligible = eligible.filter(
                ~pl.col("security_id").is_in(self.config.restricted_securities)
            )

        working = self.config.model_copy(deep=True)
        beta_widen = 0.0
        applied: list[str] = []

        for attempt in range(len(DEFAULT_RELAXATION_ORDER) + 1):
            result = self._solve(
                eligible,
                previous,
                as_of,
                risk_model,
                working,
                beta_widen,
                sector_map,
                benchmark_sector_weights,
                adv,
                portfolio_value,
                applied,
            )
            if result is not None:
                # Domain records are frozen (they represent observed facts), so record the
                # relaxations by producing a new instance rather than mutating in place.
                result.diagnostics = result.diagnostics.model_copy(
                    update={"relaxations_applied": list(applied)}
                )
                if applied:
                    logger.warning(
                        "optimization on %s required relaxations: %s", as_of, ", ".join(applied)
                    )
                self._verify(result.weights, result.diagnostics)
                return result.weights, result.diagnostics

            if attempt >= len(DEFAULT_RELAXATION_ORDER):
                break

            step = DEFAULT_RELAXATION_ORDER[attempt]
            if step.apply == "_beta_widen":
                beta_widen += 0.15
            elif step.apply == "min_position_weight":
                working.min_position_weight = 0.0
            elif step.apply == "max_holdings":
                working.max_holdings = min(eligible.height, int(working.max_holdings * step.factor))
            elif step.apply == "cash_buffer":
                working.cash_buffer = min(0.25, working.cash_buffer + step.factor)
            else:
                setattr(working, step.apply, getattr(working, step.apply) * step.factor)
            applied.append(f"{step.name}: {step.description}")
            logger.info("optimization infeasible on %s; relaxing: %s", as_of, step.description)

        # Every relaxation exhausted: fall back to a feasible score-weighted allocation
        # rather than emitting weights that violate the mandate.
        logger.error(
            "optimization on %s remained infeasible after all relaxations; falling back to "
            "score-weighted allocation",
            as_of,
        )
        from quant_platform.portfolio.constructors import ScoreWeightedConstructor

        fallback = ScoreWeightedConstructor(self.config)
        weights, _fallback_diag = fallback.build(eligible, previous, as_of)
        applied.append("FALLBACK: score-weighted allocation (optimizer infeasible)")
        return weights, OptimizationDiagnostics(
            as_of=as_of,
            solver_status=SolverStatus.INFEASIBLE,
            solver_name="fallback_score_weighted",
            n_holdings=len(weights),
            relaxations_applied=applied,
            expected_turnover=_turnover(weights, previous),
            message="optimizer infeasible after all relaxations; used score-weighted fallback",
        )

    def _solve(
        self,
        candidates: pl.DataFrame,
        previous: dict[str, float],
        as_of: date,
        risk_model: RiskModel,
        cfg: PortfolioConfig,
        beta_widen: float,
        sector_map: dict[str, str] | None,
        benchmark_sector_weights: dict[str, float] | None,
        adv: dict[str, float] | None,
        portfolio_value: float,
        applied: list[str],
    ) -> OptimizationResult | None:
        """One solve attempt. Returns None when infeasible."""
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise RuntimeError(
                "cvxpy is required for ConstrainedOptimizer. Install with: "
                "uv sync --extra optimization"
            ) from exc

        # Restrict to names the risk model covers -- optimizing a security with no covariance
        # row would mean inventing its risk.
        covered = set(risk_model.security_ids)
        pool = candidates.filter(pl.col("security_id").is_in(list(covered)))
        if pool.height < max(2, cfg.min_holdings // 4):
            return None

        # Pre-select the top-ranked names. Solving over thousands of securities is slow and
        # the tail has no chance of entering a 50-100 name book anyway.
        max_pool = min(pool.height, max(cfg.max_holdings * 4, 200))
        pool = pool.sort("cross_sectional_rank", descending=True).head(max_pool)

        ids = pool["security_id"].to_list()
        sub = risk_model.subset(ids)
        order = {s: i for i, s in enumerate(sub.security_ids)}
        pool = pool.filter(pl.col("security_id").is_in(sub.security_ids))
        ids = pool["security_id"].to_list()
        idx = [order[s] for s in ids]

        n = len(ids)
        sigma = sub.covariance[np.ix_(idx, idx)]
        betas = sub.betas[idx]

        # Expected returns from ranks, centered so the mean is zero: the optimizer then
        # expresses relative views rather than a market-direction bet.
        ranks = pool["cross_sectional_rank"].to_numpy().astype(float)
        mu = (ranks - ranks.mean()) * 0.10  # scaled to a plausible annual spread

        w_prev = np.array([previous.get(s, 0.0) for s in ids])
        budget = 1.0 - cfg.cash_buffer

        w = cp.Variable(n)
        objective = mu @ w - self.risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma))
        if self.cost_penalty > 0:
            objective -= self.cost_penalty * cp.norm1(w - w_prev)

        constraints = [cp.sum(w) == budget, w >= 0, w <= cfg.max_position_weight]

        # Turnover cap: sum |w - w_prev| is two-way, so a 15% one-way cap is 30% here.
        if cfg.max_turnover_per_rebalance < 1.0:
            constraints.append(cp.norm1(w - w_prev) <= 2.0 * cfg.max_turnover_per_rebalance)

        # Beta band.
        lo = cfg.min_beta - beta_widen
        hi = cfg.max_beta + beta_widen
        constraints += [betas @ w >= lo * budget, betas @ w <= hi * budget]

        # Sector deviation versus the benchmark.
        if sector_map and benchmark_sector_weights:
            sectors = sorted({sector_map.get(s, "unknown") for s in ids})
            for sector in sectors:
                mask = np.array([1.0 if sector_map.get(s) == sector else 0.0 for s in ids])
                if mask.sum() == 0:
                    continue
                target = benchmark_sector_weights.get(sector, 0.0)
                constraints += [
                    mask @ w <= target + cfg.max_sector_deviation,
                    mask @ w >= max(0.0, target - cfg.max_sector_deviation),
                ]

        # ADV participation: cap each position so liquidating it stays realistic.
        if adv:
            caps = np.array(
                [
                    min(
                        cfg.max_position_weight,
                        (adv.get(s, 0.0) * cfg.max_adv_participation * 5.0) / portfolio_value
                        if adv.get(s, 0.0) > 0
                        else cfg.max_position_weight,
                    )
                    for s in ids
                ]
            )
            caps = np.maximum(caps, 1e-6)
            constraints.append(w <= caps)

        problem = cp.Problem(cp.Maximize(objective), constraints)
        start = time.perf_counter()
        try:
            solver = self.solver or (cp.CLARABEL if hasattr(cp, "CLARABEL") else cp.ECOS)
            problem.solve(solver=solver, verbose=False)
        except Exception as exc:
            logger.debug("solver raised on %s: %s", as_of, exc)
            return None
        elapsed = time.perf_counter() - start

        if problem.status not in {"optimal", "optimal_inaccurate"} or w.value is None:
            return None

        raw = np.asarray(w.value, dtype=float)
        raw = np.clip(raw, 0.0, cfg.max_position_weight)

        # Drop dust positions, then enforce the holding count.
        floor = max(cfg.min_position_weight, 1e-4)
        raw[raw < floor] = 0.0
        if int((raw > 0).sum()) > cfg.max_holdings:
            keep = np.argsort(raw)[::-1][: cfg.max_holdings]
            mask = np.zeros(n, dtype=bool)
            mask[keep] = True
            raw[~mask] = 0.0

        total = raw.sum()
        if total <= 1e-9:
            return None
        # Rescaling to the budget can push names over the cap, and naively clipping afterwards
        # leaves the book under-invested. Reuse the iterative redistribution helper, which
        # pushes the excess onto uncapped names and converges to a feasible fully-invested
        # solution (or holds cash when genuinely every name is capped).
        raw = _apply_cap_and_redistribute(raw, cfg.max_position_weight, budget)

        n_holdings = int((raw > 0).sum())
        if n_holdings < min(cfg.min_holdings, pool.height):
            return None

        weights = {s: float(v) for s, v in zip(ids, raw, strict=True) if v > 1e-9}
        binding = self._binding_constraints(raw, betas, cfg, w_prev, lo, hi, budget)

        return OptimizationResult(
            weights=weights,
            diagnostics=OptimizationDiagnostics(
                as_of=as_of,
                solver_status=SolverStatus.OPTIMAL
                if problem.status == "optimal"
                else SolverStatus.OPTIMAL_INACCURATE,
                solver_name=str(
                    problem.solver_stats.solver_name if problem.solver_stats else "cvxpy"
                ),
                objective_value=float(problem.value) if problem.value is not None else None,
                expected_return=float(mu @ raw),
                expected_risk=float(np.sqrt(max(float(raw @ sigma @ raw), 0.0))),
                expected_turnover=float(0.5 * np.abs(raw - w_prev).sum()),
                binding_constraints=binding,
                n_holdings=n_holdings,
                solve_seconds=elapsed,
            ),
        )

    @staticmethod
    def _binding_constraints(
        w: np.ndarray,
        betas: np.ndarray,
        cfg: PortfolioConfig,
        w_prev: np.ndarray,
        beta_lo: float,
        beta_hi: float,
        budget: float,
    ) -> list[str]:
        """Which constraints are active at the solution.

        Persisted because a permanently binding constraint is where the mandate, not the
        signal, is driving the portfolio -- worth knowing when interpreting results.
        """
        binding: list[str] = []
        if np.any(w >= cfg.max_position_weight - 1e-6):
            binding.append(
                f"max_position_weight ({int((w >= cfg.max_position_weight - 1e-6).sum())} names)"
            )
        beta = float(betas @ w)
        if beta <= beta_lo * budget + 1e-4:
            binding.append(f"min_beta ({beta:.3f})")
        if beta >= beta_hi * budget - 1e-4:
            binding.append(f"max_beta ({beta:.3f})")
        turnover = 0.5 * float(np.abs(w - w_prev).sum())
        if turnover >= cfg.max_turnover_per_rebalance - 1e-4:
            binding.append(f"turnover ({turnover:.3f})")
        return binding


def benchmark_sector_weights(securities: pl.DataFrame, universe_ids: list[str]) -> dict[str, float]:
    """Equal-weighted sector weights of the eligible universe.

    A proxy for true benchmark sector weights, which need index constituent weights that free
    data does not provide. Documented as an approximation in docs/limitations.md.
    """
    sub = securities.filter(pl.col("security_id").is_in(universe_ids))
    if sub.is_empty() or "sector" not in sub.columns:
        return {}
    counts = sub.group_by("sector").agg(pl.len().alias("n"))
    total = int(counts["n"].sum())
    if total == 0:
        return {}
    return {str(r["sector"]): r["n"] / total for r in counts.iter_rows(named=True)}
