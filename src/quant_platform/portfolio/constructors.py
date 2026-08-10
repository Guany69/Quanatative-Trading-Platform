"""Portfolio construction (spec section 20).

Four constructors of increasing sophistication, all behind one interface so the report can
compare them on equal terms. The ordering is deliberate: the spec warns against assuming the
most complex optimizer wins, and equal-weight is a famously hard baseline to beat once costs
and estimation error are accounted for.

Every constructor returns weights that satisfy the hard constraints (long-only, capped,
budget) or raises. None of them silently renormalizes away a violated constraint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import numpy as np
import polars as pl

from quant_platform.config.models import PortfolioConfig
from quant_platform.domain.enums import SolverStatus
from quant_platform.domain.portfolio import OptimizationDiagnostics
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("portfolio")


class PortfolioConstructor(ABC):
    """Turns cross-sectional scores into target weights."""

    name: str = "base"

    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config

    @abstractmethod
    def _weights(
        self,
        candidates: pl.DataFrame,
        previous: dict[str, float],
        as_of: date,
    ) -> tuple[dict[str, float], OptimizationDiagnostics]: ...

    def build(
        self,
        candidates: pl.DataFrame,
        previous: dict[str, float] | None = None,
        as_of: date | None = None,
    ) -> tuple[dict[str, float], OptimizationDiagnostics]:
        """Build target weights.

        ``candidates`` needs security_id and cross_sectional_rank; optional columns
        (volatility, sector, adv) are used by the constructors that can exploit them.
        """
        as_of = as_of or date.today()
        previous = previous or {}

        if candidates.is_empty():
            return {}, OptimizationDiagnostics(
                as_of=as_of,
                solver_status=SolverStatus.INFEASIBLE,
                message="no eligible candidates",
            )

        eligible = candidates
        if self.config.restricted_securities:
            eligible = eligible.filter(
                ~pl.col("security_id").is_in(self.config.restricted_securities)
            )

        weights, diag = self._weights(eligible, previous, as_of)
        self._verify(weights, diag)
        return weights, diag

    def _verify(self, weights: dict[str, float], diag: OptimizationDiagnostics) -> None:
        """Assert the hard constraints actually hold on the returned weights.

        A last line of defence: if a constructor has a bug, this turns a silently invalid
        portfolio into a loud failure.
        """
        if not weights:
            return
        violations: dict[str, float] = {}
        if self.config.long_only:
            worst = min(weights.values())
            if worst < -1e-9:
                violations["long_only"] = worst
        heaviest = max(weights.values())
        if heaviest > self.config.max_position_weight + 1e-6:
            violations["max_position_weight"] = heaviest
        lightest = min(weights.values())
        if lightest < self.config.min_position_weight - 1e-6:
            violations["min_position_weight"] = lightest
        if len(weights) < self.config.min_holdings:
            violations["min_holdings"] = float(len(weights))
        if len(weights) > self.config.max_holdings:
            violations["max_holdings"] = float(len(weights))
        total = sum(weights.values())
        budget = 1.0 - self.config.cash_buffer
        if total > budget + 1e-6:
            violations["budget"] = total
        if self.config.fully_invested and abs(total - budget) > 1e-5:
            violations["fully_invested"] = total
        if violations:
            raise AssertionError(
                f"{self.name} produced weights violating hard constraints: {violations}. "
                f"This is a bug -- constraints must never be silently dropped."
            )


class EqualWeightConstructor(PortfolioConstructor):
    """Equal weight across the top-ranked names.

    Notoriously hard to beat: it has no estimation error because it estimates nothing.
    """

    name = "equal_weight"

    def _weights(self, candidates, previous, as_of):
        n = self._target_count(candidates.height)
        if n == 0:
            return {}, OptimizationDiagnostics(
                as_of=as_of,
                solver_status=SolverStatus.INFEASIBLE,
                message=f"only {candidates.height} candidates; need >= {self.config.min_holdings}",
            )
        top = candidates.sort("cross_sectional_rank", descending=True).head(n)
        budget = 1.0 - self.config.cash_buffer
        w = budget / n
        # If equal weight would breach the position cap, the cap wins and we hold cash.
        w = min(w, self.config.max_position_weight)
        weights = dict.fromkeys(top["security_id"].to_list(), w)
        return weights, OptimizationDiagnostics(
            as_of=as_of,
            solver_status=SolverStatus.OPTIMAL,
            solver_name="analytic",
            n_holdings=n,
            expected_turnover=_turnover(weights, previous),
        )

    def _target_count(self, available: int) -> int:
        if available < self.config.min_holdings:
            return 0
        return min(available, self.config.max_holdings)


class ScoreWeightedConstructor(PortfolioConstructor):
    """Weights proportional to rank above the selection threshold.

    Tilts toward conviction while staying transparent. Rank (not raw score) drives weight, so
    an outlier prediction cannot dominate the book.
    """

    name = "score_weighted"

    def _weights(self, candidates, previous, as_of):
        if candidates.height < self.config.min_holdings:
            return {}, OptimizationDiagnostics(
                as_of=as_of,
                solver_status=SolverStatus.INFEASIBLE,
                message=f"only {candidates.height} candidates",
            )
        n = min(candidates.height, self.config.max_holdings)
        top = candidates.sort("cross_sectional_rank", descending=True).head(n)

        ranks = top["cross_sectional_rank"].to_numpy().astype(float)
        # Shift so the weakest selected name gets ~0 weight and weights stay non-negative.
        scores = ranks - ranks.min() + 1e-6
        if self.config.score_transform == "power":
            scores = scores**self.config.score_power
        elif self.config.score_transform == "softmax":
            scores = np.exp((ranks - ranks.max()) / self.config.score_temperature)
        capped = _allocate_with_bounds(
            scores,
            self.config.min_position_weight,
            self.config.max_position_weight,
            1.0 - self.config.cash_buffer,
        )
        weights = {
            sid: float(w)
            for sid, w in zip(top["security_id"].to_list(), capped, strict=True)
            if w > 1e-9
        }
        return weights, OptimizationDiagnostics(
            as_of=as_of,
            solver_status=SolverStatus.OPTIMAL,
            solver_name="analytic",
            n_holdings=len(weights),
            expected_turnover=_turnover(weights, previous),
        )


class InverseVolatilityConstructor(PortfolioConstructor):
    """Weights inversely proportional to volatility (risk parity, diagonal).

    A risk-based baseline that ignores the signal's magnitude entirely.
    """

    name = "inverse_volatility"

    def __init__(self, config: PortfolioConfig, vol_column: str = "volatility_60d") -> None:
        super().__init__(config)
        self.vol_column = vol_column

    def _weights(self, candidates, previous, as_of):
        if self.vol_column not in candidates.columns:
            raise ValueError(
                f"{self.name} needs column '{self.vol_column}' but candidates have "
                f"{candidates.columns}"
            )
        if candidates.height < self.config.min_holdings:
            return {}, OptimizationDiagnostics(
                as_of=as_of,
                solver_status=SolverStatus.INFEASIBLE,
                message=f"only {candidates.height} candidates",
            )
        n = min(candidates.height, self.config.max_holdings)
        top = candidates.sort("cross_sectional_rank", descending=True).head(n)

        vol = top[self.vol_column].to_numpy().astype(float)
        # Guard against zero/NaN vol producing infinite weight.
        vol = np.where(
            np.isfinite(vol) & (vol > 1e-6),
            vol,
            np.nanmedian(vol[np.isfinite(vol)]) if np.isfinite(vol).any() else 0.25,
        )
        inv = 1.0 / vol
        capped = _allocate_with_bounds(
            inv,
            self.config.min_position_weight,
            self.config.max_position_weight,
            1.0 - self.config.cash_buffer,
        )
        weights = {
            sid: float(w)
            for sid, w in zip(top["security_id"].to_list(), capped, strict=True)
            if w > 1e-9
        }
        return weights, OptimizationDiagnostics(
            as_of=as_of,
            solver_status=SolverStatus.OPTIMAL,
            solver_name="analytic",
            n_holdings=len(weights),
            expected_turnover=_turnover(weights, previous),
        )


def _apply_cap_and_redistribute(
    raw: np.ndarray, cap: float, budget: float, max_iter: int = 100
) -> np.ndarray:
    """Enforce a position cap, pushing the excess onto uncapped names.

    Naively clipping then renormalizing re-breaches the cap; this iterates to a fixed point.
    If every name is capped, the remainder stays in cash rather than forcing a breach.
    """
    w = np.maximum(raw.copy(), 0.0)
    total = w.sum()
    if total <= 1e-12:
        return w

    # Normalize to the budget FIRST, in either direction. Scaling only downward would leave
    # the book under-invested whenever upstream steps (dust removal, holding-count truncation)
    # shrink the raw weights -- a silent cash drag rather than the intended full investment.
    w = w * (budget / total)

    # Redistribute above-cap weight onto uncapped names. Iterated because relieving one name
    # can push another over the cap; this preserves the total, so the sum stays at budget.
    for _ in range(max_iter):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over
        if not free.any() or w[free].sum() <= 1e-12:
            break  # every name is capped: the remainder legitimately stays in cash
        w[free] += excess * (w[free] / w[free].sum())

    return np.minimum(w, cap)


def _allocate_with_bounds(
    strength: np.ndarray, floor: float, cap: float, budget: float
) -> np.ndarray:
    """Allocate a budget proportionally while honoring configured position bounds."""
    values = np.maximum(np.asarray(strength, dtype=float), 0.0)
    n = len(values)
    if n == 0 or floor * n > budget + 1e-9:
        return np.zeros(n)
    remainder = max(0.0, budget - floor * n)
    if values.sum() <= 1e-12:
        values = np.ones(n)
    raw = np.full(n, floor) + remainder * values / values.sum()
    return _apply_cap_and_redistribute(raw, cap, budget)


def _turnover(new: dict[str, float], old: dict[str, float]) -> float:
    """One-way turnover between two weight vectors."""
    keys = set(new) | set(old)
    return 0.5 * sum(abs(new.get(k, 0.0) - old.get(k, 0.0)) for k in keys)


def get_constructor(name: str, config: PortfolioConfig) -> PortfolioConstructor:
    """Look up a constructor by name."""
    table: dict[str, type[PortfolioConstructor]] = {
        "equal_weight": EqualWeightConstructor,
        "score_weighted": ScoreWeightedConstructor,
        "inverse_volatility": InverseVolatilityConstructor,
    }
    if name == "constrained_optimizer":
        from quant_platform.portfolio.constrained_optimizer import ConstrainedOptimizer

        return ConstrainedOptimizer(config)
    if name not in table:
        known = [*sorted(table), "constrained_optimizer"]
        raise KeyError(f"unknown portfolio constructor '{name}'; known: {known}")
    return table[name](config)
