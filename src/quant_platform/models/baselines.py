"""Benchmark and factor-composite models (spec sections 14.1-14.2).

These exist to make the sophisticated models prove their worth. A LightGBM model with an
information ratio of 0.3 sounds impressive until a five-line momentum rule scores 0.35 on the
same data. Every baseline here runs through the identical pipeline -- same universe, same
costs, same portfolio construction -- so comparisons are apples to apples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_platform.models.base import BaseModel, ModelError


class NoSkillModel(BaseModel):
    """Predicts the training mean for every security.

    The floor. Any model that cannot beat this has no cross-sectional information at all.
    Because it predicts a constant, its cross-sectional ranking is arbitrary (ties), which is
    exactly the point: it holds an effectively random portfolio.
    """

    name = "no_skill"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._mean: float = 0.0

    def _fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        self._mean = float(np.mean(y))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._mean)


class MomentumBaseline(BaseModel):
    """Ranks on a single momentum feature.

    A transparent, well-documented anomaly with no fitting whatsoever. If the ML models
    cannot beat this, the added complexity is not earning anything.
    """

    name = "momentum_baseline"

    def __init__(self, momentum_column: str = "momentum_252d", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.momentum_column = momentum_column
        self._idx: int | None = None

    def _fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        if self.momentum_column not in feature_names:
            raise ModelError(
                f"{self.name}: '{self.momentum_column}' is not among the supplied features "
                f"({feature_names[:5]}...)"
            )
        self._idx = feature_names.index(self.momentum_column)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        assert self._idx is not None
        return X[:, self._idx]

    def get_hyperparameters(self) -> dict[str, Any]:
        return {"momentum_column": self.momentum_column}


# Which feature families feed each composite pillar, and with what sign.
# Sign is +1 when a HIGH raw value is attractive, -1 when a LOW value is.
_DEFAULT_PILLARS: dict[str, list[tuple[str, float]]] = {
    "momentum": [
        ("momentum_252d", 1.0),
        ("momentum_126d", 1.0),
        ("momentum_12m_ex1m", 1.0),
        ("dist_52w_high", 1.0),  # closer to the high (less negative) is stronger
        ("momentum_consistency_63d", 1.0),
    ],
    "value": [
        ("earnings_yield", 1.0),
        ("book_to_market", 1.0),
        ("fcf_yield", 1.0),
        ("sales_to_price", 1.0),
    ],
    "quality": [
        ("gross_profitability", 1.0),
        ("return_on_equity", 1.0),
        ("operating_margin", 1.0),
        ("debt_to_assets", -1.0),  # more leverage is lower quality
    ],
    "defensive": [
        ("volatility_252d", -1.0),  # low-volatility anomaly
        ("beta_252d", -1.0),
        ("idio_vol_252d", -1.0),
        ("max_drawdown_252d", 1.0),  # a shallower (less negative) drawdown is better
    ],
    "liquidity": [
        ("adv_21d", 1.0),
        ("amihud_illiquidity_21d", -1.0),
        ("zero_return_freq_63d", -1.0),
    ],
}


class FactorComposite(BaseModel):
    """Transparent weighted composite of normalized factor pillars.

    Deliberately not fitted to the target. Its weights are economic priors, which makes it a
    genuinely out-of-sample benchmark at every point in time and immune to the backtest
    overfitting that afflicts trained models. That is what makes it the honest bar to clear.

    Inputs are expected to be cross-sectionally normalized already (the preprocessor's rank
    transform), so pillars are commensurable before weighting.
    """

    name = "factor_composite"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        pillars: dict[str, list[tuple[str, float]]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weights = weights or {
            "momentum": 0.30,
            "value": 0.20,
            "quality": 0.25,
            "defensive": 0.15,
            "liquidity": 0.10,
        }
        self.pillars = pillars or _DEFAULT_PILLARS
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("factor composite weights must sum to a positive number")
        # Normalize so the composite is a weighted average regardless of what was passed in.
        self.weights = {k: v / total for k, v in self.weights.items()}
        self._plan: list[tuple[int, float, float]] = []  # (col_idx, sign, weight)
        self._active_pillars: list[str] = []

    def _fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        """Resolve which pillar components are actually present.

        Pillars whose inputs are entirely absent (e.g. value, when fundamentals were not
        built) are dropped and the remaining weights renormalized -- rather than silently
        contributing zeros, which would bias the composite toward the missing pillar's
        neutral value.
        """
        available: dict[str, list[tuple[int, float]]] = {}
        for pillar, components in self.pillars.items():
            if self.weights.get(pillar, 0.0) <= 0:
                continue
            found = [
                (feature_names.index(col), sign) for col, sign in components if col in feature_names
            ]
            if found:
                available[pillar] = found

        if not available:
            raise ModelError(
                f"{self.name}: none of the composite's factor inputs are present in "
                f"{feature_names[:8]}..."
            )

        active_weight = sum(self.weights[p] for p in available)
        self._plan = []
        self._active_pillars = sorted(available)
        for pillar, comps in available.items():
            w = self.weights[pillar] / active_weight  # renormalize across present pillars
            per_component = w / len(comps)
            for idx, sign in comps:
                self._plan.append((idx, sign, per_component))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        score = np.zeros(X.shape[0])
        for idx, sign, weight in self._plan:
            col = X[:, idx]
            score += sign * weight * col
        return score

    def get_hyperparameters(self) -> dict[str, Any]:
        return {"weights": self.weights, "active_pillars": self._active_pillars}

    def feature_importance(self) -> pl.DataFrame | None:
        """Effective weight of each input -- exact, since the composite is linear."""
        if self.metadata is None or not self._plan:
            return None
        names = self.metadata.feature_names
        return pl.DataFrame(
            {
                "feature": [names[i] for i, _, _ in self._plan],
                "importance": [abs(w) for _, _, w in self._plan],
                "sign": [s for _, s, _ in self._plan],
            }
        ).sort("importance", descending=True)
