"""Constrained rank ensemble (spec section 14.7).

Combines models on their **cross-sectional ranks**, not their raw predictions. This matters:
LightGBM might output values around 0.5 while an MLP outputs values around 50. Averaging
those directly lets the larger-scaled model dominate for a reason that has nothing to do with
skill. Ranks are scale-free, so each model contributes according to its ordering only.

Weighting is deliberately constrained:

* **caps** stop one model from taking over after a lucky stretch;
* **floors** keep diversity, which is the only reason to ensemble at all;
* **slow updates** damp weight changes, because reacting fast to recent performance is just
  performance-chasing at the model level;
* weights are fitted on **validation** results only -- never on the locked holdout, which
  would make the final evaluation self-selected.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_platform.models.base import BaseModel, ModelError
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("models.ensemble")


class RankEnsemble(BaseModel):
    """Weighted combination of member models' cross-sectional ranks."""

    name = "ensemble"

    def __init__(
        self,
        members: dict[str, BaseModel] | None = None,
        weights: dict[str, float] | None = None,
        max_weight: float = 0.5,
        min_weight: float = 0.1,
        smoothing: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")
        self.members: dict[str, BaseModel] = members or {}
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.smoothing = smoothing
        self.weights: dict[str, float] = {}
        if weights:
            self.weights = self._constrain(weights)
        self._weight_history: list[dict[str, Any]] = []

    def add_member(self, name: str, model: BaseModel) -> None:
        self.members[name] = model

    def _constrain(self, raw: dict[str, float]) -> dict[str, float]:
        """Normalize, then apply floors and caps, then renormalize.

        Iterated because capping one model redistributes weight that may breach another cap.
        """
        if not raw:
            return {}
        names = sorted(raw)
        n = len(names)

        # A cap below 1/n is infeasible: no allocation could sum to 1.
        if self.max_weight * n < 1.0 - 1e-9:
            raise ValueError(
                f"max_weight={self.max_weight} with {n} members cannot sum to 1.0 "
                f"(max achievable {self.max_weight * n:.2f})"
            )
        floor = min(self.min_weight, 1.0 / n)

        values = np.array([max(raw[k], 0.0) for k in names], dtype=float)
        if values.sum() <= 0:
            values = np.ones(n)  # degenerate input -> equal weight
        values = values / values.sum()

        for _ in range(100):
            values = np.clip(values, floor, self.max_weight)
            total = values.sum()
            if abs(total - 1.0) < 1e-9:
                break
            values = values / total
        values = np.clip(values, floor, self.max_weight)
        values = values / values.sum()
        return dict(zip(names, values.tolist(), strict=True))

    def fit_weights_from_validation(
        self, validation_scores: dict[str, float], is_holdout: bool = False
    ) -> dict[str, float]:
        """Update weights from out-of-sample validation performance (e.g. IC).

        Refuses holdout-derived scores outright. Choosing ensemble weights using the holdout
        is one of the specific prohibitions in the spec: it turns the final unbiased estimate
        into a selected one.
        """
        if is_holdout:
            raise ModelError(
                "Refusing to fit ensemble weights on locked-holdout results. Doing so makes "
                "the holdout part of model selection, so it can no longer serve as an "
                "unbiased final estimate (spec section 16.3)."
            )
        if not validation_scores:
            return self.weights

        # Negative-IC models get zero raw weight; the floor then reinstates a minimum so a
        # single bad window cannot permanently exile a member.
        positive = {k: max(v, 0.0) for k, v in validation_scores.items()}
        if sum(positive.values()) <= 0:
            logger.warning("no member has positive validation score; falling back to equal weights")
            positive = dict.fromkeys(validation_scores, 1.0)

        target = self._constrain(positive)

        if self.weights:
            # Slow update: blend toward the target instead of jumping to it.
            blended = {
                k: self.smoothing * self.weights.get(k, 0.0)
                + (1 - self.smoothing) * target.get(k, 0.0)
                for k in set(self.weights) | set(target)
            }
            self.weights = self._constrain(blended)
        else:
            self.weights = target

        self._weight_history.append(
            {"weights": dict(self.weights), "scores": dict(validation_scores)}
        )
        logger.info(
            "ensemble weights: %s", {k: round(v, 3) for k, v in sorted(self.weights.items())}
        )
        return self.weights

    def _fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        """Members are fitted independently; equal weights are the default if none set."""
        if not self.members:
            raise ModelError("ensemble has no members")
        if not self.weights:
            self.weights = self._constrain(dict.fromkeys(self.members, 1.0))

    def fit(
        self, train: pl.DataFrame, feature_columns: list[str], target_column: str
    ) -> RankEnsemble:
        """Fit every member on the same training frame."""
        if not self.members:
            raise ModelError("ensemble has no members")
        for name, model in self.members.items():
            if not model.is_fitted:
                logger.debug("fitting ensemble member %s", name)
                model.fit(train, feature_columns, target_column)
        super().fit(train, feature_columns, target_column)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - predict() overridden
        raise ModelError("RankEnsemble combines ranks; use predict() with a DataFrame")

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Weighted average of member ranks, renormalized to a percentile."""
        if not self.members:
            raise ModelError("ensemble has no members")
        if not self.weights:
            self.weights = self._constrain(dict.fromkeys(self.members, 1.0))

        combined: pl.DataFrame | None = None
        used: dict[str, float] = {}
        member_ranks: dict[str, np.ndarray] = {}

        for name, model in self.members.items():
            if not model.is_fitted:
                # Graceful degradation: drop the member and reweight the rest, rather than
                # failing the whole rebalance.
                logger.warning("ensemble member '%s' is not fitted; skipping", name)
                continue
            try:
                preds = model.predict(df)
            except Exception as exc:
                logger.warning("ensemble member '%s' failed to predict: %s", name, exc)
                continue

            used[name] = self.weights.get(name, 0.0)
            member_ranks[name] = preds["cross_sectional_rank"].to_numpy()
            renamed = preds.select(
                ["security_id", "as_of", pl.col("cross_sectional_rank").alias(f"_rank_{name}")]
            )
            combined = (
                renamed
                if combined is None
                else combined.join(renamed, on=["security_id", "as_of"], how="inner")
            )

        if combined is None or not used:
            raise ModelError("no ensemble member produced predictions")

        total = sum(used.values())
        if total <= 0:
            used = dict.fromkeys(used, 1.0 / len(used))
            total = 1.0

        # Fold explicitly: the builtin sum() starts from int 0, so its static type becomes
        # `Expr | int` and the result is no longer a pure Polars expression.
        score = pl.lit(0.0)
        for name, weight in used.items():
            score = score + (weight / total) * pl.col(f"_rank_{name}")
        out = combined.with_columns(score.alias("prediction")).select(
            ["security_id", "as_of", "prediction"]
        )
        out = out.with_columns(
            (pl.col("prediction").rank("average").over("as_of") / pl.len().over("as_of")).alias(
                "cross_sectional_rank"
            )
        )
        self._last_disagreement = self._disagreement(member_ranks)
        return out

    @staticmethod
    def _disagreement(member_ranks: dict[str, np.ndarray]) -> float:
        """Mean pairwise rank disagreement between members.

        Near 0 means the members are near-duplicates and the ensemble adds nothing. Very high
        disagreement means they are picking opposite names, which is worth investigating
        before trusting the blend.
        """
        names = sorted(member_ranks)
        if len(names) < 2:
            return float("nan")
        from quant_platform.evaluation.metrics import spearman_ic

        correlations = [
            spearman_ic(member_ranks[a], member_ranks[b])
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        ]
        finite = [c for c in correlations if np.isfinite(c)]
        return float(1.0 - np.mean(finite)) if finite else float("nan")

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "members": sorted(self.members),
            "weights": self.weights,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "smoothing": self.smoothing,
        }

    def feature_importance(self) -> pl.DataFrame | None:
        """Weight-blended importances across members that expose them."""
        frames = []
        for name, model in self.members.items():
            fi = model.feature_importance()
            if fi is None or "importance" not in fi.columns:
                continue
            weight = self.weights.get(name, 0.0)
            frames.append(fi.with_columns((pl.col("importance") * weight).alias("importance")))
        if not frames:
            return None
        return (
            pl.concat([f.select(["feature", "importance"]) for f in frames])
            .group_by("feature")
            .agg(pl.col("importance").sum())
            .sort("importance", descending=True)
        )

    def weight_history(self) -> pl.DataFrame:
        if not self._weight_history:
            return pl.DataFrame()
        return pl.DataFrame(
            [
                {"step": i, **{f"w_{k}": v for k, v in entry["weights"].items()}}
                for i, entry in enumerate(self._weight_history)
            ]
        )
