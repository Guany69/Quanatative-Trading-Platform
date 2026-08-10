"""Elastic Net models (spec section 14.3).

A regularized linear model sits between the transparent factor composite and the black-box
gradient booster. Its value is interpretability: the coefficients say exactly what the model
learned, and comparing coefficients across walk-forward folds reveals whether a relationship
is stable or is being refit to noise each period.

Elastic Net specifically (rather than plain OLS) because equity features are heavily
collinear -- five momentum variants measure nearly the same thing. OLS responds by assigning
huge offsetting coefficients that flip sign between folds; the L2 term shares weight among
correlated features while L1 drives the useless ones to exactly zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_platform.models.base import BaseModel, ModelError


class ElasticNetModel(BaseModel):
    """Linear model with combined L1/L2 regularization."""

    name = "elastic_net"

    def __init__(
        self,
        alpha: float = 0.001,
        l1_ratio: float = 0.5,
        max_iter: int = 5000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self._model: Any = None
        self._coefficients: np.ndarray | None = None

    def _fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        from sklearn.linear_model import ElasticNet

        self._model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_seed,
            selection="cyclic",  # deterministic; 'random' would break reproducibility
        )
        self._model.fit(X, y)
        self._coefficients = np.asarray(self._model.coef_, dtype=float)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelError(f"{self.name}: not fitted")
        return np.asarray(self._model.predict(X), dtype=float)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "l1_ratio": self.l1_ratio, "max_iter": self.max_iter}

    def feature_importance(self) -> pl.DataFrame | None:
        """Absolute coefficients -- exact importances for a linear model."""
        if self._coefficients is None or self.metadata is None:
            return None
        return pl.DataFrame(
            {
                "feature": self.metadata.feature_names,
                "importance": np.abs(self._coefficients),
                "coefficient": self._coefficients,
            }
        ).sort("importance", descending=True)

    @property
    def n_selected_features(self) -> int:
        """How many features survived L1 shrinkage."""
        if self._coefficients is None:
            return 0
        return int(np.sum(np.abs(self._coefficients) > 1e-10))


class LogisticModel(BaseModel):
    """Regularized logistic regression for the binary outperformance target.

    Produces calibrated-ish probabilities of beating the benchmark, which is the third target
    type in the spec and the natural input to probability-weighted sizing.
    """

    name = "logistic"

    def __init__(self, C: float = 1.0, max_iter: int = 2000, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        self.C = C
        self.max_iter = max_iter
        self._model: Any = None

    def _fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        from sklearn.linear_model import LogisticRegression

        classes = np.unique(y)
        if classes.size < 2:
            raise ModelError(
                f"{self.name}: target has a single class ({classes}); a classifier needs both "
                f"outcomes present in the training window"
            )
        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_seed,
            solver="lbfgs",
        )
        self._model.fit(X, (y > 0.5).astype(int))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelError(f"{self.name}: not fitted")
        # Probability of the positive class -- the quantity we actually want to rank on.
        return np.asarray(self._model.predict_proba(X)[:, 1], dtype=float)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {"C": self.C, "max_iter": self.max_iter}

    def feature_importance(self) -> pl.DataFrame | None:
        if self._model is None or self.metadata is None:
            return None
        coef = np.asarray(self._model.coef_, dtype=float).ravel()
        return pl.DataFrame(
            {
                "feature": self.metadata.feature_names,
                "importance": np.abs(coef),
                "coefficient": coef,
            }
        ).sort("importance", descending=True)


def coefficient_stability(models: list[ElasticNetModel]) -> pl.DataFrame:
    """Compare coefficients across walk-forward folds.

    A feature whose coefficient flips sign between folds is not a stable relationship -- it is
    the model refitting noise. ``sign_consistency`` near 1.0 means the feature kept the same
    direction throughout; near 0.5 means it is a coin flip.
    """
    fitted = [m for m in models if m.metadata is not None and m._coefficients is not None]
    if not fitted:
        return pl.DataFrame()

    first_metadata = fitted[0].metadata
    assert first_metadata is not None  # guaranteed by the `fitted` filter above
    names = first_metadata.feature_names
    coefficient_rows: list[np.ndarray] = [
        m._coefficients for m in fitted if m._coefficients is not None
    ]
    matrix = np.vstack(coefficient_rows)

    signs = np.sign(matrix)
    with np.errstate(invalid="ignore"):
        # Fraction of folds agreeing with the majority direction.
        consistency = np.abs(signs.sum(axis=0)) / max(len(fitted), 1)

    return pl.DataFrame(
        {
            "feature": names,
            "mean_coefficient": matrix.mean(axis=0),
            "std_coefficient": matrix.std(axis=0, ddof=1)
            if len(fitted) > 1
            else np.zeros(len(names)),
            "sign_consistency": consistency,
            "pct_nonzero": (np.abs(matrix) > 1e-10).mean(axis=0),
            "n_folds": len(fitted),
        }
    ).sort("sign_consistency", descending=True)
