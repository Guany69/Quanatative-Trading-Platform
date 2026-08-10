"""Gradient-boosted and random-forest models (spec sections 14.4-14.5).

LightGBM is the primary nonlinear model. The configuration here is deliberately conservative,
because the default settings of any boosting library are tuned for datasets with far higher
signal-to-noise than equity returns. Cross-sectional return prediction has an R^2 close to
zero; a model with deep trees and no regularization will fit the noise perfectly and
generalize not at all.

Concretely: shallow trees (depth <= 4), strong L1/L2 penalties, aggressive row and column
subsampling, and early stopping against a genuine validation fold. The hyperparameter search
is **bounded** -- the grid cannot wander into high-capacity territory, because an unrestricted
search over thousands of configurations is itself a multiple-testing problem that manufactures
apparent skill.

Random forest is a *challenger and diagnostic only*. The spec is explicit that it must not
silently become the production model, so it carries a flag saying so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_platform.models.base import BaseModel, ModelError
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("models.trees")

# Bounded search space. Every bound keeps capacity low on purpose (spec section 14.4).
LIGHTGBM_SEARCH_SPACE = {
    "num_leaves": (7, 63),
    "max_depth": (3, 6),
    "learning_rate": (0.01, 0.1),
    "min_child_samples": (50, 500),
    "subsample": (0.5, 0.9),
    "colsample_bytree": (0.4, 0.9),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (1.0, 50.0),
}


class LightGBMModel(BaseModel):
    """Gradient-boosted trees, regularized for a very low signal-to-noise setting."""

    name = "lightgbm"

    def __init__(
        self,
        num_leaves: int = 15,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        n_estimators: int = 300,
        min_child_samples: int = 100,
        subsample: float = 0.7,
        colsample_bytree: float = 0.6,
        reg_alpha: float = 0.5,
        reg_lambda: float = 10.0,
        early_stopping_rounds: int = 50,
        monotone_constraints: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.params = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.monotone_constraints = monotone_constraints or {}
        self._booster: Any = None
        self._best_iteration: int | None = None
        self._validation: tuple[np.ndarray, np.ndarray] | None = None

    def set_validation(self, X: np.ndarray, y: np.ndarray) -> None:
        """Supply a validation set for early stopping.

        Must come from the fold's validation window -- never the test or holdout period, or
        the stopping point itself is fitted to data we intend to evaluate on.
        """
        self._validation = (X, y)

    def _fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        try:
            import lightgbm as lgb
        except (ImportError, OSError) as exc:
            raise ModelError(
                f"LightGBM is unavailable ({exc}). On macOS without Homebrew run: "
                f"uv run python scripts/fix_macos_libomp.py"
            ) from exc

        constraints = None
        if self.monotone_constraints:
            # Only use these where economics genuinely dictates direction; otherwise they
            # impose a prior the data may not support.
            constraints = [self.monotone_constraints.get(f, 0) for f in feature_names]

        params: dict[str, Any] = {
            **self.params,
            "objective": "regression",
            "random_state": self.random_seed,
            "deterministic": True,
            "force_row_wise": True,  # removes threading nondeterminism
            "verbose": -1,
            "n_jobs": 1,
        }
        if constraints:
            params["monotone_constraints"] = constraints

        model = lgb.LGBMRegressor(**params)
        fit_kwargs: dict[str, Any] = {}
        if self._validation is not None:
            vx, vy = self._validation
            fit_kwargs["eval_set"] = [(vx, vy)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ]

        model.fit(X, y, **fit_kwargs)
        self._booster = model
        self._best_iteration = getattr(model, "best_iteration_", None)
        if self._best_iteration:
            logger.debug("lightgbm early-stopped at iteration %d", self._best_iteration)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self._booster is None:
            raise ModelError(f"{self.name}: not fitted")
        import warnings

        with warnings.catch_warnings():
            # LightGBM's sklearn wrapper records internal feature names at fit time and warns
            # when predicting from a bare ndarray. BaseModel already guarantees column order
            # via check_compatible, so the warning is noise rather than a real mismatch.
            warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
            return np.asarray(self._booster.predict(X), dtype=float)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {**self.params, "early_stopping_rounds": self.early_stopping_rounds}

    def feature_importance(self) -> pl.DataFrame | None:
        if self._booster is None or self.metadata is None:
            return None
        # 'gain' reflects contribution to loss reduction; 'split' only counts usage and
        # over-credits high-cardinality features.
        gains = np.asarray(self._booster.booster_.feature_importance("gain"), dtype=float)
        total = gains.sum()
        return pl.DataFrame(
            {
                "feature": self.metadata.feature_names,
                "importance": gains / total if total > 0 else gains,
                "raw_gain": gains,
            }
        ).sort("importance", descending=True)

    def shap_values(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """SHAP contributions, when the optional package is present."""
        if self._booster is None or self.metadata is None:
            return None
        try:
            import shap
        except ImportError:
            logger.info("shap not installed; skipping SHAP analysis")
            return None
        X = df.select(self.metadata.feature_names).to_numpy().astype(float)
        explainer = shap.TreeExplainer(self._booster)
        values = np.asarray(explainer.shap_values(X))
        return pl.DataFrame(
            {
                "feature": self.metadata.feature_names,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        ).sort("mean_abs_shap", descending=True)


class RandomForestModel(BaseModel):
    """Random forest / extra-trees CHALLENGER (spec section 14.5).

    Diagnostic only. Random forests fit financial noise readily and rarely beat a
    well-regularized booster out of sample, so ``is_production_candidate`` is False and the
    model selection code must not promote it automatically.
    """

    name = "random_forest"
    is_production_candidate = False

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        min_samples_leaf: int = 100,
        max_features: str | float = "sqrt",
        extra_trees: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.extra_trees = extra_trees
        self._model: Any = None

    def _fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

        cls = ExtraTreesRegressor if self.extra_trees else RandomForestRegressor
        self._model = cls(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_seed,
            n_jobs=1,  # single-threaded for determinism
        )
        self._model.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelError(f"{self.name}: not fitted")
        return np.asarray(self._model.predict(X), dtype=float)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "extra_trees": self.extra_trees,
        }

    def feature_importance(self) -> pl.DataFrame | None:
        if self._model is None or self.metadata is None:
            return None
        return pl.DataFrame(
            {
                "feature": self.metadata.feature_names,
                "importance": np.asarray(self._model.feature_importances_, dtype=float),
            }
        ).sort("importance", descending=True)


def tune_lightgbm(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    n_trials: int = 20,
    random_seed: int = 42,
    timeout_seconds: int | None = 300,
) -> dict[str, Any]:
    """Bounded hyperparameter search using Optuna.

    Three guards against the search itself becoming an overfitting engine:

    * the space is capped at low capacity (``LIGHTGBM_SEARCH_SPACE``);
    * ``n_trials`` is small by default -- each trial is another hypothesis test, and the
      Deflated Sharpe correction later needs an honest count of them;
    * selection is on **validation** rank IC, never on test or holdout data.

    Returns the best parameters plus the trial count, which the experiment registry records
    so multiple-testing corrections can use it.
    """
    try:
        import optuna
    except ImportError:
        logger.warning("optuna not installed; using default LightGBM parameters")
        return {"params": {}, "n_trials": 0, "best_value": float("nan")}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train = train.select(feature_columns).to_numpy().astype(float)
    y_train = train[target_column].to_numpy().astype(float)
    X_val = validation.select(feature_columns).to_numpy().astype(float)
    y_val = validation[target_column].to_numpy().astype(float)

    from quant_platform.evaluation.metrics import spearman_ic

    def objective(trial: Any) -> float:
        space = LIGHTGBM_SEARCH_SPACE
        model = LightGBMModel(
            num_leaves=trial.suggest_int("num_leaves", *space["num_leaves"]),
            max_depth=trial.suggest_int("max_depth", *space["max_depth"]),
            learning_rate=trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            min_child_samples=trial.suggest_int("min_child_samples", *space["min_child_samples"]),
            subsample=trial.suggest_float("subsample", *space["subsample"]),
            colsample_bytree=trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
            reg_alpha=trial.suggest_float("reg_alpha", *space["reg_alpha"]),
            reg_lambda=trial.suggest_float("reg_lambda", *space["reg_lambda"]),
            n_estimators=200,
            random_seed=random_seed,
        )
        model._fit(X_train, y_train, feature_columns)
        preds = model._predict(X_val)
        # Rank IC, because ranking is what the portfolio actually consumes.
        ic = spearman_ic(preds, y_val)
        return float(ic) if np.isfinite(ic) else -1.0

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)

    logger.info(
        "lightgbm search: %d trials, best validation IC %.4f",
        len(study.trials),
        study.best_value,
    )
    return {
        "params": study.best_params,
        "n_trials": len(study.trials),
        "best_value": float(study.best_value),
        # Every trial is retained: Deflated Sharpe needs the true number of attempts.
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
            if t.value is not None
        ],
    }
