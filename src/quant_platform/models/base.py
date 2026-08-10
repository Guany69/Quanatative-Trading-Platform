"""Common model interface (spec section 14).

Every model -- from the no-skill baseline to LightGBM -- implements this interface, so the
walk-forward harness, the ensemble, and the reports can treat them interchangeably. That
uniformity is what makes the baseline comparisons honest: the trivial predictor runs through
exactly the same pipeline, costs, and portfolio construction as the sophisticated one, so any
difference in results comes from the model rather than from a friendlier code path.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_platform.domain.enums import TargetType
from quant_platform.utilities.narrow import as_optional_date


class ModelError(RuntimeError):
    """Raised for model misuse (predicting before fit, schema drift, etc.)."""


@dataclass
class ModelMetadata:
    """Provenance travelling with a fitted model."""

    model_name: str
    model_version: str = "1.0.0"
    target_type: TargetType = TargetType.EXCESS_RETURN_RANK
    feature_names: list[str] = field(default_factory=list)
    train_start: date | None = None
    train_end: date | None = None
    n_train_rows: int = 0
    random_seed: int | None = None
    fitted_at: datetime | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "target_type": self.target_type.value,
            "feature_names": self.feature_names,
            "train_start": str(self.train_start) if self.train_start else None,
            "train_end": str(self.train_end) if self.train_end else None,
            "n_train_rows": self.n_train_rows,
            "random_seed": self.random_seed,
            "fitted_at": self.fitted_at.isoformat() if self.fitted_at else None,
            "hyperparameters": self.hyperparameters,
            "extra": self.extra,
        }


class BaseModel(ABC):
    """Abstract model.

    Subclasses implement ``_fit`` and ``_predict``; this class handles schema validation,
    metadata capture, and the fitted/unfitted guard so those cannot be forgotten.
    """

    name: str = "base"
    version: str = "1.0.0"

    def __init__(
        self, target_type: TargetType = TargetType.EXCESS_RETURN_RANK, random_seed: int = 42
    ) -> None:
        self.target_type = target_type
        self.random_seed = random_seed
        self.metadata: ModelMetadata | None = None

    # ------------------------------------------------------------------ abstract
    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None: ...

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray: ...

    # ------------------------------------------------------------------ public
    @property
    def is_fitted(self) -> bool:
        return self.metadata is not None

    def fit(self, train: pl.DataFrame, feature_columns: list[str], target_column: str) -> BaseModel:
        """Fit on a training frame.

        The frame must contain only training rows; this class does not slice by date, because
        an implicit slice is exactly how test data sneaks into training.
        """
        missing = [c for c in feature_columns if c not in train.columns]
        if missing:
            raise ModelError(f"{self.name}: training frame is missing features {missing}")
        if target_column not in train.columns:
            raise ModelError(f"{self.name}: training frame lacks target '{target_column}'")

        clean = train.drop_nulls(subset=[*feature_columns, target_column])
        if clean.is_empty():
            raise ModelError(f"{self.name}: no training rows remain after dropping nulls")

        X = clean.select(feature_columns).to_numpy().astype(np.float64)
        y = clean[target_column].to_numpy().astype(np.float64)

        if not np.isfinite(X).all():
            raise ModelError(f"{self.name}: feature matrix contains non-finite values")
        if not np.isfinite(y).all():
            raise ModelError(f"{self.name}: target contains non-finite values")

        # Guard against the target hiding in the features (a direct, total leak).
        if target_column in feature_columns:
            raise ModelError(
                f"{self.name}: target '{target_column}' is also listed as a feature -- this "
                f"would leak the answer into the model"
            )

        self._fit(X, y, list(feature_columns))

        dates = clean["as_of"] if "as_of" in clean.columns else None
        self.metadata = ModelMetadata(
            model_name=self.name,
            model_version=self.version,
            target_type=self.target_type,
            feature_names=list(feature_columns),
            train_start=as_optional_date(dates.min(), "train_start") if dates is not None else None,
            train_end=as_optional_date(dates.max(), "train_end") if dates is not None else None,
            n_train_rows=clean.height,
            random_seed=self.random_seed,
            fitted_at=datetime.now(),
            hyperparameters=self.get_hyperparameters(),
        )
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict and attach a cross-sectional rank.

        The rank -- not the raw prediction -- is the portfolio signal, so it is produced here
        once rather than being re-derived (possibly inconsistently) by each caller.
        """
        if self.metadata is None:
            raise ModelError(f"{self.name}: predict() called before fit()")

        expected = self.metadata.feature_names
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ModelError(
                f"{self.name}: prediction frame is missing features the model was trained on: "
                f"{missing}"
            )

        X = df.select(expected).to_numpy().astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        preds = np.asarray(self._predict(X), dtype=np.float64).ravel()
        if preds.shape[0] != df.height:
            raise ModelError(
                f"{self.name}: produced {preds.shape[0]} predictions for {df.height} rows"
            )

        out = df.select(["security_id", "as_of"]).with_columns(pl.Series("prediction", preds))
        return out.with_columns(
            (pl.col("prediction").rank("average").over("as_of") / pl.len().over("as_of")).alias(
                "cross_sectional_rank"
            )
        )

    def get_hyperparameters(self) -> dict[str, Any]:
        return {}

    def feature_importance(self) -> pl.DataFrame | None:
        """Feature importances, when the model exposes them."""
        return None

    def save(self, path: str | Path) -> Path:
        """Persist model and metadata side by side."""
        if self.metadata is None:
            raise ModelError(f"{self.name}: cannot save an unfitted model")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh)
        meta_path = p.with_suffix(".meta.json")
        with meta_path.open("w") as fh:
            json.dump(self.metadata.to_dict(), fh, indent=2)
        return p

    @staticmethod
    def load(path: str | Path) -> BaseModel:
        with Path(path).open("rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, BaseModel):
            raise ModelError(f"{path} does not contain a BaseModel")
        return model

    def check_compatible(self, feature_columns: list[str]) -> None:
        """Raise if the feature schema drifted since training."""
        if self.metadata is None:
            raise ModelError(f"{self.name}: model is not fitted")
        if list(feature_columns) != self.metadata.feature_names:
            raise ModelError(
                f"{self.name}: feature schema changed since training.\n"
                f"  trained on: {self.metadata.feature_names}\n"
                f"  given:      {list(feature_columns)}"
            )
