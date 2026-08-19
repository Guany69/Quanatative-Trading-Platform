from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from quant_platform.models.base import BaseModel, ModelError
from quant_platform.models.registry import production_candidates
from quant_platform.models.temporal_conv import TemporalConvModel

pytest.importorskip("torch")


FEATURES = [f"x{i}" for i in range(6)]


def _frame(rows: int = 30) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(rows, len(FEATURES)))
    data = {name: X[:, i] for i, name in enumerate(FEATURES)}
    return pl.DataFrame(
        {
            "security_id": [f"S{i % 5}" for i in range(rows)],
            "as_of": [date(2024, 1, 2) + timedelta(days=i // 5) for i in range(rows)],
            **data,
            "target": X[:, 0] - 0.5 * X[:, -1],
        }
    )


def test_fit_predict_and_save_load_round_trip(tmp_path) -> None:
    frame = _frame()
    model = TemporalConvModel(
        channels=2, lookback=3, conv_channels=(4, 3), max_epochs=2, patience=1
    )
    model.fit(frame, FEATURES, "target")
    predictions = model.predict(frame)

    assert predictions.height == frame.height
    assert predictions["prediction"].is_finite().all()
    assert model.metadata is not None
    assert model.metadata.hyperparameters["channels"] == 2
    assert model.metadata.hyperparameters["lookback"] == 3

    restored = BaseModel.load(model.save(tmp_path / "temporal.pkl"))
    assert isinstance(restored, TemporalConvModel)
    np.testing.assert_allclose(
        restored.predict(frame)["prediction"], predictions["prediction"]
    )
    assert "temporal_conv" not in production_candidates()


def test_feature_layout_is_validated() -> None:
    with pytest.raises(ModelError, match=r"channels=2, lookback=4.*got 6"):
        TemporalConvModel(channels=2, lookback=4, max_epochs=1).fit(
            _frame(), FEATURES, "target"
        )


def test_explicit_validation_drives_early_stopping() -> None:
    frame = _frame()
    X = frame.select(FEATURES).to_numpy()
    y = frame["target"].to_numpy()
    model = TemporalConvModel(
        channels=2,
        lookback=3,
        conv_channels=(4, 3),
        learning_rate=0.0,
        max_epochs=8,
        patience=2,
    )
    model.set_validation(X[-5:], y[-5:])
    model.fit(frame, FEATURES, "target")

    assert model.training_history().height == 3
    assert model._best_epoch == 0
