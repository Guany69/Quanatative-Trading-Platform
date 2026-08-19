from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from quant_platform.models.base import BaseModel, ModelError
from quant_platform.models.neural_network import NeuralNetworkModel

torch = pytest.importorskip("torch")


def _frame(rows: int = 24) -> pl.DataFrame:
    x1 = np.linspace(-1.0, 1.0, rows)
    x2 = np.cos(np.linspace(0.0, 2.0, rows))
    return pl.DataFrame(
        {
            "security_id": [f"S{i % 4}" for i in range(rows)],
            "as_of": [date(2024, 1, 2 + i // 4) for i in range(rows)],
            "x1": x1,
            "x2": x2,
            "target": 0.4 * x1 - 0.2 * x2,
        }
    )


def test_device_and_amp_round_trip_through_metadata_and_save(tmp_path) -> None:
    model = NeuralNetworkModel(hidden_sizes=(8,), max_epochs=2, patience=1)
    model.fit(_frame(), ["x1", "x2"], "target")

    assert model.metadata is not None
    assert model.metadata.hyperparameters["device"] == "cpu"
    assert model.metadata.hyperparameters["use_amp"] is False

    restored = BaseModel.load(model.save(tmp_path / "nn.pkl"))
    assert isinstance(restored, NeuralNetworkModel)
    assert restored.device == "cpu"
    assert restored.use_amp is False
    assert restored.metadata is not None
    assert restored.metadata.hyperparameters["device"] == "cpu"
    np.testing.assert_allclose(
        restored.predict(_frame())["prediction"],
        model.predict(_frame())["prediction"],
    )


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CUDA-less machine")
def test_cuda_request_fails_clearly_without_cuda() -> None:
    model = NeuralNetworkModel(device="cuda", max_epochs=1)

    with pytest.raises(ModelError, match="CUDA was requested but no CUDA device is available"):
        model.fit(_frame(), ["x1", "x2"], "target")
