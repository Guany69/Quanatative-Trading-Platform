from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_platform.models.registry import MODEL_REGISTRY, create_model
from quant_platform.paper.state import PaperTradingState
from quant_platform.research.results import ResultsStore, ResultsStoreError
from quant_platform.research.runner import TARGET_MODELS, TARGET_STRATEGIES

EXPECTED_MODELS = {
    "no_skill",
    "momentum_baseline",
    "factor_composite",
    "elastic_net",
    "logistic",
    "lightgbm",
    "random_forest",
    "neural_network",
    "ensemble",
}


def test_complete_model_and_strategy_rosters_are_registered():
    assert set(TARGET_MODELS) == EXPECTED_MODELS == set(MODEL_REGISTRY)
    assert create_model("neural_network").__class__.__module__.endswith("neural_network")
    assert create_model("lightgbm").__class__.__module__.endswith("trees")
    assert create_model("elastic_net").__class__.__module__.endswith("linear")
    assert create_model("random_forest").__class__.__module__.endswith("trees")
    assert set(TARGET_STRATEGIES) == {
        "equal_weight",
        "score_weighted",
        "inverse_volatility",
        "constrained_optimizer",
    }


def _completed_run(path) -> None:
    with ResultsStore(path) as store:
        store.create_run(
            run_id="completed",
            snapshot_id="snapshot",
            charter_hash="charter",
            code_version="commit",
            code_dirty=False,
            seed=42,
            models=["factor_composite"],
            strategies=["equal_weight"],
            scenarios=["base"],
            fold_schedule_hash="folds",
        )
        store.write_predictions(
            pl.DataFrame(
                {
                    "run_id": ["completed"],
                    "model": ["factor_composite"],
                    "fold": [0],
                    "as_of": [date(2020, 1, 2)],
                    "security_id": ["A"],
                    "score": [1.0],
                    "rank": [1.0],
                }
            )
        )
        store.finish_run("completed", "COMPLETED")


def test_production_designation_is_validated_audited_and_survives_reload(tmp_path):
    results_db = tmp_path / "research.duckdb"
    _completed_run(results_db)
    state = PaperTradingState.initialize()
    with pytest.raises(RuntimeError, match="no production model"):
        state.require_production_model(results_db)

    reference = state.designate_production_model(
        "completed", "factor_composite", "alice", results_db
    )
    assert reference.designated_by == "alice"
    path = state.save(tmp_path / "paper-state.json")
    restored = PaperTradingState.load(path)
    assert restored.require_production_model(results_db) == reference
    assert restored.production_model_history[0]["run_id"] == "completed"

    with pytest.raises(ResultsStoreError, match="unknown run_id"):
        restored.designate_production_model("missing", "factor_composite", "alice", results_db)
