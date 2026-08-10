from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant_platform.api.config import ApiSettings
from quant_platform.research.results import ResultsStore


@pytest.fixture
def api_settings(tmp_path: Path) -> ApiSettings:
    root = Path(__file__).resolve().parents[2]
    return ApiSettings(
        results_db=tmp_path / "research.duckdb",
        charter_path=root / "configs/research_charter.yaml",
        snapshot_root=tmp_path / "snapshots",
        cache_root=tmp_path / "cache",
        artifact_root=tmp_path / "artifacts",
        report_root=tmp_path / "reports",
        registry_path=tmp_path / "experiments.jsonl",
        job_state_path=tmp_path / "research-jobs.json",
        paper_state_path=tmp_path / "paper" / "state.json",
        frontend_dist=tmp_path / "frontend-dist",
    )


def seed_completed_run(settings: ApiSettings, run_id: str = "run-a") -> None:
    with ResultsStore(settings.results_db) as store:
        store.create_run(
            run_id=run_id,
            snapshot_id="snapshot-a",
            charter_hash="charter",
            code_version="commit",
            code_dirty=False,
            seed=42,
            models=["factor_composite"],
            strategies=["equal_weight"],
            scenarios=["base"],
            fold_schedule_hash="folds",
        )
        store.write_folds(
            run_id,
            pl.DataFrame(
                {
                    "fold": [0],
                    "train_start": [date(2018, 1, 1)],
                    "train_end": [date(2019, 1, 1)],
                    "validation_start": [date(2019, 1, 2)],
                    "validation_end": [date(2019, 2, 1)],
                    "test_start": [date(2019, 2, 2)],
                    "test_end": [date(2019, 3, 1)],
                    "purge": [True],
                    "embargo": [20],
                }
            ),
        )
        store.write_predictions(
            pl.DataFrame(
                {
                    "run_id": [run_id, run_id, run_id],
                    "model": ["factor_composite"] * 3,
                    "fold": [0, 0, 0],
                    "as_of": [date(2019, 2, 4), date(2019, 2, 4), date(2019, 2, 5)],
                    "security_id": ["A", "B", "A"],
                    "score": [0.7, 0.3, float("nan")],
                    "rank": [1.0, 0.5, 1.0],
                }
            )
        )
        store.write_fold_metrics(
            pl.DataFrame(
                {
                    "run_id": [run_id],
                    "model": ["factor_composite"],
                    "fold": [0],
                    "metric": ["mean_ic"],
                    "value": [0.1],
                }
            )
        )
        store.write_strategy_results(
            pl.DataFrame(
                {
                    "run_id": [run_id],
                    "model": ["factor_composite"],
                    "strategy": ["equal_weight"],
                    "cost_scenario": ["base"],
                    "metric": ["sharpe_ratio"],
                    "value": [0.5],
                }
            )
        )
        store.finish_run(run_id, "COMPLETED")


class FakeOrchestrator:
    """Small deterministic stand-in that exercises the real store lifecycle."""

    def __init__(self, settings: ApiSettings, *, delay: float = 0.03) -> None:
        self.settings = settings
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.order: list[str] = []

    def run(self, request, **kwargs):
        run_id = kwargs["run_id"]
        progress = kwargs["progress"]
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.order.append(run_id)
        try:
            with ResultsStore(self.settings.results_db) as store:
                store.create_run(
                    run_id=run_id,
                    snapshot_id=request.snapshot_id or "auto",
                    charter_hash="charter",
                    code_version="test",
                    code_dirty=False,
                    seed=42,
                    models=list(request.models),
                    strategies=list(request.strategies),
                    scenarios=list(request.scenarios),
                    fold_schedule_hash="folds",
                    created_at=kwargs["created_at"],
                )
                for stage in ("PIT_CACHE", "MODEL_TRAINING", "REPORTING"):
                    store.update_stage(run_id, stage)
                    progress(stage)
                    time.sleep(self.delay)
                if request.fixture_securities == 21:
                    store.finish_run(
                        run_id,
                        "FAILED",
                        failure_stage="MODEL_TRAINING",
                        failure_cause="TrainingTaskError: requested test failure",
                    )
                    raise RuntimeError("requested test failure")
                store.finish_run(run_id, "COMPLETED")
        finally:
            self.active -= 1
