from __future__ import annotations

import os
import pickle
from datetime import date

import polars as pl
import pytest

from quant_platform.config import load_charter
from quant_platform.research import executor as executor_module
from quant_platform.research.executor import (
    TrainingExecutor,
    TrainingTaskError,
    deterministic_task_seed,
    make_task,
)
from quant_platform.research.pit_cache import FoldPanelRef
from quant_platform.validation.walk_forward import Fold


def _panel(tmp_path, fold_index: int) -> FoldPanelRef:
    securities = [f"S{i:03d}" for i in range(30)]

    def frame(as_of: date) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "security_id": securities,
                "as_of": [as_of] * len(securities),
                "feature": [float(i) for i in range(len(securities))],
                "excess_return_rank": [(i + 1) / len(securities) for i in range(len(securities))],
            }
        )

    directory = tmp_path / f"fold-{fold_index}"
    directory.mkdir()
    train_path = directory / "train.parquet"
    validation_path = directory / "validation.parquet"
    test_path = directory / "test.parquet"
    frame(date(2019, 1, 2)).write_parquet(train_path)
    frame(date(2020, 1, 2)).write_parquet(validation_path)
    frame(date(2020, 2, 3)).write_parquet(test_path)
    fold = Fold(
        fold_index,
        date(2019, 1, 1),
        date(2019, 12, 31),
        date(2020, 1, 1),
        date(2020, 1, 31),
        date(2020, 2, 1),
        date(2020, 2, 28),
    )
    return FoldPanelRef(fold, train_path, validation_path, test_path, 30, 30, 30)


def test_process_and_serial_execution_are_deterministic_and_isolated(tmp_path):
    charter = load_charter("configs/research_charter.yaml")
    panels = [_panel(tmp_path, 0), _panel(tmp_path, 1)]
    tasks = [
        make_task(
            run_id="run",
            model="no_skill",
            panel=panel,
            feature_columns=["feature"],
            target_column="excess_return_rank",
            charter=charter,
        )
        for panel in panels
    ]
    serial = TrainingExecutor(max_workers=1).map(tasks)
    parallel = TrainingExecutor(max_workers=2).map(tasks)

    assert [result.seed for result in serial] == [result.seed for result in parallel]
    assert serial[0].seed == deterministic_task_seed(charter.random_seed, "no_skill", 0)
    assert all(
        left.predictions.equals(right.predictions)
        for left, right in zip(serial, parallel, strict=True)
    )
    assert all(result.worker_pid != os.getpid() for result in parallel)
    artifact = pickle.loads(parallel[0].artifact)
    assert set(artifact) == {"model", "preprocessor"}
    assert parallel[0].metadata["extra"]["fold_windows"]["test_start"] == "2020-02-01"
    assert "quant_platform.research.results" not in pickle.dumps(tasks[0]).decode(
        "latin1", errors="ignore"
    )


def test_task_failure_retries_exactly_once(tmp_path, monkeypatch):
    charter = load_charter("configs/research_charter.yaml")
    task = make_task(
        run_id="run",
        model="no_skill",
        panel=_panel(tmp_path, 0),
        feature_columns=["feature"],
        target_column="excess_return_rank",
        charter=charter,
    )
    real_execute = executor_module._execute_task
    calls = 0

    def flaky(current):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient")
        return real_execute(current)

    monkeypatch.setattr(executor_module, "_execute_task", flaky)
    result = TrainingExecutor(max_workers=1, use_processes=False).map([task])
    assert result[0].model == "no_skill"
    assert calls == 2


def test_persistent_task_failure_returns_no_partial_success(tmp_path, monkeypatch):
    charter = load_charter("configs/research_charter.yaml")
    task = make_task(
        run_id="run",
        model="no_skill",
        panel=_panel(tmp_path, 0),
        feature_columns=["feature"],
        target_column="excess_return_rank",
        charter=charter,
    )
    calls = 0

    def broken(_current):
        nonlocal calls
        calls += 1
        raise RuntimeError("persistent")

    monkeypatch.setattr(executor_module, "_execute_task", broken)
    with pytest.raises(TrainingTaskError, match=r"after one retry.*no_skill/fold-0"):
        TrainingExecutor(max_workers=1, use_processes=False).map([task])
    assert calls == 2
