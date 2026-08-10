"""Deterministic process-level model/fold training executor."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from quant_platform.config.models import ResearchCharter
from quant_platform.evaluation.metrics import spearman_ic
from quant_platform.features.preprocessing import CrossSectionalPreprocessor
from quant_platform.models.registry import create_model
from quant_platform.research.pit_cache import FoldPanelRef


class TrainingTaskError(RuntimeError):
    """A model/fold task failed on both its initial attempt and single retry."""


@dataclass(frozen=True)
class TrainingTask:
    run_id: str
    model: str
    fold: int
    train_path: str
    validation_path: str
    test_path: str
    feature_columns: list[str]
    target_column: str
    charter: dict[str, Any]
    seed: int
    fold_windows: dict[str, str]

    @property
    def identity(self) -> str:
        return f"{self.model}/fold-{self.fold}"


@dataclass
class TaskResult:
    run_id: str
    model: str
    fold: int
    seed: int
    worker_pid: int
    predictions: pl.DataFrame
    validation_predictions: pl.DataFrame
    metadata: dict[str, Any]
    artifact: bytes
    artifact_extension: str = ".pkl"


def deterministic_task_seed(run_seed: int, model: str, fold: int) -> int:
    """Stable seed independent of task ordering or process scheduling."""
    payload = f"{run_seed}:{model}:{fold}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def make_task(
    *,
    run_id: str,
    model: str,
    panel: FoldPanelRef,
    feature_columns: list[str],
    target_column: str,
    charter: ResearchCharter,
) -> TrainingTask:
    return TrainingTask(
        run_id=run_id,
        model=model,
        fold=panel.fold.index,
        train_path=str(panel.train_path),
        validation_path=str(panel.validation_path),
        test_path=str(panel.test_path),
        feature_columns=list(feature_columns),
        target_column=target_column,
        charter=charter.to_dict(),
        seed=deterministic_task_seed(charter.random_seed, model, panel.fold.index),
        fold_windows={
            "train_start": str(panel.fold.train_start),
            "train_end": str(panel.fold.train_end),
            "validation_start": str(panel.fold.validation_start),
            "validation_end": str(panel.fold.validation_end),
            "test_start": str(panel.fold.test_start),
            "test_end": str(panel.fold.test_end),
        },
    )


def _execute_task(task: TrainingTask) -> TaskResult:
    """Worker entry point.  It deliberately has no ResultsStore dependency."""
    charter = ResearchCharter.model_validate(task.charter)
    train = pl.read_parquet(task.train_path)
    validation = pl.read_parquet(task.validation_path)
    test = pl.read_parquet(task.test_path)

    preprocessor = CrossSectionalPreprocessor(
        task.feature_columns,
        winsorize_quantile=charter.features.winsorize_quantile,
        method="rank",
        min_cross_section=charter.features.min_cross_section,
    )
    train_t = preprocessor.fit_transform(train)
    validation_t = preprocessor.transform(validation)
    test_t = preprocessor.transform(test)

    model = create_model(
        task.model,
        target_type=charter.label.target_type,
        random_seed=task.seed,
    )
    # Early stopping is allowed to inspect the fold's validation window, never its test.
    if hasattr(model, "set_validation"):
        clean = validation_t.drop_nulls(subset=[*task.feature_columns, task.target_column])
        if not clean.is_empty():
            model.set_validation(  # type: ignore[attr-defined]
                clean.select(task.feature_columns).to_numpy().astype(float),
                clean[task.target_column].to_numpy().astype(float),
            )
    model.fit(train_t, task.feature_columns, task.target_column)
    validation_predictions = model.predict(validation_t)
    predictions = model.predict(test_t)
    assert model.metadata is not None
    model.metadata.extra.update(
        {
            "run_id": task.run_id,
            "fold": task.fold,
            "worker_pid": os.getpid(),
            "validation_source": "fold_validation_only",
            "fold_windows": task.fold_windows,
        }
    )
    return TaskResult(
        run_id=task.run_id,
        model=task.model,
        fold=task.fold,
        seed=task.seed,
        worker_pid=os.getpid(),
        predictions=predictions,
        validation_predictions=validation_predictions,
        metadata=model.metadata.to_dict(),
        artifact=pickle.dumps(
            {"model": model, "preprocessor": preprocessor}, protocol=pickle.HIGHEST_PROTOCOL
        ),
    )


class TrainingExecutor:
    """Map complete model/fold tasks with one retry and all-or-nothing return."""

    def __init__(self, max_workers: int | None = None, *, use_processes: bool = True) -> None:
        available = os.cpu_count() or 1
        self.max_workers = max(1, min(max_workers or available, available))
        self.use_processes = use_processes

    def map(self, tasks: list[TrainingTask]) -> list[TaskResult]:
        if not tasks:
            return []
        pending = list(tasks)
        results: dict[tuple[str, int], TaskResult] = {}
        failures: dict[tuple[str, int], BaseException] = {}

        for attempt in range(2):
            failures = {}
            if self.use_processes and self.max_workers > 1:
                with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = {pool.submit(_execute_task, task): task for task in pending}
                    for future in as_completed(futures):
                        task = futures[future]
                        key = (task.model, task.fold)
                        try:
                            results[key] = future.result()
                        except BaseException as exc:  # preserve worker traceback as cause
                            failures[key] = exc
            else:
                for task in pending:
                    key = (task.model, task.fold)
                    try:
                        results[key] = _execute_task(task)
                    except BaseException as exc:
                        failures[key] = exc

            if not failures:
                break
            if attempt == 0:
                pending = [task for task in tasks if (task.model, task.fold) in failures]

        if failures:
            identities = ", ".join(
                f"{model}/fold-{fold}: {type(exc).__name__}: {exc}"
                for (model, fold), exc in sorted(failures.items())
            )
            raise TrainingTaskError(f"training task failed after one retry: {identities}")

        # Nothing is returned to the orchestrator until every requested task succeeded.
        return [results[(task.model, task.fold)] for task in tasks]


def assemble_ensemble_result(
    *,
    run_id: str,
    fold_panel: FoldPanelRef,
    members: dict[str, TaskResult],
    run_seed: int,
    max_weight: float = 0.5,
    min_weight: float = 0.1,
    smoothing: float = 0.7,
    target_column: str = "excess_return_rank",
) -> TaskResult:
    """Build one leakage-safe ensemble from independently completed member outputs."""
    if len(members) < 2:
        raise TrainingTaskError("ensemble requires at least two completed member models")
    validation = pl.read_parquet(fold_panel.validation_path)
    validation_scores: dict[str, float] = {}
    for name, result in sorted(members.items()):
        joined = result.validation_predictions.join(
            validation.select(["security_id", "as_of", target_column]),
            on=["security_id", "as_of"],
            how="inner",
        )
        validation_scores[name] = spearman_ic(
            joined["prediction"].to_numpy(), joined[target_column].to_numpy()
        )

    from quant_platform.models.ensemble import RankEnsemble

    ensemble = RankEnsemble(
        max_weight=max_weight,
        min_weight=min_weight,
        smoothing=smoothing,
        random_seed=deterministic_task_seed(run_seed, "ensemble", fold_panel.fold.index),
    )
    weights = ensemble.fit_weights_from_validation(validation_scores, is_holdout=False)
    predictions = _combine_member_predictions(members, weights, validation=False)
    validation_predictions = _combine_member_predictions(members, weights, validation=True)
    seed = deterministic_task_seed(run_seed, "ensemble", fold_panel.fold.index)
    metadata = {
        "model_name": "ensemble",
        "model_version": "1.0.0",
        "target_type": "excess_return_rank",
        "feature_names": [],
        "random_seed": seed,
        "hyperparameters": {
            "members": sorted(members),
            "weights": weights,
            "max_weight": max_weight,
            "min_weight": min_weight,
            "smoothing": smoothing,
        },
        "extra": {
            "run_id": run_id,
            "fold": fold_panel.fold.index,
            "weights_fit_from": "validation_only",
            "validation_scores": validation_scores,
            "fold_windows": {
                "train_start": str(fold_panel.fold.train_start),
                "train_end": str(fold_panel.fold.train_end),
                "validation_start": str(fold_panel.fold.validation_start),
                "validation_end": str(fold_panel.fold.validation_end),
                "test_start": str(fold_panel.fold.test_start),
                "test_end": str(fold_panel.fold.test_end),
            },
        },
    }
    artifact = json.dumps({"weights": weights, "metadata": metadata}, sort_keys=True).encode()
    return TaskResult(
        run_id=run_id,
        model="ensemble",
        fold=fold_panel.fold.index,
        seed=seed,
        worker_pid=os.getpid(),
        predictions=predictions,
        validation_predictions=validation_predictions,
        metadata=metadata,
        artifact=artifact,
        artifact_extension=".json",
    )


def _combine_member_predictions(
    members: dict[str, TaskResult], weights: dict[str, float], *, validation: bool
) -> pl.DataFrame:
    combined: pl.DataFrame | None = None
    for name, result in sorted(members.items()):
        frame = result.validation_predictions if validation else result.predictions
        ranked = frame.select(
            "security_id",
            "as_of",
            pl.col("cross_sectional_rank").alias(f"_rank_{name}"),
        )
        combined = (
            ranked
            if combined is None
            else combined.join(ranked, on=["security_id", "as_of"], how="inner")
        )
    if combined is None or combined.is_empty():
        raise TrainingTaskError("ensemble members produced no aligned predictions")
    score = pl.lit(0.0)
    for name in sorted(members):
        score += weights[name] * pl.col(f"_rank_{name}")
    return (
        combined.with_columns(score.alias("prediction"))
        .select("security_id", "as_of", "prediction")
        .with_columns(
            (pl.col("prediction").rank("average").over("as_of") / pl.len().over("as_of")).alias(
                "cross_sectional_rank"
            )
        )
    )


def artifact_paths(root: str | Path, result: TaskResult) -> tuple[Path, Path]:
    directory = Path(root) / result.run_id / result.model / f"fold_{result.fold:03d}"
    return directory / f"model{result.artifact_extension}", directory / "metadata.json"
