"""Experiment tracking (spec section 15).

Records **every** experiment, including rejected ones. That is not bookkeeping tidiness -- it
is a statistical requirement. The Deflated Sharpe Ratio and the Probability of Backtest
Overfitting both take the *number of configurations tried* as an input. A registry that keeps
only winners makes those corrections impossible to compute and hides selection bias entirely:
the best of 500 random strategies always looks impressive.

Storage is a JSONL file so runs append safely and the history is diffable and greppable.
MLflow is supported when installed, but is never required.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_platform.domain.enums import CandidateStatus
from quant_platform.utilities.reproducibility import (
    dependency_versions,
    get_logger,
    git_commit,
    git_is_dirty,
)

logger = get_logger("experiments")


@dataclass
class Trial:
    """One recorded experiment."""

    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    name: str = ""
    model_class: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    git_commit: str | None = field(default_factory=git_commit)
    git_dirty: bool | None = field(default_factory=git_is_dirty)
    config_hash: str | None = None
    data_snapshot_id: str | None = None
    feature_version: str | None = None
    label_version: str | None = None
    universe: str | None = None
    train_start: str | None = None
    train_end: str | None = None
    validation_start: str | None = None
    validation_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    cost_assumptions: dict[str, Any] = field(default_factory=dict)
    portfolio_settings: dict[str, Any] = field(default_factory=dict)
    forecast_metrics: dict[str, float] = field(default_factory=dict)
    portfolio_metrics: dict[str, float] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    status: str = CandidateStatus.RESEARCH_CANDIDATE.value
    rejection_reason: str | None = None
    dependency_versions: dict[str, str] = field(default_factory=dependency_versions)
    notes: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)


class ExperimentRegistry:
    """Append-only JSONL registry of all trials."""

    def __init__(self, path: str | Path = "artifacts/experiments.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trial: Trial) -> Trial:
        """Append a trial. Append-only so history cannot be quietly rewritten."""
        with self.path.open("a") as fh:
            fh.write(trial.to_json() + "\n")
        logger.debug("recorded trial %s (%s)", trial.experiment_id, trial.status)
        return trial

    def record_rejection(self, trial: Trial, reason: str) -> Trial:
        """Record a rejected configuration. These count toward the trial total."""
        trial.status = CandidateStatus.REJECTED.value
        trial.rejection_reason = reason
        return self.record(trial)

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("skipping malformed registry line")
        return rows

    def to_frame(self) -> pl.DataFrame:
        rows = self.load_all()
        if not rows:
            return pl.DataFrame()
        flat = [
            {
                "experiment_id": r.get("experiment_id"),
                "run_id": r.get("run_id"),
                "created_at": r.get("created_at"),
                "name": r.get("name"),
                "model_class": r.get("model_class"),
                "status": r.get("status"),
                "rejection_reason": r.get("rejection_reason"),
                "test_start": r.get("test_start"),
                "test_end": r.get("test_end"),
                "mean_ic": (r.get("forecast_metrics") or {}).get("mean_ic"),
                "ic_ir": (r.get("forecast_metrics") or {}).get("ic_ir"),
                "sharpe_ratio": (r.get("portfolio_metrics") or {}).get("sharpe_ratio"),
                "information_ratio": (r.get("portfolio_metrics") or {}).get("information_ratio"),
                "cagr": (r.get("portfolio_metrics") or {}).get("cagr"),
            }
            for r in rows
        ]
        return pl.DataFrame(flat)

    def trial_count(self) -> int:
        """Total configurations tried -- the input to multiple-testing corrections.

        Counts rejected trials too. Excluding them would understate the search and inflate
        the Deflated Sharpe Ratio.
        """
        return len(self.load_all())

    def run_ids(self) -> set[str]:
        """Research runs linked into the append-only history."""
        return {str(row["run_id"]) for row in self.load_all() if row.get("run_id")}

    def sharpe_ratios(self) -> list[float]:
        """Every recorded Sharpe ratio, for Deflated Sharpe's variance-across-trials term."""
        out: list[float] = []
        for row in self.load_all():
            value = (row.get("portfolio_metrics") or {}).get("sharpe_ratio")
            if value is not None:
                try:
                    fv = float(value)
                except (TypeError, ValueError):
                    continue
                if fv == fv:  # skip NaN
                    out.append(fv)
        return out

    def summary(self) -> dict[str, Any]:
        rows = self.load_all()
        by_status: dict[str, int] = {}
        for r in rows:
            key = str(r.get("status", "unknown"))
            by_status[key] = by_status.get(key, 0) + 1
        return {
            "total_trials": len(rows),
            "by_status": by_status,
            "path": str(self.path),
        }


class MLflowTracker:
    """Optional MLflow mirror.

    A convenience layer only: the JSONL registry remains the source of truth, so the platform
    behaves identically whether or not MLflow is installed.
    """

    def __init__(self, tracking_uri: str | None = None, experiment: str = "quant-platform") -> None:
        self.enabled = False
        try:
            import mlflow
        except ImportError:
            logger.info("mlflow not installed; using the file registry only")
            return
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.enabled = True

    def log_trial(self, trial: Trial) -> None:
        if not self.enabled:
            return
        with self._mlflow.start_run(run_name=trial.name or trial.experiment_id):
            self._mlflow.log_params({k: str(v)[:500] for k, v in trial.hyperparameters.items()})
            self._mlflow.log_params(
                {
                    "model_class": trial.model_class,
                    "status": trial.status,
                    "git_commit": trial.git_commit or "unknown",
                    "config_hash": trial.config_hash or "unknown",
                }
            )
            for key, value in {**trial.forecast_metrics, **trial.portfolio_metrics}.items():
                try:
                    fv = float(value)
                    if fv == fv:
                        self._mlflow.log_metric(key, fv)
                except (TypeError, ValueError):
                    continue
