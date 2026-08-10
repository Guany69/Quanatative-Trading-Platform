"""One-shot locked-holdout evaluation with permanent DuckDB consumption record."""

from __future__ import annotations

import json
from pathlib import Path

from quant_platform.config import config_hash, load_charter
from quant_platform.data.snapshots import SnapshotStore
from quant_platform.paper.coordinator import _ensemble_predictions, _single_artifact_predictions
from quant_platform.pipeline import build_training_frame, forecast_ic, load_snapshot_data
from quant_platform.research.results import ResultsReader, ResultsStore
from quant_platform.validation.walk_forward import WalkForwardSplitter


def evaluate_locked_holdout(
    *,
    run_id: str,
    model: str,
    charter_path: str | Path,
    acknowledged_by: str,
    acknowledge: bool,
    results_db: str | Path = "research.duckdb",
    snapshot_root: str | Path = "data/snapshots",
    report_root: str | Path = "reports",
) -> dict[str, float]:
    charter = load_charter(charter_path)
    if charter.validation.holdout is None:
        raise ValueError("no locked holdout is configured")
    with ResultsReader(results_db) as reader:
        run = reader.query("research_run", run_id=run_id, limit=1)
        if run["status"][0] != "COMPLETED":
            raise RuntimeError(f"run {run_id} is not completed")
        if str(run["charter_hash"][0]) != config_hash(charter):
            raise RuntimeError(
                "current charter does not match the selected research run; holdout blocked"
            )
        artifacts = reader.query("artifact", run_id=run_id, model=model, limit=1000).sort(
            "fold", descending=True
        )
    snapshot = SnapshotStore(snapshot_root).load(str(run["snapshot_id"][0]))
    data = load_snapshot_data(snapshot, charter)
    panel = build_training_frame(data)
    holdout = WalkForwardSplitter(charter.validation, data.calendar).holdout(
        panel, acknowledge=acknowledge
    )
    if holdout.is_empty():
        raise RuntimeError("configured holdout contains no model-ready observations")
    fold = int(artifacts["fold"][0])
    if model == "ensemble":
        predictions, _metadata = _ensemble_predictions(run_id, fold, holdout, results_db)
    else:
        predictions, _metadata = _single_artifact_predictions(
            Path(str(artifacts["artifact_path"][0])), holdout
        )
    metrics = forecast_ic(predictions, holdout, "excess_return_rank")
    with ResultsStore(results_db) as store:
        store.record_holdout_evaluation(run_id, model, acknowledged_by, metrics)
    path = Path(report_root) / run_id / "holdout.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": model,
                "acknowledged_by": acknowledged_by,
                "metrics": metrics,
                "warning": "The locked holdout is now permanently consumed.",
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    return metrics
