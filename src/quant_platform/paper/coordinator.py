"""Forward paper coordinator linkage to designated, persisted research artifacts."""

from __future__ import annotations

import json
import pickle
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_platform.config import config_hash
from quant_platform.config.models import ResearchCharter
from quant_platform.data.snapshots import SnapshotStore
from quant_platform.paper.state import PaperTradingState, ProductionModelReference
from quant_platform.pipeline import PipelineData, build_prediction_frame
from quant_platform.research.panels import (
    build_fold_panels,
    derive_fold_schedule,
    make_pit_cache_key,
    pipeline_data_from_cache,
)
from quant_platform.research.pit_cache import PitPanelCache, fold_schedule_hash
from quant_platform.research.results import ResultsReader
from quant_platform.utilities.narrow import as_date


class PaperDataError(RuntimeError):
    """Raised when production lineage or the forward data panel is invalid/stale."""


def production_predictions(
    state: PaperTradingState,
    charter: ResearchCharter,
    *,
    results_db: str | Path = "research.duckdb",
    snapshot_root: str | Path = "data/snapshots",
    cache_root: str | Path = "data/interim/pit_cache",
) -> tuple[ProductionModelReference, PipelineData, pl.DataFrame, dict[str, Any]]:
    """Load the designated run artifact and score the latest validated PIT cross-section."""
    reference = state.require_production_model(results_db)
    with ResultsReader(results_db) as reader:
        run = reader.query("research_run", run_id=reference.run_id, limit=1)
        snapshot_id = str(run["snapshot_id"][0])
        artifacts = reader.query(
            "artifact", run_id=reference.run_id, model=reference.model, limit=1000
        ).sort("fold", descending=True)
    if str(run["charter_hash"][0]) != config_hash(charter):
        raise PaperDataError(
            "current charter does not match the designated research run; paper scoring blocked"
        )
    snapshot = SnapshotStore(snapshot_root).load(snapshot_id)
    folds, calendar = derive_fold_schedule(snapshot, charter)
    schedule_hash = fold_schedule_hash(folds)
    if str(run["fold_schedule_hash"][0]) != schedule_hash:
        raise PaperDataError(
            "current fold policy does not match the designated research run; paper scoring blocked"
        )
    cache = PitPanelCache(cache_root)
    cached = cache.get_or_build(
        make_pit_cache_key(snapshot, charter, schedule_hash),
        lambda: build_fold_panels(snapshot, charter, folds),
    )
    data = pipeline_data_from_cache(cached, charter, calendar)
    panel = build_prediction_frame(data)
    if panel.is_empty():
        raise PaperDataError("latest validated production panel is empty")
    signal_date = _latest_executable_signal_date(panel, data, charter)
    latest = panel.filter(pl.col("as_of") == signal_date)
    if latest.is_empty():
        raise PaperDataError("latest validated production cross-section is empty")

    if _is_real_data_snapshot(snapshot) and (datetime.now(UTC).date() - signal_date).days > 7:
        raise PaperDataError(
            f"latest production data is stale ({signal_date}); proposal generation blocked"
        )

    fold = int(artifacts["fold"][0])
    if reference.model == "ensemble":
        predictions, metadata = _ensemble_predictions(
            reference.run_id,
            fold,
            latest,
            results_db,
        )
    else:
        predictions, metadata = _single_artifact_predictions(
            Path(str(artifacts["artifact_path"][0])), latest
        )
    return reference, data, predictions, metadata


def _is_real_data_snapshot(snapshot: Any) -> bool:
    """Named helper keeps the real-data staleness gate explicit and fixture-only bypass narrow."""
    return "fixture_synthetic" not in snapshot.manifest.get("sources", [])


def _latest_executable_signal_date(
    panel: pl.DataFrame, data: PipelineData, charter: ResearchCharter
) -> date:
    """Latest feature date whose delayed fill session exists in the admitted snapshot."""
    latest_price = as_date(data.prices["observation_date"].max(), "latest price date")
    for value in panel["as_of"].unique().sort(descending=True):
        candidate = as_date(value, "paper signal date")
        fill_date = data.calendar.shift(
            candidate, charter.backtest.rebalance.execution_delay_sessions
        )
        if fill_date is not None and fill_date <= latest_price:
            return candidate
    raise PaperDataError("no validated signal cross-section has a later admitted execution session")


def _single_artifact_predictions(
    artifact_path: Path, latest: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if not artifact_path.exists():
        raise PaperDataError(f"designated model artifact is missing: {artifact_path}")
    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not {"model", "preprocessor"}.issubset(payload):
        raise PaperDataError(f"artifact {artifact_path} lacks model/preprocessor state")
    model = payload["model"]
    transformed = payload["preprocessor"].transform(latest)
    predictions = model.predict(transformed)
    metadata = model.metadata.to_dict() if model.metadata else {}
    return predictions, metadata


def _ensemble_predictions(
    run_id: str,
    fold: int,
    latest: pl.DataFrame,
    results_db: str | Path,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    with ResultsReader(results_db) as reader:
        ensemble_rows = reader.query("artifact", run_id=run_id, model="ensemble", limit=1000)
        ensemble_row = ensemble_rows.filter(pl.col("fold") == fold)
        if ensemble_row.is_empty():
            raise PaperDataError("designated ensemble artifact is missing")
        payload = json.loads(Path(str(ensemble_row["artifact_path"][0])).read_text())
        weights = payload["weights"]
        member_frames: dict[str, pl.DataFrame] = {}
        for member in weights:
            rows = reader.query("artifact", run_id=run_id, model=member, limit=1000)
            row = rows.filter(pl.col("fold") == fold)
            if row.is_empty():
                raise PaperDataError(f"ensemble member artifact missing: {member}/fold-{fold}")
            member_frames[member], _metadata = _single_artifact_predictions(
                Path(str(row["artifact_path"][0])), latest
            )
    combined: pl.DataFrame | None = None
    for member, frame in sorted(member_frames.items()):
        ranked = frame.select(
            "security_id",
            "as_of",
            pl.col("cross_sectional_rank").alias(f"_rank_{member}"),
        )
        combined = (
            ranked
            if combined is None
            else combined.join(ranked, on=["security_id", "as_of"], how="inner")
        )
    if combined is None or combined.is_empty():
        raise PaperDataError("ensemble members produced no aligned production predictions")
    score = pl.lit(0.0)
    for member, weight in sorted(weights.items()):
        score += float(weight) * pl.col(f"_rank_{member}")
    predictions = (
        combined.with_columns(score.alias("prediction"))
        .select("security_id", "as_of", "prediction")
        .with_columns(
            (pl.col("prediction").rank("average").over("as_of") / pl.len().over("as_of")).alias(
                "cross_sectional_rank"
            )
        )
    )
    return predictions, payload.get("metadata", {})
