"""Shared snapshot-to-cache path used by research and forward paper scoring."""

from __future__ import annotations

import hashlib
import json

import polars as pl

from quant_platform.config.models import ResearchCharter
from quant_platform.data.snapshots import DataSnapshot
from quant_platform.domain.enums import BenchmarkSource
from quant_platform.features.registry import REGISTRY
from quant_platform.pipeline import PipelineData, build_training_frame, load_snapshot_data
from quant_platform.research.pit_cache import (
    BuiltFoldPanels,
    CachedFoldPanels,
    PitCacheKey,
)
from quant_platform.utilities.calendar import TradingCalendar
from quant_platform.utilities.narrow import as_date
from quant_platform.validation.walk_forward import Fold, WalkForwardSplitter


def derive_fold_schedule(
    snapshot: DataSnapshot, charter: ResearchCharter
) -> tuple[list[Fold], TradingCalendar]:
    benchmark = snapshot.read("benchmark")
    start = as_date(benchmark["observation_date"].min(), "snapshot start")
    end = as_date(benchmark["observation_date"].max(), "snapshot end")
    calendar = TradingCalendar(start, end)
    signal_dates = calendar.rebalance_sessions(
        charter.backtest.rebalance.frequency,
        charter.backtest.rebalance.signal_day,
    )
    usable_end = calendar.shift(
        signal_dates[-1],
        -(
            charter.label.forecast_horizon_sessions
            + charter.backtest.rebalance.execution_delay_sessions
        ),
    )
    if usable_end is None:
        raise ValueError("snapshot cannot support the configured label horizon")
    panel_end = max(signal for signal in signal_dates if signal <= usable_end)
    return (
        WalkForwardSplitter(charter.validation, calendar).generate_folds(
            signal_dates[0], panel_end
        ),
        calendar,
    )


def make_pit_cache_key(
    snapshot: DataSnapshot, charter: ResearchCharter, schedule_hash: str
) -> PitCacheKey:
    universe_payload = json.dumps(
        charter.universe.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return PitCacheKey(
        snapshot_id=snapshot.snapshot_id,
        universe_version=hashlib.sha256(universe_payload.encode()).hexdigest()[:16],
        feature_versions={
            "registry": charter.features.feature_version,
            **{definition.name: definition.version for definition in REGISTRY.all()},
        },
        label_parameters=charter.label.model_dump(mode="json"),
        validation_parameters=charter.validation.model_dump(mode="json"),
        purge_parameters={
            "enabled": charter.validation.purge_enabled,
            "label_horizon": charter.label.forecast_horizon_sessions,
        },
        embargo_parameters={"sessions": charter.validation.embargo_sessions},
        fold_schedule_hash=schedule_hash,
    )


def build_fold_panels(
    snapshot: DataSnapshot, charter: ResearchCharter, folds: list[Fold]
) -> BuiltFoldPanels:
    data = load_snapshot_data(snapshot, charter)
    panel = build_training_frame(data)
    splitter = WalkForwardSplitter(charter.validation, data.calendar)
    materialized = []
    for fold in folds:
        train, validation, test = splitter.split(panel, fold)
        splitter.assert_holdout_untouched(train)
        splitter.assert_holdout_untouched(validation)
        splitter.assert_holdout_untouched(test)
        materialized.append((fold, train, validation, test))
    return BuiltFoldPanels(
        panels=materialized,
        auxiliaries={
            "prices": data.prices,
            "benchmark": data.benchmark,
            "securities": data.securities,
            "corporate_actions": data.corporate_actions,
            "features": data.features,
            "labels": data.labels,
            "universe_panel": data.universe_panel,
            "identifiers": data.identifiers,
            "membership": data.membership,
            "fundamentals": data.fundamentals,
            "macro": data.macro,
        },
        pit_validated=True,
        provenance={
            "snapshot_id": snapshot.snapshot_id,
            "survivorship_biased": snapshot.manifest.get("survivorship_biased", False),
            "is_synthetic": data.is_synthetic,
            "benchmark_source": data.benchmark_source.value,
            "builder": "validated_snapshot_pipeline",
        },
    )


def pipeline_data_from_cache(
    cached: CachedFoldPanels, charter: ResearchCharter, calendar: TradingCalendar
) -> PipelineData:
    frames = {name: cached.read_auxiliary(name) for name in cached.auxiliary_paths}
    features = frames["features"]
    provenance = cached.manifest.get("provenance", {})
    is_synthetic = bool(provenance.get("is_synthetic", False))
    benchmark_source = provenance.get("benchmark_source")
    return PipelineData(
        prices=frames["prices"],
        benchmark=frames["benchmark"],
        securities=frames["securities"],
        corporate_actions=frames["corporate_actions"],
        features=features,
        labels=frames["labels"],
        universe_panel=frames["universe_panel"],
        calendar=calendar,
        is_synthetic=is_synthetic,
        feature_columns=[
            column for column in features.columns if column not in {"security_id", "as_of"}
        ],
        identifiers=frames.get("identifiers", pl.DataFrame()),
        membership=frames.get("membership", pl.DataFrame()),
        fundamentals=frames.get("fundamentals", pl.DataFrame()),
        macro=frames.get("macro", pl.DataFrame()),
        snapshot_id=cached.key.snapshot_id,
        benchmark_source=(
            BenchmarkSource(benchmark_source)
            if benchmark_source
            else (
                BenchmarkSource.FIXTURE_SYNTHETIC
                if is_synthetic
                else charter.benchmark.preferred_source
            )
        ),
    )
