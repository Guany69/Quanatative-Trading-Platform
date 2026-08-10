from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import date

import polars as pl
import pytest

from quant_platform.research.pit_cache import (
    BuiltFoldPanels,
    PitCacheError,
    PitCacheKey,
    PitPanelCache,
)
from quant_platform.validation.walk_forward import Fold


def _key() -> PitCacheKey:
    return PitCacheKey(
        snapshot_id="snapshot-a",
        universe_version="universe-1",
        feature_versions={"price": "1"},
        label_parameters={"horizon": 20, "version": "1"},
        validation_parameters={"min_train_years": 5},
        purge_parameters={"enabled": True, "horizon": 20},
        embargo_parameters={"sessions": 5},
        fold_schedule_hash="folds-a",
    )


def _built(*, safe: bool = True) -> BuiltFoldPanels:
    fold = Fold(
        index=0,
        train_start=date(2020, 1, 1),
        train_end=date(2020, 1, 10),
        validation_start=date(2020, 2, 1),
        validation_end=date(2020, 2, 10),
        test_start=date(2020, 3, 1),
        test_end=date(2020, 3, 10),
    )
    train = pl.DataFrame(
        {"security_id": ["A"], "as_of": [date(2020, 1, 1)], "window_end": [date(2020, 1, 20)]}
    )
    validation = pl.DataFrame({"security_id": ["A"], "as_of": [date(2020, 2, 3)]})
    test = pl.DataFrame({"security_id": ["A"], "as_of": [date(2020, 3, 2)]})
    return BuiltFoldPanels(
        panels=[(fold, train, validation, test)],
        auxiliaries={"features": train.drop("window_end")},
        pit_validated=safe,
        provenance={"builder": "validated_snapshot_pipeline", "snapshot_id": "snapshot-a"},
    )


def test_cache_hits_only_for_identical_correctness_identity(tmp_path):
    cache = PitPanelCache(tmp_path)
    calls = 0

    def build() -> BuiltFoldPanels:
        nonlocal calls
        calls += 1
        return _built()

    first = cache.get_or_build(_key(), build)
    second = cache.get_or_build(_key(), build)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == 1

    variants = [
        replace(_key(), snapshot_id="snapshot-b"),
        replace(_key(), universe_version="universe-2"),
        replace(_key(), feature_versions={"price": "2"}),
        replace(_key(), label_parameters={"horizon": 21, "version": "1"}),
        replace(_key(), purge_parameters={"enabled": False, "horizon": 20}),
        replace(_key(), embargo_parameters={"sessions": 6}),
        replace(_key(), fold_schedule_hash="folds-b"),
    ]
    assert len({key.digest for key in [_key(), *variants]}) == len(variants) + 1


def test_cache_refuses_payload_outside_validated_pit_builder(tmp_path):
    with pytest.raises(PitCacheError, match="did not pass"):
        PitPanelCache(tmp_path).get_or_build(_key(), lambda: _built(safe=False))


def test_cache_deletion_changes_only_rebuild_work(tmp_path):
    cache = PitPanelCache(tmp_path)
    first = cache.get_or_build(_key(), _built)
    expected = pl.read_parquet(first.panels[0].test_path)
    shutil.rmtree(first.path)

    rebuilt = cache.get_or_build(_key(), _built)
    assert rebuilt.cache_hit is False
    assert pl.read_parquet(rebuilt.panels[0].test_path).equals(expected)
