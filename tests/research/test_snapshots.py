from __future__ import annotations

import polars as pl
import pytest

from quant_platform.data.adapters.fixture import FixtureDataProvider, FixtureSpec
from quant_platform.data.snapshots import SnapshotStore, SurvivorshipBiasError


def _frames():
    dataset = FixtureDataProvider(FixtureSpec(n_securities=30, seed=17)).generate()
    return {
        "prices": dataset.prices,
        "benchmark": dataset.benchmark,
        "securities": dataset.securities,
        "identifiers": dataset.identifiers,
        "membership": dataset.membership,
        "corporate_actions": dataset.corporate_actions,
        "fundamentals": dataset.fundamentals,
        "macro": dataset.macro,
    }


def test_snapshot_identity_is_content_stable_and_immutable(tmp_path):
    store = SnapshotStore(tmp_path)
    frames = _frames()
    first = store.create(frames, sources=["fixture_synthetic"], provenance={"seed": 17})
    second = store.create(
        {name: frame.reverse() for name, frame in frames.items()},
        sources=["fixture_synthetic"],
        provenance={"seed": 17},
    )

    assert first.snapshot_id == second.snapshot_id
    assert first.path == second.path
    assert first.manifest["immutable"] is True
    assert store.load(first.snapshot_id).read("prices").equals(first.read("prices"))
    assert store.find_by_provenance(sources=["fixture_synthetic"], provenance={"seed": 17}) == first

    changed = dict(frames)
    changed["prices"] = (
        frames["prices"]
        .with_row_index("_row")
        .with_columns(
            pl.when(pl.col("_row") == 0)
            .then(pl.col("close") + 0.01)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        .drop("_row")
    )
    third = store.create(changed, sources=["fixture_synthetic"], provenance={"seed": 17})
    assert third.snapshot_id != first.snapshot_id


def test_survivorship_biased_source_requires_explicit_acknowledgement(tmp_path):
    store = SnapshotStore(tmp_path)
    with pytest.raises(SurvivorshipBiasError, match="refusing survivorship-biased"):
        store.create(_frames(), sources=["current_membership"], survivorship_biased=True)

    snapshot = store.create(
        _frames(),
        sources=["current_membership"],
        survivorship_biased=True,
        acknowledge_survivorship_bias=True,
    )
    assert snapshot.manifest["survivorship_biased"] is True
    assert snapshot.manifest["survivorship_caveat"]
