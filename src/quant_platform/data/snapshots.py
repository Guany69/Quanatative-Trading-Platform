"""Immutable point-in-time research snapshots.

A snapshot is the root of research identity.  Provider data is admitted only after the
existing quality gates pass, normalized to the three-date audit contract where applicable,
written as Parquet, and bound to a content-derived ``snapshot_id`` by a JSON manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_platform.data.validation.checks import validate_dataset


class SnapshotError(RuntimeError):
    """Base error for snapshot admission or integrity failures."""


class SurvivorshipBiasError(SnapshotError):
    """Raised when a biased membership source is not explicitly acknowledged."""


_SORT_KEYS: dict[str, list[str]] = {
    "prices": ["security_id", "observation_date"],
    "benchmark": ["observation_date"],
    "securities": ["security_id"],
    "identifiers": ["security_id", "start_date", "identifier_value"],
    "membership": ["security_id", "universe", "start_date"],
    "corporate_actions": ["security_id", "ex_date", "action_type"],
    "fundamentals": ["security_id", "observation_date", "available_at"],
    "macro": ["series_id", "observation_date", "available_at"],
}


@dataclass(frozen=True)
class DataSnapshot:
    """A verified immutable snapshot directory."""

    snapshot_id: str
    path: Path
    manifest: dict[str, Any]

    def read(self, name: str) -> pl.DataFrame:
        tables = self.manifest.get("tables", {})
        if name not in tables:
            return pl.DataFrame()
        return pl.read_parquet(self.path / tables[name]["file"])

    @property
    def table_names(self) -> list[str]:
        return sorted(self.manifest.get("tables", {}))


class SnapshotStore:
    """Sole writer and integrity checker for immutable snapshot directories."""

    manifest_version = 1

    def __init__(self, root: str | Path = "data/snapshots") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        frames: dict[str, pl.DataFrame],
        *,
        sources: list[str],
        provenance: dict[str, Any] | None = None,
        survivorship_biased: bool = False,
        acknowledge_survivorship_bias: bool = False,
    ) -> DataSnapshot:
        """Validate and persist one immutable content-addressed snapshot.

        ``survivorship_biased`` is deliberately an admission parameter rather than an
        informational label.  A biased source is refused unless the caller makes the
        acknowledgement explicit; accepted snapshots retain the caveat forever.
        """
        if survivorship_biased and not acknowledge_survivorship_bias:
            raise SurvivorshipBiasError(
                "refusing survivorship-biased membership source; pass an explicit "
                "acknowledgement to persist a permanently caveated snapshot"
            )

        normalized = {
            name: self._canonicalize(name, frame)
            for name, frame in frames.items()
            if frame is not None and not frame.is_empty()
        }
        required = {"prices", "benchmark", "securities", "membership", "corporate_actions"}
        missing = required - set(normalized)
        if missing:
            raise SnapshotError(f"snapshot is missing required tables: {sorted(missing)}")

        sessions = normalized["benchmark"]["observation_date"].to_list()
        report = validate_dataset(
            prices=normalized.get("prices"),
            benchmark=normalized.get("benchmark"),
            membership=normalized.get("membership"),
            fundamentals=normalized.get("fundamentals"),
            macro=normalized.get("macro"),
            corporate_actions=normalized.get("corporate_actions"),
            calendar_sessions=sessions,
        )
        report.raise_if_critical()

        table_meta: dict[str, dict[str, Any]] = {}
        for name, frame in sorted(normalized.items()):
            table_meta[name] = {
                "file": f"{name}.parquet",
                "rows": frame.height,
                "columns": frame.columns,
                "schema": {key: str(value) for key, value in frame.schema.items()},
                "content_hash": _frame_digest(frame),
                "date_range": _date_range(frame),
            }

        identity = {
            "manifest_version": self.manifest_version,
            "sources": sorted(set(sources)),
            "provenance": provenance or {},
            "survivorship_biased": survivorship_biased,
            "tables": {
                name: {
                    "content_hash": meta["content_hash"],
                    "rows": meta["rows"],
                    "schema": meta["schema"],
                }
                for name, meta in table_meta.items()
            },
        }
        snapshot_id = hashlib.sha256(_canonical_json(identity)).hexdigest()[:24]
        target = self.root / snapshot_id
        if target.exists():
            return self.load(snapshot_id)

        manifest = {
            **identity,
            "snapshot_id": snapshot_id,
            "created_at": datetime.now(UTC).isoformat(),
            "immutable": True,
            "survivorship_caveat": (
                "Historical membership may be survivorship-biased; results must retain this caveat."
                if survivorship_biased
                else None
            ),
            "validation": {
                "critical": len(report.critical),
                "warnings": len(report.warnings),
                "summary": report.summary(),
            },
            "tables": table_meta,
        }

        temp_dir = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=self.root))
        try:
            for name, frame in normalized.items():
                frame.write_parquet(temp_dir / f"{name}.parquet")
            (temp_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
            )
            try:
                os.replace(temp_dir, target)
            except OSError:
                if not target.exists():
                    raise
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        return self.load(snapshot_id)

    def load(self, snapshot_id: str, *, verify: bool = True) -> DataSnapshot:
        path = self.root / snapshot_id
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise SnapshotError(f"unknown snapshot_id '{snapshot_id}' in {self.root}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("snapshot_id") != snapshot_id or not manifest.get("immutable"):
            raise SnapshotError(f"invalid snapshot manifest at {manifest_path}")
        snapshot = DataSnapshot(snapshot_id, path, manifest)
        if verify:
            for name, meta in manifest.get("tables", {}).items():
                file_path = path / meta["file"]
                if not file_path.exists():
                    raise SnapshotError(f"snapshot {snapshot_id} is missing {file_path.name}")
                actual = _frame_digest(pl.read_parquet(file_path))
                if actual != meta["content_hash"]:
                    raise SnapshotError(
                        f"snapshot {snapshot_id} table {name} failed its content hash"
                    )
        return snapshot

    def find_by_provenance(
        self, *, sources: list[str], provenance: dict[str, Any]
    ) -> DataSnapshot | None:
        """Return an existing immutable snapshot for an identical provider request."""
        expected_sources = sorted(set(sources))
        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if (
                manifest.get("immutable") is True
                and manifest.get("sources") == expected_sources
                and manifest.get("provenance") == provenance
            ):
                return self.load(str(manifest["snapshot_id"]))
        return None

    @staticmethod
    def _canonicalize(name: str, frame: pl.DataFrame) -> pl.DataFrame:
        out = frame
        # Provider frames predate the explicit audit field in some adapters.  The platform
        # records deterministic synthetic ingestion at publication time; real adapters may
        # supply their actual ingestion timestamp.
        if {"observation_date", "available_at"}.issubset(out.columns) and "ingested_at" not in out:
            out = out.with_columns(pl.col("available_at").alias("ingested_at"))
        keys = [key for key in _SORT_KEYS.get(name, []) if key in out.columns]
        return out.sort(keys) if keys else out


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _frame_digest(frame: pl.DataFrame) -> str:
    hasher = hashlib.sha256()
    hasher.update(_canonical_json({key: str(value) for key, value in frame.schema.items()}))
    hasher.update(str(frame.shape).encode())
    if frame.height:
        hasher.update(frame.hash_rows(seed=0, seed_1=1, seed_2=2, seed_3=3).to_numpy().tobytes())
    return hasher.hexdigest()


def _date_range(frame: pl.DataFrame) -> dict[str, str] | None:
    for column in ("observation_date", "as_of", "start_date", "ex_date"):
        if column in frame.columns and frame.height:
            lo, hi = frame[column].min(), frame[column].max()
            return {"column": column, "start": str(lo), "end": str(hi)}
    return None
