"""Content-keyed point-in-time fold-panel cache.

This is deliberately not a provider download cache.  It materializes the exact
train/validation/test frames consumed by model tasks and refuses payloads that have not been
produced through the validated PIT builder.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_platform.utilities.narrow import as_date
from quant_platform.validation.walk_forward import Fold


class PitCacheError(RuntimeError):
    """Raised for unsafe, corrupt, or incomplete cached panels."""


@dataclass(frozen=True)
class PitCacheKey:
    snapshot_id: str
    universe_version: str
    feature_versions: dict[str, str]
    label_parameters: dict[str, Any]
    validation_parameters: dict[str, Any]
    purge_parameters: dict[str, Any]
    embargo_parameters: dict[str, Any]
    fold_schedule_hash: str
    contract_version: str = "1"

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(asdict(self))).hexdigest()[:32]


@dataclass(frozen=True)
class FoldPanelRef:
    fold: Fold
    train_path: Path
    validation_path: Path
    test_path: Path
    train_rows: int
    validation_rows: int
    test_rows: int


@dataclass(frozen=True)
class CachedFoldPanels:
    key: PitCacheKey
    path: Path
    panels: list[FoldPanelRef]
    auxiliary_paths: dict[str, Path]
    cache_hit: bool
    manifest: dict[str, Any]

    def read_auxiliary(self, name: str) -> pl.DataFrame:
        path = self.auxiliary_paths.get(name)
        return pl.read_parquet(path) if path else pl.DataFrame()


@dataclass
class BuiltFoldPanels:
    """Validated builder payload accepted by the cache manager."""

    panels: list[tuple[Fold, pl.DataFrame, pl.DataFrame, pl.DataFrame]]
    auxiliaries: dict[str, pl.DataFrame]
    pit_validated: bool
    provenance: dict[str, Any]


class PitPanelCache:
    """Sole writer for derived fold-ready panel materializations."""

    manifest_version = 1

    def __init__(self, root: str | Path = "data/interim/pit_cache") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_build(
        self,
        key: PitCacheKey,
        builder: Callable[[], BuiltFoldPanels],
    ) -> CachedFoldPanels:
        target = self.root / key.digest
        if (target / "manifest.json").exists():
            return self._load(key, target, cache_hit=True, increment_hit=True)

        built = builder()
        if not built.pit_validated:
            raise PitCacheError(
                "refusing to cache panels that did not pass the point-in-time validation path"
            )
        if built.provenance.get("builder") != "validated_snapshot_pipeline":
            raise PitCacheError("refusing panels without validated_snapshot_pipeline provenance")
        provenance_snapshot = built.provenance.get("snapshot_id")
        if provenance_snapshot is not None and provenance_snapshot != key.snapshot_id:
            raise PitCacheError("panel provenance snapshot does not match the cache key")
        if not built.panels:
            raise PitCacheError("refusing to cache an empty fold schedule")
        for fold, train, validation, test in built.panels:
            self._assert_fold_safe(fold, train, validation, test, key)

        temp_dir = Path(tempfile.mkdtemp(prefix=".pit-cache-", dir=self.root))
        files: dict[str, dict[str, Any]] = {}
        try:
            for fold, train, validation, test in built.panels:
                fold_dir = temp_dir / f"fold_{fold.index:03d}"
                fold_dir.mkdir(parents=True)
                for split, frame in (("train", train), ("validation", validation), ("test", test)):
                    relative = f"fold_{fold.index:03d}/{split}.parquet"
                    file_path = temp_dir / relative
                    frame.write_parquet(file_path)
                    files[relative] = {
                        "rows": frame.height,
                        "sha256": _file_digest(file_path),
                    }
            for name, frame in sorted(built.auxiliaries.items()):
                relative = f"aux/{name}.parquet"
                file_path = temp_dir / relative
                file_path.parent.mkdir(parents=True, exist_ok=True)
                frame.write_parquet(file_path)
                files[relative] = {"rows": frame.height, "sha256": _file_digest(file_path)}

            manifest = {
                "manifest_version": self.manifest_version,
                "cache_key": asdict(key),
                "cache_digest": key.digest,
                "created_at": datetime.now(UTC).isoformat(),
                "pit_validated": True,
                "provenance": built.provenance,
                "hit_count": 0,
                "last_hit_at": None,
                "folds": [asdict(fold) for fold, *_frames in built.panels],
                "files": files,
            }
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
        return self._load(key, target, cache_hit=False, increment_hit=False)

    def _load(
        self,
        key: PitCacheKey,
        target: Path,
        *,
        cache_hit: bool,
        increment_hit: bool,
    ) -> CachedFoldPanels:
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("cache_digest") != key.digest or manifest.get("cache_key") != json.loads(
            json.dumps(asdict(key), default=str)
        ):
            raise PitCacheError(f"cache manifest identity mismatch at {target}")
        if not manifest.get("pit_validated"):
            raise PitCacheError(f"unsafe cache entry at {target}: PIT validation marker absent")
        for relative, metadata in manifest.get("files", {}).items():
            file_path = target / relative
            if not file_path.exists() or _file_digest(file_path) != metadata["sha256"]:
                raise PitCacheError(f"cache file failed integrity check: {file_path}")

        panels: list[FoldPanelRef] = []
        for row in manifest.get("folds", []):
            fold = Fold(
                index=int(row["index"]),
                train_start=date.fromisoformat(row["train_start"]),
                train_end=date.fromisoformat(row["train_end"]),
                validation_start=date.fromisoformat(row["validation_start"]),
                validation_end=date.fromisoformat(row["validation_end"]),
                test_start=date.fromisoformat(row["test_start"]),
                test_end=date.fromisoformat(row["test_end"]),
            )
            prefix = f"fold_{fold.index:03d}"
            panels.append(
                FoldPanelRef(
                    fold=fold,
                    train_path=target / prefix / "train.parquet",
                    validation_path=target / prefix / "validation.parquet",
                    test_path=target / prefix / "test.parquet",
                    train_rows=int(manifest["files"][f"{prefix}/train.parquet"]["rows"]),
                    validation_rows=int(manifest["files"][f"{prefix}/validation.parquet"]["rows"]),
                    test_rows=int(manifest["files"][f"{prefix}/test.parquet"]["rows"]),
                )
            )
        auxiliary_paths = {
            Path(relative).stem: target / relative
            for relative in manifest.get("files", {})
            if relative.startswith("aux/")
        }

        if increment_hit:
            manifest["hit_count"] = int(manifest.get("hit_count", 0)) + 1
            manifest["last_hit_at"] = datetime.now(UTC).isoformat()
            temp_manifest = manifest_path.with_suffix(".json.tmp")
            temp_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
            )
            temp_manifest.replace(manifest_path)
        return CachedFoldPanels(key, target, panels, auxiliary_paths, cache_hit, manifest)

    @staticmethod
    def _assert_fold_safe(
        fold: Fold,
        train: pl.DataFrame,
        validation: pl.DataFrame,
        test: pl.DataFrame,
        key: PitCacheKey,
    ) -> None:
        if train.is_empty() or validation.is_empty() or test.is_empty():
            raise PitCacheError(f"fold {fold.index} contains an empty required split")
        train_minimum = as_date(train["as_of"].min(), "training minimum")
        train_maximum = as_date(train["as_of"].max(), "training maximum")
        if train_minimum < fold.train_start or train_maximum > fold.train_end:
            raise PitCacheError(f"fold {fold.index} training rows escape declared boundaries")
        for name, frame, start, end in (
            ("validation", validation, fold.validation_start, fold.validation_end),
            ("test", test, fold.test_start, fold.test_end),
        ):
            if "as_of" not in frame.columns:
                raise PitCacheError(f"fold {fold.index} {name} frame lacks as_of")
            minimum = as_date(frame["as_of"].min(), f"fold {fold.index} {name} minimum")
            maximum = as_date(frame["as_of"].max(), f"fold {fold.index} {name} maximum")
            if minimum < start or maximum > end:
                raise PitCacheError(f"fold {fold.index} {name} rows escape declared boundaries")
        if (
            key.purge_parameters.get("enabled", False)
            and "window_end" in train.columns
            and as_date(train["window_end"].max(), "training window end") >= fold.validation_start
        ):
            raise PitCacheError(f"fold {fold.index} contains overlapping training labels")
        if train_maximum >= fold.validation_start:
            raise PitCacheError(f"fold {fold.index} training rows cross validation start")


def fold_schedule_hash(folds: list[Fold]) -> str:
    return hashlib.sha256(_canonical_json([asdict(fold) for fold in folds])).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()
