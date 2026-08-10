import re

from fastapi import APIRouter, Depends

from quant_platform.api.config import ApiSettings
from quant_platform.api.dependencies import settings
from quant_platform.api.errors import ApiError
from quant_platform.api.schemas.common import json_safe
from quant_platform.api.schemas.metadata import SnapshotDetail, SnapshotSummary
from quant_platform.data.snapshots import DataSnapshot, SnapshotError, SnapshotStore

router = APIRouter(tags=["snapshots"])
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _summary(snapshot: DataSnapshot) -> SnapshotSummary:
    manifest = snapshot.manifest
    return SnapshotSummary(
        snapshot_id=snapshot.snapshot_id,
        created_at=manifest.get("created_at", ""),
        sources=list(manifest.get("sources") or []),
        survivorship_biased=bool(manifest.get("survivorship_biased")),
        table_count=len(manifest.get("tables") or {}),
        validation=json_safe(manifest.get("validation") or {}),
    )


@router.get("/snapshots", response_model=list[SnapshotSummary])
def list_snapshots(config: ApiSettings = Depends(settings)) -> list[SnapshotSummary]:
    return [_summary(snapshot) for snapshot in SnapshotStore(config.snapshot_root).list()]


@router.get("/snapshots/{snapshot_id}", response_model=SnapshotDetail)
def get_snapshot(snapshot_id: str, config: ApiSettings = Depends(settings)) -> SnapshotDetail:
    if not _SAFE_ID.fullmatch(snapshot_id):
        raise ApiError("SNAPSHOT_NOT_FOUND", "Unknown snapshot identifier.", status_code=404)
    try:
        snapshot = SnapshotStore(config.snapshot_root).load(snapshot_id, verify=False)
    except SnapshotError as exc:
        raise ApiError("SNAPSHOT_NOT_FOUND", str(exc), status_code=404) from exc
    summary = _summary(snapshot)
    return SnapshotDetail(
        **summary.model_dump(),
        provenance=json_safe(snapshot.manifest.get("provenance") or {}),
        tables=json_safe(snapshot.manifest.get("tables") or {}),
        survivorship_caveat=snapshot.manifest.get("survivorship_caveat"),
    )
