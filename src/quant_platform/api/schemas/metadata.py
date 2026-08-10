from __future__ import annotations

from typing import Any

from pydantic import Field

from quant_platform.api.schemas.common import ApiModel


class InventoryItem(ApiModel):
    id: str
    name: str
    description: str | None = None
    production_eligible: bool | None = None


class FeatureItem(ApiModel):
    name: str
    family: str
    lookback_sessions: int
    min_observations: int
    missing_policy: str
    version: str
    description: str


class MetadataResponse(ApiModel):
    models: list[InventoryItem]
    strategies: list[InventoryItem]
    cost_scenarios: list[InventoryItem]
    features: list[FeatureItem]
    portfolio_models: list[str]
    default_charter_id: str


class ConfigResponse(ApiModel):
    id: str
    config_hash: str
    config: dict[str, Any]


class SnapshotSummary(ApiModel):
    snapshot_id: str
    created_at: str
    sources: list[str]
    survivorship_biased: bool
    table_count: int
    validation: dict[str, Any]


class SnapshotDetail(SnapshotSummary):
    provenance: dict[str, Any]
    tables: dict[str, Any]
    survivorship_caveat: str | None = None


class FeatureListResponse(ApiModel):
    items: list[FeatureItem] = Field(default_factory=list)
