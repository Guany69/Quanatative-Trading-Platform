"""Frontend-safe inventories derived from backend registries and configuration."""

from typing import Any

from quant_platform.api.config import ApiSettings
from quant_platform.api.schemas.common import json_safe
from quant_platform.api.schemas.metadata import (
    ConfigResponse,
    FeatureItem,
    InventoryItem,
    MetadataResponse,
)
from quant_platform.config import config_hash, load_charter
from quant_platform.features import compute as _compute  # noqa: F401
from quant_platform.features import fundamental as _fundamental  # noqa: F401
from quant_platform.features import macro as _macro  # noqa: F401
from quant_platform.features.registry import REGISTRY
from quant_platform.models.registry import available_models, production_candidates
from quant_platform.research.runner import TARGET_STRATEGIES

_SENSITIVE_PARTS = ("secret", "password", "token", "api_key", "apikey", "credential")

STRATEGY_DESCRIPTIONS = {
    "equal_weight": "Equal allocation across the highest-ranked eligible securities.",
    "score_weighted": "Allocation proportional to transformed model scores.",
    "inverse_volatility": "Rank-selected allocation scaled by trailing volatility.",
    "constrained_optimizer": "Cost-aware constrained optimization with visible relaxations.",
}


class MetadataService:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings

    def metadata(self) -> MetadataResponse:
        charter = load_charter(self.settings.charter_path)
        candidates = set(production_candidates())
        features = [
            FeatureItem(
                name=item.name,
                family=item.family.value,
                lookback_sessions=item.lookback_sessions,
                min_observations=item.min_observations,
                missing_policy=item.missing_policy.value,
                version=item.version,
                description=item.description,
            )
            for item in REGISTRY.all()
        ]
        return MetadataResponse(
            models=[
                InventoryItem(
                    id=name,
                    name=name.replace("_", " ").title(),
                    production_eligible=name in candidates,
                )
                for name in available_models()
            ],
            strategies=[
                InventoryItem(
                    id=name,
                    name=name.replace("_", " ").title(),
                    description=STRATEGY_DESCRIPTIONS.get(name),
                )
                for name in TARGET_STRATEGIES
            ],
            cost_scenarios=[
                InventoryItem(
                    id=name,
                    name=name.title(),
                    description=f"{multiplier:g}x configured transaction costs.",
                )
                for name, multiplier in charter.costs.scenario_multipliers.items()
            ],
            features=features,
            portfolio_models=available_models(),
            default_charter_id="default",
        )

    def config(self) -> ConfigResponse:
        charter = load_charter(self.settings.charter_path)
        return ConfigResponse(
            id="default",
            config_hash=config_hash(charter),
            config=_redact(charter.to_dict()),
        )


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if any(part in str(key).lower() for part in _SENSITIVE_PARTS)
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return json_safe(value)
