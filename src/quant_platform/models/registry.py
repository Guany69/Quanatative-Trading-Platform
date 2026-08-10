"""Model registry: name -> constructor (spec section 14).

Keeps the CLI's `--model X` and the config files decoupled from import paths, and keeps the
set of production-eligible models explicit. `RandomForestModel` is registered but flagged
non-production, so `production_candidates()` will not return it.
"""

from __future__ import annotations

from typing import Any

from quant_platform.models.base import BaseModel
from quant_platform.models.baselines import (
    FactorComposite,
    MomentumBaseline,
    NoSkillModel,
)
from quant_platform.models.ensemble import RankEnsemble
from quant_platform.models.linear import ElasticNetModel, LogisticModel
from quant_platform.models.neural_network import NeuralNetworkModel
from quant_platform.models.trees import LightGBMModel, RandomForestModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "no_skill": NoSkillModel,
    "momentum_baseline": MomentumBaseline,
    "factor_composite": FactorComposite,
    "elastic_net": ElasticNetModel,
    "logistic": LogisticModel,
    "lightgbm": LightGBMModel,
    "random_forest": RandomForestModel,
    "neural_network": NeuralNetworkModel,
    "ensemble": RankEnsemble,
}

# Models that exist purely as controls. They must appear in every comparison so a complex
# model has to prove it beats something trivial.
BASELINE_MODELS = frozenset({"no_skill", "momentum_baseline", "factor_composite"})


def create_model(name: str, **kwargs: Any) -> BaseModel:
    """Instantiate a model by registry name."""
    key = name.replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)


def production_candidates() -> list[str]:
    """Models eligible to become the production signal.

    Excludes anything flagged `is_production_candidate = False` (the random-forest
    diagnostic), per spec section 14.5.
    """
    return sorted(
        name
        for name, cls in MODEL_REGISTRY.items()
        if getattr(cls, "is_production_candidate", True)
    )


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)
