"""Feature registry (spec section 12).

Every feature declares its contract up front -- inputs, lookback, minimum observations,
missing-value policy, version -- so that:

* the pipeline can compute the warm-up period instead of silently emitting features from
  too little history (a 252-session momentum built from 30 bars is noise wearing a label);
* changing a formula forces a version bump, so stale cached features cannot silently mix
  with new ones;
* the universe's ``min_history_sessions`` can be checked against what features actually need.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import polars as pl


class FeatureFamily(StrEnum):
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VALUE = "value"
    QUALITY = "quality"
    INVESTMENT = "investment"
    RISK = "risk"
    LIQUIDITY = "liquidity"
    MACRO = "macro"


class MissingPolicy(StrEnum):
    """What to do when a feature cannot be computed."""

    DROP = "drop"  # exclude the observation entirely
    ZERO = "zero"  # neutral value after normalization
    CROSS_SECTIONAL_MEDIAN = "cross_sectional_median"  # impute from the same date's peers
    KEEP_NULL = "keep_null"  # let the model handle it (LightGBM can)


@dataclass(frozen=True)
class FeatureDefinition:
    """The declared contract of one feature."""

    name: str
    family: FeatureFamily
    required_inputs: tuple[str, ...]
    lookback_sessions: int
    min_observations: int
    description: str
    missing_policy: MissingPolicy = MissingPolicy.CROSS_SECTIONAL_MEDIAN
    requires_cross_sectional_normalization: bool = True
    update_frequency: str = "daily"
    version: str = "1.0.0"
    # Computed from a per-security frame sorted by date; returns one column.
    compute: Callable[..., pl.Expr] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.min_observations > self.lookback_sessions and self.lookback_sessions > 0:
            raise ValueError(
                f"feature '{self.name}' requires {self.min_observations} observations but "
                f"only looks back {self.lookback_sessions} sessions"
            )


class FeatureRegistry:
    """Registry of feature definitions, queryable by family."""

    def __init__(self) -> None:
        self._features: dict[str, FeatureDefinition] = {}

    def register(self, definition: FeatureDefinition) -> FeatureDefinition:
        if definition.name in self._features:
            raise ValueError(f"feature '{definition.name}' is already registered")
        self._features[definition.name] = definition
        return definition

    def get(self, name: str) -> FeatureDefinition:
        if name not in self._features:
            raise KeyError(f"unknown feature: {name}")
        return self._features[name]

    def names(self) -> list[str]:
        return sorted(self._features)

    def all(self) -> list[FeatureDefinition]:
        return [self._features[n] for n in self.names()]

    def by_family(self, family: FeatureFamily | str) -> list[FeatureDefinition]:
        fam = FeatureFamily(family)
        return [f for f in self.all() if f.family == fam]

    def families(self) -> list[str]:
        return sorted({f.family.value for f in self.all()})

    def for_families(self, families: list[str]) -> list[FeatureDefinition]:
        wanted = {FeatureFamily(f) for f in families}
        return [f for f in self.all() if f.family in wanted]

    def max_lookback(self, families: list[str] | None = None) -> int:
        """Longest lookback across selected features -- the pipeline's warm-up requirement."""
        defs = self.for_families(families) if families else self.all()
        return max((f.lookback_sessions for f in defs), default=0)

    def to_frame(self) -> pl.DataFrame:
        """Registry as a table, for documentation and reports."""
        return pl.DataFrame(
            [
                {
                    "name": f.name,
                    "family": f.family.value,
                    "lookback_sessions": f.lookback_sessions,
                    "min_observations": f.min_observations,
                    "missing_policy": f.missing_policy.value,
                    "normalize": f.requires_cross_sectional_normalization,
                    "version": f.version,
                    "description": f.description,
                }
                for f in self.all()
            ]
        )


# The global registry. Feature modules populate it on import.
REGISTRY = FeatureRegistry()


def register(**kwargs: Any) -> FeatureDefinition:
    """Convenience wrapper to build and register in one call."""
    return REGISTRY.register(FeatureDefinition(**kwargs))
