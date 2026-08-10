"""Validation: walk-forward splitting, purging, embargo, and overfitting statistics."""

from quant_platform.validation.walk_forward import (
    Fold,
    HoldoutViolationError,
    WalkForwardSplitter,
    apply_embargo,
    purge_overlapping_labels,
)

__all__ = [
    "Fold",
    "HoldoutViolationError",
    "WalkForwardSplitter",
    "apply_embargo",
    "purge_overlapping_labels",
]
