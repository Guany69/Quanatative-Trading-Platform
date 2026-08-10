# ADR 0006: LightGBM as the primary nonlinear model

**Status:** Accepted

## Context
Cross-sectional equity prediction has R^2 near zero, heavy collinearity, and fat tails.

## Decision
LightGBM, configured conservatively: shallow trees (depth <= 4), strong L1/L2, row and column
subsampling, early stopping on a genuine validation fold, and a **bounded** hyperparameter
search.

## Alternatives
- XGBoost/CatBoost: comparable; LightGBM is faster on wide panels.
- Deep networks: too little signal and too little data.
- Linear only: misses interactions, though ElasticNet is retained for interpretability.

## Consequences
- Library defaults would overfit badly, so every capacity knob is pinned down.
- Search bounds are themselves a bias control: an unrestricted sweep manufactures skill.
- Requires `libomp` on macOS (see ADR 0011).
