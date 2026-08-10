# Modeling

## Target

The primary target is **forward benchmark-relative excess return** over 20 trading sessions:

    y[i,t] = R[i, t->t+20] - R[benchmark, t->t+20]

Predicting excess return rather than price level matters. Absolute return is dominated by the
market factor, which is not a stock-selection skill: a model can look excellent simply by
being long during a bull market. Excess return isolates the part a stock picker controls.

Three target types share one interface (`LabelGenerator`):
1. numeric excess return
2. **cross-sectional percentile rank** (the primary portfolio signal)
3. probability of outperformance

The rank is what the portfolio consumes. Raw predictions are unstable in scale across regimes;
ranks are bounded, scale-free, and comparable over time.

## Features

28 price-derived features (momentum, reversal, risk, liquidity) plus fundamental (value,
quality, investment) and macro/regime families. Each declares its lookback, minimum
observations, missing-value policy, and version in the registry, so warm-up requirements are
computed rather than assumed and formula changes force a version bump.

## Preprocessing

Fitted on the **training window only**:
- winsorization bounds from training quantiles
- imputation from the same date's cross-section, falling back to training medians
- cross-sectional rank or robust z-score normalization
- missingness indicators (missingness is often informative)

Cross-sectional operations are inherently safe (they use only contemporaneous data); panel
statistics are the ones that must respect the train/test boundary.

## Model hierarchy

| Model | Role |
|---|---|
| No-skill mean | Floor. Predicts a constant; its ranking is arbitrary. |
| Momentum baseline | A single transparent anomaly, no fitting. |
| Factor composite | Weighted pillars from economic priors. Not fitted, so genuinely out-of-sample at every date. **The bar ML must clear.** |
| Elastic Net | Interpretable; coefficient stability across folds reveals refitting. |
| LightGBM | Primary nonlinear model; deliberately low capacity. |
| Random forest | **Diagnostic only** — never auto-promoted to production. |
| Neural network | Shallow MLP challenger (2-3 layers). |
| Ensemble | Constrained rank blend with caps, floors, and slow updates. |

## Validation

Expanding-window walk-forward with **purging** (drop training samples whose forward label
windows overlap the test period) and an **embargo** (drop the buffer immediately before the
test window, since serial correlation makes adjacent samples near-duplicates).

The locked holdout is structurally excluded from fold generation and requires an explicit
config flag plus acknowledgement to touch, because it can only be evaluated once.

## Known limitations

- Value and quality pillars are inert unless fundamentals are wired into the panel; the
  composite renormalizes across the pillars actually present rather than silently treating
  missing ones as neutral.
- Neural-network importances come from first-layer weights only — a rough attribution.
- Monotonic constraints are available but off by default; they impose a prior the data may
  not support.
