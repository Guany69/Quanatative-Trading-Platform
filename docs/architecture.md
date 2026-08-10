# Architecture

## Data flow

```
providers (fixture | local files | open data)
  -> data validation           (fail fast on CRITICAL issues)
  -> security master           (permanent IDs, symbol history, delistings)
  -> point-in-time universe    (membership + eligibility as of each date)
  -> features                  (trailing windows only; available_at gating)
  -> labels                    (20-session forward excess return + delay)
  -> walk-forward split        (purge overlapping windows + embargo)
  -> preprocessing             (FITTED ON TRAIN ONLY)
  -> models                    (baselines + ML + ensemble)
  -> predictions               (cross-sectional ranks)
  -> portfolio construction    (equal / score / inverse-vol / optimizer)
  -> backtest engine           (execution delay, costs, corporate actions)
  -> evaluation                (forecast + portfolio metrics)
  -> stress + overfitting stats
  -> reports                   (JSON, CSV, Markdown, HTML, charts)
  -> paper trading             (proposals -> HUMAN APPROVAL -> simulated fills)
```

## The five timestamps

The backtest keeps these strictly separated. Collapsing them is the single most common source
of fake alpha:

| Timestamp | Meaning |
|---|---|
| information | when the data became knowable (`available_at`) |
| signal | when the model formed a view |
| order | when the order was submitted (>= signal + delay) |
| fill | when it executed, at that session's price |
| accounting | when it hit the books |

## Module boundaries

| Module | Responsibility |
|---|---|
| `domain/` | Frozen, validated records. Rejects impossible data at construction. |
| `config/` | Strongly validated YAML; rejects incoherent mandates at load time. |
| `data/` | Provider-neutral adapters, validation, security master. |
| `universe/` | Point-in-time eligibility resolution. |
| `features/` | Registry + computation + train-only preprocessing. |
| `labels/` | Leakage-safe target generation with delisting handling. |
| `models/` | Common interface across baselines, ML, and the ensemble. |
| `validation/` | Walk-forward, purging, embargo, PSR/DSR/PBO, stress suite. |
| `risk/` | Shrinkage covariance, beta, exposures, concentration. |
| `costs/` | Composable commission / spread / slippage / impact. |
| `portfolio/` | Constructors and the CVXPY optimizer with logged relaxation. |
| `backtest/` | Event-aware engine with full accounting. |
| `evaluation/` | Metrics, factor diagnostics, reports. |
| `execution/` + `paper/` | Broker interface, approval gate, persistent paper state. |
| `experiments/` | Append-only trial registry (feeds multiple-testing corrections). |

## Point-in-time handling

Every time-dependent record carries three distinct dates: `observation_date` (the period it
describes), `available_at` (when it became knowable), and `ingested_at` (audit only). Features
at time *t* may only consume records with `available_at <= t`, enforced through as-of joins
rather than plain joins on fiscal period.

## Extension points

- **New data provider**: implement the Protocols in `data/base.py` and declare a
  `SourceProvenance` stating what it can and cannot guarantee.
- **New feature**: register a `FeatureDefinition` (declares lookback, missing policy, version)
  and add its computation.
- **New model**: subclass `BaseModel`, implement `_fit`/`_predict`, register it.
- **New portfolio method**: subclass `PortfolioConstructor`; hard constraints are verified
  centrally by `_verify`.
- **New broker**: implement `BrokerAdapter`. The approval gate applies regardless.
