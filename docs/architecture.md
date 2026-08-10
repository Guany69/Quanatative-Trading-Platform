# Architecture

## Data flow

```text
Researcher -> CLI -> ExperimentRunOrchestrator (sole batch owner)
  -> SnapshotStore: validated immutable provider frames + hashes
  -> PIT builder -> PitPanelCache: content-keyed train/validation/test Parquet
  -> WalkForwardSplitter: expanding folds + purge + embargo + locked holdout exclusion
  -> TrainingExecutor: process-level (model, fold) tasks; train-only preprocessing
       -> ModelRegistry -> run/model/fold artifacts
       <- predictions + validation-safe ensemble inputs (workers never write DuckDB)
  -> four PortfolioConstructors -> shared risk/cost inputs -> BacktestEngine
  -> stress / DSR / PBO / placebo / regime evaluation
  -> ResultsStore: parent-process Arrow batches -> authoritative DuckDB
  -> ExperimentRegistry: append-only trial linkage, including rejected runs
  -> persisted-results reporter -> run-scoped HTML/MD/JSON/CSV/charts

CLI -> PaperTradingCoordinator
  -> designated completed run + immutable snapshot + stored model/preprocessor artifact
  -> shared PIT feature path -> strategy -> recorded proposals
  -> HUMAN APPROVAL/REJECTION -> >=1-session delayed SimulatedBroker only
  -> fills + unfilled orders + reconciliation -> atomic forward-only PaperTradingState
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
| `research/` | Run orchestration, process executor, PIT cache, DuckDB ownership, holdout. |

## Storage ownership

| Store | Role | Writer |
|---|---|---|
| `data/snapshots/<snapshot_id>/` | Permanent immutable Parquet + manifest root | `SnapshotStore` |
| `data/interim/pit_cache/<key>/` | Disposable validated fold panels | `PitPanelCache` |
| `research.duckdb` | Authoritative runs, predictions, metrics, strategies and trades | Parent `ResultsStore` |
| `artifacts/models/<run>/<model>/<fold>/` | Reloadable model/preprocessor and metadata | Orchestrator |
| `artifacts/experiments.jsonl` | Append-only attempted/rejected trial history | `ExperimentRegistry` |
| `paper_state/state.json` | Separate forward paper decisions/fills/reconciliation | `PaperTradingState` |
| `reports/<run>/` | Regenerable presentation exports, never authority | Persisted reporter |

Workers do not receive or import `ResultsStore`; the owning parent writes complete logical
stages through Arrow-backed transactions. Failed runs remain queryable with stage and cause.

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
