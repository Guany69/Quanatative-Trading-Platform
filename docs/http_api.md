# Local HTTP API and React workstation

## Boundary and dependency direction

The HTTP layer extends the modular monolith; it is not a second quantitative engine:

```text
React -> HTTP/JSON -> routes/DTOs -> application services
      -> existing orchestrators/domain components -> existing stores
```

React never reads DuckDB, snapshots, Parquet, report directories, artifacts, or paper JSON.
The API has no arbitrary-SQL endpoint and accepts identifiers rather than filesystem paths.
Model training, PIT construction, portfolio construction, cost calculation, simulated fills,
approval enforcement, and holdout enforcement remain in Python.

## Starting the application

Install Python and frontend dependencies once:

```bash
uv sync --all-extras
cd frontend && npm ci && cd ..
```

Development uses Vite's same-machine proxy:

```bash
# terminal 1, repository root
uv run quant-platform-api

# terminal 2
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. The backend defaults to `127.0.0.1:8000`; it does not enable a
permissive CORS policy or bind to all interfaces. Configuration uses `QUANT_PLATFORM_`-prefixed
environment variables, such as `QUANT_PLATFORM_PORT=8123`.

For a local production-style single origin:

```bash
cd frontend
npm ci
npm run build
cd ..
uv run quant-platform-api
```

Open `http://127.0.0.1:8000`. FastAPI serves the compiled SPA and keeps `/api/*` reserved for
the application boundary. Swagger UI is `/api/docs`, ReDoc is `/api/redoc`, and the generated
contract is `/api/openapi.json`. Refresh TypeScript definitions with `npm run generate:api`
while the API is running on its default port.

## Research queue and DuckDB ownership

`POST /api/runs` validates a registry-backed request, durably records it in
`.quant-platform/research-jobs.json`, and returns `202 QUEUED` without running research in the
request thread. Exactly one FIFO worker calls the existing `ExperimentRunOrchestrator`.
Workers used by model/fold computation return data to that owning process and never write
DuckDB.

The job manager has one re-entrant writer lock shared by research persistence and locked
holdout evaluation. On API startup it requeues durable `QUEUED` jobs and records interrupted
`RUNNING` jobs as `FAILED` with `STALE_RUN_RECOVERY`; stale DuckDB runs are recovered only at
this lifecycle boundary. Real stage transitions are persisted as `INITIALIZATION`,
`PIT_CACHE`, `MODEL_TRAINING`, `PORTFOLIO_SIMULATION`, `OVERFITTING_STRESS`, and `REPORTING`.
The UI polls only a run's identity while it is non-terminal and does not fabricate progress.

## Paper mutation and safety guarantees

`PaperService` serializes each authoritative paper load/mutate/save cycle under a process-local
lock. Paper state remains separate from replayable research state and is atomically replaced on
disk. A production model must be explicitly designated from a completed run. Rebalancing first
creates proposals. Each proposal must then receive an explicit `APPROVE` or `REJECT` decision;
execution is refused while any proposal is undecided and sends only the approved subset to the
existing `SimulatedBroker` approval gate. Fills and reconciliation are persisted as forward
audit records.

The holdout endpoint requires deliberate acknowledgement and calls the existing one-shot
holdout service while holding the DuckDB writer lock. The browser confirmation is usability,
not authority: the backend continues to reject missing acknowledgement and repeat consumption.

## Endpoint inventory

All result lists that can grow are bounded or paginated; filters are implemented as
parameterized purpose-specific queries.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Process health. |
| GET | `/api/meta` | Model, strategy, cost, feature, and portfolio inventories. |
| GET | `/api/config` | Redacted resolved configuration and identity. |
| GET | `/api/features` | Feature registry metadata. |
| GET | `/api/snapshots` | Safe snapshot summaries. |
| GET | `/api/snapshots/{snapshot_id}` | One server-owned snapshot manifest. |
| POST | `/api/runs` | Validate and enqueue an asynchronous research run. |
| GET | `/api/runs` | Paginated run history with optional state filter. |
| GET | `/api/runs/{run_id}` | Run state, stage, provenance, and failure. |
| GET | `/api/runs/{run_id}/summary` | Result counts, cache state, and timings. |
| GET | `/api/runs/{run_id}/folds` | Paginated walk-forward boundaries. |
| GET | `/api/runs/{run_id}/model-metrics` | Filtered model/fold metrics. |
| GET | `/api/runs/{run_id}/predictions` | Paginated, filtered predictions. |
| GET | `/api/runs/{run_id}/strategy-metrics` | Filtered strategy metrics. |
| GET | `/api/runs/{run_id}/equity` | Paginated, filtered equity series. |
| GET | `/api/runs/{run_id}/trades` | Paginated, filtered trade ledger. |
| GET | `/api/runs/{run_id}/weights` | Paginated, filtered target weights. |
| GET | `/api/runs/{run_id}/optimizer-diagnostics` | Constraint and relaxation audit. |
| GET | `/api/runs/{run_id}/overfitting` | Persisted overfitting/stress values. |
| GET | `/api/runs/{run_id}/artifacts` | Opaque artifact availability metadata. |
| POST | `/api/runs/{run_id}/holdout-evaluations` | Audited one-shot holdout evaluation. |
| GET | `/api/runs/{run_id}/reports` | Run-owned report inventory with opaque IDs. |
| GET | `/api/runs/{run_id}/reports/{report_id}` | Safely serve one verified report. |
| GET | `/api/paper` | Authoritative paper account summary. |
| POST | `/api/paper/init` | Explicitly initialize the paper account. |
| GET | `/api/paper/production-model` | Current explicit production designation. |
| POST | `/api/paper/production-model` | Designate a completed run/model. |
| POST | `/api/paper/rebalances` | Generate proposals without approving or executing. |
| GET | `/api/paper/rebalances` | Rebalance audit records. |
| GET | `/api/paper/proposals` | Proposal inbox, optionally filtered by status. |
| POST | `/api/paper/proposals/{order_id}/decision` | Persist an explicit human decision. |
| POST | `/api/paper/rebalances/{rebalance_id}/execute` | Execute only an all-decided approved subset. |
| GET | `/api/paper/positions` | Authoritative positions. |
| GET | `/api/paper/fills` | Fill audit records. |
| GET | `/api/paper/reconciliations` | Reconciliation audit records. |
| GET | `/api/paper/history` | Consolidated forward audit history. |

Errors use `{ "error": { "code", "message", "details" } }`; validation errors are normalized
to the same envelope. Non-finite numeric results are serialized as JSON `null`.

## Frontend routes

| Route | View |
|---|---|
| `/dashboard` | Health, queue, recent runs, failures, and paper summary. |
| `/research/new` | Registry-driven governed run form. |
| `/research/runs` | Filterable, sortable, paginated run history. |
| `/research/runs/:runId` | Progress, persisted result tabs, reports, and holdout action. |
| `/paper` | Account, positions, model designation, and proposal generation. |
| `/paper/proposals` | Human decisions and explicitly gated execution. |
| `/paper/history` | Decisions, rebalances, fills, and designations. |
| `/scorecard` | Link to the intentionally retained legacy descriptive scorecard. |

The legacy scorecard remains separate because it is auxiliary and already owns its analysis
logic. No scorecard computation was duplicated in TypeScript.
