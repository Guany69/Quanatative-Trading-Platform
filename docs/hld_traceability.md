# Target HLD requirement ledger

Source: `/Users/gyansaxsna/Downloads/Final_HLD.pdf`, version 1.0, 27 pages, SHA-256
`61fdbde5ba65993e83727a4d8d85e0ee8883866b3d1b4956032f9459072d7d08`. The complete
document and Figure 5-1 were reviewed before implementation and again after implementation.
This ledger is the implementation control document for the migration.

Legend: current state is `SATISFIED`, `PARTIAL`, `MISSING`, `CONFLICTING`, or `OBSOLETE`.
Action is `REUSE`, `MODIFY`, `REPLACE`, `ADD`, or `REMOVE`.

| ID | HLD section | Exact requirement / constraint | Components; current -> target files | Current state / action | Implementation status | Verification / acceptance evidence |
|---|---|---|---|---|---|---|
| HLD-001 | Purpose, 1.1 | Local single-researcher U.S. equity research and paper-trading platform; 20-session benchmark-relative ranking; no real-money decisions. | Whole system; package and docs remain a modular local application. | SATISFIED / REUSE | VERIFIED EXISTING | Package metadata, charter, README, no live broker implementation. |
| HLD-002 | Purpose, 1.1 | Target stack roles: Python 3.12, Polars, pandas, NumPy, scikit-learn, LightGBM, PyTorch, CVXPY, DuckDB. | `pyproject.toml`, model/data/portfolio/results modules. | PARTIAL / MODIFY | IMPLEMENTED | Dependency inspection; library-specific model and store tests. |
| HLD-003 | 1.2, 5.1 | Modular monolith staged pipeline with batch research and forward paper orchestration planes over shared storage. | Existing modular packages + `paper/*`; add `research/*`. | PARTIAL / ADD | IMPLEMENTED | Dependency-direction review and end-to-end tests. |
| HLD-004 | 1.3, 14.3 | No services, brokers, distributed compute, external cache, GPU requirement, hosted API, or multi-user serving. | Whole repository. | SATISFIED / REUSE | VERIFIED EXISTING | Architecture/import audit; Nautilus seam remains non-functional. |
| HLD-005 | 1.4 | Consume OHLCV, benchmark, corporate actions/delistings, security metadata, PIT membership, optional fundamentals/macro/factors, charter, approvals. | `data/*`, `config/*`, `execution/*`; snapshot admission path. | PARTIAL / MODIFY | IMPLEMENTED | Snapshot manifest and integration tests. |
| HLD-006 | 1.4 | Produce the full model comparison with predictions and fold metrics. | Registry exists; add executor, orchestrator, DB. | PARTIAL / ADD | IMPLEMENTED | Roster/executor/results tests and research-run smoke. |
| HLD-007 | 1.4 | Produce all-four strategy comparison, gross/net, scenarios, curves, trades, relaxations. | Three constructors wired today; optimizer exists but is not registered in common path. | CONFLICTING / MODIFY | IMPLEMENTED | Four-strategy fan-out and persistence tests. |
| HLD-008 | 1.4 | Produce DSR, PBO, placebo, cost and regime stress using trial history. | Statistics exist but have no governed caller. | PARTIAL / MODIFY | IMPLEMENTED | Automatic evaluation/results tests. |
| HLD-009 | 1.4, 11.3 | HTML, Markdown, CSV, JSON and charts are run-scoped regenerable reports, not authority. | `evaluation/report.py`; reports currently consume in-memory/demo results. | PARTIAL / MODIFY | IMPLEMENTED | Regenerate report from DuckDB test. |
| HLD-010 | 1.4 | Persist proposals, decisions, simulated fills, reconciliation, account state, and audit history. | `paper/*` persists state but clears proposal details and does not retain fill/reconciliation ledgers. | PARTIAL / MODIFY | IMPLEMENTED | Reload/audit-history tests. |
| HLD-011 | 1.4 | Persist run_id, snapshot, config/code/seeds/folds/models/status and trial linkage. | `RunMetadata` is report-file metadata only. | MISSING / ADD | IMPLEMENTED | Results schema and run-lifecycle tests. |
| HLD-012 | 2 | Preserve Researcher and architecturally distinct Trade Approver roles; no decision means no execution. | CLI and approval fields/gate. | SATISFIED / REUSE | VERIFIED EXISTING | `tests/execution`, new persisted-decision test. |
| HLD-013 | UC-1 | Validated immutable PIT snapshot (Parquet + manifest + id, sources/ranges/provenance/hashes); critical validation refuses it. | Validation exists; no snapshot store. Add `data/snapshots.py`. | MISSING / ADD | IMPLEMENTED | Snapshot immutability/hash/refusal tests. |
| HLD-014 | UC-1 | Survivorship-biased membership refused by default; acknowledged fallback permanently caveated. | Universe config/builder enforce policy; no permanent snapshot caveat. | PARTIAL / MODIFY | IMPLEMENTED | Snapshot admission/caveat tests. |
| HLD-015 | UC-2 | Governed research run resolves charter/snapshot/cache/folds, all requested model-fold work, persistence, artifacts and trial linkage. | Serial CLI path only. Add runner. | MISSING / ADD | IMPLEMENTED | End-to-end research-run integration. |
| HLD-016 | UC-2, 12 | All required model-fold tasks complete; holdout/infeasible schedule/worker failure sets FAILED with cause; partial output never appears complete. | Current walk-forward skips empty folds and has no run state. | CONFLICTING / REPLACE | IMPLEMENTED | Failure/retry/run-status tests. |
| HLD-017 | UC-3 | Same prediction set/calendar/prices/delay/cost scenario fans to four constructors; infeasible variants fail loudly and independently. | Demo shares inputs for three variants; optimizer disconnected. | PARTIAL / MODIFY | IMPLEMENTED | Fairness and independent-failure tests. |
| HLD-018 | UC-4 | DSR/PBO use complete append-only true trial count including rejected configurations; gaps mark statistics optimistic. | Registry counts rejected trials but is unused and has no run linkage/gap check. | PARTIAL / MODIFY | IMPLEMENTED | Registry linkage/gap/optimistic-flag tests. |
| HLD-019 | UC-5 | Holdout structurally excluded; explicit acknowledgement; one-shot evaluation permanently recorded; unauthorized access raises domain error. | Splitter gates access; CLI only prints acknowledgement and records nothing. | PARTIAL / MODIFY | IMPLEMENTED | Unique holdout-record and violation tests. |
| HLD-020 | UC-6 | Paper proposal refreshes same PIT/validation path, loads explicit production model, constructs/persists PROPOSED orders, then stops. | Demo rebuilds fixture path and guesses factor composite; stages proposals. | CONFLICTING / MODIFY | IMPLEMENTED | Missing/invalid designation, stale data, propose-only tests. |
| HLD-021 | UC-7 | Only explicitly approved proposals reach simulated broker; later-session price; fills/reconciliation/state; unfilled orders reported. | Approval gate/reconciliation exist; broker does not itself reject same-session timing. | PARTIAL / MODIFY | IMPLEMENTED | Approval, delayed-fill, unfilled/reload tests. |
| HLD-022 | UC-8 | Query persisted results across runs/models/folds/strategies/dates/scenarios; unknown/empty explicit; no rerun. | No DB or query interface. | MISSING / ADD | IMPLEMENTED | ResultsReader/CLI/export tests. |
| HLD-023 | 4 | Permanent security identity, symbol history, listing/delisting referenced across panels/results/trades. | Security master and IDs exist. | SATISFIED / REUSE | VERIFIED EXISTING | Domain/security tests; persisted IDs inspection. |
| HLD-024 | 4, bias 1 | Three-date PIT contract: observation_date, available_at, ingested_at; use only available_at <= as-of. | Domain contract has all three; adapter frames often omit ingested_at. | PARTIAL / MODIFY | IMPLEMENTED | Snapshot normalization and PIT contract tests. |
| HLD-025 | 4 | Snapshot is first-class immutable root of cache/run/reproducibility identity. | Only lightweight `dataset_snapshot_id`. | MISSING / ADD | IMPLEMENTED | Store identity and foreign-reference tests. |
| HLD-026 | 4 | Interval-based historical universe membership answers eligibility as of t. | `universe/builder.py`. | SATISFIED / REUSE | VERIFIED EXISTING | `test_universe_pit.py`. |
| HLD-027 | 4 | Versioned features; every output-sensitive version/config enters cache identity; PIT-safe as-of construction. | Feature versions and joins exist; no consumed cache identity. | PARTIAL / MODIFY | IMPLEMENTED | Cache invalidation matrix and availability tests. |
| HLD-028 | 4 | 20-session benchmark-relative excess label is execution-delay and delisting aware; horizon enters purge/cache. | Label generator implements semantics. | SATISFIED / REUSE | VERIFIED EXISTING | Label/integration/delisting tests plus cache-key test. |
| HLD-029 | 4 | Fold contains expanding train/validation/test, purge, embargo; fundamental training/prediction/artifact/cache unit. | Fold/splitter exist. | SATISFIED / REUSE | VERIFIED EXISTING | Splitter and new fold-persistence tests. |
| HLD-030 | 4, 11 | Cached fold panel is first-class, deterministic, safe, train/validation/test Parquet + manifest. | No implementation. | MISSING / ADD | IMPLEMENTED | `research/pit_cache.py` tests. |
| HLD-031 | 4, 7 | Model metadata/artifact includes identity, schema/version, windows, seed, hyperparameters, provenance; trained per fold. | Base metadata exists; flat artifact path and no fold/run fields. | PARTIAL / MODIFY | IMPLEMENTED | Artifact layout/reload/metadata tests. |
| HLD-032 | 4, 11 | Persist prediction fields run_id/model/fold/as_of/security_id/score/rank. | Predictions transient/CSV. | MISSING / ADD | IMPLEMENTED | DB schema/write/query tests. |
| HLD-033 | 4, 9 | Target weights include mandate/date/weights/verification/relaxations and feed simulation/paper. | Constructors return weights/diagnostics, not persisted. | PARTIAL / MODIFY | IMPLEMENTED | Target/diagnostic persistence tests. |
| HLD-034 | 4, 10 | Trade records retain five timestamps, security/side/shares/prices and cost components. | Domain order chronology exists; backtest TradeRecord stores only three dates. | PARTIAL / MODIFY | IMPLEMENTED | Timestamp chain/schema tests. |
| HLD-035 | 4, 12 | ResearchRun is persistent parent for folds/predictions/metrics/strategies/trades/artifacts/trial. | Missing. | MISSING / ADD | IMPLEMENTED | Relational schema/lifecycle tests. |
| HLD-036 | 4, 11 | ExperimentTrial is permanent append-only, every tried/rejected config, full provenance and run_id. | JSONL registry exists, no run_id/automatic calls. | PARTIAL / MODIFY | IMPLEMENTED | Automatic success/failure entries; rejection count. |
| HLD-037 | 4, 10 | Proposal records PROPOSED/APPROVED/REJECTED plus approver identity/time and expected cost. | Runtime object has fields; persisted pending representation loses approval metadata/status history. | PARTIAL / MODIFY | IMPLEMENTED | State serialization/transition tests. |
| HLD-038 | 4, 10, 12 | Paper state owns positions, cash, proposals, decisions, fills, rebalances, reconciliation and production model reference; survives restarts. | Positions/cash/rebalances exist; other histories/designation missing. | PARTIAL / MODIFY | IMPLEMENTED | Reload/audit and designation tests. |
| HLD-039 | 4 | Named commission/spread/slippage/impact cost scenarios are comparison dimensions. | Cost model/config exists. | SATISFIED / REUSE | VERIFIED EXISTING | Cost scenario and strategy persistence tests. |
| HLD-040 | 5.2 | Research CLI remains single governed entry point; loopback scorecard remains auxiliary/outside correctness boundary; no HTTP API added. | Typer + auxiliary web UI exist. | SATISFIED / MODIFY | IMPLEMENTED | CLI command inventory/smoke tests. |
| HLD-041 | 5.2 | Orchestrator sequences only; financial/model formulas remain domain modules; sole results-store writer. | No orchestrator. | MISSING / ADD | IMPLEMENTED | Dependency review and worker/store isolation tests. |
| HLD-042 | 5.2, 12 | Parallel executor uses process-level `(model, fold)` tasks, deterministic seeds, train-only preprocessing, returns values; retry once then fail with identity. | No executor. | MISSING / ADD | IMPLEMENTED | Serial/parallel equivalence, seed, retry tests. |
| HLD-043 | 5.2, 14 | Models remain internally single-threaded/deterministic; run/model/fold seed independent of scheduling. | Trees/torch use one thread; CLI uses one common seed. | PARTIAL / MODIFY | IMPLEMENTED | Deterministic seed derivation and equivalence tests. |
| HLD-044 | 5.2, 14 | Replace per-security adjustment loops with vectorized Polars joins/window/cumulative operations and regression equivalence. | `pipeline.build_adjusted_prices` loops groups/actions in Python. | CONFLICTING / REPLACE | IMPLEMENTED | Existing adjustment suite + reference-equivalence fixture. |
| HLD-045 | 5.2, 11 | Cache key includes snapshot, universe version, feature versions, label params/version, validation, purge, embargo, fold schedule hash. | Missing. | MISSING / ADD | IMPLEMENTED | One-change-at-a-time invalidation tests. |
| HLD-046 | 5.2, 11 | Cache is sole-writer, safe/disposable; deleting changes runtime only, never outputs; unsafe panels cannot enter. | Missing. | MISSING / ADD | IMPLEMENTED | Safety marker, deletion equivalence tests. |
| HLD-047 | 5.2, bias 5 | Preprocessor fits TRAIN ONLY inside isolated task; validation/test only transform; no fitted state shared. | Serial helper fits train only. | PARTIAL / MODIFY | IMPLEMENTED | Existing leakage tests + task isolation test. |
| HLD-048 | 7 | All registry names share the fit/predict/schema/provenance contract. | Exact registry exists. | SATISFIED / REUSE | VERIFIED EXISTING | Registry roster/common-contract tests. |
| HLD-049 | 7 | `no_skill` baseline/control runs through governed comparison. | Implemented/registered. | SATISFIED / REUSE | VERIFIED EXISTING | Roster and executor tests. |
| HLD-050 | 7 | `momentum_baseline` rule model runs through governed comparison. | Implemented/registered. | SATISFIED / REUSE | VERIFIED EXISTING | Roster and executor tests. |
| HLD-051 | 7 | `factor_composite` multi-factor rule model runs through governed comparison. | Implemented/registered. | SATISFIED / REUSE | VERIFIED EXISTING | Roster and executor tests. |
| HLD-052 | 7 | `elastic_net` uses scikit-learn per fold. | Implemented/registered. | SATISFIED / REUSE | VERIFIED EXISTING | Library/roster/executor tests. |
| HLD-053 | 7 | `logistic` uses scikit-learn per fold and probability ranking. | Implemented/registered. | SATISFIED / REUSE | VERIFIED EXISTING | Library/roster/executor tests. |
| HLD-054 | 7 | `lightgbm` uses LightGBM deterministically with internal single-threading. | Implemented/registered, `n_jobs=1`. | SATISFIED / REUSE | VERIFIED EXISTING | Library/seed/executor tests. |
| HLD-055 | 7 | `random_forest` diagnostic uses scikit-learn and remains non-production. | Implemented/registered/non-production. | SATISFIED / REUSE | VERIFIED EXISTING | Registry production-candidate test. |
| HLD-056 | 7 | `neural_network` is real seeded CPU PyTorch MLP. | Implemented/registered. | SATISFIED / REUSE | VERIFIED EXISTING | PyTorch/module/executor test. |
| HLD-057 | 7 | `ensemble` combines member ranks; weights fit on validation only; parallel member outputs consumed deterministically. | Implementation has validation guard but registry creates it with no members and it has no caller. | CONFLICTING / MODIFY | IMPLEMENTED | Holdout refusal and parallel-output ensemble tests. |
| HLD-058 | 7 | Future artifacts at `artifacts/models/<run>/<model>/<fold>/model + metadata`; reloadable where supported. | Flat single-model writes. | CONFLICTING / REPLACE | IMPLEMENTED | Path, metadata and reload tests. |
| HLD-059 | 8.1 | Structural look-ahead control: available_at gates, as-of joins, trailing features; safe cache only. | Core joins/features exist; cache absent. | PARTIAL / MODIFY | IMPLEMENTED | PIT join and cache-admission tests. |
| HLD-060 | 8.2 | Structural survivorship control: historical intervals/default refusal/permanent caveat. | Refusal/intervals exist; caveat not snapshot-persistent. | PARTIAL / MODIFY | IMPLEMENTED | Universe + snapshot tests. |
| HLD-061 | 8.3 | Structural delisting control: terminal events/returns including -100%; optimization retains event boundaries. | Labels/backtest handle delistings. | SATISFIED / REUSE | VERIFIED EXISTING | Corporate-action/accounting tests. |
| HLD-062 | 8.4 | Structural same-bar control: >=1 session, later price, five timestamps, historical and paper paths. | Backtest delay exists; paper broker trusts proposal dates and demo can fall back to same date. | PARTIAL / MODIFY | IMPLEMENTED | Same-session broker refusal and chronology tests. |
| HLD-063 | 8.5 | Structural preprocessing-leakage control. | Train-only serial helper exists. | PARTIAL / MODIFY | IMPLEMENTED | Executor-local fit/state-isolation tests. |
| HLD-064 | 8.6 | Structural overlapping-label control: horizon purge + embargo, both in cache identity. | Splitter exists; cache absent. | PARTIAL / MODIFY | IMPLEMENTED | Split/cache policy tests. |
| HLD-065 | 8.7 | Structural overfitting control: DSR/PBO/placebo/locked holdout/true automatic trial history. | Individual implementations exist but are disconnected/manual. | PARTIAL / MODIFY | IMPLEMENTED | Governed run and gap tests. |
| HLD-066 | 8.8 | Structural accidental-execution control at broker boundary, including approved/by/at. | Broker gate and fields exist. | SATISFIED / MODIFY | IMPLEMENTED | Existing execution tests + persisted audit. |
| HLD-067 | 9.1 | Constructors consume only persisted model scores/ranks, never raw features/labels. | Current constructor API uses predictions. | SATISFIED / REUSE | VERIFIED EXISTING | Fan-out/dependency test. |
| HLD-068 | 9.2 | `equal_weight` registered and compared. | Implemented/registered. | SATISFIED / REUSE | VERIFIED EXISTING | Four-strategy test. |
| HLD-069 | 9.2 | `score_weighted` supports configured score transformation. | Rank-linear implementation, no transform config. | PARTIAL / MODIFY | IMPLEMENTED | Transform parameter tests. |
| HLD-070 | 9.2 | `inverse_volatility` registered and supplied trailing volatility. | Implemented and joined by pipeline. | SATISFIED / REUSE | VERIFIED EXISTING | Four-strategy test. |
| HLD-071 | 9.2 | `constrained_optimizer` uses CVXPY/risk/cost/constraints and logged configurable relaxation; prohibited relaxation fails loudly. | Optimizer exists, common registry omits it, always relaxes/falls back. | CONFLICTING / MODIFY | IMPLEMENTED | Registration, diagnostics, prohibited-relaxation tests. |
| HLD-072 | 9.3 | Shared verification enforces long-only, holdings, cap, sector, beta, turnover/budget where applicable. | `_verify` covers only long-only/cap/budget; optimizer covers remaining constraints internally. | PARTIAL / MODIFY | IMPLEMENTED | Shared invariants and optimizer tests. |
| HLD-073 | 9.3 | Persist target history, gross/net curve, trade log, turnover/costs/metrics/relaxations keyed by run/model/strategy/scenario. | Generated transient/report files. | MISSING / ADD | IMPLEMENTED | DB schema and comparison query tests. |
| HLD-074 | 9.3 | Historical book is transient; only paper account is durable authoritative portfolio state. | Current architecture follows this. | SATISFIED / REUSE | VERIFIED EXISTING | Storage ownership audit. |
| HLD-075 | 10.1 | Exact paper lifecycle PROPOSE -> APPROVE/REJECT -> EXECUTE APPROVED -> RECONCILE -> RECORD. | Runtime flow exists; explicit rejection/persistent stage histories incomplete. | PARTIAL / MODIFY | IMPLEMENTED | State transition and reload tests. |
| HLD-076 | 10.1 | Never sell more than held; expected costs; approved-but-unfilled retained and warned. | Proposal/broker/reconcile implement behavior. | SATISFIED / REUSE | VERIFIED EXISTING | Paper execution tests. |
| HLD-077 | 10.2, 12 | Explicit recorded production designation `(run_id, model)`; missing/invalid reference blocks proposals; never guess. | Missing; demo hard-codes factor composite. | CONFLICTING / ADD | IMPLEMENTED | Designation validation/blocking tests. |
| HLD-078 | 10.2 | Paper uses same PIT validation/cache/model/portfolio/cost semantics as research. | Shares model/portfolio/cost helpers but not snapshot/cache and validation governance. | PARTIAL / MODIFY | IMPLEMENTED | Coordinator/shared-path integration test. |
| HLD-079 | 11.2 | DuckDB is authoritative research system of record; CSV only export; analytical cross-run joins; Polars/pandas/Arrow compatible. | DuckDB optional and unused. | MISSING / ADD | IMPLEMENTED | ResultsStore schema/query/export/report tests. |
| HLD-080 | 11.2 | Results schema includes research_run, fold, prediction, fold_metric, strategy_result and trade logical fields. | Missing. | MISSING / ADD | IMPLEMENTED | Schema introspection and round-trip tests. |
| HLD-081 | 11.2 | Orchestrator is sole DuckDB writer; workers never mutate shared stores. | Missing. | MISSING / ADD | IMPLEMENTED | PID/owner guard and executor API tests. |
| HLD-082 | 11.3 | Snapshots permanent partitioned Parquet; cache disposable Parquet/JSON; artifacts filesystem; registry JSONL; paper JSON separate; provider caches retained. | Partial stores exist. | PARTIAL / MODIFY | IMPLEMENTED | Storage-map/owner tests and docs. |
| HLD-083 | 11.3 | Reports live at `reports/<run_id>/` and regenerate from authoritative results. | Fixed `reports/demo`, `verify`, etc. | CONFLICTING / REPLACE | IMPLEMENTED | Run-scoped persisted report smoke. |
| HLD-084 | 12 | RUNNING -> COMPLETED/FAILED only; cause stored; stale RUNNING recovered as failed; status tells truth. | Missing persistent run state. | MISSING / ADD | IMPLEMENTED | Lifecycle/stale recovery/failure tests. |
| HLD-085 | 12 | Result buffers owned/flushed by orchestrator per stage; crashes do not corrupt stores. | Missing. | MISSING / ADD | IMPLEMENTED | Transaction/stage-write tests. |
| HLD-086 | 12 | Cache owner writes content key once; key changes invalidate; deletion only recomputes. | Missing. | MISSING / ADD | IMPLEMENTED | Cache lifecycle tests. |
| HLD-087 | 13.2 | Add `research-run` with charter/snapshot/models/strategies/scenarios and governed outputs/failures. | Missing. | MISSING / ADD | IMPLEMENTED | Typer runner smoke/failure tests. |
| HLD-088 | 13.2 | Modify `walk-forward` into thin diagnostic over governed runner; no duplicate engine. | Serial duplicate path writes CSV. | CONFLICTING / REPLACE | IMPLEMENTED | CLI routing test. |
| HLD-089 | 13.2 | Add `results` filters/safe SQL/CSV export with explicit unknown/empty behavior. | Missing. | MISSING / ADD | IMPLEMENTED | CLI query/export tests. |
| HLD-090 | 13.2 | Preserve targeted backtest/stress/holdout, data, paper, introspection and auxiliary commands with actionable nonzero errors. | Commands exist. | SATISFIED / MODIFY | IMPLEMENTED | CLI inventory and smoke tests. |
| HLD-091 | 13.3 | Narrow owner interfaces: cache get/build, ResultsStore writes/query, TrainingExecutor map; no generic row DAL. | Missing. | MISSING / ADD | IMPLEMENTED | API/import review. |
| HLD-092 | 14 | Cache-hit research begins at fold training; every correctness change misses. | No fold-panel cache. | MISSING / ADD | IMPLEMENTED | Cold/warm benchmark and invalidation tests. |
| HLD-093 | 14 | Process pool parallelizes up to model x fold, read-only cached panels, deterministic equivalent results. | Serial only. | MISSING / ADD | IMPLEMENTED | Parallel scheduling/equivalence benchmark. |
| HLD-094 | 14 | Price/data generation is Polars-vectorized and regression-equivalent. | Features vectorized; adjustments loop. | PARTIAL / REPLACE | IMPLEMENTED | Adjustment equivalence and timing benchmark. |
| HLD-095 | 14 | Simulation pre-indexes matrices, vectorizes inter-event valuation, shares prepared inputs, retains exact rebalance/action/delisting/fill events and accounting. | Dict/session hot path, no shared prepared input. | PARTIAL / MODIFY | IMPLEMENTED | Reference equivalence/accounting/event tests. |
| HLD-096 | 14 | Reproducible benchmark reports cold/warm/build/train/sim/total/cache/parallelism/memory where practical; no fabricated 42->12 claim. | No harness. | MISSING / ADD | IMPLEMENTED | Benchmark command and recorded local timings. |
| HLD-097 | 15 | Absorb `pipeline.py`/`demo.py` sequencing into runner while retaining fixture demo/legacy usefulness. | Sequencing duplicated. | CONFLICTING / MODIFY | IMPLEMENTED | CLI routing and existing integration tests. |
| HLD-098 | 15-16 | Supersede write-only `data/processed` feature/label outputs with consumed fold cache; preserve CSV as export. | CLI writes processed files that no path reads. | OBSOLETE / REPLACE | IMPLEMENTED | Import/caller search and cache use test. |
| HLD-099 | 15-16 | Remove empty unreferenced `monitoring/__init__.py` only if still unused. | Empty, no imports. | OBSOLETE / REMOVE | IMPLEMENTED | `rg` import audit. |
| HLD-100 | 16 | Generated historical report/model files do not drive architecture; future output run-scoped; caches ignored; retain setup scripts. | Generated files present but source paths do not import them. | PARTIAL / MODIFY | IMPLEMENTED | `.gitignore`, caller audit, new output paths. |
| HLD-101 | 5.3, Fig 5-1 | Every store has one writing owner; no generic repository-wide DAL; dependencies follow CLI -> orchestrators -> domain/store owners. | Existing stores narrow; new research stores absent. | PARTIAL / ADD | IMPLEMENTED | Ownership/dependency audit. |
| HLD-102 | Fig 5-1 | Principal arrows: providers->builder/cache; runner->splitter/executor/models; predictions->portfolio/risk/cost->engine; writer->DB/query; paper->approval->broker->state. | Most domain arrows exist; governed runner/cache/DB arrows absent. | PARTIAL / ADD | IMPLEMENTED | End-to-end call-path trace. |
| HLD-103 | Failures | Critical data, unsafe membership, unauthorized holdout, task failure, infeasible no-relax strategy, unapproved order, unfilled order, unknown run, invalid production reference and registry gaps surface explicitly. | Mixed: several domain refusals exist; run/query/designation/gap failures absent. | PARTIAL / MODIFY | IMPLEMENTED | Dedicated unsafe-path tests. |
| HLD-104 | Testing | Add target coverage for data, cache, folds, preprocessing, nine models, executor, store, four portfolios, simulation, paper, overfitting, run state and CLI. | Strong legacy tests; target-plane tests absent. | PARTIAL / ADD | IMPLEMENTED | New test modules plus full suite. |
| HLD-105 | Acceptance | Run formatter, linter, configured type checks, full tests, integration, CLI smoke, research fixture, paper fixture and benchmark; report actual outcomes. | Baseline pytest passes; target not implemented. | PARTIAL / MODIFY | IMPLEMENTED | Exact final command log. |
| HLD-106 | Completion audit | Second page-1-through-27 audit plus must/required/transition/failure/field/path/arrow/model/strategy/command/performance searches. | Not yet applicable. | MISSING / ADD | IMPLEMENTED | Final ledger dispositions and audit notes. |

## Baseline evidence before edits

- Existing full suite: passed on 2026-08-10.
- Registered model names: all HLD names plus the experimental temporal convolution model exist.
- Governed research-plane search: no `research` package, ResultsStore, DuckDB schema,
  process-pool executor, `research-run`, `results`, or benchmark harness exists.
- Current `walk-forward`: one model, serial folds, CSV output, skips empty folds.
- Current demo: three models and three portfolio variants.
- `ConstrainedOptimizer` exists but `get_constructor` registers only three strategies.
- `ExperimentRegistry`, DSR and PBO have no runtime callers outside tests.
- `monitoring/__init__.py` is empty and unreferenced.

## Completion-audit evidence

- Second pass: all 27 PDF pages reread with PDFKit after implementation; pages 1-27 and the
  rendered Figure 5-1 relationship graph were checked against the final repository.
- Independent searches: requirements/“must”, ADD/MODIFY/REPLACE/REMOVE transitions, failure
  behavior, persistence fields, state transitions, named paths, models, strategies, commands,
  eight bias controls, and four performance mechanisms were reconciled to the 106 rows above.
- Audit fixes: paper now consumes the shared content-keyed PIT cache and blocks charter/fold
  policy mismatch; forward scoring is label-free; PBO consumes persisted experiment-history
  curves; legacy data commands consume the PIT cache by default; legacy training writes only
  run-scoped artifacts; artifact metadata carries all fold windows.
- Final local fixture benchmark (`80` securities, all ten models, six folds, all four
  strategies, base costs, two workers): cold run `cf371530de2a49cd` completed in `154.291s`
  with a `2.559s` panel build; warm run `d0f6d9e9895e43a4` completed in `131.920s` with a
  `0.048s` cache load. Peak process RSS was `1,057,439,744` bytes on macOS. This is not the
  HLD reference workload or hardware and does not claim
  to reproduce the illustrative 42-to-12-minute comparison.
- Final target smoke after the shared-path refactor: runs `58ee694efc804d8d` (cold) and
  `570939ba8d1b43eb` (warm) both `COMPLETED`; PBO details recorded two historical
  configurations across two runs.
- Paper fixture: designated `cf371530de2a49cd:factor_composite`; 75 proposals; zero fills
  before approval; 75 delayed simulated fills after named approval; reconciliation and model
  designation survived state reload.
