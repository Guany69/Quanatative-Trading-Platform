# quant-platform

A quantitative U.S. equity research and paper-trading platform. It predicts **forward
benchmark-relative excess return** over a 20-session horizon and trades the **cross-sectional
ranking** of those predictions.

> ## Read this first
>
> The bundled demo runs on **randomly generated synthetic data**. Its results measure whether
> the *software* works — they say nothing about markets and carry no investment meaning.
>
> This platform makes **no claim** that any strategy is profitable or will outperform the
> S&P 500. All historical results are **simulated**. Simulated performance does not guarantee
> future results. This is a research tool, not investment advice, and nothing here is
> validated for live capital.

## Purpose

Most backtests are wrong in the same handful of ways: they peek at the future, they only
include companies that survived, they never book the bankruptcies, they trade at prices that
generated the signal, and they report the best of hundreds of attempts without saying so. This
platform is built around preventing those specific failures, and the test suite exists to
prove the preventions actually work.

## What is enforced (not just documented)

| Bias | Control | Verified by |
|---|---|---|
| Look-ahead | `available_at` gating; as-of joins; trailing windows only | `tests/feature_availability/`, `tests/data_leakage/` |
| Survivorship | true membership intervals; biased data **refused** by default | `tests/data_leakage/test_universe_pit.py` |
| Delisting | terminal returns booked (a bankruptcy costs -100%) | `tests/corporate_actions/`, `tests/portfolio_accounting/` |
| Execution | fills happen >= 1 session after the signal | `tests/portfolio_accounting/` |
| Scaler leakage | preprocessing fitted on training data only | `tests/data_leakage/test_preprocessing_leakage.py` |
| Label overlap | purging + embargo around every fold | `tests/integration/`, `validation/walk_forward.py` |
| Overfitting | Deflated Sharpe, PBO, placebo tests | `validation/overfitting.py`, `validation/stress.py` |
| Accidental trading | orders require explicit human approval | `tests/execution/` |

The strongest single check is the **placebo test**: with the planted signal removed, the
pipeline scores IC ~= 0. A pipeline that leaks would find "signal" in pure noise.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra ml --extra optimization --extra data --extra reporting --extra dev

# macOS only, and only if LightGBM fails to import (no Homebrew required):
uv run python scripts/fix_macos_libomp.py

uv run quant-platform validate-environment
```

## Quick start — governed research

Runs end to end with **no credentials and no network access**:

```bash
uv run quant-platform research-run --models all --strategies all --workers 2
```

This creates or resolves an immutable data snapshot, validates and content-caches PIT fold
panels, trains all nine models in model/fold worker processes, compares four portfolio
constructors on the same inputs, persists the complete run to `research.duckdb`, links the
append-only experiment registry, and generates run-scoped reports. `run-demo` remains as a
smaller legacy presentation workflow; it is not the research system of record.

## Local web workstation

The browser application is a typed adapter over the same research orchestrator, results
store, and paper coordinator used by the CLI. It binds to loopback by default.

```bash
# terminal 1: API at http://127.0.0.1:8000 (interactive docs at /api/docs)
uv run quant-platform-api

# terminal 2: Vite development server at http://127.0.0.1:5173
cd frontend
npm ci
npm run dev
```

For a single-origin local build, run `npm run build` in `frontend/`, return to the repository
root, and start `uv run quant-platform-api`. FastAPI then serves both `frontend/dist` and
`/api/*` at `http://127.0.0.1:8000`. See [docs/http_api.md](docs/http_api.md) for queue,
storage, mutation, and route semantics.

## Commands

```bash
# environment and inspection
uv run quant-platform validate-environment
uv run quant-platform inspect-reference-catalog
uv run quant-platform show-config
uv run quant-platform show-features
uv run quant-platform list-models

# data
uv run quant-platform bootstrap-demo-data --securities 120
uv run quant-platform validate-data --strict
uv run quant-platform build-universe
uv run quant-platform build-features
uv run quant-platform build-labels

# modeling
uv run quant-platform research-run --models all --strategies all --workers 2
uv run quant-platform results --table research_run
uv run quant-platform results --run-id RUN_ID --table fold_metric --export metrics.csv
uv run quant-platform train --model factor-composite
uv run quant-platform train --model elastic-net
uv run quant-platform train --model lightgbm
uv run quant-platform train --model neural-network
uv run quant-platform walk-forward --model lightgbm

# evaluation
uv run quant-platform backtest --model lightgbm --portfolio equal_weight --cost double
uv run quant-platform stress-test
uv run quant-platform generate-report
uv run quant-platform evaluate-holdout --run-id RUN_ID --acknowledged-by NAME \
  --i-understand-this-can-only-be-done-once

# paper trading
uv run quant-platform paper-init
uv run quant-platform paper-designate-model --run-id RUN_ID --model factor_composite \
  --designated-by NAME
uv run quant-platform paper-rebalance                # proposes only
uv run quant-platform paper-rebalance --approve-all  # approves + simulates fills
uv run quant-platform paper-status
```

## Configuration

Validated YAML in `configs/`, loaded through `research_charter.yaml`. Incoherent mandates are
rejected **at load time** rather than producing plausible-but-invalid results — e.g. 30 names
capped at 2% cannot reach full investment, and that fails immediately.

Key defaults: 20-session horizon, weekly rebalance, 1-session execution delay, long-only,
50-100 holdings, 2% max position, ±3% sector deviation, 0.90-1.10 beta, 15% turnover cap.

## Real-data workflow

Open-data adapters exist for prices (yfinance), macro (FRED), fundamentals (SEC EDGAR), and
factors (Ken French). Copy `.env.example` to `.env` and supply credentials:

```bash
QP_FRED_API_KEY=...
QP_SEC_USER_AGENT="Your Name your@email.com"
```

**Critical caveat:** free price data is survivorship-biased (delisted companies cannot be
retrieved), and no open source provides historical S&P 500 constituents. Supply a vendor file
via `LocalConstituentSource`, or results are explicitly marked survivorship-biased. See
[docs/limitations.md](docs/limitations.md).

## Testing

```bash
uv run pytest                          # everything
uv run pytest tests/data_leakage/      # the bias controls
uv run pytest tests/portfolio_accounting/
uv run pytest -m "not slow"

cd frontend
npm run typecheck
npm run lint
npm test
npm run build
npm run test:e2e
```

## Benchmarking

Run a fresh cold/warm benchmark with the complete roster:

```bash
uv run python scripts/benchmark_research.py --securities 120 --workers 2
```

The command writes its workload, run IDs, cache status, stage timings, and platform peak-RSS
value beneath `artifacts/benchmarks/`. It measures the local synthetic fixture; it does not
claim to reproduce another machine's reference timings.

## Code quality

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
```

Or `make check`.

## Documentation

- [architecture.md](docs/architecture.md) — data flow, module boundaries, the five timestamps
- [data_contracts.md](docs/data_contracts.md) — schemas and time semantics
- [modeling.md](docs/modeling.md) — target, features, models, validation
- [backtesting.md](docs/backtesting.md) — event sequence, costs, bias controls
- [paper_trading.md](docs/paper_trading.md) — the approval gate and reconciliation
- [limitations.md](docs/limitations.md) — **read before interpreting any output**
- [reproducibility.md](docs/reproducibility.md) — seeding and known nondeterminism
- [http_api.md](docs/http_api.md) — browser API, queue semantics, routes, and local operation
- [adr/](docs/adr/) — architecture decision records

## Troubleshooting

**`OSError: Library not loaded: @rpath/libomp.dylib`** — LightGBM's macOS wheel expects
Homebrew's libomp. Run `uv run python scripts/fix_macos_libomp.py`, which reuses the copy
already vendored by scikit-learn/PyTorch. No Homebrew needed.

**`torch ... has no wheels with a matching platform tag`** — PyTorch >= 2.12 requires
macOS 14+. The project pins `torch<2.12`; keep that bound on macOS 13.

**`universe is not selective`** — the eligible universe must be meaningfully larger than
`max_holdings`, or every portfolio holds everything and the ranking is never consulted. Raise
`--securities` or lower `max_holdings`.

**Optimizer reports relaxations** — expected when a mandate is infeasible for that day's
universe. Every relaxation is logged; check `relaxations_applied` in the diagnostics.

## Investment risk disclaimer

This software is for research and education. It is not investment advice, not a
recommendation, and not validated for managing real money. Simulated and historical results do
not guarantee future performance. Trading equities involves risk of loss, including total loss
of capital. Any decision to deploy capital is yours alone.

## License

MIT — see [LICENSE](LICENSE).
