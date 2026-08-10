# ADR 0013: Library selection from the awesome-quant catalog

**Status:** Accepted

## Context
The provided `awesome-quant` archive is a curated **catalog** of quant libraries, not a
platform. It was inspected to identify useful dependencies.

## Decision

**Adopted:** polars, pyarrow, pydantic, typer, scikit-learn, LightGBM, PyTorch, Optuna, CVXPY,
matplotlib, DuckDB, yfinance, pytest, ruff, mypy.

**Rejected or deferred, with reasons:**

| Library | Reason |
|---|---|
| vectorbt | Python 3.13 support uncertain; execution-delay semantics awkward (ADR 0008). |
| alphalens-reloaded | Factor diagnostics implemented internally to avoid an optional dependency on the critical path. |
| skfolio / riskfolio-lib | CVXPY gives more direct constraint control (ADR 0009). |
| quantstats / empyrical | Metrics implemented internally with documented annualization. |
| nautilus_trader | Heavy; interface stub only, since no live trading is in scope. |
| zipline | Unmaintained for current Python. |

No third-party source code was copied; dependencies are consumed through the package manager,
respecting their licenses.

## Consequences
- The catalog informed selection without becoming a dependency itself.
- `quant-platform inspect-reference-catalog` summarizes it.
