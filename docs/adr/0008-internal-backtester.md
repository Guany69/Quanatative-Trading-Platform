# ADR 0008: Internal event-aware backtester

**Status:** Accepted

## Context
The backtester must enforce the five-timestamp separation and book delisting returns. Existing
vectorized libraries make those hard to guarantee.

## Decision
Write an internal engine that separates information/signal/order/fill/accounting timestamps,
with a vectorized comparison for simple strategies as a reconciliation check.

## Alternatives
- vectorbt: fast, but the execution-delay and delisting semantics we require are awkward, and
  it does not currently install cleanly on Python 3.13.
- Zipline: unmaintained for current Python versions.

## Consequences
- Full control over the semantics that matter for bias control.
- More code to test, hence `tests/portfolio_accounting/`.
- Slower than a pure vectorized engine; acceptable at this scale.
