# ADR 0014: Add a local HTTP application adapter for React

**Status:** Accepted

## Context

The platform was CLI-first, but a browser is now an explicit application consumer. Direct
browser access to DuckDB, snapshots, reports, or paper JSON would bypass typed validation,
single-writer ownership, and the domain approval/holdout gates. Reimplementing research or
portfolio logic in a web tier would create a second system with divergent correctness rules.

## Decision

Add FastAPI/Pydantic routes and thin application services inside the existing modular
monolith. Add a React/TypeScript/Vite presentation layer whose contract is generated from
OpenAPI and whose server state is owned by TanStack Query.

Research POSTs enter one durable in-process FIFO worker. That worker remains the only research
persistence owner and shares its writer lock with holdout consumption. Purpose-specific,
parameterized readers expose bounded result data; there is no general SQL API. Paper mutations
are serialized load/mutate/save cycles that delegate to existing domain components. The API
binds to loopback and serves the compiled SPA from the same origin when present.

## Consequences

- The CLI and React UI reuse the same research, model, portfolio, execution, and paper code.
- Async jobs survive queued-process restarts, while interrupted active jobs become explicit
  stale-recovery failures rather than silently resuming partial writes.
- The API process is intentionally single-instance for this local MVP; its in-process locks are
  not a distributed coordination mechanism.
- Approval and holdout controls remain authoritative in Python even if a client bypasses UI
  confirmation or button state.
- OpenAPI generation prevents most backend/frontend DTO drift.
- The legacy scorecard remains separate until its Python analysis is exposed as JSON; its logic
  is not duplicated in React.
