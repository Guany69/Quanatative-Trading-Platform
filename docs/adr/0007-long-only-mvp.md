# ADR 0007: Long-only, benchmark-aware MVP

**Status:** Accepted

## Context
Long/short introduces borrow costs, borrow availability, short squeezes, and margin — each a
substantial modelling problem with no free data.

## Decision
Long-only, fully invested, no leverage, 50-100 holdings, benchmark-aware constraints.

## Alternatives
- Market-neutral long/short: higher fidelity to institutional practice, far more modelling
  surface, and borrow data is not freely available.

## Consequences
- Portfolio beta is near 1, so excess return is the meaningful measure.
- Cash can never go negative, which makes accounting invariants checkable.
- Short-side alpha is out of scope.
