# ADR 0009: CVXPY optimizer with ordered, logged relaxation

**Status:** Accepted

## Context
Real mandates are frequently infeasible for a given universe on a given day. The dangerous
failure is silently dropping a constraint.

## Decision
CVXPY (CLARABEL) for direct constraint control. On infeasibility, apply an **ordered**
relaxation policy — minimum position, holding count, sector bounds, beta band, turnover, cash —
logging every step, and falling back to a feasible score-weighted allocation if all else fails.
Long-only and the position cap are never relaxed.

## Alternatives
- Skfolio/Riskfolio: higher level, less direct control, Python 3.13 uncertainty.
- Analytic weighting only: cannot express sector/beta/turnover constraints jointly.

## Consequences
- Expected returns come from ranks, not raw predictions, limiting mean-variance sensitivity.
- Every relaxation appears in the diagnostics; violations are never silent.
- Post-solve dust removal requires iterative cap redistribution to stay fully invested.
