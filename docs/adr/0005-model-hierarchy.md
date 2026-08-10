# ADR 0005: Baselines are mandatory, not optional

**Status:** Accepted

## Context
A model with IR 0.3 sounds good until a five-line momentum rule scores 0.35 on the same data.
Sophistication is frequently reported without a control.

## Decision
Every comparison includes a no-skill predictor, a single-factor momentum baseline, and a
transparent factor composite, all running through the identical pipeline, universe, costs,
and portfolio construction.

## Alternatives
- Benchmark against the index only: does not isolate whether the *model* adds anything.
- Optional baselines: they would be omitted exactly when inconvenient.

## Consequences
- Every report shows what simplicity achieves.
- The no-skill model doubles as a leakage detector: if it scores well, something is wrong.
