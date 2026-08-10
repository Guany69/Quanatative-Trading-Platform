# ADR 0010: File-based registry recording every trial

**Status:** Accepted

## Context
The Deflated Sharpe Ratio and PBO both require the *number of configurations tried*. Keeping
only winners makes those corrections impossible and hides selection bias.

## Decision
Append-only JSONL registry recording every trial including rejected ones, with MLflow as an
optional mirror.

## Alternatives
- MLflow required: adds a heavy dependency and a server for a local research tool.
- Ad-hoc logging: not machine-readable, and trial counts get lost.

## Consequences
- Trial counts are available for honest multiple-testing corrections.
- History is diffable, greppable, and append-only.
- Trials run outside the registry silently make DSR optimistic — a documented discipline
  requirement.
