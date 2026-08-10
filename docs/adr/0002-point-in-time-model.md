# ADR 0002: Explicit point-in-time availability model

**Status:** Accepted

## Context
Look-ahead bias is the most common and most damaging backtest error, and it is usually
invisible: joining fundamentals on fiscal period silently gives the model a filing weeks
before publication.

## Decision
Every time-dependent record carries `observation_date`, `available_at`, `ingested_at`, and
`revision_id`. Features may only consume records with `available_at <= t`, implemented with
as-of (backward) joins. Records whose `available_at` precedes `observation_date` are rejected
at construction.

## Alternatives
- Applying a fixed lag: simple, but wrong for filings with variable delays.
- Trusting providers: they routinely serve latest-revised values with no vintage.

## Consequences
- Adapters must supply real publication dates or explicitly declare an approximation.
- As-of joins are more expensive than plain joins.
- Restatements and macro revisions are handled correctly by construction.
