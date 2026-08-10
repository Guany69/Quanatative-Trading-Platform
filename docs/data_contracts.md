# Data contracts

Every time-dependent record carries the point-in-time triple:

| Field | Meaning |
|---|---|
| `observation_date` | the period the information describes |
| `available_at` | when it first became publicly knowable |
| `ingested_at` | when this platform stored it (audit only, never used for filtering) |
| `revision_id` | vintage counter; 0 = original print |

**Selection rule:** the correct record for date *t* is the latest revision with
`available_at <= t` — never simply the latest revision.

## prices
Key `(security_id, observation_date)`. Columns: `open/high/low/close`, `adjusted_close`,
`volume`, `dollar_volume`, `shares_outstanding`, `market_cap`,
`cumulative_adjustment_factor`.
`close` is **unadjusted** (used by eligibility screens); `adjusted_close` drives returns.
Rules: positive prices, `high >= low`, no duplicate keys, non-increasing adjustment factors.

## fundamentals
Key `(security_id, fiscal_period_end, available_at)`. `available_at` **must** derive from the
filing date, never the period end. Multiple rows per period are legitimate — they are
restatements, and each becomes visible only from its own publication date.

## macro
Key `(series_id, observation_date, revision_id)`. Revisions share `observation_date` but
differ in `available_at` and value.

## membership
Key `(security_id, universe, start_date)`. Half-open intervals `[start, end)` so consecutive
memberships abut without double-counting. `is_survivorship_biased` is **mandatory**; the
universe builder refuses biased data unless explicitly opted into.

## corporate_actions
Key `(security_id, action_type, ex_date)`. `ex_date` drives price adjustment; `available_at`
drives what the model may know (delistings and splits are announced in advance).

## labels
Key `(security_id, as_of)`. Stores `window_start`/`window_end` so the walk-forward splitter can
purge samples whose forward windows overlap the test period.

## Storage
Partitioned Parquet via PyArrow. Frames are Polars; pandas is used only where a library
requires it.
