# Backtesting

## Event sequence

For each trading session:

1. **Corporate actions** — dividends credited to cash; delisted names liquidated at their
   terminal value (including a -100% wipeout).
2. **Mark to market** — pre-trade positions valued at today's close. This is the *gross*
   return: what markets did, before any trading friction.
3. **Trade** — only if today is a scheduled fill date (signal date + `execution_delay_sessions`).
   Sells execute before buys, and buys are capped at available cash.
4. **Post-trade valuation** — costs are now deducted, giving the *net* return.
5. **Snapshot** — value, cash, positions, turnover, cost decomposition, benchmark return.

## Execution assumptions

- Fills occur at the delayed session's close, adjusted by the cost model. A signal formed on
  the Friday close cannot fill at that Friday close.
- Cost always works against the trader: buys fill higher, sells lower.
- Long-only, no margin: cash never goes negative, verified by tests.

## Costs

Decomposed rather than a single fudge factor, because components scale differently and fail
differently:

| Component | Behaviour |
|---|---|
| Commission | linear in notional/shares; charged in cash |
| Spread | half the quoted spread; widens with illiquidity |
| Slippage | scales with volatility and participation |
| Market impact | **concave** (square root) in participation |

Scenarios: `zero` (diagnostic only, never achievable), `base`, `double`, `triple`.

## Corporate actions

Splits restate history so returns stay continuous; dividends scale pre-ex-date prices;
delistings book the terminal return. Eligibility screens use the **unadjusted** close, because
the $5 rule concerns the price that actually traded.

## Bias controls

| Bias | Control |
|---|---|
| Look-ahead | `available_at` gating; as-of joins; trailing windows only |
| Survivorship | true membership intervals; biased data refused unless explicitly opted into |
| Delisting | terminal returns booked, not dropped |
| Execution | mandatory >= 1 session delay |
| Overfitting | purge + embargo + locked holdout + DSR/PBO |
| Selection | every trial recorded, including rejected ones |

## Reconciliation

`tests/portfolio_accounting/` verifies the conservation identity: final value equals initial
capital minus total costs in a flat market. Cash, positions, dividends, and delistings all
reconcile, and cost scenarios are monotonic.
