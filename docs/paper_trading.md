# Paper trading

## Workflow

```
refresh -> validate -> features -> predict -> targets -> PROPOSE
                                                          |
                                                 [HUMAN APPROVAL]
                                                          |
                                            execute -> reconcile -> monitor
```

Nothing crosses the approval boundary automatically. `SimulatedBroker.submit()` raises
`OrderNotApprovedError` for any order that has not been through `approve()`. This is enforced
in code because it is the boundary where software becomes money.

## Commands

```bash
uv run quant-platform paper-init                    # create the account
uv run quant-platform paper-rebalance               # propose only (default)
uv run quant-platform paper-rebalance --approve-all # approve + simulate fills
uv run quant-platform paper-status                  # monitoring report
```

Partial approval is supported: a reviewer can approve a subset by order ID rather than facing
an all-or-nothing choice.

## State

Persisted to `paper_state/state.json` via atomic write (temp file + rename), so an interrupted
save cannot truncate the account history. Tracks cash, positions with average cost, every
rebalance, pending proposals, equity history, realized P&L, and cumulative costs.

## Reconciliation

After execution, compares intent against outcome: weight deviations, cash balance, unfilled
orders, fill shortfall, and **cost surprise** (realized vs expected).

Cost surprise is the point. A backtest can never reveal that its cost model is wrong, because
the same model produces both the expectation and the outcome. Paper trading can — and
persistent positive surprise means every backtest built on that model is optimistic.

## Safety

- No broker integration exists. `NautilusBrokerAdapter` raises on instantiation.
- Long-only; cash cannot go negative (regression-tested).
- ADV participation limits cap fill sizes.
- All fills are simulated: real execution differs in price, timing, and liquidity.
