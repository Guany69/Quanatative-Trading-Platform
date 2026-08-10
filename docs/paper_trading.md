# Paper trading

## Workflow

```text
designated completed run -> immutable snapshot -> shared validate/features path
  -> stored fold artifact -> predict -> targets -> record PROPOSALS
                                                   |
                                         HUMAN APPROVE / REJECT
                                                   |
                         approved only -> delayed simulation -> reconcile -> persist
```

Nothing crosses the approval boundary automatically. `SimulatedBroker.submit()` raises
`OrderNotApprovedError` for any order that has not been through `approve()`. This is enforced
in code because it is the boundary where software becomes money.

## Commands

```bash
uv run quant-platform paper-init                    # create the account
uv run quant-platform paper-designate-model --run-id RUN_ID \
  --model factor_composite --designated-by NAME
uv run quant-platform paper-rebalance               # propose only (default)
uv run quant-platform paper-rebalance --approve-all # approve + simulate fills
uv run quant-platform paper-status                  # monitoring report
```

Partial approval is supported: a reviewer can approve a subset by order ID rather than facing
an all-or-nothing choice.

## State

Persisted to `paper_state/state.json` via atomic write (temp file + rename), so an interrupted
save cannot truncate the account history. Tracks production designations, proposal history,
approval/rejection identity and timestamps, fills, reconciliation, cash, positions, every
rebalance, unfilled proposals, equity history, realized P&L, and cumulative costs. Reloading
state validates the designation against the completed DuckDB run before new proposals.

## Reconciliation

After execution, compares intent against outcome: weight deviations, cash balance, unfilled
orders, fill shortfall, and **cost surprise** (realized vs expected).

Cost surprise is the point. A backtest can never reveal that its cost model is wrong, because
the same model produces both the expectation and the outcome. Paper trading can — and
persistent positive surprise means every backtest built on that model is optimistic.

## Safety

- No broker integration exists. `NautilusBrokerAdapter` raises on instantiation.
- No valid production designation means no proposal generation.
- Real-data snapshots older than the staleness gate block proposal generation.
- Same-session order/fill dates raise `ExecutionTimingError`.
- Long-only; cash cannot go negative (regression-tested).
- ADV participation limits cap fill sizes.
- All fills are simulated: real execution differs in price, timing, and liquidity.
