# ADR 0012: Accept open-data gaps explicitly rather than paper over them

**Status:** Accepted

## Context
Free data cannot supply historical index constituents, delisting returns, or macro vintages.
The tempting workaround — using today's S&P 500 members as if they were historical — produces
a badly survivorship-biased backtest that looks excellent.

## Decision
Every adapter declares a `SourceProvenance` stating what it cannot guarantee. The universe
builder **refuses** survivorship-biased membership unless explicitly opted into, and any
result derived from it is marked biased throughout. `LocalConstituentSource` exists so users
can supply a vendor extract.

## Alternatives
- Silently using current membership: the single most common way backtests inflate returns.
- Refusing to run without vendor data: makes the platform unusable for exploration.

## Consequences
- Reports always state data completeness alongside the numbers.
- The fixture provider has none of these gaps, which is precisely why fixture results carry
  no investment meaning.
