# ADR 0001: Cross-sectional excess-return ranking, not price prediction

**Status:** Accepted

## Context
The obvious framing of "predict stocks" is forecasting future prices. That target is dominated
by the market factor and by scale differences between securities, and it rewards being long in
a rising market rather than picking better stocks.

## Decision
Predict **forward benchmark-relative excess return** over 20 sessions, and trade the
**cross-sectional percentile rank** of those predictions.

## Alternatives
- Absolute return: conflates market timing with selection.
- Price level: scale-dependent and dominated by the current price.
- Binary outperformance: supported as a target type, but discards magnitude.

## Consequences
- Model quality is measured by rank IC, which is what portfolio construction consumes.
- Results are naturally benchmark-relative, matching a long-only mandate.
- Market-timing skill is explicitly out of scope.
