# ADR 0004: Polars primary, pandas at library boundaries

**Status:** Accepted

## Context
Polars is substantially faster for panel operations and has stricter, more predictable typing.
Much of the quant ecosystem still expects pandas.

## Decision
Polars for all internal transformations; convert to pandas only where a library requires it.

## Alternatives
- Pandas throughout: slower, and its silent dtype coercion has caused real bugs.
- Polars only: would exclude useful libraries.

## Consequences
- Expression-based, mostly vectorized internals.
- Conversion cost at boundaries, kept narrow deliberately.
- `utilities/narrow.py` exists because Polars scalar returns are broadly typed.
