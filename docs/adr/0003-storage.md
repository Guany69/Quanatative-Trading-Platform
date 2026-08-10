# ADR 0003: Partitioned Parquet for local storage

**Status:** Accepted

## Context
Panels reach millions of rows. Storage must be columnar, typed, and portable.

## Decision
Parquet via PyArrow, with DuckDB available for ad-hoc analytical queries.

## Alternatives
- CSV: untyped, large, loses dtypes (dates become strings).
- SQLite: row-oriented, poor for wide analytical scans.
- A database server: unjustified operational weight for local research.

## Consequences
- Fast columnar reads and good compression; types survive round trips.
- Parquet files are gitignored; only fixtures are regenerable from seed.
