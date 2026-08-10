# ADR 0003: Role-specific local persistence

**Status:** Accepted

## Context
Panels reach millions of rows. Storage must be columnar, typed, and portable.

## Decision
Immutable provider snapshots and disposable PIT fold caches use Parquet plus content-hashed
JSON manifests. DuckDB is the authoritative research-results system of record and is written
only by the parent run orchestrator through Arrow batches and transactions. Model artifacts
use run/model/fold filesystem paths, the trial registry is append-only JSONL, paper state is
separate atomic JSON, and CSV is a regenerable export only.

## Alternatives
- CSV: untyped, large, loses dtypes (dates become strings).
- SQLite: row-oriented, poor for wide analytical scans.
- A database server: unjustified operational weight for local research.

## Consequences
- Fast columnar reads and good compression; types survive round trips.
- Analytical cross-run joins do not depend on report files.
- Cache deletion affects performance, not correctness or authoritative run history.
- A local single-writer design avoids unjustified database-server or distributed complexity.
