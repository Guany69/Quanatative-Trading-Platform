# Reproducibility

## What is recorded per run

Run ID, UTC timestamp, git commit **and dirty flag**, config hash, immutable dataset snapshot
ID, fold schedule hash, selected models/strategies/scenarios, random seeds, and artifact paths.
A dirty working tree is recorded explicitly, since the commit hash then does not fully
describe the code that ran.

The snapshot manifest contains source/provenance, table schemas, row counts, date ranges and
content hashes. Cache identity includes snapshot, universe, feature and label versions,
validation, purge, embargo and the complete fold schedule. Model task seeds are derived from
`(run seed, model name, fold)` and are independent of process scheduling.

## Seeding

`set_global_seeds()` seeds Python `random`, NumPy, and PyTorch (with deterministic algorithms
enabled where available). LightGBM is seeded per training call with `deterministic=True` and
`force_row_wise=True`, which removes threading-order nondeterminism.

## Known nondeterminism

- `PYTHONHASHSEED` only takes effect at interpreter start, so setting it mid-process has no
  retroactive effect. Its value is recorded rather than silently assumed.
- BLAS reductions may reorder floating-point sums across platforms and thread counts, so
  bit-level identity is not guaranteed across machines. Single-threaded settings are used for
  the models where it matters.
- Results are reproducible **on the same machine with the same dependency versions**; the
  dependency manifest is captured to make divergence detectable.

## Verifying determinism

`tests/integration/test_end_to_end.py::TestDeterminism` asserts that identical seeds produce
identical fixture data and identical predictions.

## Reproducing a prior run

Query `research.duckdb` for the run and use its `snapshot_id`, `charter_hash`, code state,
seed and fold schedule. Each report directory contains `resolved_config.json` (the fully
resolved configuration—not the source YAML, since includes and defaults change what ran),
and the artifacts are scoped by run/model/fold.

## Cold/warm benchmark

```bash
uv run python scripts/benchmark_research.py --securities 120 --workers 2
```

The harness creates a fresh output root, repeats the identical request against the same
snapshot/cache, and records panel, training, simulation and total timings plus cache status.
Its numbers describe only the actual local synthetic workload.
