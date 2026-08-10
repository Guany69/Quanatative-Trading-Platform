# Reproducibility

## What is recorded per run

Run ID, UTC timestamp, git commit **and dirty flag**, config hash, dataset snapshot ID,
feature/label versions, random seeds, dependency versions, platform, and artifact paths.
A dirty working tree is recorded explicitly, since the commit hash then does not fully
describe the code that ran.

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

Each report directory contains `run_metadata.json` (environment and versions) and
`resolved_config.json` (the fully resolved configuration — not the source YAML, since
includes, environment expansion, and defaults all change what actually executed).
