# Contributing

## Setup

```bash
uv sync --extra ml --extra optimization --extra data --extra reporting --extra dev
uv run python scripts/fix_macos_libomp.py   # macOS, if LightGBM fails to import
make check
```

## The rules that matter

This is a bias-control system before it is a modelling system. Changes are judged primarily on
whether they preserve these invariants:

1. **Never let the future leak backward.** Any new data path must respect `available_at`. If
   you add a join, ask what happens when a record is published *after* the feature date.
2. **Never fit on data you will evaluate on.** Preprocessing, hyperparameters, model
   selection, and ensemble weights all come from training/validation only.
3. **Never silently drop a constraint.** If a portfolio constraint cannot be met, relax it in
   the documented order and log it.
4. **Never fabricate a number.** If a statistic's assumptions are not met, return NaN and say
   so. An honest gap beats a plausible-looking approximation.
5. **Never claim performance.** Reports state data source, cost assumptions, evaluation stage,
   and completeness alongside every metric.

## Adding a feature

Register a `FeatureDefinition` declaring lookback, minimum observations, missing policy, and
version. Bump the version when you change a formula — stale cached features must not silently
mix with new ones. Use only trailing windows.

## Adding a model

Subclass `BaseModel`, implement `_fit`/`_predict`, register it. Set
`is_production_candidate = False` for diagnostic-only models.

## Adding a data source

Implement the relevant Protocol in `data/base.py` and declare a `SourceProvenance` that
honestly states what it cannot guarantee. Defaults are pessimistic on purpose: a source that
forgets to describe itself is treated as incomplete.

## Tests

New bias controls need a test that **fails when the control is removed**. A test that passes
either way proves nothing. The placebo pattern in
`tests/data_leakage/test_no_signal_control.py` is the model to follow: verify the pipeline
finds nothing when there is nothing to find, *and* finds a planted signal that genuinely
exists.

## Style

Ruff (format + lint) and mypy must pass. Comments should explain *why*, especially where the
reasoning is financial rather than mechanical.
