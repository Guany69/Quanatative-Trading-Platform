# ADR 0011: Optional heavy dependencies behind thin adapters

**Status:** Accepted

## Context
The spec lists many libraries. Several do not support Python 3.13 or the target platform, and
the core platform must remain runnable regardless.

## Decision
Core = numpy, pandas, polars, pyarrow, pydantic, typer, scikit-learn, scipy, matplotlib.
Everything heavier lives in extras (`ml`, `optimization`, `data`, `reporting`, `tracking`)
behind small internal interfaces. **All critical metrics are implemented internally** — IC,
Sharpe, Sortino, PSR, DSR, PBO, factor diagnostics — so no essential capability depends on an
optional package.

Platform-specific decisions:
- **PyTorch pinned `<2.12`**: from 2.12 the macOS arm64 wheels require macOS 14+, and this
  project targets macOS 13. Verified empirically: 2.11 resolves, 2.12 does not.
- **LightGBM `libomp`**: the wheel's rpath points only at Homebrew/MacPorts locations. With no
  Homebrew present, `import lightgbm` fails. `scripts/fix_macos_libomp.py` vendors the
  `libomp.dylib` already shipped by scikit-learn/PyTorch, adds an `@loader_path` rpath, and
  re-signs — no system package manager needed.

## Alternatives
- Requiring everything: the platform would not install here at all.
- Vendoring third-party code: licensing and maintenance burden.

## Consequences
- `validate-environment` reports exactly what is present and what is missing.
- Not installed: vectorbt, alphalens-reloaded, skfolio, riskfolio-lib, nautilus_trader,
  quantstats — documented as capability gaps in `docs/limitations.md`.
