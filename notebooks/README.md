# Notebooks

Intentionally empty. Research logic belongs in `src/quant_platform/` where it can be tested,
versioned, and reused; notebooks are for exploration only and should not become the place
where analysis actually lives.

If you add one, import from the package rather than redefining logic inline — a notebook that
reimplements a feature is a notebook that will silently disagree with the pipeline.
