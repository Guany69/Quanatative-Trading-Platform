"""Command-line interface (spec section 27).

Every command loads a validated charter, emits structured logs, persists artifacts, and
returns a non-zero exit code on failure so the CLI composes correctly in scripts and CI.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from quant_platform.utilities.narrow import as_date, as_float
from quant_platform.utilities.reproducibility import configure_logging, get_logger, load_dotenv

# Load .env before any command runs, so credentials are available without shell sourcing.
load_dotenv()

app = typer.Typer(
    name="quant-platform",
    help="Quantitative U.S. equity research and paper-trading platform. "
    "Predicts forward benchmark-relative excess return and trades the cross-sectional ranking.",
    no_args_is_help=True,
    add_completion=False,
)

logger = get_logger("cli")

ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", help="Path to the research charter YAML.")
]
OutOpt = Annotated[Path, typer.Option("--out", "-o", help="Output directory.")]


def _echo(msg: str, kind: str = "info") -> None:
    colors = {
        "info": typer.colors.CYAN,
        "ok": typer.colors.GREEN,
        "warn": typer.colors.YELLOW,
        "err": typer.colors.RED,
    }
    typer.secho(msg, fg=colors.get(kind, typer.colors.WHITE))


@app.command("validate-environment")
def validate_environment() -> None:
    """Check the Python environment and report which optional stacks are available."""
    configure_logging("INFO")
    import importlib
    import platform

    _echo(f"Python  : {sys.version.split()[0]}", "info")
    _echo(f"Platform: {platform.platform()}", "info")
    print()

    core = [
        "numpy",
        "pandas",
        "polars",
        "pyarrow",
        "pydantic",
        "typer",
        "sklearn",
        "scipy",
        "matplotlib",
        "yaml",
        "duckdb",
    ]
    optional = {
        "lightgbm": "primary nonlinear model",
        "torch": "neural-network challenger",
        "optuna": "hyperparameter search",
        "cvxpy": "constrained optimizer",
        "yfinance": "open price data",
        "plotly": "interactive charts",
        "mlflow": "experiment tracking",
    }

    failures = 0
    _echo("Core dependencies (required):", "info")
    for mod in core:
        try:
            m = importlib.import_module(mod)
            print(f"  OK    {mod:12s} {getattr(m, '__version__', '')}")
        except Exception as exc:
            failures += 1
            _echo(f"  FAIL  {mod:12s} {type(exc).__name__}: {exc}", "err")

    print()
    _echo("Optional dependencies (platform degrades gracefully without these):", "info")
    for mod, purpose in optional.items():
        try:
            m = importlib.import_module(mod)
            print(f"  OK    {mod:12s} {getattr(m, '__version__', ''):10s} -- {purpose}")
        except Exception:
            _echo(f"  MISS  {mod:12s} {'':10s} -- {purpose} UNAVAILABLE", "warn")

    # LightGBM on macOS needs libomp; surface the fix rather than a raw dlopen error.
    try:
        importlib.import_module("lightgbm")
    except Exception as exc:
        if "libomp" in str(exc):
            print()
            _echo("LightGBM cannot load libomp. Fix without Homebrew:", "warn")
            _echo("    uv run python scripts/fix_macos_libomp.py", "warn")

    print()
    if failures:
        _echo(f"{failures} core dependency failure(s).", "err")
        raise typer.Exit(1)
    _echo("Environment OK: all core dependencies present.", "ok")


def _csv_option(value: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if value.strip().lower() in {"all", "*"}:
        return defaults
    return tuple(item.strip().replace("-", "_") for item in value.split(",") if item.strip())


def _cached_fixture_data(config: Path, securities: int):
    """Resolve the legacy data-inspection commands through the consumed PIT cache."""
    from dataclasses import asdict

    from quant_platform.config import load_charter
    from quant_platform.data.adapters.fixture import (
        FIXTURE_SOURCE,
        FixtureDataProvider,
        FixtureSpec,
    )
    from quant_platform.data.snapshots import SnapshotStore
    from quant_platform.research.panels import (
        build_fold_panels,
        derive_fold_schedule,
        make_pit_cache_key,
        pipeline_data_from_cache,
    )
    from quant_platform.research.pit_cache import PitPanelCache, fold_schedule_hash

    charter = load_charter(config)
    spec = FixtureSpec(n_securities=securities, seed=charter.random_seed)
    provenance = {"provider": "FixtureDataProvider", "spec": asdict(spec)}
    snapshots = SnapshotStore()
    snapshot = snapshots.find_by_provenance(sources=[FIXTURE_SOURCE], provenance=provenance)
    if snapshot is None:
        dataset = FixtureDataProvider(spec).generate()
        snapshot = snapshots.create(
            {
                name: getattr(dataset, name)
                for name in (
                    "securities",
                    "identifiers",
                    "prices",
                    "benchmark",
                    "membership",
                    "corporate_actions",
                    "fundamentals",
                    "macro",
                )
            },
            sources=[FIXTURE_SOURCE],
            provenance=provenance,
        )
    folds, calendar = derive_fold_schedule(snapshot, charter)
    schedule_hash = fold_schedule_hash(folds)
    cache = PitPanelCache(Path(charter.data_sources.cache_dir) / "pit_cache")
    cached = cache.get_or_build(
        make_pit_cache_key(snapshot, charter, schedule_hash),
        lambda: build_fold_panels(snapshot, charter, folds),
    )
    return charter, snapshot, cached, pipeline_data_from_cache(cached, charter, calendar)


@app.command("inspect-reference-catalog")
def inspect_reference_catalog(
    path: Annotated[
        Path, typer.Option("--path", help="Path to the awesome-quant checkout or README.")
    ] = Path("../awesome-quant/README.md"),
) -> None:
    """Summarize the awesome-quant reference catalog (spec section 1)."""
    configure_logging("INFO")
    p = Path(path)
    if p.is_dir():
        p = p / "README.md"
    if not p.exists():
        _echo(f"Reference catalog not found at {p}.", "err")
        _echo("It is a catalog of libraries, not a dependency. Pass --path to locate it.", "warn")
        raise typer.Exit(1)

    text = p.read_text(errors="replace")
    lines = text.splitlines()
    sections = [ln.lstrip("#").strip() for ln in lines if ln.startswith("## ")]
    links = sum(1 for ln in lines if "](http" in ln)

    _echo(f"Reference catalog: {p}", "info")
    print(f"  {len(lines):,} lines, {len(sections)} sections, ~{links:,} linked projects")
    print("\n  Sections:")
    for s in sections[:25]:
        print(f"    - {s}")
    print(
        "\n  This is a curated CATALOG of third-party libraries, not a platform.\n"
        "  Libraries selected for this build are recorded in docs/adr/."
    )


@app.command("bootstrap-demo-data")
def bootstrap_demo_data(
    out: OutOpt = Path("data/fixtures"),
    securities: Annotated[int, typer.Option("--securities", "-n")] = 120,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Generate the deterministic synthetic dataset and write it to Parquet."""
    configure_logging("INFO")
    from dataclasses import asdict

    from quant_platform.data.adapters.fixture import FixtureDataProvider, FixtureSpec
    from quant_platform.data.snapshots import SnapshotStore

    out.mkdir(parents=True, exist_ok=True)
    spec = FixtureSpec(n_securities=securities, seed=seed)
    _echo(
        f"Generating fixture: {securities} securities, {spec.start}..{spec.end}, seed={seed}",
        "info",
    )
    ds = FixtureDataProvider(spec).generate()

    for name in (
        "securities",
        "identifiers",
        "prices",
        "benchmark",
        "membership",
        "corporate_actions",
        "fundamentals",
        "macro",
    ):
        frame = getattr(ds, name)
        path = out / f"{name}.parquet"
        frame.write_parquet(path)
        print(f"  {name:18s} {frame.shape!s:>16s} -> {path}")

    snapshot = SnapshotStore().create(
        {
            name: getattr(ds, name)
            for name in (
                "securities",
                "identifiers",
                "prices",
                "benchmark",
                "membership",
                "corporate_actions",
                "fundamentals",
                "macro",
            )
        },
        sources=["fixture_synthetic"],
        provenance={"provider": "FixtureDataProvider", "spec": asdict(spec)},
    )

    _echo("\nSYNTHETIC DATA: randomly generated. No investment meaning.", "warn")
    _echo(f"Fixture written. immutable snapshot_id={snapshot.snapshot_id}", "ok")


@app.command("run-demo")
def run_demo(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    out: OutOpt = Path("reports/demo"),
    securities: Annotated[int, typer.Option("--securities", "-n")] = 400,
    test_start: Annotated[str, typer.Option("--test-start")] = "2021-01-04",
) -> None:
    """Run the full synthetic demonstration end to end (spec section 36).

    Needs no credentials and no network access.
    """
    configure_logging("INFO")
    from quant_platform.demo import run_full_demo

    try:
        result = run_full_demo(
            config_path=config,
            out_dir=out,
            n_securities=securities,
            test_start=date.fromisoformat(test_start),
        )
    except Exception as exc:
        logger.exception("demo failed")
        _echo(f"Demo failed: {exc}", "err")
        raise typer.Exit(1) from exc

    _echo(f"\nDemo complete. Reports: {result['dir']}", "ok")
    _echo("Results are SYNTHETIC and carry no investment meaning.", "warn")


@app.command("build-features")
def build_features(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    export: Annotated[
        Path | None,
        typer.Option("--export", "--out", help="Optional Parquet export path."),
    ] = None,
    securities: Annotated[int, typer.Option("--securities", "-n")] = 120,
) -> None:
    """Build/load the feature panel through the governed PIT cache."""
    configure_logging("INFO")
    _charter, snapshot, cached, data = _cached_fixture_data(config, securities)
    _echo(
        f"features {data.features.shape} snapshot={snapshot.snapshot_id} "
        f"cache={'HIT' if cached.cache_hit else 'MISS'}",
        "ok",
    )
    if export:
        export.parent.mkdir(parents=True, exist_ok=True)
        data.features.write_parquet(export)
        _echo(f"diagnostic export -> {export}", "info")


@app.command("build-labels")
def build_labels(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    export: Annotated[
        Path | None,
        typer.Option("--export", "--out", help="Optional Parquet export path."),
    ] = None,
    securities: Annotated[int, typer.Option("--securities", "-n")] = 120,
) -> None:
    """Build/load forward labels through the governed PIT cache."""
    configure_logging("INFO")
    charter, snapshot, cached, data = _cached_fixture_data(config, securities)
    _echo(
        f"labels {data.labels.shape} snapshot={snapshot.snapshot_id} "
        f"cache={'HIT' if cached.cache_hit else 'MISS'}",
        "ok",
    )
    if export:
        export.parent.mkdir(parents=True, exist_ok=True)
        data.labels.write_parquet(export)
        _echo(f"diagnostic export -> {export}", "info")
    _echo(
        f"target={charter.label.target_type.value} "
        f"horizon={charter.label.forecast_horizon_sessions} sessions "
        f"delay={charter.backtest.rebalance.execution_delay_sessions}",
        "info",
    )


@app.command("show-config")
def show_config(config: ConfigOpt = Path("configs/research_charter.yaml")) -> None:
    """Load, validate, and print the resolved configuration with its hash."""
    configure_logging("WARNING")
    import json

    from quant_platform.config import config_hash, load_charter

    try:
        charter = load_charter(config)
    except Exception as exc:
        _echo(f"Invalid configuration: {exc}", "err")
        raise typer.Exit(1) from exc

    _echo(f"config_hash: {config_hash(charter)}", "ok")
    print(json.dumps(charter.to_dict(), indent=2, default=str))


@app.command("show-features")
def show_features() -> None:
    """List the registered features and their declared contracts."""
    configure_logging("WARNING")
    import quant_platform.features.compute  # noqa: F401  (registers features on import)
    from quant_platform.features.registry import REGISTRY

    frame = REGISTRY.to_frame()
    _echo(f"{frame.height} registered features across {len(REGISTRY.families())} families", "ok")
    with __import__("polars").Config(tbl_rows=100, fmt_str_lengths=60, tbl_width_chars=200):
        print(frame.select(["name", "family", "lookback_sessions", "version"]))
    print(f"\nMax lookback: {REGISTRY.max_lookback()} sessions (pipeline warm-up requirement)")


@app.command("validate-data")
def validate_data(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    securities: Annotated[int, typer.Option("--securities", "-n")] = 120,
    strict: Annotated[bool, typer.Option("--strict/--no-strict")] = True,
) -> None:
    """Run all data-quality checks. Exits non-zero on critical issues when --strict."""
    configure_logging("INFO")
    from quant_platform.config import load_charter
    from quant_platform.data.validation import DataQualityError, validate_dataset
    from quant_platform.pipeline import load_fixture_data

    charter = load_charter(config)
    data = load_fixture_data(charter, n_securities=securities)
    report = validate_dataset(
        prices=data.prices,
        benchmark=data.benchmark,
        corporate_actions=data.corporate_actions,
        calendar_sessions=data.calendar.sessions,
    )
    print(report.to_frame())
    _echo(f"\n{report.summary()}", "ok" if report.is_clean else "warn")
    if strict:
        try:
            report.raise_if_critical()
        except DataQualityError as exc:
            _echo(str(exc), "err")
            raise typer.Exit(1) from exc


@app.command("build-universe")
def build_universe(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    export: Annotated[
        Path | None,
        typer.Option("--export", "--out", help="Optional Parquet export path."),
    ] = None,
    securities: Annotated[int, typer.Option("--securities", "-n")] = 120,
) -> None:
    """Build/load the point-in-time universe through the governed PIT cache."""
    configure_logging("INFO")
    import polars as pl

    _charter, snapshot, cached, data = _cached_fixture_data(config, securities)
    if export:
        export.parent.mkdir(parents=True, exist_ok=True)
        data.universe_panel.write_parquet(export)

    sizes = data.universe_panel.group_by("as_of").agg(pl.len().alias("n"))["n"]
    _echo(
        f"universe {data.universe_panel.shape} snapshot={snapshot.snapshot_id} "
        f"cache={'HIT' if cached.cache_hit else 'MISS'}",
        "ok",
    )
    if export:
        _echo(f"diagnostic export -> {export}", "info")
    _echo(
        f"eligible names per date: min={as_float(sizes.min(), context='min universe size'):.0f} "
        f"median={as_float(sizes.median(), context='median universe size'):.0f} "
        f"max={as_float(sizes.max(), context='max universe size'):.0f}",
        "info",
    )


@app.command("research-run")
def research_run(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    snapshot_id: Annotated[str, typer.Option("--snapshot-id")] = "auto",
    models: Annotated[
        str, typer.Option("--models", help="Comma-separated names or 'all'.")
    ] = "all",
    strategies: Annotated[
        str, typer.Option("--strategies", help="Comma-separated names or 'all'.")
    ] = "all",
    scenarios: Annotated[str, typer.Option("--scenarios")] = "base",
    portfolio_model: Annotated[str, typer.Option("--portfolio-model")] = "factor_composite",
    securities: Annotated[int, typer.Option("--securities", "-n")] = 120,
    workers: Annotated[int | None, typer.Option("--workers")] = None,
    results_db: Annotated[Path, typer.Option("--results-db")] = Path("research.duckdb"),
) -> None:
    """Run the governed cached, parallel, persisted research and strategy comparison."""
    configure_logging("INFO")
    from quant_platform.research.runner import (
        TARGET_MODELS,
        TARGET_STRATEGIES,
        ExperimentRunOrchestrator,
        ResearchRunRequest,
    )

    request = ResearchRunRequest(
        charter_path=config,
        snapshot_id=snapshot_id,
        models=_csv_option(models, TARGET_MODELS),
        strategies=_csv_option(strategies, TARGET_STRATEGIES),
        scenarios=_csv_option(scenarios, ("base", "double", "triple")),
        portfolio_model=portfolio_model.replace("-", "_"),
        fixture_securities=securities,
        max_workers=workers,
    )
    try:
        result = ExperimentRunOrchestrator(results_db=results_db).run(request)
    except Exception as exc:
        logger.exception("research run failed")
        _echo(f"Research run failed: {exc}", "err")
        raise typer.Exit(1) from exc
    _echo(f"run_id={result.run_id} status={result.status}", "ok")
    _echo(
        f"snapshot={result.snapshot_id} cache={'HIT' if result.cache_hit else 'MISS'} "
        f"total={result.timings.get('total_seconds', float('nan')):.2f}s",
        "info",
    )
    print(result.model_comparison)
    print(result.strategy_comparison)
    if result.failed_variants:
        _echo(f"{len(result.failed_variants)} strategy variant(s) failed loudly:", "warn")
        for failure in result.failed_variants:
            print(f"    {failure}")
    for name, path in sorted(result.report_paths.items()):
        print(f"    {name:16s} {path}")


@app.command("results")
def results_query(
    table: Annotated[str, typer.Option("--table")] = "research_run",
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    strategy: Annotated[str | None, typer.Option("--strategy")] = None,
    scenario: Annotated[str | None, typer.Option("--scenario")] = None,
    sql: Annotated[
        str | None, typer.Option("--sql", help="One read-only SELECT statement.")
    ] = None,
    export: Annotated[Path | None, typer.Option("--export")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 1000,
    results_db: Annotated[Path, typer.Option("--results-db")] = Path("research.duckdb"),
) -> None:
    """Query authoritative persisted results across runs, models, folds and strategies."""
    from quant_platform.research.results import ResultsReader, ResultsStoreError

    try:
        with ResultsReader(results_db) as reader:
            frame = (
                reader.query_sql(sql)
                if sql
                else reader.query(
                    table,
                    run_id=run_id,
                    model=model.replace("-", "_") if model else None,
                    strategy=strategy,
                    cost_scenario=scenario,
                    limit=limit,
                )
            )
            if export:
                reader.export_csv(frame, export)
    except ResultsStoreError as exc:
        _echo(str(exc), "err")
        raise typer.Exit(1) from exc
    print(frame)
    if export:
        _echo(f"exported {frame.height} row(s) -> {export}", "ok")


@app.command("train")
def train(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model name from the registry.")
    ] = "factor-composite",
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    out: OutOpt = Path("artifacts/models"),
    securities: Annotated[int, typer.Option("--securities", "-n")] = 200,
    test_start: Annotated[str, typer.Option("--test-start")] = "2021-01-04",
) -> None:
    """Run one model through governed folds and persist run-scoped artifacts."""
    configure_logging("INFO")
    from quant_platform.research.runner import ExperimentRunOrchestrator, ResearchRunRequest

    normalized = model.replace("-", "_")
    try:
        result = ExperimentRunOrchestrator(artifact_root=out).run(
            ResearchRunRequest(
                charter_path=config,
                models=(normalized,),
                strategies=("equal_weight",),
                scenarios=("base",),
                portfolio_model=normalized,
                fixture_securities=securities,
            )
        )
    except Exception as exc:
        _echo(f"Governed training run failed: {exc}", "err")
        raise typer.Exit(1) from exc
    _echo(
        f"run_id={result.run_id} model={normalized} status={result.status}; "
        f"artifacts -> {out / result.run_id / normalized}",
        "ok",
    )
    if test_start != "2021-01-04":
        _echo(
            "--test-start is retained for CLI compatibility; governed fold dates come from "
            "the validated charter.",
            "warn",
        )
    print(result.model_comparison)


@app.command("walk-forward")
def walk_forward(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    model: Annotated[str, typer.Option("--model", "-m")] = "factor-composite",
    securities: Annotated[int, typer.Option("--securities", "-n")] = 200,
    out: OutOpt = Path("reports/walk_forward"),
) -> None:
    """Thin fold-diagnostics view over the governed research-run infrastructure."""
    configure_logging("INFO")
    from quant_platform.research.runner import ExperimentRunOrchestrator, ResearchRunRequest

    normalized = model.replace("-", "_")
    try:
        result = ExperimentRunOrchestrator().run(
            ResearchRunRequest(
                charter_path=config,
                models=(normalized,),
                strategies=("equal_weight",),
                scenarios=("base",),
                portfolio_model=normalized,
                fixture_securities=securities,
            )
        )
    except Exception as exc:
        _echo(f"Walk-forward run failed: {exc}", "err")
        raise typer.Exit(1) from exc
    frame = result.model_comparison
    out.mkdir(parents=True, exist_ok=True)
    export = out / f"{result.run_id}_walk_forward.csv"
    frame.write_csv(export)
    print(frame)
    _echo(f"governed run {result.run_id}; fold diagnostics -> {export}", "ok")


@app.command("backtest")
def backtest(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    model: Annotated[str, typer.Option("--model", "-m")] = "factor-composite",
    portfolio: Annotated[str, typer.Option("--portfolio", "-p")] = "equal_weight",
    cost_scenario: Annotated[str, typer.Option("--cost")] = "base",
    securities: Annotated[int, typer.Option("--securities", "-n")] = 200,
    test_start: Annotated[str, typer.Option("--test-start")] = "2021-01-04",
) -> None:
    """Backtest one model/portfolio combination."""
    configure_logging("INFO")
    from quant_platform.config import load_charter
    from quant_platform.domain.enums import EvaluationStage
    from quant_platform.models.registry import create_model
    from quant_platform.pipeline import (
        build_training_frame,
        evaluate,
        fit_and_predict,
        load_fixture_data,
        run_backtest_for_predictions,
        split_train_test,
    )
    from quant_platform.utilities.reproducibility import set_global_seeds

    charter = load_charter(config)
    set_global_seeds(charter.random_seed)
    data = load_fixture_data(charter, n_securities=securities)
    panel = build_training_frame(data)
    start = date.fromisoformat(test_start)
    train_df, test_df = split_train_test(
        panel, start, charter.validation.embargo_sessions, data.calendar
    )
    instance = create_model(model, random_seed=charter.random_seed)
    preds = fit_and_predict(
        instance, train_df, test_df, data.feature_columns, "excess_return_rank", charter
    )
    result = run_backtest_for_predictions(preds, data, charter, portfolio, cost_scenario)
    report = evaluate(
        result,
        charter,
        data,
        EvaluationStage.WALK_FORWARD_TEST,
        start,
        as_date(test_df["as_of"].max(), "test_end"),
    )

    _echo(f"\n{model} / {portfolio} / cost={cost_scenario}", "ok")
    for key in (
        "cagr",
        "gross_cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "information_ratio",
        "beta",
        "total_costs",
        "n_trades",
    ):
        value = report.metrics.get(key)
        if value is not None:
            print(f"    {key:24s} {value:>14,.4f}")
    _echo(f"\n{report.disclosure_note}", "warn")


@app.command("stress-test")
def stress_test(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    model: Annotated[str, typer.Option("--model", "-m")] = "factor-composite",
    securities: Annotated[int, typer.Option("--securities", "-n")] = 200,
    out: OutOpt = Path("reports/stress"),
) -> None:
    """Run the adversarial stress suite (costs, delays, ablations, randomization)."""
    configure_logging("INFO")
    from quant_platform.demo import run_stress_suite

    try:
        frame = run_stress_suite(config, model=model, n_securities=securities, out_dir=out)
    except Exception as exc:
        logger.exception("stress test failed")
        _echo(f"Stress test failed: {exc}", "err")
        raise typer.Exit(1) from exc

    print(frame)
    _echo(f"\nstress results -> {out}", "ok")


@app.command("evaluate-holdout")
def evaluate_holdout(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    run_id: Annotated[str, typer.Option("--run-id")] = "",
    model: Annotated[str, typer.Option("--model")] = "factor_composite",
    acknowledged_by: Annotated[str, typer.Option("--acknowledged-by")] = "",
    results_db: Annotated[Path, typer.Option("--results-db")] = Path("research.duckdb"),
    acknowledge: Annotated[
        bool,
        typer.Option(
            "--i-understand-this-can-only-be-done-once",
            help="Required acknowledgement that evaluating the holdout consumes it.",
        ),
    ] = False,
) -> None:
    """Evaluate the LOCKED HOLDOUT. Can only be done honestly once."""
    configure_logging("INFO")
    from quant_platform.research.holdout import evaluate_locked_holdout

    if not run_id or not acknowledged_by:
        _echo("--run-id and --acknowledged-by are required for a permanent holdout record", "err")
        raise typer.Exit(1)
    try:
        metrics = evaluate_locked_holdout(
            run_id=run_id,
            model=model.replace("-", "_"),
            charter_path=config,
            acknowledged_by=acknowledged_by,
            acknowledge=acknowledge,
            results_db=results_db,
        )
    except Exception as exc:
        _echo(f"Holdout evaluation refused/failed: {exc}", "err")
        raise typer.Exit(1) from exc
    _echo("Holdout evaluation completed and permanently recorded.", "warn")
    for key, value in metrics.items():
        print(f"    {key:20s} {value}")


@app.command("paper-init")
def paper_init(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    capital: Annotated[float, typer.Option("--capital")] = 1_000_000.0,
) -> None:
    """Initialize a paper-trading account."""
    configure_logging("INFO")
    from quant_platform.config import load_charter
    from quant_platform.paper.state import PaperTradingState

    charter = load_charter(config)
    state_dir = Path(charter.paper_trading.state_dir)
    path = state_dir / "state.json"
    if path.exists():
        _echo(f"Paper state already exists at {path}. Delete it to start over.", "err")
        raise typer.Exit(1)
    state = PaperTradingState.initialize(capital)
    state.save(path)
    _echo(f"initialized paper account with ${capital:,.0f} -> {path}", "ok")


@app.command("paper-status")
def paper_status(config: ConfigOpt = Path("configs/research_charter.yaml")) -> None:
    """Show paper-trading account status and monitoring metrics."""
    configure_logging("WARNING")
    from quant_platform.config import load_charter
    from quant_platform.paper.rebalance import monitoring_report
    from quant_platform.paper.state import PaperTradingState

    charter = load_charter(config)
    path = Path(charter.paper_trading.state_dir) / "state.json"
    try:
        state = PaperTradingState.load(path)
    except FileNotFoundError as exc:
        _echo(str(exc), "err")
        raise typer.Exit(1) from exc

    report = monitoring_report(state)
    _echo(f"Paper account {state.account_id}", "ok")
    for key, value in report.items():
        if key == "disclaimer":
            continue
        formatted = f"{value:,.4f}" if isinstance(value, float) else str(value)
        print(f"    {key:26s} {formatted}")
    if state.pending_proposals:
        _echo(
            f"\n{len(state.pending_proposals)} order(s) awaiting approval. "
            f"Review them before executing.",
            "warn",
        )
    _echo(f"\n{report['disclaimer']}", "warn")


@app.command("paper-designate-model")
def paper_designate_model(
    run_id: Annotated[str, typer.Option("--run-id")],
    model: Annotated[str, typer.Option("--model")],
    designated_by: Annotated[str, typer.Option("--designated-by")],
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    results_db: Annotated[Path, typer.Option("--results-db")] = Path("research.duckdb"),
) -> None:
    """Explicitly designate the auditable research model used by the paper account."""
    from quant_platform.config import load_charter
    from quant_platform.paper.state import PaperTradingState

    charter = load_charter(config)
    path = Path(charter.paper_trading.state_dir) / "state.json"
    try:
        state = PaperTradingState.load(path)
        reference = state.designate_production_model(
            run_id, model.replace("-", "_"), designated_by, results_db
        )
        state.save(path)
    except Exception as exc:
        _echo(f"Production-model designation failed: {exc}", "err")
        raise typer.Exit(1) from exc
    _echo(f"designated {reference.run_id}:{reference.model} -> {path}", "ok")


@app.command("paper-rebalance")
def paper_rebalance(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    approve_all: Annotated[
        bool, typer.Option("--approve-all", help="Approve every proposed order.")
    ] = False,
    securities: Annotated[int, typer.Option("--securities", "-n")] = 200,
    results_db: Annotated[Path, typer.Option("--results-db")] = Path("research.duckdb"),
) -> None:
    """Generate paper-trading order proposals; execute only what is approved."""
    configure_logging("INFO")
    from quant_platform.demo import run_paper_rebalance

    try:
        summary = run_paper_rebalance(
            config,
            approve_all=approve_all,
            n_securities=securities,
            results_db=results_db,
        )
    except Exception as exc:
        logger.exception("paper rebalance failed")
        _echo(f"Paper rebalance failed: {exc}", "err")
        raise typer.Exit(1) from exc

    _echo(f"\nproposed {summary['n_proposed']} orders", "ok")
    for line in summary["proposals"][:15]:
        print(f"    {line}")
    if summary["n_proposed"] > 15:
        print(f"    ... and {summary['n_proposed'] - 15} more")

    if not approve_all:
        _echo(
            "\nNo orders were executed. Orders remain PROPOSALS until approved. "
            "Re-run with --approve-all to execute them in the simulator.",
            "warn",
        )
    else:
        _echo(
            f"\nexecuted {summary['n_filled']} fills; realized cost "
            f"${summary['realized_cost']:,.0f}",
            "ok",
        )
        _echo("Fills are SIMULATED. Real execution differs.", "warn")


@app.command("generate-report")
def generate_report_cmd(
    config: ConfigOpt = Path("configs/research_charter.yaml"),
    out: OutOpt = Path("reports/demo"),
    securities: Annotated[int, typer.Option("--securities", "-n")] = 400,
) -> None:
    """Generate the full report set (alias for run-demo's reporting stage)."""
    run_demo(config=config, out=out, securities=securities, test_start="2021-01-04")


@app.command("list-models")
def list_models() -> None:
    """List available models and which are production-eligible."""
    configure_logging("WARNING")
    from quant_platform.models.registry import (
        BASELINE_MODELS,
        available_models,
        production_candidates,
    )

    production = set(production_candidates())
    _echo("Registered models:", "info")
    for name in available_models():
        tags = []
        if name in BASELINE_MODELS:
            tags.append("baseline/control")
        if name not in production:
            tags.append("DIAGNOSTIC ONLY -- not production-eligible")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        print(f"    {name:20s}{suffix}")


@app.command("analyze")
def analyze(
    tickers: Annotated[str, typer.Argument(help="Ticker(s), comma-separated. e.g. AAPL,MSFT")],
    peers: Annotated[
        str | None,
        typer.Option("--peers", help="Comma-separated peer tickers, or a file with one per line."),
    ] = None,
    lookback_days: Annotated[int, typer.Option("--lookback-days")] = 600,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write JSON here.")] = None,
    explain: Annotated[
        bool, typer.Option("--explain", help="Explain what every value means.")
    ] = False,
) -> None:
    """Factor scorecard for real stocks, ranked against a peer universe.

    Downloads real prices and reports where each stock sits relative to its peers on
    momentum, reversal, defensive, and liquidity factors.

    Requires network access. Results are DESCRIPTIVE, not a forecast or recommendation.
    """
    configure_logging("INFO")
    import json

    from quant_platform.analysis import (
        DEFAULT_PEER_UNIVERSE,
        analyze_tickers,
        compare_scorecards,
        format_scorecard,
    )

    requested = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not requested:
        _echo("No tickers supplied.", "err")
        raise typer.Exit(1)

    peer_list: list[str] | None = None
    if peers:
        peer_path = Path(peers)
        if peer_path.exists():
            peer_list = [
                line.strip().upper()
                for line in peer_path.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
            _echo(f"using {len(peer_list)} peers from {peer_path}", "info")
        else:
            peer_list = [t.strip().upper() for t in peers.split(",") if t.strip()]

    _echo(
        f"Analyzing {', '.join(requested)} against "
        f"{len(peer_list) if peer_list else len(DEFAULT_PEER_UNIVERSE)} peers...",
        "info",
    )

    try:
        cards, failures = analyze_tickers(requested, peer_list, lookback_days)
    except Exception as exc:
        logger.exception("analysis failed")
        _echo(f"Analysis failed: {exc}", "err")
        _echo("This command needs network access to download prices.", "warn")
        raise typer.Exit(1) from exc

    for failure in failures:
        _echo(f"  could not score {failure}", "warn")
    if not cards:
        _echo("No tickers could be scored.", "err")
        raise typer.Exit(1)

    for card in cards:
        print(format_scorecard(card, explain=explain))

    if len(cards) > 1:
        _echo("COMPARISON (percentiles vs peers)", "ok")
        with __import__("polars").Config(tbl_rows=50, tbl_width_chars=160):
            print(compare_scorecards(cards))

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump([c.to_dict() for c in cards], fh, indent=2, default=str)
        _echo(f"\nJSON written to {out}", "ok")

    _echo(
        "\nDescriptive factor standings only. NOT investment advice, NOT a forecast, and "
        "NOT a recommendation to buy or sell anything.",
        "warn",
    )


@app.command("serve")
def serve_cmd(
    port: Annotated[int, typer.Option("--port", "-p")] = 8000,
    no_browser: Annotated[bool, typer.Option("--no-browser")] = False,
) -> None:
    """Serve the factor scorecard as a local web UI.

    Binds to 127.0.0.1 only -- the interface has no authentication and is not intended to be
    reachable from anywhere but this machine.
    """
    configure_logging("INFO")
    from quant_platform.web import HOST, serve

    _echo(f"Starting local server at http://{HOST}:{port}", "ok")
    _echo("Loopback only. Press Ctrl+C to stop.", "info")
    try:
        serve(port=port, open_browser=not no_browser)
    except RuntimeError as exc:
        _echo(str(exc), "err")
        raise typer.Exit(1) from exc


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
