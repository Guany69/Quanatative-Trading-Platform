"""End-to-end governed experiment-run orchestrator."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_platform.backtest.engine import BacktestEngine
from quant_platform.config import config_hash, load_charter, snapshot_config
from quant_platform.config.models import ResearchCharter
from quant_platform.data.adapters.fixture import FIXTURE_SOURCE, FixtureDataProvider, FixtureSpec
from quant_platform.data.snapshots import DataSnapshot, SnapshotStore
from quant_platform.domain.enums import CandidateStatus, EvaluationStage
from quant_platform.evaluation.metrics import sharpe_ratio
from quant_platform.evaluation.persisted import generate_persisted_run_report
from quant_platform.experiments.tracking import ExperimentRegistry, Trial
from quant_platform.models.registry import available_models
from quant_platform.pipeline import (
    PipelineData,
    evaluate,
    forecast_ic,
    run_backtest_for_predictions,
)
from quant_platform.research.executor import (
    TaskResult,
    TrainingExecutor,
    artifact_paths,
    assemble_ensemble_result,
    make_task,
)
from quant_platform.research.panels import (
    build_fold_panels,
    derive_fold_schedule,
    make_pit_cache_key,
    pipeline_data_from_cache,
)
from quant_platform.research.pit_cache import CachedFoldPanels, PitPanelCache, fold_schedule_hash
from quant_platform.research.results import ResultsStore
from quant_platform.utilities.narrow import as_date
from quant_platform.utilities.reproducibility import git_commit, git_is_dirty, set_global_seeds
from quant_platform.validation.overfitting import (
    analyze_sharpe,
    probability_of_backtest_overfitting,
)
from quant_platform.validation.stress import randomize_predictions
from quant_platform.validation.walk_forward import Fold

TARGET_MODELS = tuple(sorted(available_models()))
TARGET_STRATEGIES = (
    "equal_weight",
    "score_weighted",
    "inverse_volatility",
    "constrained_optimizer",
)
ENSEMBLE_MEMBERS = ("elastic_net", "lightgbm", "neural_network")


@dataclass(frozen=True)
class ResearchRunRequest:
    charter_path: str | Path = "configs/research_charter.yaml"
    snapshot_id: str | None = None
    models: tuple[str, ...] = TARGET_MODELS
    strategies: tuple[str, ...] = TARGET_STRATEGIES
    scenarios: tuple[str, ...] = ("base",)
    portfolio_model: str = "factor_composite"
    fixture_securities: int = 120
    max_workers: int | None = None


@dataclass
class ResearchRunResult:
    run_id: str
    status: str
    snapshot_id: str
    cache_hit: bool
    timings: dict[str, float]
    model_comparison: pl.DataFrame
    strategy_comparison: pl.DataFrame
    report_paths: dict[str, str]
    failed_variants: list[dict[str, str]] = field(default_factory=list)


class ExperimentRunOrchestrator:
    """Sole owner of one batch run and every shared research-result write."""

    def __init__(
        self,
        *,
        results_db: str | Path = "research.duckdb",
        snapshot_root: str | Path = "data/snapshots",
        cache_root: str | Path = "data/interim/pit_cache",
        artifact_root: str | Path = "artifacts/models",
        report_root: str | Path = "reports",
        registry_path: str | Path = "artifacts/experiments.jsonl",
    ) -> None:
        self.results_db = Path(results_db)
        self.snapshots = SnapshotStore(snapshot_root)
        self.cache = PitPanelCache(cache_root)
        self.artifact_root = Path(artifact_root)
        self.report_root = Path(report_root)
        self.registry = ExperimentRegistry(registry_path)

    def run(self, request: ResearchRunRequest) -> ResearchRunResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        charter = load_charter(request.charter_path)
        self._validate_request(request, charter)
        set_global_seeds(charter.random_seed)

        snapshot_started = time.perf_counter()
        snapshot = self._resolve_snapshot(request, charter)
        timings["snapshot_seconds"] = time.perf_counter() - snapshot_started
        folds, calendar = derive_fold_schedule(snapshot, charter)
        schedule_hash = fold_schedule_hash(folds)
        run_id = uuid.uuid4().hex[:16]
        stage = "initialization"

        with ResultsStore(self.results_db) as store:
            store.recover_stale_runs()
            store.create_run(
                run_id=run_id,
                snapshot_id=snapshot.snapshot_id,
                charter_hash=config_hash(charter),
                code_version=git_commit(),
                code_dirty=git_is_dirty(),
                seed=charter.random_seed,
                models=list(request.models),
                strategies=list(request.strategies),
                scenarios=list(request.scenarios),
                fold_schedule_hash=schedule_hash,
            )
            try:
                stage = "pit_cache"
                cache_started = time.perf_counter()
                cached = self.cache.get_or_build(
                    make_pit_cache_key(snapshot, charter, schedule_hash),
                    lambda: build_fold_panels(snapshot, charter, folds),
                )
                timings["panel_cache_seconds"] = time.perf_counter() - cache_started
                data = pipeline_data_from_cache(cached, charter, calendar)
                store.write_folds(run_id, _fold_frame(folds, charter))

                stage = "model_training"
                training_started = time.perf_counter()
                task_results = self._train_models(run_id, request, charter, cached)
                timings["model_training_seconds"] = time.perf_counter() - training_started
                comparison_results = [
                    result for result in task_results if result.model in set(request.models)
                ]
                self._persist_model_stage(store, comparison_results, task_results, cached, data)

                stage = "portfolio_simulation"
                simulation_started = time.perf_counter()
                failed_variants, curves, strategy_metrics = self._run_portfolios(
                    store, run_id, request, charter, cached, data, comparison_results
                )
                timings["portfolio_simulation_seconds"] = time.perf_counter() - simulation_started

                stage = "overfitting_stress"
                self._persist_overfitting(
                    store,
                    run_id,
                    request,
                    cached,
                    comparison_results,
                    curves,
                    strategy_metrics,
                    charter,
                )

                trial = self._trial(
                    run_id,
                    request,
                    charter,
                    snapshot,
                    comparison_results,
                    strategy_metrics,
                    CandidateStatus.RESEARCH_CANDIDATE.value,
                )
                self.registry.record(trial)

                stage = "reporting"
                report_dir = self.report_root / run_id
                snapshot_config(charter, report_dir / "resolved_config.json")
                report_paths = generate_persisted_run_report(store, run_id, report_dir)
                timings["total_seconds"] = time.perf_counter() - started
                benchmark_path = report_dir / "benchmark.json"
                benchmark_path.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "cache_hit": cached.cache_hit,
                            "max_workers": request.max_workers or (os.cpu_count() or 1),
                            "model_fold_tasks": len(task_results),
                            "timings": timings,
                            "reference_workload_note": (
                                "Measured on the requested local fixture; not a fabricated "
                                "claim about the HLD reference hardware/workload."
                            ),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                report_paths["benchmark"] = str(benchmark_path)
                store.finish_run(run_id, "COMPLETED", report_path=str(report_dir))

                model_comparison = store.query(
                    "SELECT model, fold, metric, value FROM fold_metric WHERE run_id=? "
                    "ORDER BY model, fold, metric",
                    [run_id],
                )
                strategy_comparison = store.query(
                    "SELECT model, strategy, cost_scenario, metric, value FROM strategy_result "
                    "WHERE run_id=? ORDER BY strategy, cost_scenario, metric",
                    [run_id],
                )
                return ResearchRunResult(
                    run_id=run_id,
                    status="COMPLETED",
                    snapshot_id=snapshot.snapshot_id,
                    cache_hit=cached.cache_hit,
                    timings=timings,
                    model_comparison=model_comparison,
                    strategy_comparison=strategy_comparison,
                    report_paths=report_paths,
                    failed_variants=failed_variants,
                )
            except Exception as exc:
                try:
                    store.finish_run(
                        run_id,
                        "FAILED",
                        failure_stage=stage,
                        failure_cause=f"{type(exc).__name__}: {exc}",
                    )
                finally:
                    self.registry.record_rejection(
                        self._trial(
                            run_id,
                            request,
                            charter,
                            snapshot,
                            [],
                            [],
                            CandidateStatus.REJECTED.value,
                        ),
                        f"{stage}: {type(exc).__name__}: {exc}",
                    )
                raise

    @staticmethod
    def _validate_request(request: ResearchRunRequest, charter: ResearchCharter) -> None:
        unknown_models = set(request.models) - set(TARGET_MODELS)
        if unknown_models or not request.models:
            raise ValueError(f"invalid model selection: {sorted(unknown_models)}")
        unknown_strategies = set(request.strategies) - set(TARGET_STRATEGIES)
        if unknown_strategies or not request.strategies:
            raise ValueError(f"invalid strategy selection: {sorted(unknown_strategies)}")
        if request.portfolio_model not in request.models:
            raise ValueError("portfolio_model must be included in the requested model set")
        unknown_scenarios = set(request.scenarios) - set(charter.costs.scenario_multipliers)
        if unknown_scenarios or not request.scenarios:
            raise ValueError(f"invalid cost scenarios: {sorted(unknown_scenarios)}")

    def _resolve_snapshot(
        self, request: ResearchRunRequest, charter: ResearchCharter
    ) -> DataSnapshot:
        if request.snapshot_id and request.snapshot_id not in {"fixture", "auto"}:
            return self.snapshots.load(request.snapshot_id)
        spec = FixtureSpec(n_securities=request.fixture_securities, seed=charter.random_seed)
        provenance = {"provider": "FixtureDataProvider", "spec": asdict(spec)}
        existing = self.snapshots.find_by_provenance(
            sources=[FIXTURE_SOURCE], provenance=provenance
        )
        if existing is not None:
            return existing
        dataset = FixtureDataProvider(spec).generate()
        return self.snapshots.create(
            {
                "securities": dataset.securities,
                "identifiers": dataset.identifiers,
                "prices": dataset.prices,
                "benchmark": dataset.benchmark,
                "membership": dataset.membership,
                "corporate_actions": dataset.corporate_actions,
                "fundamentals": dataset.fundamentals,
                "macro": dataset.macro,
            },
            sources=[FIXTURE_SOURCE],
            provenance=provenance,
        )

    @staticmethod
    def _train_models(
        run_id: str,
        request: ResearchRunRequest,
        charter: ResearchCharter,
        cached: CachedFoldPanels,
    ) -> list[TaskResult]:
        base_models = set(request.models) - {"ensemble"}
        if "ensemble" in request.models:
            base_models.update(ENSEMBLE_MEMBERS)
        feature_columns = [
            column
            for column in cached.read_auxiliary("features").columns
            if column not in {"security_id", "as_of"}
        ]
        tasks = [
            make_task(
                run_id=run_id,
                model=model,
                panel=panel,
                feature_columns=feature_columns,
                target_column="excess_return_rank",
                charter=charter,
            )
            for model in sorted(base_models)
            for panel in cached.panels
        ]
        results = TrainingExecutor(max_workers=request.max_workers).map(tasks)
        if "ensemble" in request.models:
            by_key = {(result.model, result.fold): result for result in results}
            for panel in cached.panels:
                members = {name: by_key[(name, panel.fold.index)] for name in ENSEMBLE_MEMBERS}
                results.append(
                    assemble_ensemble_result(
                        run_id=run_id,
                        fold_panel=panel,
                        members=members,
                        run_seed=charter.random_seed,
                    )
                )
        return sorted(results, key=lambda result: (result.model, result.fold))

    def _persist_model_stage(
        self,
        store: ResultsStore,
        comparison_results: list[TaskResult],
        all_results: list[TaskResult],
        cached: CachedFoldPanels,
        data: PipelineData,
    ) -> None:
        artifact_rows = []
        for result in all_results:
            artifact_path, metadata_path = artifact_paths(self.artifact_root, result)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(result.artifact)
            metadata_path.write_text(json.dumps(result.metadata, indent=2, default=str) + "\n")
            artifact_rows.append(
                {
                    "run_id": result.run_id,
                    "model": result.model,
                    "fold": result.fold,
                    "artifact_path": str(artifact_path),
                    "metadata_path": str(metadata_path),
                }
            )
        store.write_artifacts(pl.DataFrame(artifact_rows))

        panel_by_fold = {panel.fold.index: panel for panel in cached.panels}
        prediction_frames = []
        metric_rows = []
        for result in comparison_results:
            prediction_frames.append(
                result.predictions.rename(
                    {"prediction": "score", "cross_sectional_rank": "rank"}
                ).with_columns(
                    pl.lit(result.run_id).alias("run_id"),
                    pl.lit(result.model).alias("model"),
                    pl.lit(result.fold).alias("fold"),
                )
            )
            test = pl.read_parquet(panel_by_fold[result.fold].test_path)
            metrics = forecast_ic(result.predictions, test, "excess_return_rank")
            result.metadata.setdefault("extra", {})["forecast_metrics"] = metrics
            for metric, value in metrics.items():
                metric_rows.append(
                    {
                        "run_id": result.run_id,
                        "model": result.model,
                        "fold": result.fold,
                        "metric": metric,
                        "value": value,
                    }
                )
        store.write_predictions(pl.concat(prediction_frames, how="vertical_relaxed"))
        store.write_fold_metrics(pl.DataFrame(metric_rows))

    def _run_portfolios(
        self,
        store: ResultsStore,
        run_id: str,
        request: ResearchRunRequest,
        charter: ResearchCharter,
        cached: CachedFoldPanels,
        data: PipelineData,
        task_results: list[TaskResult],
    ) -> tuple[list[dict[str, str]], dict[str, pl.DataFrame], list[dict[str, Any]]]:
        selected = [result for result in task_results if result.model == request.portfolio_model]
        predictions = pl.concat([result.predictions for result in selected]).unique(
            ["security_id", "as_of"], keep="last"
        )
        if predictions.is_empty():
            raise RuntimeError(f"no predictions for portfolio model {request.portfolio_model}")
        start = as_date(predictions["as_of"].min(), "portfolio start")
        end = as_date(predictions["as_of"].max(), "portfolio end")
        preparer = BacktestEngine(charter.backtest, charter.costs, data.calendar)
        prepared = preparer.prepare_inputs(
            data.prices, data.benchmark, data.securities, data.corporate_actions
        )
        failed: list[dict[str, str]] = []
        curves: dict[str, pl.DataFrame] = {}
        metric_rows: list[dict[str, Any]] = []
        execution_rows: list[dict[str, Any]] = []
        equity_frames: list[pl.DataFrame] = []
        trade_frames: list[pl.DataFrame] = []
        target_frames: list[pl.DataFrame] = []
        diagnostic_rows: list[dict[str, Any]] = []

        for strategy in request.strategies:
            persisted_targets = False
            for scenario in request.scenarios:
                key = f"{strategy}:{scenario}"
                try:
                    result = run_backtest_for_predictions(
                        predictions,
                        data,
                        charter,
                        strategy,
                        scenario,
                        strategy_name=strategy,
                        prepared=prepared,
                    )
                    report = evaluate(
                        result,
                        charter,
                        data,
                        EvaluationStage.WALK_FORWARD_TEST,
                        start,
                        end,
                    )
                    curves[key] = result.equity_curve().filter(
                        (pl.col("as_of") >= start) & (pl.col("as_of") <= end)
                    )
                    for metric, value in report.metrics.items():
                        metric_rows.append(
                            {
                                "run_id": run_id,
                                "model": request.portfolio_model,
                                "strategy": strategy,
                                "cost_scenario": scenario,
                                "metric": metric,
                                "value": value,
                            }
                        )
                    equity_frames.append(
                        curves[key].with_columns(
                            pl.lit(run_id).alias("run_id"),
                            pl.lit(request.portfolio_model).alias("model"),
                            pl.lit(strategy).alias("strategy"),
                            pl.lit(scenario).alias("cost_scenario"),
                        )
                    )
                    trades = result.trade_log()
                    if not trades.is_empty():
                        trade_frames.append(
                            trades.with_columns(
                                pl.lit(run_id).alias("run_id"),
                                pl.lit(request.portfolio_model).alias("model"),
                                pl.lit(strategy).alias("strategy"),
                                pl.lit(scenario).alias("cost_scenario"),
                            )
                        )
                    if not persisted_targets:
                        target_rows = [
                            {
                                "run_id": run_id,
                                "model": request.portfolio_model,
                                "strategy": strategy,
                                "as_of": as_of,
                                "security_id": security_id,
                                "weight": weight,
                            }
                            for as_of, weights in result.target_weights.items()
                            for security_id, weight in weights.items()
                        ]
                        if target_rows:
                            target_frames.append(pl.DataFrame(target_rows))
                        for as_of, diag in result.diagnostics.items():
                            diagnostic_rows.append(
                                {
                                    "run_id": run_id,
                                    "model": request.portfolio_model,
                                    "strategy": strategy,
                                    "as_of": as_of,
                                    "solver_status": diag.solver_status.value,
                                    "solver_name": diag.solver_name,
                                    "objective_value": diag.objective_value,
                                    "expected_return": diag.expected_return,
                                    "expected_risk": diag.expected_risk,
                                    "expected_turnover": diag.expected_turnover,
                                    "expected_cost": diag.expected_cost,
                                    "relaxations_json": json.dumps(diag.relaxations_applied),
                                    "message": diag.message,
                                }
                            )
                        persisted_targets = True
                    execution_rows.append(
                        {
                            "run_id": run_id,
                            "model": request.portfolio_model,
                            "strategy": strategy,
                            "cost_scenario": scenario,
                            "status": "COMPLETED",
                            "failure_cause": None,
                        }
                    )
                except Exception as exc:
                    failure = {
                        "strategy": strategy,
                        "cost_scenario": scenario,
                        "cause": f"{type(exc).__name__}: {exc}",
                    }
                    failed.append(failure)
                    execution_rows.append(
                        {
                            "run_id": run_id,
                            "model": request.portfolio_model,
                            "strategy": strategy,
                            "cost_scenario": scenario,
                            "status": "FAILED",
                            "failure_cause": failure["cause"],
                        }
                    )

        store.write_strategy_execution(pl.DataFrame(execution_rows))
        if not metric_rows:
            raise RuntimeError(f"every requested strategy variant failed: {failed}")
        store.write_strategy_results(pl.DataFrame(metric_rows))
        if equity_frames:
            store.write_strategy_equity(pl.concat(equity_frames, how="vertical_relaxed"))
        if trade_frames:
            store.write_trades(pl.concat(trade_frames, how="vertical_relaxed"))
        if target_frames:
            store.write_target_weights(pl.concat(target_frames, how="vertical_relaxed"))
        if diagnostic_rows:
            store.write_optimizer_diagnostics(
                pl.DataFrame(diagnostic_rows, infer_schema_length=None)
            )
        return failed, curves, metric_rows

    def _persist_overfitting(
        self,
        store: ResultsStore,
        run_id: str,
        request: ResearchRunRequest,
        cached: CachedFoldPanels,
        task_results: list[TaskResult],
        curves: dict[str, pl.DataFrame],
        strategy_metrics: list[dict[str, Any]],
        charter: ResearchCharter,
    ) -> None:
        registry_ids = self.registry.run_ids()
        prior = store.query(
            "SELECT run_id FROM research_run WHERE run_id<>? AND status IN ('COMPLETED','FAILED')",
            [run_id],
        )
        missing_links = (
            set(prior["run_id"].to_list()) - registry_ids if not prior.is_empty() else set()
        )
        optimistic = bool(missing_links)
        rows: list[dict[str, Any]] = []

        primary_curve = next(iter(curves.values()))
        returns = primary_curve["net_return"].fill_null(0.0).to_numpy()
        n_trials = self.registry.trial_count() + 1
        sharpe_stats = analyze_sharpe(
            returns,
            n_trials=n_trials,
            all_trial_sharpes=[*self.registry.sharpe_ratios(), sharpe_ratio(returns)],
        )
        rows.append(
            _overfit_row(
                run_id,
                "deflated_sharpe",
                sharpe_stats.deflated_sharpe,
                optimistic or sharpe_stats.deflated_sharpe is None,
                {"n_trials": n_trials, "missing_run_links": sorted(missing_links)},
            )
        )

        history = store.query(
            "SELECT run_id, model, strategy, cost_scenario, as_of, net_return "
            "FROM strategy_equity ORDER BY run_id, model, strategy, cost_scenario, as_of"
        )
        aligned = [
            group["net_return"].fill_null(0.0).to_numpy()
            for _identity, group in history.group_by(
                ["run_id", "model", "strategy", "cost_scenario"], maintain_order=True
            )
            if group.height > 1
        ]
        if len(aligned) < 2:
            aligned = [frame["net_return"].fill_null(0.0).to_numpy() for frame in curves.values()]
        minimum = min(map(len, aligned))
        matrix = np.column_stack([values[:minimum] for values in aligned])
        pbo = probability_of_backtest_overfitting(matrix)
        pbo["n_historical_configurations"] = float(len(aligned))
        pbo["n_historical_runs"] = float(history["run_id"].n_unique())
        rows.append(_overfit_row(run_id, "pbo", pbo.get("pbo"), optimistic, pbo))

        selected = [result for result in task_results if result.model == request.portfolio_model]
        predictions = pl.concat([result.predictions for result in selected])
        tests = pl.concat([pl.read_parquet(panel.test_path) for panel in cached.panels])
        placebo = forecast_ic(
            randomize_predictions(predictions, charter.random_seed),
            tests,
            "excess_return_rank",
        )
        rows.append(
            _overfit_row(run_id, "placebo_mean_ic", placebo.get("mean_ic"), optimistic, placebo)
        )

        cagr_values = [
            float(row["value"])
            for row in strategy_metrics
            if row["metric"] == "cagr" and row["value"] is not None
        ]
        rows.append(
            _overfit_row(
                run_id,
                "cost_stress_worst_cagr",
                min(cagr_values) if cagr_values else None,
                optimistic,
                {"scenarios": list(request.scenarios)},
            )
        )
        midpoint = len(returns) // 2
        rows.extend(
            [
                _overfit_row(
                    run_id,
                    "regime_first_half_sharpe",
                    sharpe_ratio(returns[:midpoint]),
                    optimistic,
                    {"observations": midpoint},
                ),
                _overfit_row(
                    run_id,
                    "regime_second_half_sharpe",
                    sharpe_ratio(returns[midpoint:]),
                    optimistic,
                    {"observations": len(returns) - midpoint},
                ),
            ]
        )
        store.write_overfitting_metrics(pl.DataFrame(rows))

    @staticmethod
    def _trial(
        run_id: str,
        request: ResearchRunRequest,
        charter: ResearchCharter,
        snapshot: DataSnapshot,
        task_results: list[TaskResult],
        strategy_metrics: list[dict[str, Any]],
        status: str,
    ) -> Trial:
        fold_metrics: dict[str, float] = {}
        portfolio_metrics: dict[str, float] = {}
        metric_values: dict[str, list[float]] = {}
        for result in task_results:
            for metric, value in (
                result.metadata.get("extra", {}).get("forecast_metrics", {}).items()
            ):
                if isinstance(value, (int, float)) and np.isfinite(value):
                    metric_values.setdefault(metric, []).append(float(value))
        fold_metrics.update(
            {metric: float(np.mean(values)) for metric, values in metric_values.items()}
        )
        for row in strategy_metrics:
            if row.get("metric") in {"sharpe_ratio", "information_ratio", "cagr"}:
                portfolio_metrics[f"{row['strategy']}:{row['cost_scenario']}:{row['metric']}"] = (
                    row["value"]
                )
                if (
                    row["strategy"] == request.strategies[0]
                    and row["cost_scenario"] == request.scenarios[0]
                ):
                    portfolio_metrics[str(row["metric"])] = float(row["value"])
        return Trial(
            run_id=run_id,
            name=charter.name,
            model_class=",".join(request.models),
            hyperparameters={
                result.model: result.metadata.get("hyperparameters", {})
                for result in task_results
                if result.fold == 0
            },
            random_seed=charter.random_seed,
            config_hash=config_hash(charter),
            data_snapshot_id=snapshot.snapshot_id,
            feature_version=charter.features.feature_version,
            label_version=charter.label.label_version,
            universe=charter.universe.name,
            cost_assumptions=charter.costs.model_dump(mode="json"),
            portfolio_settings=charter.portfolio.model_dump(mode="json"),
            forecast_metrics=fold_metrics,
            portfolio_metrics=portfolio_metrics,
            status=status,
        )


def _fold_frame(folds: list[Fold], charter: ResearchCharter) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "fold": fold.index,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "purge": charter.validation.purge_enabled,
                "embargo": charter.validation.embargo_sessions,
            }
            for fold in folds
        ]
    )


def _overfit_row(
    run_id: str,
    metric: str,
    value: float | None,
    optimistic: bool,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "metric": metric,
        "value": float(value) if value is not None and np.isfinite(value) else None,
        "optimistic": optimistic,
        "details_json": json.dumps(details, default=str, sort_keys=True),
    }
