"""Purpose-specific research reads over the authoritative ResultsReader."""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
from pydantic import BaseModel

from quant_platform.api.config import ApiSettings
from quant_platform.api.errors import ApiError
from quant_platform.api.schemas.common import Page, json_safe, page_of
from quant_platform.api.schemas.research import (
    ArtifactRecord,
    EquityRecord,
    FoldRecord,
    ModelMetricRecord,
    OptimizerDiagnosticRecord,
    OverfittingRecord,
    PredictionRecord,
    RunDetail,
    RunFailure,
    RunSummaryResponse,
    StrategyMetricRecord,
    TradeRecord,
    WeightRecord,
)
from quant_platform.api.services.research_jobs import ResearchJob, ResearchRunJobManager
from quant_platform.research.results import ResultsReader

STAGES = (
    "INITIALIZATION",
    "PIT_CACHE",
    "MODEL_TRAINING",
    "PORTFOLIO_SIMULATION",
    "OVERFITTING_STRESS",
    "REPORTING",
)

TModel = TypeVar("TModel", bound=BaseModel)


class ResearchQueryService:
    def __init__(self, settings: ApiSettings, jobs: ResearchRunJobManager) -> None:
        self.settings = settings
        self.jobs = jobs

    def list_runs(self, *, page: int, limit: int, state: str | None = None) -> Page[RunDetail]:
        rows = self._fetch("SELECT * FROM research_run ORDER BY created_at DESC")
        db_by_id = {str(row["run_id"]): row for row in rows.to_dicts()}
        job_by_id = {job.run_id: job for job in self.jobs.list()}
        run_ids = set(db_by_id) | set(job_by_id)
        items = [
            self._run_detail(run_id, db_by_id.get(run_id), job_by_id.get(run_id))
            for run_id in run_ids
        ]
        if state:
            items = [item for item in items if item.state == state]
        items.sort(key=lambda item: item.created_at, reverse=True)
        total = len(items)
        start = (page - 1) * limit
        return page_of(items[start : start + limit], page=page, limit=limit, total=total)

    def run_detail(self, run_id: str) -> RunDetail:
        job = self.jobs.get(run_id)
        rows = self._fetch("SELECT * FROM research_run WHERE run_id=?", [run_id])
        row = rows.to_dicts()[0] if rows.height else None
        if row is None and job is None:
            raise ApiError("RUN_NOT_FOUND", f"Unknown research run '{run_id}'.", status_code=404)
        return self._run_detail(run_id, row, job)

    def summary(self, run_id: str) -> RunSummaryResponse:
        detail = self.run_detail(run_id)
        counts: dict[str, int] = {}
        for table in (
            "fold",
            "prediction",
            "fold_metric",
            "strategy_result",
            "strategy_equity",
            "trade",
            "target_weight",
            "optimizer_diagnostic",
            "overfitting_metric",
        ):
            row = self._fetch(f'SELECT count(*) AS count FROM "{table}" WHERE run_id=?', [run_id])
            counts[table] = int(row["count"][0])
        cache_hit: bool | None = None
        timings: dict[str, float | None] = {}
        benchmark = self.settings.report_root / run_id / "benchmark.json"
        if benchmark.is_file() and benchmark.parent.resolve().is_relative_to(
            self.settings.report_root.resolve()
        ):
            try:
                payload = json.loads(benchmark.read_text())
                cache_hit = bool(payload.get("cache_hit"))
                timings = json_safe(payload.get("timings") or {})
            except (OSError, ValueError):
                pass
        return RunSummaryResponse(
            run=detail,
            result_counts=counts,
            cache_hit=cache_hit,
            timings=timings,
        )

    def folds(self, run_id: str, page: int, limit: int) -> Page[FoldRecord]:
        return self._page(
            run_id,
            table="fold",
            columns=(
                "fold",
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
                "purge",
                "embargo",
            ),
            model=FoldRecord,
            page=page,
            limit=limit,
            order="fold",
        )

    def model_metrics(
        self, run_id: str, page: int, limit: int, model_name: str | None = None
    ) -> Page[ModelMetricRecord]:
        return self._page(
            run_id,
            table="fold_metric",
            columns=("model", "fold", "metric", "value"),
            model=ModelMetricRecord,
            page=page,
            limit=limit,
            order="model, fold, metric",
            filters={"model": model_name},
        )

    def predictions(
        self,
        run_id: str,
        page: int,
        limit: int,
        *,
        model_name: str | None = None,
        fold: int | None = None,
        security: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Page[PredictionRecord]:
        return self._page(
            run_id,
            table="prediction",
            columns=("model", "fold", "as_of", "security_id", "score", "rank"),
            model=PredictionRecord,
            page=page,
            limit=limit,
            order="as_of, security_id, model, fold",
            filters={"model": model_name, "fold": fold, "security_id": security},
            date_column="as_of",
            start_date=start_date,
            end_date=end_date,
        )

    def strategy_metrics(
        self, run_id: str, page: int, limit: int, **filters: Any
    ) -> Page[StrategyMetricRecord]:
        return self._page(
            run_id,
            table="strategy_result",
            columns=("model", "strategy", "cost_scenario", "metric", "value"),
            model=StrategyMetricRecord,
            page=page,
            limit=limit,
            order="model, strategy, cost_scenario, metric",
            filters=filters,
        )

    def equity(self, run_id: str, page: int, limit: int, **filters: Any) -> Page[EquityRecord]:
        return self._page(
            run_id,
            table="strategy_equity",
            columns=(
                "model",
                "strategy",
                "cost_scenario",
                "as_of",
                "total_value",
                "gross_return",
                "net_return",
                "benchmark_return",
                "turnover",
                "costs",
            ),
            model=EquityRecord,
            page=page,
            limit=limit,
            order="as_of",
            filters=filters,
        )

    def trades(
        self, run_id: str, page: int, limit: int, *, security: str | None = None, **filters: Any
    ) -> Page[TradeRecord]:
        filters["security_id"] = security
        return self._page(
            run_id,
            table="trade",
            columns=(
                "model",
                "strategy",
                "cost_scenario",
                "information_date",
                "signal_date",
                "order_date",
                "fill_date",
                "accounting_date",
                "security_id",
                "side",
                "shares",
                "reference_price",
                "fill_price",
                "notional",
                "commission",
                "spread_cost",
                "slippage_cost",
                "impact_cost",
            ),
            model=TradeRecord,
            page=page,
            limit=limit,
            order="fill_date DESC, security_id",
            filters=filters,
        )

    def weights(
        self,
        run_id: str,
        page: int,
        limit: int,
        *,
        security: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **filters: Any,
    ) -> Page[WeightRecord]:
        filters["security_id"] = security
        return self._page(
            run_id,
            table="target_weight",
            columns=("model", "strategy", "as_of", "security_id", "weight"),
            model=WeightRecord,
            page=page,
            limit=limit,
            order="as_of DESC, security_id",
            filters=filters,
            date_column="as_of",
            start_date=start_date,
            end_date=end_date,
        )

    def optimizer_diagnostics(
        self, run_id: str, page: int, limit: int, **filters: Any
    ) -> Page[OptimizerDiagnosticRecord]:
        self.run_detail(run_id)
        predicates, values = self._predicates(run_id, filters)
        total = self._count("optimizer_diagnostic", predicates, values)
        sql = (
            "SELECT model, strategy, as_of, solver_status, solver_name, objective_value, "
            "expected_return, expected_risk, expected_turnover, expected_cost, "
            "relaxations_json, message FROM optimizer_diagnostic WHERE "
            + " AND ".join(predicates)
            + " ORDER BY as_of DESC LIMIT ? OFFSET ?"
        )
        rows = self._fetch(sql, [*values, limit, (page - 1) * limit]).to_dicts()
        items = []
        for row in rows:
            row["relaxations"] = json.loads(row.pop("relaxations_json") or "[]")
            items.append(OptimizerDiagnosticRecord.model_validate(json_safe(row)))
        return page_of(items, page=page, limit=limit, total=total)

    def overfitting(self, run_id: str) -> list[OverfittingRecord]:
        self.run_detail(run_id)
        rows = self._fetch(
            "SELECT metric, value, optimistic, details_json FROM overfitting_metric "
            "WHERE run_id=? ORDER BY metric",
            [run_id],
        ).to_dicts()
        return [
            OverfittingRecord(
                metric=row["metric"],
                value=json_safe(row["value"]),
                optimistic=bool(row["optimistic"]),
                details=json_safe(json.loads(row["details_json"] or "{}")),
            )
            for row in rows
        ]

    def artifacts(self, run_id: str) -> list[ArtifactRecord]:
        self.run_detail(run_id)
        rows = self._fetch(
            "SELECT model, fold, artifact_path, metadata_path FROM artifact "
            "WHERE run_id=? ORDER BY model, fold",
            [run_id],
        ).to_dicts()
        return [
            ArtifactRecord(
                id=f"{row['model']}:{row['fold']}",
                model=row["model"],
                fold=row["fold"],
                artifact_available=Path(row["artifact_path"]).is_file(),
                metadata_available=Path(row["metadata_path"]).is_file(),
            )
            for row in rows
        ]

    def _page(
        self,
        run_id: str,
        *,
        table: str,
        columns: tuple[str, ...],
        model: type[TModel],
        page: int,
        limit: int,
        order: str,
        filters: dict[str, Any] | None = None,
        date_column: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Page[TModel]:
        self.run_detail(run_id)
        predicates, values = self._predicates(run_id, filters or {})
        if date_column and start_date:
            predicates.append(f"{date_column} >= ?")
            values.append(start_date)
        if date_column and end_date:
            predicates.append(f"{date_column} <= ?")
            values.append(end_date)
        total = self._count(table, predicates, values)
        selected = ", ".join(columns)
        sql = (
            f'SELECT {selected} FROM "{table}" WHERE {" AND ".join(predicates)} '
            f"ORDER BY {order} LIMIT ? OFFSET ?"
        )
        rows = self._fetch(sql, [*values, limit, (page - 1) * limit]).to_dicts()
        items = [model.model_validate(json_safe(row)) for row in rows]
        return page_of(items, page=page, limit=limit, total=total)

    @staticmethod
    def _predicates(run_id: str, filters: dict[str, Any]) -> tuple[list[str], list[Any]]:
        predicates = ["run_id = ?"]
        values: list[Any] = [run_id]
        for column, value in filters.items():
            if value is not None:
                predicates.append(f'"{column}" = ?')
                values.append(value)
        return predicates, values

    def _count(self, table: str, predicates: list[str], values: list[Any]) -> int:
        row = self._fetch(
            f'SELECT count(*) AS count FROM "{table}" WHERE {" AND ".join(predicates)}', values
        )
        return int(row["count"][0])

    def _fetch(self, sql: str, parameters: list[Any] | None = None) -> pl.DataFrame:
        if not self.settings.results_db.exists():
            return pl.DataFrame()
        with ResultsReader(self.settings.results_db) as reader:
            return reader.fetch(sql, parameters)

    @staticmethod
    def _json_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return list(json.loads(value))
        return list(value)

    def _run_detail(
        self, run_id: str, row: dict[str, Any] | None, job: ResearchJob | None
    ) -> RunDetail:
        if row is None and job is None:
            raise ApiError("RUN_NOT_FOUND", f"Unknown research run '{run_id}'.", status_code=404)
        row_data = row or {}
        job_request = job.request if job else {}
        state = job.state if job else str(row_data["status"])
        stage = job.stage if job and job.stage else row_data.get("current_stage")
        if state == "COMPLETED":
            completed_stages = len(STAGES)
        elif stage in STAGES:
            completed_stages = STAGES.index(stage)
        else:
            completed_stages = 0
        failure_message = (
            job.failure_message if job and job.failure_message else row_data.get("failure_cause")
        )
        failure_stage = row_data.get("failure_stage") or stage
        failure = None
        if failure_message:
            failure = RunFailure(
                code=(job.failure_code if job and job.failure_code else "RUN_FAILED"),
                message=failure_message,
                stage=failure_stage,
            )
        created = job.created_at if job else json_safe(row_data["created_at"])
        started = (
            job.started_at if job and job.started_at else json_safe(row_data.get("started_at"))
        )
        completed = (
            job.completed_at
            if job and job.completed_at
            else json_safe(row_data.get("completed_at"))
        )
        updated = (
            job.updated_at
            if job
            else json_safe(row_data.get("updated_at") or row_data["created_at"])
        )
        duration = None
        if started and completed:
            with suppress(ValueError):
                duration = (
                    datetime.fromisoformat(completed) - datetime.fromisoformat(started)
                ).total_seconds()
        return RunDetail(
            run_id=run_id,
            state=state,
            stage=stage,
            completed_stages=completed_stages,
            total_stages=len(STAGES),
            created_at=created,
            started_at=started,
            completed_at=completed,
            updated_at=updated,
            failure=failure,
            snapshot_id=str(row_data["snapshot_id"]) if row else job_request.get("snapshot_id"),
            charter_hash=str(row_data["charter_hash"]) if row else None,
            code_version=row_data.get("code_version") if row else None,
            code_dirty=row_data.get("code_dirty") if row else None,
            seed=int(row_data["seed"]) if row else None,
            models=self._json_list(row_data.get("models_json"))
            if row
            else list(job_request.get("models", [])),
            strategies=self._json_list(row_data.get("strategies_json"))
            if row
            else list(job_request.get("strategies", [])),
            cost_scenarios=self._json_list(row_data.get("scenarios_json"))
            if row
            else list(job_request.get("scenarios", [])),
            fold_schedule_hash=row_data.get("fold_schedule_hash") if row else None,
            duration_seconds=duration,
        )
