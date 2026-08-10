from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Query, status

from quant_platform.api.config import ApiSettings
from quant_platform.api.dependencies import jobs, queries, settings
from quant_platform.api.errors import ApiError
from quant_platform.api.schemas.common import Page
from quant_platform.api.schemas.research import (
    ArtifactRecord,
    CreateRunRequest,
    CreateRunResponse,
    EquityRecord,
    FoldRecord,
    HoldoutEvaluationRequest,
    HoldoutEvaluationResponse,
    ModelMetricRecord,
    OptimizerDiagnosticRecord,
    OverfittingRecord,
    PredictionRecord,
    RunDetail,
    RunSummaryResponse,
    StrategyMetricRecord,
    TradeRecord,
    WeightRecord,
)
from quant_platform.api.services.research_jobs import ResearchRunJobManager
from quant_platform.api.services.research_queries import ResearchQueryService
from quant_platform.config import load_charter
from quant_platform.data.snapshots import SnapshotError, SnapshotStore
from quant_platform.research.holdout import evaluate_locked_holdout
from quant_platform.research.runner import ExperimentRunOrchestrator, ResearchRunRequest

router = APIRouter(tags=["research"])
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def pagination(
    page: int = Query(default=1, ge=1), limit: int = Query(default=100, ge=1, le=500)
) -> tuple[int, int]:
    return page, limit


@router.post("/runs", response_model=CreateRunResponse, status_code=status.HTTP_202_ACCEPTED)
def create_run(
    body: CreateRunRequest,
    manager: ResearchRunJobManager = Depends(jobs),
    config: ApiSettings = Depends(settings),
) -> CreateRunResponse:
    if not _SAFE_ID.fullmatch(body.snapshot_id):
        raise ApiError(
            "INVALID_RESEARCH_CONFIGURATION",
            "snapshotId is not a safe identifier.",
            status_code=422,
        )
    if body.snapshot_id not in {"auto", "fixture"}:
        try:
            SnapshotStore(config.snapshot_root).load(body.snapshot_id, verify=False)
        except SnapshotError as exc:
            raise ApiError("SNAPSHOT_NOT_FOUND", str(exc), status_code=404) from exc
    request = ResearchRunRequest(
        charter_path=config.charter_path,
        snapshot_id=body.snapshot_id,
        models=tuple(body.models),
        strategies=tuple(body.strategies),
        scenarios=tuple(body.cost_scenarios),
        portfolio_model=body.portfolio_model,
        fixture_securities=body.fixture_securities,
        max_workers=body.max_workers,
    )
    try:
        ExperimentRunOrchestrator._validate_request(request, load_charter(config.charter_path))
    except Exception as exc:
        raise ApiError("INVALID_RESEARCH_CONFIGURATION", str(exc), status_code=422) from exc
    job = manager.submit(request)
    return CreateRunResponse(run_id=job.run_id, state="QUEUED")


@router.get("/runs", response_model=Page[RunDetail])
def list_runs(
    paging: tuple[int, int] = Depends(pagination),
    state_filter: str | None = Query(default=None, alias="state"),
    service: ResearchQueryService = Depends(queries),
) -> Page[RunDetail]:
    return service.list_runs(page=paging[0], limit=paging[1], state=state_filter)


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str, service: ResearchQueryService = Depends(queries)) -> RunDetail:
    return service.run_detail(run_id)


@router.get("/runs/{run_id}/summary", response_model=RunSummaryResponse)
def get_summary(
    run_id: str, service: ResearchQueryService = Depends(queries)
) -> RunSummaryResponse:
    return service.summary(run_id)


@router.get("/runs/{run_id}/folds", response_model=Page[FoldRecord])
def get_folds(
    run_id: str,
    paging: tuple[int, int] = Depends(pagination),
    service: ResearchQueryService = Depends(queries),
) -> Page[FoldRecord]:
    return service.folds(run_id, *paging)


@router.get("/runs/{run_id}/model-metrics", response_model=Page[ModelMetricRecord])
def get_model_metrics(
    run_id: str,
    paging: tuple[int, int] = Depends(pagination),
    model: str | None = None,
    service: ResearchQueryService = Depends(queries),
) -> Page[ModelMetricRecord]:
    return service.model_metrics(run_id, *paging, model_name=model)


@router.get("/runs/{run_id}/predictions", response_model=Page[PredictionRecord])
def get_predictions(
    run_id: str,
    paging: tuple[int, int] = Depends(pagination),
    model: str | None = None,
    fold: int | None = None,
    security: str | None = None,
    start_date: str | None = Query(default=None, alias="startDate"),
    end_date: str | None = Query(default=None, alias="endDate"),
    service: ResearchQueryService = Depends(queries),
) -> Page[PredictionRecord]:
    return service.predictions(
        run_id,
        *paging,
        model_name=model,
        fold=fold,
        security=security,
        start_date=start_date,
        end_date=end_date,
    )


@router.get("/runs/{run_id}/strategy-metrics", response_model=Page[StrategyMetricRecord])
def get_strategy_metrics(
    run_id: str,
    paging: tuple[int, int] = Depends(pagination),
    model: str | None = None,
    strategy: str | None = None,
    cost_scenario: str | None = Query(default=None, alias="costScenario"),
    service: ResearchQueryService = Depends(queries),
) -> Page[StrategyMetricRecord]:
    return service.strategy_metrics(
        run_id, *paging, model=model, strategy=strategy, cost_scenario=cost_scenario
    )


@router.get("/runs/{run_id}/equity", response_model=Page[EquityRecord])
def get_equity(
    run_id: str,
    paging: tuple[int, int] = Depends(pagination),
    model: str | None = None,
    strategy: str | None = None,
    cost_scenario: str | None = Query(default=None, alias="costScenario"),
    service: ResearchQueryService = Depends(queries),
) -> Page[EquityRecord]:
    return service.equity(
        run_id, *paging, model=model, strategy=strategy, cost_scenario=cost_scenario
    )


@router.get("/runs/{run_id}/trades", response_model=Page[TradeRecord])
def get_trades(
    run_id: str,
    paging: tuple[int, int] = Depends(pagination),
    model: str | None = None,
    strategy: str | None = None,
    cost_scenario: str | None = Query(default=None, alias="costScenario"),
    security: str | None = None,
    service: ResearchQueryService = Depends(queries),
) -> Page[TradeRecord]:
    return service.trades(
        run_id,
        *paging,
        model=model,
        strategy=strategy,
        cost_scenario=cost_scenario,
        security=security,
    )


@router.get("/runs/{run_id}/weights", response_model=Page[WeightRecord])
def get_weights(
    run_id: str,
    paging: tuple[int, int] = Depends(pagination),
    model: str | None = None,
    strategy: str | None = None,
    security: str | None = None,
    start_date: str | None = Query(default=None, alias="startDate"),
    end_date: str | None = Query(default=None, alias="endDate"),
    service: ResearchQueryService = Depends(queries),
) -> Page[WeightRecord]:
    return service.weights(
        run_id,
        *paging,
        model=model,
        strategy=strategy,
        security=security,
        start_date=start_date,
        end_date=end_date,
    )


@router.get("/runs/{run_id}/optimizer-diagnostics", response_model=Page[OptimizerDiagnosticRecord])
def get_optimizer_diagnostics(
    run_id: str,
    paging: tuple[int, int] = Depends(pagination),
    model: str | None = None,
    strategy: str | None = None,
    service: ResearchQueryService = Depends(queries),
) -> Page[OptimizerDiagnosticRecord]:
    return service.optimizer_diagnostics(run_id, *paging, model=model, strategy=strategy)


@router.get("/runs/{run_id}/overfitting", response_model=list[OverfittingRecord])
def get_overfitting(
    run_id: str, service: ResearchQueryService = Depends(queries)
) -> list[OverfittingRecord]:
    return service.overfitting(run_id)


@router.get("/runs/{run_id}/artifacts", response_model=list[ArtifactRecord])
def get_artifacts(
    run_id: str, service: ResearchQueryService = Depends(queries)
) -> list[ArtifactRecord]:
    return service.artifacts(run_id)


@router.post("/runs/{run_id}/holdout-evaluations", response_model=HoldoutEvaluationResponse)
def evaluate_holdout(
    run_id: str,
    body: HoldoutEvaluationRequest,
    manager: ResearchRunJobManager = Depends(jobs),
    config: ApiSettings = Depends(settings),
) -> HoldoutEvaluationResponse:
    if not body.acknowledge_one_shot_evaluation:
        raise ApiError(
            "HOLDOUT_VIOLATION",
            "Locked holdout evaluation requires explicit one-shot acknowledgement.",
            status_code=422,
        )

    def operation() -> dict[str, float]:
        return evaluate_locked_holdout(
            run_id=run_id,
            model=body.model,
            charter_path=config.charter_path,
            acknowledged_by=body.acknowledged_by,
            acknowledge=body.acknowledge_one_shot_evaluation,
            results_db=config.results_db,
            snapshot_root=config.snapshot_root,
            report_root=config.report_root,
        )

    try:
        metrics = manager.run_writer_operation(operation)
    except Exception as exc:
        text = str(exc)
        code = "RUN_NOT_FOUND" if "unknown run_id" in text else "HOLDOUT_VIOLATION"
        raise ApiError(code, text, status_code=404 if code == "RUN_NOT_FOUND" else 422) from exc
    return HoldoutEvaluationResponse(run_id=run_id, model=body.model, metrics=metrics)
