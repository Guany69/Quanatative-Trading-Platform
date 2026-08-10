from typing import Any

from fastapi import APIRouter, Depends, Query, status

from quant_platform.api.dependencies import paper
from quant_platform.api.schemas.paper import (
    CreateRebalanceRequest,
    HistoryResponse,
    PaperInitRequest,
    PaperSummary,
    PositionRecord,
    ProductionModel,
    ProductionModelRequest,
    ProposalDecisionRequest,
    ProposalRecord,
    RebalanceResponse,
)
from quant_platform.api.services.paper_service import PaperService

router = APIRouter(tags=["paper"])


@router.get("/paper", response_model=PaperSummary)
def get_paper(service: PaperService = Depends(paper)) -> PaperSummary:
    return service.summary()


@router.post("/paper/init", response_model=PaperSummary, status_code=status.HTTP_201_CREATED)
def initialize_paper(
    body: PaperInitRequest, service: PaperService = Depends(paper)
) -> PaperSummary:
    return service.initialize(body.initial_capital, body.account_id)


@router.get("/paper/production-model", response_model=ProductionModel | None)
def get_production_model(
    service: PaperService = Depends(paper),
) -> ProductionModel | None:
    return service.production_model()


@router.post("/paper/production-model", response_model=ProductionModel)
def set_production_model(
    body: ProductionModelRequest, service: PaperService = Depends(paper)
) -> ProductionModel:
    return service.designate_model(body.run_id, body.model, body.designated_by)


@router.post("/paper/rebalances", response_model=RebalanceResponse, status_code=201)
def create_rebalance(
    body: CreateRebalanceRequest, service: PaperService = Depends(paper)
) -> RebalanceResponse:
    return service.create_rebalance(body.strategy)


@router.get("/paper/rebalances", response_model=list[dict[str, Any]])
def get_rebalances(service: PaperService = Depends(paper)) -> list[dict[str, Any]]:
    return service.rebalances()


@router.get("/paper/proposals", response_model=list[ProposalRecord])
def get_proposals(
    proposal_status: str | None = Query(default=None, alias="status"),
    service: PaperService = Depends(paper),
) -> list[ProposalRecord]:
    return service.proposals(proposal_status)


@router.post("/paper/proposals/{order_id}/decision", response_model=ProposalRecord)
def decide_proposal(
    order_id: str,
    body: ProposalDecisionRequest,
    service: PaperService = Depends(paper),
) -> ProposalRecord:
    return service.decide(order_id, body)


@router.post("/paper/rebalances/{rebalance_id}/execute", response_model=RebalanceResponse)
def execute_rebalance(
    rebalance_id: str, service: PaperService = Depends(paper)
) -> RebalanceResponse:
    return service.execute_rebalance(rebalance_id)


@router.get("/paper/positions", response_model=list[PositionRecord])
def get_positions(service: PaperService = Depends(paper)) -> list[PositionRecord]:
    return service.positions()


@router.get("/paper/fills", response_model=list[dict[str, Any]])
def get_fills(service: PaperService = Depends(paper)) -> list[dict[str, Any]]:
    return service.fills()


@router.get("/paper/reconciliations", response_model=list[dict[str, Any]])
def get_reconciliations(service: PaperService = Depends(paper)) -> list[dict[str, Any]]:
    return service.reconciliations()


@router.get("/paper/history", response_model=HistoryResponse)
def get_history(service: PaperService = Depends(paper)) -> HistoryResponse:
    return service.history()
