from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

from fastapi.testclient import TestClient

from quant_platform.api.app import create_app
from quant_platform.api.schemas.paper import ProposalDecisionRequest
from quant_platform.api.services.paper_service import PaperService
from quant_platform.api.services.research_jobs import ResearchRunJobManager
from quant_platform.domain.enums import OrderSide
from quant_platform.execution.base import OrderProposal
from quant_platform.paper.state import PaperTradingState

from .conftest import FakeOrchestrator, seed_completed_run


def _stage_paper_state(api_settings):
    state = PaperTradingState.initialize(100_000.0)
    proposals = [
        OrderProposal(
            order_id=f"ORD-{security}",
            security_id=security,
            side=OrderSide.BUY,
            quantity=10,
            signal_date=date(2024, 1, 2),
            order_date=date(2024, 1, 3),
            target_weight=0.01,
            current_weight=0.0,
            reference_price=100.0,
        )
        for security in ("A", "B")
    ]
    state.stage_proposals(
        proposals,
        context={
            "rebalance_id": "RB-TEST",
            "strategy": "equal_weight",
            "signal_date": "2024-01-02",
            "order_date": "2024-01-03",
            "target_weights": {"A": 0.01, "B": 0.01},
            "prices": {"A": 100.0, "B": 100.0},
            "model_name": "factor_composite",
            "model_version": "test",
            "feature_version": "test",
        },
    )
    state.save(api_settings.paper_state_path)


def test_unapproved_orders_cannot_execute_and_decisions_persist(api_settings):
    _stage_paper_state(api_settings)
    manager = ResearchRunJobManager(api_settings, orchestrator=FakeOrchestrator(api_settings))
    with TestClient(create_app(api_settings, job_manager=manager)) as client:
        refused = client.post("/api/paper/rebalances/RB-TEST/execute")
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "ORDER_NOT_APPROVED"

        approved = client.post(
            "/api/paper/proposals/ORD-A/decision",
            json={"decision": "APPROVE", "actor": "alice"},
        )
        rejected = client.post(
            "/api/paper/proposals/ORD-B/decision",
            json={"decision": "REJECT", "actor": "alice", "reason": "concentration"},
        )
        assert approved.json()["status"] == "APPROVED"
        assert rejected.json()["status"] == "REJECTED"

        executed = client.post("/api/paper/rebalances/RB-TEST/execute")
        assert executed.status_code == 200
        assert executed.json()["filled"] == 1
        restored = PaperTradingState.load(api_settings.paper_state_path)
        assert restored.fill_history[0]["order_id"] == "ORD-A"
        assert any(row["order_id"] == "ORD-B" for row in restored.decision_history)


def test_concurrent_paper_decisions_do_not_lose_updates(api_settings):
    _stage_paper_state(api_settings)
    service = PaperService(api_settings)

    def decide(order_id: str):
        return service.decide(
            order_id,
            ProposalDecisionRequest(decision="APPROVE", actor="concurrent-reviewer"),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(decide, ["ORD-A", "ORD-B"]))

    restored = PaperTradingState.load(api_settings.paper_state_path)
    assert {row["order_id"] for row in restored.decision_history} == {"ORD-A", "ORD-B"}
    assert all(row["status"] == "approved" for row in restored.pending_proposals)


def test_production_model_reference_is_validated_through_api(api_settings):
    seed_completed_run(api_settings)
    PaperTradingState.initialize().save(api_settings.paper_state_path)
    manager = ResearchRunJobManager(api_settings, orchestrator=FakeOrchestrator(api_settings))
    with TestClient(create_app(api_settings, job_manager=manager)) as client:
        response = client.post(
            "/api/paper/production-model",
            json={
                "runId": "run-a",
                "model": "factor_composite",
                "designatedBy": "alice",
            },
        )
        assert response.status_code == 200
        assert response.json()["runId"] == "run-a"
        invalid = client.post(
            "/api/paper/production-model",
            json={"runId": "missing", "model": "factor_composite", "designatedBy": "alice"},
        )
        assert invalid.status_code == 404
