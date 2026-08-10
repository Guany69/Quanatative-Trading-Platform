from __future__ import annotations

import time

from fastapi.testclient import TestClient

from quant_platform.api.app import create_app
from quant_platform.api.services.research_jobs import ResearchRunJobManager

from .conftest import FakeOrchestrator


def _request(**overrides):
    payload = {
        "snapshotId": "auto",
        "models": ["factor_composite"],
        "strategies": ["equal_weight"],
        "costScenarios": ["base"],
        "portfolioModel": "factor_composite",
        "maxWorkers": 1,
        "fixtureSecurities": 20,
    }
    payload.update(overrides)
    return payload


def _wait_terminal(client: TestClient, run_id: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        if response.json()["state"] in {"COMPLETED", "FAILED"}:
            return response.json()
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not become terminal")


def test_research_post_is_async_fifo_and_never_creates_two_writers(api_settings):
    orchestrator = FakeOrchestrator(api_settings, delay=0.04)
    manager = ResearchRunJobManager(api_settings, orchestrator=orchestrator)
    with TestClient(create_app(api_settings, job_manager=manager)) as client:
        started = time.monotonic()
        first = client.post("/api/runs", json=_request())
        elapsed = time.monotonic() - started
        second = client.post("/api/runs", json=_request())
        assert first.status_code == second.status_code == 202
        assert elapsed < 0.5
        first_id = first.json()["runId"]
        second_id = second.json()["runId"]
        assert first.json()["state"] == "QUEUED"

        assert _wait_terminal(client, first_id)["state"] == "COMPLETED"
        assert _wait_terminal(client, second_id)["state"] == "COMPLETED"
        assert orchestrator.max_active == 1
        assert orchestrator.order == [first_id, second_id]


def test_failed_job_retains_real_failure_stage_and_cause(api_settings):
    manager = ResearchRunJobManager(
        api_settings, orchestrator=FakeOrchestrator(api_settings, delay=0.005)
    )
    with TestClient(create_app(api_settings, job_manager=manager)) as client:
        response = client.post("/api/runs", json=_request(fixtureSecurities=21))
        result = _wait_terminal(client, response.json()["runId"])
        assert result["state"] == "FAILED"
        assert result["stage"] == "REPORTING"
        assert result["failure"]["stage"] == "MODEL_TRAINING"
        assert "requested test failure" in result["failure"]["message"]


def test_a_second_post_does_not_run_stale_recovery_against_active_job(api_settings):
    orchestrator = FakeOrchestrator(api_settings, delay=0.08)
    manager = ResearchRunJobManager(api_settings, orchestrator=orchestrator)
    with TestClient(create_app(api_settings, job_manager=manager)) as client:
        first = client.post("/api/runs", json=_request()).json()["runId"]
        deadline = time.monotonic() + 1
        while client.get(f"/api/runs/{first}").json()["state"] != "RUNNING":
            assert time.monotonic() < deadline
            time.sleep(0.005)
        client.post("/api/runs", json=_request())
        active = client.get(f"/api/runs/{first}").json()
        assert active["state"] == "RUNNING"
        assert active["failure"] is None
