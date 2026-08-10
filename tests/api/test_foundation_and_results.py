from __future__ import annotations

from fastapi.testclient import TestClient

from quant_platform.api.app import create_app
from quant_platform.api.services.research_jobs import ResearchRunJobManager

from .conftest import FakeOrchestrator, seed_completed_run


def test_health_metadata_openapi_and_consistent_errors(api_settings):
    manager = ResearchRunJobManager(api_settings, orchestrator=FakeOrchestrator(api_settings))
    with TestClient(create_app(api_settings, job_manager=manager)) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        metadata = client.get("/api/meta")
        assert metadata.status_code == 200
        assert {item["id"] for item in metadata.json()["models"]} >= {
            "factor_composite",
            "lightgbm",
        }
        assert metadata.json()["strategies"]
        schema = client.get("/api/openapi.json").json()
        assert "/api/runs" in schema["paths"]
        assert not any("sql" in path.lower() for path in schema["paths"])

        missing = client.get("/api/runs/missing")
        assert missing.status_code == 404
        assert missing.json() == {
            "error": {
                "code": "RUN_NOT_FOUND",
                "message": "Unknown research run 'missing'.",
                "details": {},
            }
        }


def test_purpose_specific_results_are_filtered_paginated_and_json_safe(api_settings):
    seed_completed_run(api_settings)
    manager = ResearchRunJobManager(api_settings, orchestrator=FakeOrchestrator(api_settings))
    with TestClient(create_app(api_settings, job_manager=manager)) as client:
        runs = client.get("/api/runs?limit=10").json()
        assert runs["total"] == 1
        assert runs["items"][0]["state"] == "COMPLETED"

        first = client.get("/api/runs/run-a/predictions?limit=1&security=A").json()
        assert first["limit"] == 1
        assert first["total"] == 2
        assert first["items"][0]["securityId"] == "A"

        second = client.get("/api/runs/run-a/predictions?limit=1&page=2&security=A").json()
        assert second["page"] == 2
        assert second["items"][0]["score"] is None

        metrics = client.get("/api/runs/run-a/model-metrics").json()
        assert metrics["items"][0]["metric"] == "mean_ic"
        summary = client.get("/api/runs/run-a/summary").json()
        assert summary["resultCounts"]["prediction"] == 3


def test_report_ids_are_server_owned_and_traversal_cannot_escape(api_settings):
    seed_completed_run(api_settings)
    report_dir = api_settings.report_root / "run-a"
    report_dir.mkdir(parents=True)
    (report_dir / "report.html").write_text("<h1>safe</h1>")
    outside = api_settings.report_root / "secret.txt"
    outside.write_text("secret")
    manager = ResearchRunJobManager(api_settings, orchestrator=FakeOrchestrator(api_settings))
    with TestClient(create_app(api_settings, job_manager=manager)) as client:
        reports = client.get("/api/runs/run-a/reports").json()
        assert [item["name"] for item in reports] == ["report.html"]
        response = client.get(reports[0]["url"])
        assert response.status_code == 200
        assert "safe" in response.text
        traversal = client.get("/api/runs/run-a/reports/..%2Fsecret.txt")
        assert traversal.status_code in {404, 422}
        assert "secret" not in traversal.text


def test_holdout_requires_explicit_backend_acknowledgement(api_settings):
    seed_completed_run(api_settings)
    manager = ResearchRunJobManager(api_settings, orchestrator=FakeOrchestrator(api_settings))
    with TestClient(create_app(api_settings, job_manager=manager)) as client:
        response = client.post(
            "/api/runs/run-a/holdout-evaluations",
            json={
                "model": "factor_composite",
                "acknowledgedBy": "reviewer",
                "acknowledgeOneShotEvaluation": False,
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "HOLDOUT_VIOLATION"
