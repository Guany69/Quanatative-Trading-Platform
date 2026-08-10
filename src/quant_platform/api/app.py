"""FastAPI bootstrap for the local browser application."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from quant_platform.api.config import ApiSettings
from quant_platform.api.errors import install_error_handlers
from quant_platform.api.routes import health, metadata, paper, reports, research, snapshots
from quant_platform.api.services.metadata import MetadataService
from quant_platform.api.services.paper_service import PaperService
from quant_platform.api.services.report_service import ReportService
from quant_platform.api.services.research_jobs import ResearchRunJobManager
from quant_platform.api.services.research_queries import ResearchQueryService


def create_app(
    settings: ApiSettings | None = None,
    *,
    job_manager: ResearchRunJobManager | None = None,
) -> FastAPI:
    config = settings or ApiSettings()
    manager = job_manager or ResearchRunJobManager(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        manager.start()
        yield
        manager.stop()

    app = FastAPI(
        title="quant-platform API",
        version="0.1.0",
        description=(
            "Loopback application adapter over the quantitative research and paper-trading "
            "modular monolith. No arbitrary SQL or filesystem APIs are exposed."
        ),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = config
    app.state.jobs = manager
    app.state.queries = ResearchQueryService(config, manager)
    app.state.metadata = MetadataService(config)
    app.state.paper = PaperService(config)
    app.state.reports = ReportService(config, app.state.queries)

    install_error_handlers(app)
    for router in (
        health.router,
        metadata.router,
        snapshots.router,
        research.router,
        reports.router,
        paper.router,
    ):
        app.include_router(router, prefix="/api")

    _install_frontend(app, config.frontend_dist)
    return app


def _install_frontend(app: FastAPI, dist: Path) -> None:
    if not dist.is_dir():

        @app.get("/", include_in_schema=False)
        def api_only_root() -> JSONResponse:
            return JSONResponse(
                {"name": "quant-platform", "apiDocs": "/api/docs", "frontendBuilt": False}
            )

        return

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    index = dist / "index.html"

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend(frontend_path: str) -> FileResponse:
        candidate = (dist / frontend_path).resolve()
        if frontend_path and candidate.is_relative_to(dist.resolve()) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


app = create_app()
