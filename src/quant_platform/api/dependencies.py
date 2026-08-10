from fastapi import Request

from quant_platform.api.config import ApiSettings
from quant_platform.api.services.metadata import MetadataService
from quant_platform.api.services.paper_service import PaperService
from quant_platform.api.services.report_service import ReportService
from quant_platform.api.services.research_jobs import ResearchRunJobManager
from quant_platform.api.services.research_queries import ResearchQueryService


def settings(request: Request) -> ApiSettings:
    return request.app.state.settings


def jobs(request: Request) -> ResearchRunJobManager:
    return request.app.state.jobs


def queries(request: Request) -> ResearchQueryService:
    return request.app.state.queries


def metadata(request: Request) -> MetadataService:
    return request.app.state.metadata


def paper(request: Request) -> PaperService:
    return request.app.state.paper


def reports(request: Request) -> ReportService:
    return request.app.state.reports
