from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from quant_platform.api.dependencies import reports
from quant_platform.api.schemas.reports import ReportRecord
from quant_platform.api.services.report_service import ReportService

router = APIRouter(tags=["reports"])


@router.get("/runs/{run_id}/reports", response_model=list[ReportRecord])
def list_reports(run_id: str, service: ReportService = Depends(reports)) -> list[ReportRecord]:
    return service.list_reports(run_id)


@router.get("/runs/{run_id}/reports/{report_id}", response_class=FileResponse)
def get_report(
    run_id: str, report_id: str, service: ReportService = Depends(reports)
) -> FileResponse:
    report = service.resolve(run_id, report_id)
    return FileResponse(report.path, filename=report.record.name)
