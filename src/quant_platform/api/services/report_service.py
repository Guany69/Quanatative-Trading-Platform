"""Path-safe discovery and retrieval of run-owned report files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from quant_platform.api.config import ApiSettings
from quant_platform.api.errors import ApiError
from quant_platform.api.schemas.reports import ReportRecord
from quant_platform.api.services.research_queries import ResearchQueryService


@dataclass(frozen=True)
class ResolvedReport:
    record: ReportRecord
    path: Path


class ReportService:
    def __init__(self, settings: ApiSettings, queries: ResearchQueryService) -> None:
        self.settings = settings
        self.queries = queries

    def list_reports(self, run_id: str) -> list[ReportRecord]:
        self.queries.run_detail(run_id)
        return [item.record for item in self._resolved(run_id)]

    def resolve(self, run_id: str, report_id: str) -> ResolvedReport:
        self.queries.run_detail(run_id)
        for item in self._resolved(run_id):
            if item.record.id == report_id:
                return item
        raise ApiError(
            "REPORT_NOT_FOUND",
            f"Report '{report_id}' does not belong to run '{run_id}'.",
            status_code=404,
        )

    def _resolved(self, run_id: str) -> list[ResolvedReport]:
        root = self.settings.report_root.resolve()
        run_root = (root / run_id).resolve()
        if not run_root.is_relative_to(root) or not run_root.is_dir():
            return []
        result: list[ResolvedReport] = []
        for path in sorted(run_root.iterdir()):
            resolved = path.resolve()
            if not path.is_file() or not resolved.is_relative_to(run_root):
                continue
            identifier = hashlib.sha256(path.name.encode()).hexdigest()[:16]
            suffix = path.suffix.lower().lstrip(".") or "binary"
            result.append(
                ResolvedReport(
                    record=ReportRecord(
                        id=identifier,
                        name=path.name,
                        type=suffix,
                        size=path.stat().st_size,
                        url=f"/api/runs/{run_id}/reports/{identifier}",
                    ),
                    path=resolved,
                )
            )
        return result
