"""Configuration for the loopback-only HTTP application."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUANT_PLATFORM_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    results_db: Path = Path("research.duckdb")
    charter_path: Path = Path("configs/research_charter.yaml")
    snapshot_root: Path = Path("data/snapshots")
    cache_root: Path = Path("data/interim/pit_cache")
    artifact_root: Path = Path("artifacts/models")
    report_root: Path = Path("reports")
    registry_path: Path = Path("artifacts/experiments.jsonl")
    job_state_path: Path = Path(".quant-platform/research-jobs.json")
    paper_state_path: Path | None = None
    frontend_dist: Path = Path("frontend/dist")
