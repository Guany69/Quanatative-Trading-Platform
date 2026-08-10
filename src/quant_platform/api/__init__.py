"""Browser-facing HTTP adapter for the quant-platform modular monolith."""

from quant_platform.api.app import create_app

__all__ = ["create_app"]
