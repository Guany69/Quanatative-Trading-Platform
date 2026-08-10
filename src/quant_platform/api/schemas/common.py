from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        extra="forbid",
    )


class HealthResponse(ApiModel):
    status: str = "ok"


class ErrorBody(ApiModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(ApiModel):
    error: ErrorBody


class Page[T](ApiModel):
    items: list[T]
    page: int = Field(ge=1)
    limit: int = Field(ge=1)
    total: int = Field(ge=0)
    total_pages: int = Field(ge=0)


def page_of[T](items: list[T], *, page: int, limit: int, total: int) -> Page[T]:
    return Page(
        items=items,
        page=page,
        limit=limit,
        total=total,
        total_pages=math.ceil(total / limit) if total else 0,
    )


def json_safe(value: Any) -> Any:
    """Normalize domain/library values to strict JSON (notably NaN -> null)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.name
    if is_dataclass(value):
        return json_safe(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)
