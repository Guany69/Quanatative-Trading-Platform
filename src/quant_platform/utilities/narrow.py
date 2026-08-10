"""Type-narrowing helpers for Polars scalar extraction.

Polars aggregations such as ``Series.max()`` and ``Series.quantile()`` are typed as returning
a broad union (``int | float | Decimal | date | ... | None``) because the return type depends
on the Series' runtime dtype, which the type checker cannot know. Call sites that legitimately
know the dtype would otherwise need a ``# type: ignore`` on every line, which suppresses real
errors along with the noise.

These helpers narrow the union *and* validate at runtime, so a genuine dtype surprise raises a
clear error instead of propagating a wrong type into a date comparison or a float computation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def as_date(value: Any, context: str = "value") -> date:
    """Narrow a Polars scalar to ``date``, raising if it is not one."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"{context}: expected a date, got {type(value).__name__} ({value!r})")


def as_optional_date(value: Any, context: str = "value") -> date | None:
    """Narrow a Polars scalar to ``date | None``."""
    if value is None:
        return None
    return as_date(value, context)


def as_float(value: Any, default: float | None = None, context: str = "value") -> float:
    """Narrow a Polars scalar to ``float``.

    ``default`` is returned for None (common for an aggregation over an all-null column).
    Without a default, None is an error rather than a silent 0.0, because a silently-zero
    winsorization bound or scale would corrupt every downstream feature.
    """
    if value is None:
        if default is None:
            raise TypeError(f"{context}: expected a float, got None")
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{context}: expected a float-like value, got {type(value).__name__} ({value!r})"
        ) from exc


def as_int(value: Any, default: int | None = None, context: str = "value") -> int:
    """Narrow a Polars scalar to ``int``."""
    if value is None:
        if default is None:
            raise TypeError(f"{context}: expected an int, got None")
        return default
    if isinstance(value, int | float):
        return int(value)
    raise TypeError(f"{context}: expected an int, got {type(value).__name__} ({value!r})")
