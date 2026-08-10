"""Base records and point-in-time (PIT) semantics.

The single most important invariant in this platform is the separation of three distinct
timestamps, which naive backtests collapse into one and thereby leak the future:

* ``observation_date`` -- the period the information *describes* (e.g. a fiscal quarter end,
  or the session a price bar covers).
* ``available_at``     -- the moment the information first became *publicly knowable*. A
  feature computed at time ``t`` may only consume records with ``available_at <= t``.
* ``ingested_at``      -- the moment *this platform* stored the record. Used for audit and
  reproducibility, never for filtering features.

``revision_id`` distinguishes vintages of the same observation (macro series get revised;
the original print and the revision have the same ``observation_date`` but different
``available_at``). Selecting the correct vintage means "latest revision whose
``available_at <= t``", never "latest revision".
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_platform.domain.enums import DataQualityStatus


class FrozenModel(BaseModel):
    """Immutable, strictly-validated base for every domain record.

    Records are frozen because they represent observed facts; correcting a fact means
    emitting a new revision, not mutating history in place.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        str_strip_whitespace=True,
    )

    def to_row(self) -> dict[str, Any]:
        """Flatten to a Parquet-friendly dict (enums -> str, nested models excluded).

        Arrow cannot store Python enums or arbitrary objects, so enums are coerced to their
        string values and dates are left as native date/datetime for Arrow to map.
        """
        row: dict[str, Any] = {}
        for name, value in self.model_dump(mode="python").items():
            if isinstance(value, BaseModel):
                continue
            row[name] = _coerce_scalar(value)
        return row


def _coerce_scalar(value: Any) -> Any:
    """Coerce a value into something Arrow can represent."""
    from enum import Enum

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list | tuple):
        return [_coerce_scalar(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_scalar(v) for k, v in value.items()}
    return value


class PointInTimeRecord(FrozenModel):
    """Any observation whose usability depends on when it became known.

    Subclasses inherit the availability contract enforced in ``_check_availability``:
    information cannot become available before the period it describes.
    """

    security_id: str | None = Field(
        default=None,
        description="Permanent internal identifier. None for non-security records (e.g. macro).",
    )
    symbol: str | None = Field(
        default=None,
        description="Ticker as of observation_date. Display/debug only -- never a join key, "
        "because tickers are recycled and reassigned.",
    )
    observation_date: date = Field(description="The period this information describes.")
    available_at: datetime = Field(
        description="When this information first became publicly knowable. Features at time t "
        "may only use records with available_at <= t."
    )
    source: str = Field(description="Provider/adapter that produced the record.")
    source_record_id: str | None = Field(
        default=None, description="Provider-side identifier, for traceability back to origin."
    )
    revision_id: int = Field(
        default=0,
        ge=0,
        description="Vintage counter. 0 = original print; higher = later revision of the same "
        "observation_date.",
    )
    ingested_at: datetime | None = Field(
        default=None, description="When this platform stored the record. Audit only."
    )
    is_adjusted: bool = Field(
        default=False, description="Whether corporate-action adjustments have been applied."
    )
    data_quality_status: DataQualityStatus = Field(default=DataQualityStatus.OK)

    @model_validator(mode="after")
    def _check_availability(self) -> PointInTimeRecord:
        """Information cannot be available before the period it describes.

        This catches the most common PIT bug at construction time: stamping a fundamental
        with available_at == fiscal period end, which would let the model read a filing
        weeks before it was published.
        """
        obs_as_dt = datetime.combine(self.observation_date, datetime.min.time())
        available_naive = self.available_at.replace(tzinfo=None)
        if available_naive < obs_as_dt:
            raise ValueError(
                f"available_at ({self.available_at}) precedes observation_date "
                f"({self.observation_date}) for security_id={self.security_id}: information "
                f"cannot be known before the period it describes."
            )
        return self

    def is_available_at(self, as_of: datetime | date) -> bool:
        """True if this record was knowable at ``as_of``."""
        as_of_dt = (
            datetime.combine(as_of, datetime.max.time())
            if not isinstance(as_of, datetime)
            else as_of
        )
        return self.available_at.replace(tzinfo=None) <= as_of_dt.replace(tzinfo=None)
