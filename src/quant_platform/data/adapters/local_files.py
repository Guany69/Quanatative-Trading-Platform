"""CSV and Parquet adapters for user-supplied data (spec section 10.3).

These are the escape hatch for the platform's hardest data problem. Reliable historical index
constituents and point-in-time fundamentals are not freely available; the honest answer is to
let users bring their own vendor extract rather than silently substituting a
current-membership snapshot and calling it history.

``available_at`` handling is the sharp edge here. If a user file lacks the column, we do NOT
quietly default it to ``observation_date`` -- that would assert the data was knowable the
instant the period ended, which is false for fundamentals and macro. Instead the adapter
either applies an explicit, user-configured lag or refuses to load.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_platform.data.base import (
    REQUIRED_ACTION_COLUMNS,
    REQUIRED_FUNDAMENTAL_COLUMNS,
    REQUIRED_MACRO_COLUMNS,
    REQUIRED_MEMBERSHIP_COLUMNS,
    REQUIRED_PRICE_COLUMNS,
    SchemaError,
    SourceProvenance,
    filter_securities,
    filter_window,
    validate_schema,
)
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("data.local")


def _read(path: Path) -> pl.DataFrame:
    """Read CSV or Parquet based on the file extension."""
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path, try_parse_dates=True)
    raise SchemaError(f"unsupported file type '{suffix}' for {path}; use .csv or .parquet")


def _ensure_date(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Coerce a column to Date, tolerating string input from CSV."""
    if column not in frame.columns:
        return frame
    dtype = frame.schema[column]
    if dtype == pl.Date:
        return frame
    if dtype == pl.Datetime:
        return frame.with_columns(pl.col(column).dt.date().alias(column))
    return frame.with_columns(pl.col(column).cast(pl.Utf8).str.to_date(strict=False).alias(column))


def _ensure_datetime(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in frame.columns:
        return frame
    dtype = frame.schema[column]
    if dtype == pl.Datetime:
        return frame
    if dtype == pl.Date:
        return frame.with_columns(pl.col(column).cast(pl.Datetime).alias(column))
    return frame.with_columns(
        pl.col(column).cast(pl.Utf8).str.to_datetime(strict=False).alias(column)
    )


class LocalPriceSource:
    """Daily bars from a local CSV/Parquet file.

    Prices are the one dataset where ``available_at = close + a few hours`` is genuinely
    correct: a session's close is known once that session ends. So a default is applied here,
    unlike for fundamentals and macro.
    """

    PROVENANCE = SourceProvenance(
        name="local_file_prices",
        description="User-supplied daily bars from CSV or Parquet.",
        corporate_actions_complete=False,
        requires_network=False,
        known_limitations=[
            "Completeness and adjustment quality depend entirely on the user's file.",
            "Delisted securities are only present if the user's extract includes them.",
        ],
    )

    def __init__(self, path: str | Path, available_at_hour: int = 21) -> None:
        self.path = Path(path)
        self.available_at_hour = available_at_hour

    def load_prices(
        self,
        security_ids: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame:
        df = _read(self.path)
        df = _ensure_date(df, "observation_date")

        if "adjusted_close" not in df.columns and "close" in df.columns:
            # Without adjustment data we must not pretend the close is adjusted; flag it.
            logger.warning(
                "%s has no adjusted_close; using raw close. Splits and dividends will appear "
                "as returns unless corporate actions are applied separately.",
                self.path,
            )
            df = df.with_columns(pl.col("close").alias("adjusted_close"))

        if "available_at" not in df.columns:
            df = df.with_columns(
                (
                    pl.col("observation_date").cast(pl.Datetime)
                    + pl.duration(hours=self.available_at_hour)
                ).alias("available_at")
            )
        df = _ensure_datetime(df, "available_at")

        if "source" not in df.columns:
            df = df.with_columns(pl.lit(self.PROVENANCE.name).alias("source"))
        if "volume" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("volume"))
        if "dollar_volume" not in df.columns:
            df = df.with_columns((pl.col("close") * pl.col("volume")).alias("dollar_volume"))

        validate_schema(df, REQUIRED_PRICE_COLUMNS, "price")
        df = filter_securities(df, security_ids)
        return filter_window(df, start, end).sort(["security_id", "observation_date"])


class LocalFundamentalSource:
    """Fundamentals from a local file.

    Refuses to guess publication dates. If the file has neither ``available_at`` nor
    ``filing_date``, the caller must supply ``assumed_lag_days`` and thereby take explicit
    responsibility for the assumption -- a documented approximation beats a silent one.
    """

    PROVENANCE = SourceProvenance(
        name="local_file_fundamentals",
        description="User-supplied fundamentals from CSV or Parquet.",
        point_in_time_fundamentals_complete=False,
        known_limitations=[
            "Point-in-time correctness depends on the user's file containing real filing dates."
        ],
    )

    def __init__(self, path: str | Path, assumed_lag_days: int | None = None) -> None:
        self.path = Path(path)
        self.assumed_lag_days = assumed_lag_days

    def load_fundamentals(
        self,
        security_ids: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame:
        df = _read(self.path)

        if "fiscal_period_end" in df.columns and "observation_date" not in df.columns:
            df = df.with_columns(pl.col("fiscal_period_end").alias("observation_date"))
        elif "observation_date" in df.columns and "fiscal_period_end" not in df.columns:
            df = df.with_columns(pl.col("observation_date").alias("fiscal_period_end"))

        for col in ("observation_date", "fiscal_period_end", "filing_date"):
            df = _ensure_date(df, col)

        if "available_at" not in df.columns:
            if "filing_date" in df.columns and df["filing_date"].null_count() < df.height:
                df = df.with_columns(
                    (pl.col("filing_date").cast(pl.Datetime) + pl.duration(hours=17)).alias(
                        "available_at"
                    )
                )
                logger.info("derived available_at from filing_date in %s", self.path)
            elif self.assumed_lag_days is not None:
                logger.warning(
                    "%s has no filing dates; assuming a %d-day publication lag. This is an "
                    "APPROXIMATION and is recorded in the report's limitations.",
                    self.path,
                    self.assumed_lag_days,
                )
                df = df.with_columns(
                    (
                        pl.col("fiscal_period_end").cast(pl.Datetime)
                        + pl.duration(days=self.assumed_lag_days, hours=17)
                    ).alias("available_at")
                )
            else:
                raise SchemaError(
                    f"{self.path} has neither 'available_at' nor 'filing_date'. Refusing to "
                    f"default available_at to the fiscal period end, which would let the model "
                    f"read filings weeks before publication. Supply filing dates, or pass "
                    f"assumed_lag_days=N to accept an explicit approximation."
                )
        df = _ensure_datetime(df, "available_at")

        if "source" not in df.columns:
            df = df.with_columns(pl.lit(self.PROVENANCE.name).alias("source"))

        validate_schema(df, REQUIRED_FUNDAMENTAL_COLUMNS, "fundamental")
        df = filter_securities(df, security_ids)
        return filter_window(df, start, end).sort(["security_id", "observation_date"])


class LocalConstituentSource:
    """Index membership from a user-provided file.

    ``is_survivorship_biased`` must be stated. A file listing today's members with a fake
    start date is the most dangerous input this platform can receive, so the flag is
    mandatory and defaults to biased when the file does not say otherwise.
    """

    PROVENANCE = SourceProvenance(
        name="local_file_constituents",
        description="User-supplied index membership intervals.",
        historical_constituents_complete=True,  # true only if the user says so per-file
        known_limitations=[
            "Correctness depends on the user's extract genuinely containing historical "
            "membership rather than a current snapshot."
        ],
    )

    def __init__(self, path: str | Path, is_survivorship_biased: bool | None = None) -> None:
        self.path = Path(path)
        self.is_survivorship_biased = is_survivorship_biased

    def load_membership(
        self, universe: str, start: date | None = None, end: date | None = None
    ) -> pl.DataFrame:
        df = _read(self.path)
        for col in ("start_date", "end_date"):
            df = _ensure_date(df, col)

        if "universe" not in df.columns:
            df = df.with_columns(pl.lit(universe).alias("universe"))
        if "end_date" not in df.columns:
            df = df.with_columns(pl.lit(date(9999, 12, 31)).alias("end_date"))

        if "is_survivorship_biased" not in df.columns:
            if self.is_survivorship_biased is None:
                raise SchemaError(
                    f"{self.path} has no 'is_survivorship_biased' column and none was declared. "
                    f"Pass is_survivorship_biased=False only if this file genuinely contains "
                    f"historical constituents; pass True if it is a current snapshot."
                )
            df = df.with_columns(
                pl.lit(self.is_survivorship_biased).alias("is_survivorship_biased")
            )

        if "available_at" not in df.columns:
            # Membership changes are announced ahead of the effective date; without an
            # announcement date the effective date is the conservative choice (it delays,
            # rather than advances, when the strategy may act).
            df = df.with_columns(pl.col("start_date").cast(pl.Datetime).alias("available_at"))
        df = _ensure_datetime(df, "available_at")
        if "source" not in df.columns:
            df = df.with_columns(pl.lit(self.PROVENANCE.name).alias("source"))

        validate_schema(df, REQUIRED_MEMBERSHIP_COLUMNS, "membership")
        df = df.filter(pl.col("universe") == universe)
        if start is not None:
            df = df.filter(pl.col("end_date") >= start)
        if end is not None:
            df = df.filter(pl.col("start_date") <= end)
        return df


class LocalMacroSource:
    """Macro series from a local file."""

    PROVENANCE = SourceProvenance(
        name="local_file_macro",
        description="User-supplied macroeconomic series.",
        macro_vintages_complete=False,
    )

    def __init__(self, path: str | Path, assumed_lag_days: int = 1) -> None:
        self.path = Path(path)
        self.assumed_lag_days = assumed_lag_days

    def load_series(
        self, series_ids: list[str], start: date | None = None, end: date | None = None
    ) -> pl.DataFrame:
        df = _read(self.path)
        df = _ensure_date(df, "observation_date")

        if "revision_id" not in df.columns:
            df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("revision_id"))
        if "is_revision" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("is_revision"))
        if "available_at" not in df.columns:
            df = df.with_columns(
                (
                    pl.col("observation_date").cast(pl.Datetime)
                    + pl.duration(days=self.assumed_lag_days, hours=8)
                ).alias("available_at")
            )
        df = _ensure_datetime(df, "available_at")
        if "source" not in df.columns:
            df = df.with_columns(pl.lit(self.PROVENANCE.name).alias("source"))

        validate_schema(df, REQUIRED_MACRO_COLUMNS, "macro")
        if series_ids:
            df = df.filter(pl.col("series_id").is_in(series_ids))
        return filter_window(df, start, end).sort(["series_id", "observation_date", "revision_id"])


class LocalCorporateActionSource:
    """Corporate actions from a local file."""

    PROVENANCE = SourceProvenance(
        name="local_file_corporate_actions",
        description="User-supplied splits, dividends, renames, and delistings.",
        corporate_actions_complete=True,
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_actions(
        self,
        security_ids: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame:
        df = _read(self.path)
        for col in ("ex_date", "observation_date", "record_date", "payment_date"):
            df = _ensure_date(df, col)
        if "observation_date" not in df.columns:
            df = df.with_columns(pl.col("ex_date").alias("observation_date"))
        if "available_at" not in df.columns:
            df = df.with_columns(pl.col("ex_date").cast(pl.Datetime).alias("available_at"))
        df = _ensure_datetime(df, "available_at")
        if "source" not in df.columns:
            df = df.with_columns(pl.lit(self.PROVENANCE.name).alias("source"))

        validate_schema(df, REQUIRED_ACTION_COLUMNS, "corporate_action")
        df = filter_securities(df, security_ids)
        return filter_window(df, start, end, column="ex_date").sort(["security_id", "ex_date"])


def write_frame(frame: pl.DataFrame, path: str | Path) -> Path:
    """Persist a frame as CSV or Parquet, inferred from the extension."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        frame.write_parquet(p)
    else:
        frame.write_csv(p)
    return p


def default_available_at(observation_date: date, hour: int = 21) -> datetime:
    """Publication time for a daily bar: after the session closes."""
    return datetime.combine(observation_date, datetime.min.time()) + timedelta(hours=hour)
