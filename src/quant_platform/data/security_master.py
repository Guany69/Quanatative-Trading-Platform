"""Security master: permanent identity across renames and delistings (spec section 11).

The single rule this module exists to enforce: **a ticker is not an identity**. Tickers get
changed by their issuer, reassigned to different companies, and recycled years after a
delisting. Keying research data on tickers therefore merges unrelated companies and splits
single ones, and does so silently.

Resolution here is always an interval lookup -- "which security traded as X on date D" -- never
a dictionary lookup, because the answer genuinely depends on the date.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_platform.domain.securities import FAR_FUTURE
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("data.security_master")


class SecurityMaster:
    """Identity, classification, and lifecycle for the security universe."""

    def __init__(self, securities: pl.DataFrame, identifiers: pl.DataFrame | None = None) -> None:
        if "security_id" not in securities.columns:
            raise ValueError("securities frame must have a security_id column")
        dupes = securities.group_by("security_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
        if not dupes.is_empty():
            raise ValueError(
                f"security_id must be unique; {dupes.height} duplicated: "
                f"{dupes['security_id'].head(5).to_list()}"
            )
        self.securities = securities
        self.identifiers = (
            identifiers
            if identifiers is not None
            else pl.DataFrame(
                schema={
                    "security_id": pl.Utf8,
                    "identifier_type": pl.Utf8,
                    "identifier_value": pl.Utf8,
                    "start_date": pl.Date,
                    "end_date": pl.Date,
                    "is_primary": pl.Boolean,
                }
            )
        )

    # ------------------------------------------------------------------ identity
    def resolve_symbol(
        self, symbol: str, as_of: date, identifier_type: str = "ticker"
    ) -> str | None:
        """Which security traded under ``symbol`` on ``as_of``?

        Returns None when the symbol was unassigned then -- which is the correct answer for a
        ticker before its IPO or after it was retired, and is precisely the case a naive
        dictionary lookup gets wrong.
        """
        if self.identifiers.is_empty():
            return None
        hit = self.identifiers.filter(
            (pl.col("identifier_type") == identifier_type)
            & (pl.col("identifier_value") == symbol)
            & (pl.col("start_date") <= as_of)
            & (pl.col("end_date") > as_of)
        )
        if hit.is_empty():
            return None
        if hit.height > 1:
            logger.warning(
                "symbol %s maps to %d securities on %s; returning the primary",
                symbol,
                hit.height,
                as_of,
            )
            primary = hit.filter(pl.col("is_primary"))
            if not primary.is_empty():
                hit = primary
        return str(hit["security_id"][0])

    def symbol_for(
        self, security_id: str, as_of: date, identifier_type: str = "ticker"
    ) -> str | None:
        """What was this security's ticker on ``as_of``?"""
        if self.identifiers.is_empty():
            return None
        hit = self.identifiers.filter(
            (pl.col("security_id") == security_id)
            & (pl.col("identifier_type") == identifier_type)
            & (pl.col("start_date") <= as_of)
            & (pl.col("end_date") > as_of)
        )
        return None if hit.is_empty() else str(hit["identifier_value"][0])

    def symbol_history(self, security_id: str) -> pl.DataFrame:
        """Every symbol this security has used, in order."""
        return self.identifiers.filter(pl.col("security_id") == security_id).sort("start_date")

    def has_been_renamed(self, security_id: str) -> bool:
        return self.symbol_history(security_id).height > 1

    # ------------------------------------------------------------------ lifecycle
    def is_listed_on(self, security_id: str, as_of: date) -> bool:
        row = self.securities.filter(pl.col("security_id") == security_id)
        if row.is_empty():
            return False
        first = row["first_trade_date"][0] if "first_trade_date" in row.columns else None
        delist = row["delisting_date"][0] if "delisting_date" in row.columns else None
        if first is not None and as_of < first:
            return False
        # Delisted names drop out going forward but remain fully present in history.
        return not (delist is not None and as_of > delist)

    def listed_on(self, as_of: date) -> list[str]:
        """All securities tradeable on ``as_of``.

        Delisted names are excluded going forward but remain fully present in history, which
        is what keeps a backtest survivorship-free.
        """
        df = self.securities
        if "first_trade_date" in df.columns:
            df = df.filter(
                pl.col("first_trade_date").is_null() | (pl.col("first_trade_date") <= as_of)
            )
        if "delisting_date" in df.columns:
            df = df.filter(pl.col("delisting_date").is_null() | (pl.col("delisting_date") >= as_of))
        return sorted(df["security_id"].to_list())

    def delisted_securities(self) -> pl.DataFrame:
        if "delisting_date" not in self.securities.columns:
            return self.securities.head(0)
        return self.securities.filter(pl.col("delisting_date").is_not_null())

    def delisting_return(self, security_id: str) -> float | None:
        row = self.securities.filter(pl.col("security_id") == security_id)
        if row.is_empty() or "delisting_return" not in row.columns:
            return None
        value = row["delisting_return"][0]
        return None if value is None else float(value)

    # ------------------------------------------------------------------ classification
    def sector_of(self, security_id: str) -> str | None:
        row = self.securities.filter(pl.col("security_id") == security_id)
        if row.is_empty() or "sector" not in row.columns:
            return None
        return str(row["sector"][0])

    def sector_map(self) -> dict[str, str]:
        if "sector" not in self.securities.columns:
            return {}
        return dict(zip(self.securities["security_id"], self.securities["sector"], strict=True))

    def by_security_type(self, security_type: str) -> list[str]:
        if "security_type" not in self.securities.columns:
            return []
        return sorted(
            self.securities.filter(pl.col("security_type") == security_type)[
                "security_id"
            ].to_list()
        )

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_prices(cls, prices: pl.DataFrame, source: str = "derived") -> SecurityMaster:
        """Derive a minimal master from a price panel.

        A fallback for datasets with no master. It cannot recover renames or delisting
        returns -- absences that are recorded honestly rather than papered over.
        """
        agg = prices.group_by("security_id").agg(
            [
                pl.col("observation_date").min().alias("first_trade_date"),
                pl.col("observation_date").max().alias("_last_date"),
            ]
        )
        securities = agg.with_columns(
            [
                pl.lit("common_stock").alias("security_type"),
                pl.lit("unknown").alias("sector"),
                pl.lit(None, dtype=pl.Date).alias("delisting_date"),
                pl.lit(None, dtype=pl.Utf8).alias("delisting_reason"),
                pl.lit(None, dtype=pl.Float64).alias("delisting_return"),
                pl.col("security_id").alias("primary_symbol"),
            ]
        ).drop("_last_date")
        logger.warning(
            "SecurityMaster derived from prices alone: renames and delisting returns are "
            "UNKNOWN, so survivorship and delisting bias cannot be controlled from this data."
        )
        return cls(securities)

    def to_frame(self) -> pl.DataFrame:
        return self.securities

    def __len__(self) -> int:
        return self.securities.height


def build_identifier_intervals(
    symbol_changes: pl.DataFrame, securities: pl.DataFrame
) -> pl.DataFrame:
    """Turn symbol-change events into identifier intervals.

    Each rename closes the previous interval and opens the next, so the same ``security_id``
    spans both -- the rename does not create a new economic security.
    """
    rows: list[dict[str, object]] = []
    changes = symbol_changes.filter(pl.col("action_type") == "symbol_change").sort(
        ["security_id", "ex_date"]
    )

    for row in securities.iter_rows(named=True):
        sec_id = row["security_id"]
        start = row.get("first_trade_date") or date(1900, 1, 1)
        end = row.get("delisting_date") or FAR_FUTURE
        sec_changes = changes.filter(pl.col("security_id") == sec_id)

        if sec_changes.is_empty():
            rows.append(
                {
                    "security_id": sec_id,
                    "identifier_type": "ticker",
                    "identifier_value": row.get("primary_symbol") or sec_id,
                    "start_date": start,
                    "end_date": end,
                    "is_primary": True,
                }
            )
            continue

        cursor = start
        for change in sec_changes.iter_rows(named=True):
            rows.append(
                {
                    "security_id": sec_id,
                    "identifier_type": "ticker",
                    "identifier_value": change["old_symbol"],
                    "start_date": cursor,
                    "end_date": change["ex_date"],
                    "is_primary": True,
                }
            )
            cursor = change["ex_date"]
        last = sec_changes.tail(1).to_dicts()[0]
        rows.append(
            {
                "security_id": sec_id,
                "identifier_type": "ticker",
                "identifier_value": last["new_symbol"],
                "start_date": cursor,
                "end_date": end,
                "is_primary": True,
            }
        )

    return pl.DataFrame(
        rows,
        schema_overrides={"start_date": pl.Date, "end_date": pl.Date, "identifier_value": pl.Utf8},
    )
