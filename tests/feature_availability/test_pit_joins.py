"""Point-in-time feature availability tests (spec sections 16.4, 31.2).

These tests target the two joins where look-ahead most often enters a quant pipeline:

1. **Fundamentals** -- joining on fiscal period instead of filing date lets a model read a
   quarter's earnings weeks before they were published.
2. **Macro revisions** -- using the final revised value instead of the original print gives
   the model a number that systematically moved toward what actually happened.

Each test builds data where the future value is *materially different* from the visible one,
so a leak would be unmistakable in the assertion rather than hidden in noise.
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from quant_platform.features.fundamental import (
    attach_point_in_time_fundamentals,
    build_trailing_fundamentals,
    compute_fundamental_features,
)
from quant_platform.features.macro import attach_macro_series, select_macro_vintage


def _panel(dates: list[date], security_id: str = "S1") -> pl.DataFrame:
    return pl.DataFrame({"security_id": [security_id] * len(dates), "as_of": dates}).with_columns(
        pl.col("as_of").cast(pl.Date)
    )


def _fundamental(period_end: date, filed: date, net_income: float, security_id: str = "S1") -> dict:
    return {
        "security_id": security_id,
        "observation_date": period_end,
        "fiscal_period_end": period_end,
        "filing_date": filed,
        "available_at": datetime.combine(filed, datetime.min.time()),
        "source": "test",
        "net_income": net_income,
        "revenue": net_income * 10,
        "total_assets": 1_000_000.0,
        "total_equity": 500_000.0,
    }


def _fundamentals(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema_overrides={
            "observation_date": pl.Date,
            "fiscal_period_end": pl.Date,
            "filing_date": pl.Date,
            "available_at": pl.Datetime,
        },
    )


class TestFundamentalAvailability:
    def test_filing_is_invisible_before_publication(self):
        """The core PIT test: Q4 ending 31 Dec, filed 15 Feb, must be invisible in January."""
        funds = _fundamentals(
            [
                _fundamental(date(2020, 9, 30), date(2020, 11, 10), net_income=100.0),
                _fundamental(date(2020, 12, 31), date(2021, 2, 15), net_income=999.0),
            ]
        )
        panel = _panel([date(2021, 1, 20), date(2021, 3, 1)])
        joined = attach_point_in_time_fundamentals(panel, funds)

        jan = joined.filter(pl.col("as_of") == date(2021, 1, 20))
        mar = joined.filter(pl.col("as_of") == date(2021, 3, 1))

        # In January only Q3 is public, so the Q4 value of 999 must NOT appear.
        assert jan["net_income"][0] == pytest.approx(100.0), (
            "January sees the unpublished Q4 figure: the join is using fiscal period rather "
            "than filing date, which is look-ahead bias"
        )
        # By March the Q4 filing is public.
        assert mar["net_income"][0] == pytest.approx(999.0)

    def test_no_fundamentals_visible_before_first_filing(self):
        """Before any filing exists, the fields must be null -- not back-filled."""
        funds = _fundamentals(
            [_fundamental(date(2020, 12, 31), date(2021, 2, 15), net_income=500.0)]
        )
        panel = _panel([date(2020, 6, 1)])
        joined = attach_point_in_time_fundamentals(panel, funds)
        assert joined["net_income"][0] is None

    def test_restatement_only_visible_from_its_own_publication(self):
        """A restatement of an old period must not rewrite what earlier dates could see."""
        funds = _fundamentals(
            [
                # Original print of Q1, filed May.
                _fundamental(date(2020, 3, 31), date(2020, 5, 10), net_income=100.0),
                # Restatement of the SAME period, filed in November.
                _fundamental(date(2020, 3, 31), date(2020, 11, 20), net_income=55.0),
            ]
        )
        joined = attach_point_in_time_fundamentals(
            _panel([date(2020, 7, 1), date(2020, 12, 1)]), funds
        )
        july = joined.filter(pl.col("as_of") == date(2020, 7, 1))["net_income"][0]
        december = joined.filter(pl.col("as_of") == date(2020, 12, 1))["net_income"][0]

        assert july == pytest.approx(100.0), (
            "July must see the ORIGINAL print, not the later restatement"
        )
        assert december == pytest.approx(55.0), "December must see the restated figure"

    def test_exact_boundary_is_inclusive_of_publication_date(self):
        """On the publication date itself the filing is knowable."""
        funds = _fundamentals(
            [_fundamental(date(2020, 12, 31), date(2021, 2, 15), net_income=42.0)]
        )
        joined = attach_point_in_time_fundamentals(_panel([date(2021, 2, 15)]), funds)
        assert joined["net_income"][0] == pytest.approx(42.0)

        # ...and the day before, it is not.
        before = attach_point_in_time_fundamentals(_panel([date(2021, 2, 14)]), funds)
        assert before["net_income"][0] is None

    def test_ttm_aggregates_use_only_published_quarters(self):
        """Trailing-twelve-month sums must not include an unpublished quarter."""
        rows = [
            _fundamental(date(2020, 3, 31), date(2020, 5, 10), 100.0),
            _fundamental(date(2020, 6, 30), date(2020, 8, 10), 100.0),
            _fundamental(date(2020, 9, 30), date(2020, 11, 10), 100.0),
            _fundamental(date(2020, 12, 31), date(2021, 2, 15), 100.0),
        ]
        enriched = build_trailing_fundamentals(_fundamentals(rows))
        # TTM needs 4 quarters; it becomes available only with the 4th filing (Feb 2021).
        ttm = enriched.filter(pl.col("net_income_ttm").is_not_null())
        assert ttm.height == 1
        assert ttm["net_income_ttm"][0] == pytest.approx(400.0)
        assert ttm["available_at"][0] == datetime(2021, 2, 15)

    def test_features_computed_only_from_visible_data(self):
        """End to end: ratios in January reflect Q3, not the unpublished Q4."""
        funds = _fundamentals(
            [
                _fundamental(date(2020, 9, 30), date(2020, 11, 10), 100.0),
                _fundamental(date(2020, 12, 31), date(2021, 2, 15), 5000.0),
            ]
        )
        panel = _panel([date(2021, 1, 20)]).with_columns(pl.lit(1_000_000.0).alias("market_cap"))
        joined = attach_point_in_time_fundamentals(panel, funds)
        feats = compute_fundamental_features(joined)
        # ROE uses the visible quarter's equity/income, so it cannot reflect the 5000 figure.
        assert feats["net_income"][0] == pytest.approx(100.0)


class TestMacroVintages:
    def _macro(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                # Original print of 2020Q1 GDP, published 30 April, noisy.
                {
                    "series_id": "GDP",
                    "observation_date": date(2020, 3, 31),
                    "available_at": datetime(2020, 4, 30),
                    "value": 21000.0,
                    "revision_id": 0,
                    "is_revision": False,
                    "source": "test",
                },
                # Revision published 29 June, materially different.
                {
                    "series_id": "GDP",
                    "observation_date": date(2020, 3, 31),
                    "available_at": datetime(2020, 6, 29),
                    "value": 19500.0,
                    "revision_id": 1,
                    "is_revision": True,
                    "source": "test",
                },
            ],
            schema_overrides={"observation_date": pl.Date, "available_at": pl.Datetime},
        )

    def test_revision_is_invisible_before_it_is_published(self):
        """Using the revised value early is the classic macro leak."""
        joined = attach_macro_series(
            _panel([date(2020, 5, 15), date(2020, 8, 1)]), self._macro(), ["GDP"]
        )
        may = joined.filter(pl.col("as_of") == date(2020, 5, 15))["gdp"][0]
        august = joined.filter(pl.col("as_of") == date(2020, 8, 1))["gdp"][0]

        assert may == pytest.approx(21000.0), (
            "May sees the June revision: the pipeline is using revised macro data before it "
            "was published"
        )
        assert august == pytest.approx(19500.0)

    def test_nothing_visible_before_first_publication(self):
        joined = attach_macro_series(_panel([date(2020, 4, 1)]), self._macro(), ["GDP"])
        assert joined["gdp"][0] is None

    def test_all_vintages_are_retained_for_selection(self):
        """Vintages must not be collapsed; the correct one depends on the query date."""
        vintages = select_macro_vintage(self._macro(), "GDP")
        assert vintages.height == 2
        assert sorted(vintages["revision_id"].to_list()) == [0, 1]


class TestFixtureEndToEnd:
    def test_fixture_fundamentals_respect_filing_lag(self):
        """On real fixture data, no visible filing may postdate the panel date."""
        from quant_platform.data.adapters.fixture import FixtureDataProvider, FixtureSpec

        ds = FixtureDataProvider(FixtureSpec(n_securities=20)).generate()
        dates = [date(2019, 3, 15), date(2020, 7, 1), date(2022, 11, 4)]
        panel = pl.concat(
            [_panel(dates, security_id=s) for s in ds.securities["security_id"].head(20)]
        )
        joined = attach_point_in_time_fundamentals(panel, ds.fundamentals)

        visible = joined.filter(pl.col("filing_date").is_not_null())
        assert visible.height > 0, "no fundamentals joined; the test would be vacuous"

        # THE invariant: nothing visible was filed after the date we asked about.
        violations = visible.filter(pl.col("filing_date") > pl.col("as_of"))
        assert violations.height == 0, (
            f"{violations.height} rows expose a filing dated after the panel date -- "
            f"look-ahead bias in the fundamental join"
        )

    def test_fixture_macro_respects_publication(self):
        from quant_platform.data.adapters.fixture import FixtureDataProvider, FixtureSpec

        ds = FixtureDataProvider(FixtureSpec(n_securities=10)).generate()
        gdp = ds.macro.filter(pl.col("series_id") == "GDP")
        panel = _panel([date(2019, 5, 1), date(2020, 9, 15), date(2023, 2, 1)])
        joined = attach_macro_series(panel, gdp, ["GDP"])

        for row in joined.iter_rows(named=True):
            if row["gdp"] is None:
                continue
            # The value shown must come from a vintage published on or before as_of.
            eligible = gdp.filter(
                pl.col("available_at") <= datetime.combine(row["as_of"], datetime.max.time())
            )
            assert not eligible.is_empty()
            assert row["gdp"] in eligible["value"].to_list(), (
                f"GDP value {row['gdp']} on {row['as_of']} does not match any vintage "
                f"published by then"
            )
