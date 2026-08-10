"""Regression tests against stored fixture baselines (spec section 31.7).

These pin down deterministic properties of the fixture pipeline so that an unintended change
in behaviour is caught rather than absorbed silently.

The baselines are deliberately expressed as **invariants and tolerances** rather than exact
metric values. Pinning an exact Sharpe ratio to 12 decimals would fail on any harmless
refactor (a different summation order in NumPy is enough), which trains people to update
baselines reflexively -- and a baseline that is always updated protects nothing. What is
pinned here are the properties that should genuinely never drift.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_platform.config import load_charter
from quant_platform.data.adapters.fixture import FixtureDataProvider, FixtureSpec
from quant_platform.pipeline import (
    build_adjusted_prices,
    build_training_frame,
    default_models,
    fit_and_predict,
    forecast_ic,
    load_fixture_data,
    run_backtest_for_predictions,
    split_train_test,
)
from quant_platform.utilities.reproducibility import set_global_seeds

BASELINE_PATH = Path(__file__).parent / "baselines.json"
TEST_START = date(2021, 1, 4)
SEED = 42
N_SECURITIES = 100


@pytest.fixture(scope="module")
def fixture_dataset():
    return FixtureDataProvider(FixtureSpec(n_securities=N_SECURITIES, seed=SEED)).generate()


@pytest.fixture(scope="module")
def charter():
    return load_charter("configs/research_charter.yaml")


@pytest.fixture(scope="module")
def pipeline_data(charter):
    set_global_seeds(SEED)
    return load_fixture_data(charter, n_securities=N_SECURITIES)


def _load_baselines() -> dict:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())
    return {}


class TestFixtureShapeStability:
    def test_fixture_dimensions_are_stable(self, fixture_dataset):
        """The generator's output shape must not drift for a fixed seed and size."""
        ds = fixture_dataset
        assert ds.securities.height == N_SECURITIES
        # Hazards must remain present -- they are what the pipeline is tested against.
        delisted = ds.securities.filter(pl.col("delisting_date").is_not_null())
        assert delisted.height == 2, "the fixture should contain exactly two delistings"
        assert set(delisted["delisting_reason"].to_list()) == {"bankruptcy", "merger"}
        assert -1.0 in delisted["delisting_return"].to_list(), "the total loss is missing"

        actions = ds.corporate_actions["action_type"].unique().to_list()
        for required in ("stock_split", "cash_dividend", "symbol_change", "delisting"):
            assert required in actions, f"fixture lost its {required} events"

    def test_macro_revisions_still_present(self, fixture_dataset):
        gdp = fixture_dataset.macro.filter(pl.col("series_id") == "GDP")
        revisions = gdp.filter(pl.col("revision_id") > 0)
        assert revisions.height > 0, "GDP revisions vanished; the vintage hazard is untested"

    def test_deterministic_across_runs(self):
        """Identical seed -> identical data, to full float precision."""
        a = FixtureDataProvider(FixtureSpec(n_securities=30, seed=SEED)).generate()
        b = FixtureDataProvider(FixtureSpec(n_securities=30, seed=SEED)).generate()
        assert a.prices["close"].sum() == pytest.approx(b.prices["close"].sum(), rel=1e-15)
        assert a.fundamentals["net_income"].sum() == pytest.approx(
            b.fundamentals["net_income"].sum(), rel=1e-15
        )

    def test_different_seeds_produce_different_data(self):
        """Guards against a seeding bug that would make the seed inert."""
        a = FixtureDataProvider(FixtureSpec(n_securities=30, seed=1)).generate()
        b = FixtureDataProvider(FixtureSpec(n_securities=30, seed=2)).generate()
        assert a.prices["close"].sum() != pytest.approx(b.prices["close"].sum(), rel=1e-6)


class TestCorporateActionInvariants:
    def test_adjusted_returns_have_no_split_sized_jumps(self, fixture_dataset):
        """A 2-for-1 split leaves a ~50% artifact if adjustment breaks."""
        adj = build_adjusted_prices(fixture_dataset.prices, fixture_dataset.corporate_actions)
        rets = adj.sort(["security_id", "observation_date"]).with_columns(
            (pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1.0)
            .over("security_id")
            .alias("ret")
        )
        split_ids = (
            fixture_dataset.corporate_actions.filter(pl.col("action_type") == "stock_split")[
                "security_id"
            ]
            .unique()
            .to_list()
        )
        for sec_id in split_ids:
            series = rets.filter(pl.col("security_id") == sec_id)["ret"].drop_nulls().to_numpy()
            assert float(np.abs(series).max()) < 0.25

    def test_market_cap_is_continuous_through_splits(self, fixture_dataset):
        """Price halves and share count doubles, so market cap must not jump."""
        splits = fixture_dataset.corporate_actions.filter(pl.col("action_type") == "stock_split")
        for row in splits.iter_rows(named=True):
            sub = fixture_dataset.prices.filter(pl.col("security_id") == row["security_id"]).sort(
                "observation_date"
            )
            sub = sub.with_columns((pl.col("close") * pl.col("shares_outstanding")).alias("mcap"))
            before = sub.filter(pl.col("observation_date") < row["ex_date"])["mcap"]
            after = sub.filter(pl.col("observation_date") >= row["ex_date"])["mcap"]
            if before.len() == 0 or after.len() == 0:
                continue
            # Adjacent days differ only by the day's return, not by the split ratio.
            assert after[0] == pytest.approx(before[-1], rel=0.15)


class TestPipelineInvariants:
    def test_purge_boundary_holds(self, pipeline_data, charter):
        panel = build_training_frame(pipeline_data)
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, pipeline_data.calendar
        )
        assert train["window_end"].max() < TEST_START
        assert train["as_of"].max() < test["as_of"].min()

    def test_signal_models_beat_the_no_skill_control(self, pipeline_data, charter):
        """The core end-to-end property: information survives the whole pipeline."""
        panel = build_training_frame(pipeline_data)
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, pipeline_data.calendar
        )
        models = default_models(charter)
        composite = fit_and_predict(
            models["factor_composite"],
            train,
            test,
            pipeline_data.feature_columns,
            "excess_return_rank",
            charter,
        )
        ic = forecast_ic(composite, test, "excess_return_rank")["mean_ic"]
        assert ic > 0.0, f"factor composite IC collapsed to {ic:.4f}"

        no_skill = fit_and_predict(
            models["no_skill"],
            train,
            test,
            pipeline_data.feature_columns,
            "excess_return_rank",
            charter,
        )
        assert no_skill["prediction"].n_unique() == 1

    def test_costs_reduce_returns_monotonically(self, pipeline_data, charter):
        """zero > base > double > triple, always."""
        panel = build_training_frame(pipeline_data)
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, pipeline_data.calendar
        )
        preds = fit_and_predict(
            default_models(charter)["factor_composite"],
            train,
            test,
            pipeline_data.feature_columns,
            "excess_return_rank",
            charter,
        )
        finals = {}
        for scenario in ("zero", "base", "double", "triple"):
            result = run_backtest_for_predictions(
                preds, pipeline_data, charter, "equal_weight", scenario
            )
            finals[scenario] = result.snapshots[-1].total_value
        assert finals["zero"] > finals["base"] > finals["double"] > finals["triple"], finals

    def test_execution_delay_always_applied(self, pipeline_data, charter):
        panel = build_training_frame(pipeline_data)
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, pipeline_data.calendar
        )
        preds = fit_and_predict(
            default_models(charter)["factor_composite"],
            train,
            test,
            pipeline_data.feature_columns,
            "excess_return_rank",
            charter,
        )
        result = run_backtest_for_predictions(preds, pipeline_data, charter, "equal_weight", "base")
        assert result.trades
        assert all(t.fill_date > t.signal_date for t in result.trades)

    def test_cash_stays_non_negative(self, pipeline_data, charter):
        panel = build_training_frame(pipeline_data)
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, pipeline_data.calendar
        )
        preds = fit_and_predict(
            default_models(charter)["factor_composite"],
            train,
            test,
            pipeline_data.feature_columns,
            "excess_return_rank",
            charter,
        )
        result = run_backtest_for_predictions(preds, pipeline_data, charter, "equal_weight", "base")
        worst = min(s.cash for s in result.snapshots)
        assert worst >= -1.0, f"cash reached {worst:,.2f}; leverage was implied"


class TestStoredBaselines:
    """Tolerance-based comparison against recorded metrics, when a baseline file exists."""

    def test_metrics_within_tolerance(self, pipeline_data, charter):
        baselines = _load_baselines()
        if not baselines:
            pytest.skip("no baselines.json recorded yet")

        panel = build_training_frame(pipeline_data)
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, pipeline_data.calendar
        )
        preds = fit_and_predict(
            default_models(charter)["factor_composite"],
            train,
            test,
            pipeline_data.feature_columns,
            "excess_return_rank",
            charter,
        )
        ic = forecast_ic(preds, test, "excess_return_rank")

        expected = baselines.get("factor_composite_mean_ic")
        if expected is not None:
            assert ic["mean_ic"] == pytest.approx(expected, abs=0.02), (
                f"factor composite IC moved from {expected:.4f} to {ic['mean_ic']:.4f}. If "
                f"this change is intentional, update tests/regression/baselines.json."
            )
