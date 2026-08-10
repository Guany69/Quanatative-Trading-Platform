"""End-to-end integration test (spec section 31.6).

Runs the complete pipeline on a small deterministic dataset:

    ingestion -> validation -> universe -> features -> labels -> training
    -> predictions -> optimization -> backtest -> evaluation -> report

Unit tests verify components in isolation; this verifies they actually compose. Most real
pipeline failures are interface mismatches -- a join key that silently drops rows, a date
column that changes dtype -- and only an end-to-end run catches them.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant_platform.config import load_charter
from quant_platform.config.models import ResearchCharter
from quant_platform.data.validation import validate_dataset
from quant_platform.domain.enums import EvaluationStage
from quant_platform.evaluation.report import ReportBundle, generate_report
from quant_platform.pipeline import (
    build_training_frame,
    default_models,
    evaluate,
    fit_and_predict,
    forecast_ic,
    load_fixture_data,
    run_backtest_for_predictions,
    split_train_test,
)
from quant_platform.utilities.reproducibility import set_global_seeds

TEST_START = date(2021, 1, 4)


@pytest.fixture(scope="module")
def charter() -> ResearchCharter:
    return load_charter("configs/research_charter.yaml")


@pytest.fixture(scope="module")
def data(charter: ResearchCharter):
    """A small but realistic dataset -- enough names for selection to be meaningful."""
    set_global_seeds(charter.random_seed)
    return load_fixture_data(charter, n_securities=150)


@pytest.fixture(scope="module")
def panel(data):
    return build_training_frame(data)


class TestPipelineStages:
    def test_data_passes_validation(self, data):
        """Stage 1-2: ingestion and point-in-time validation."""
        report = validate_dataset(
            prices=data.prices,
            benchmark=data.benchmark,
            corporate_actions=data.corporate_actions,
            calendar_sessions=data.calendar.sessions,
        )
        assert report.is_clean, f"critical data issues: {[i.message for i in report.critical]}"

    def test_universe_is_point_in_time_and_selective(self, data, charter):
        """Stage 3: the universe varies over time and is smaller than the full security set."""
        assert not data.universe_panel.is_empty()
        sizes = data.universe_panel.group_by("as_of").agg(pl.len().alias("n"))["n"]
        assert sizes.min() > 0
        # Membership and eligibility change over time, so the universe must not be constant.
        assert sizes.n_unique() > 1, "universe size never changes; PIT filtering may be inert"

    def test_features_computed_without_lookahead(self, data):
        """Stage 4: features exist and warm-up nulls appear only at the start."""
        assert not data.features.is_empty()
        assert data.feature_columns
        early = data.features.filter(pl.col("as_of") < date(2015, 6, 1))
        late = data.features.filter(pl.col("as_of") > date(2020, 1, 1))
        col = "momentum_252d"
        # Long-lookback features must be null early (no history) and populated later.
        assert early[col].null_count() > 0
        assert late[col].null_count() < late.height * 0.1

    def test_labels_have_correct_horizon(self, data, charter):
        """Stage 5: 20-session forward windows with execution delay applied."""
        assert not data.labels.is_empty()
        row = data.labels.head(1).to_dicts()[0]
        sessions = data.calendar.count_between(row["window_start"], row["window_end"])
        # horizon + both endpoints inclusive
        assert sessions == charter.label.forecast_horizon_sessions + 1
        assert row["window_start"] > row["as_of"], "label window must start after the signal"

    def test_panel_joins_cleanly(self, panel, data):
        """The universe/feature/label join must retain a usable amount of data."""
        assert not panel.is_empty()
        assert "excess_return_rank" in panel.columns
        for col in data.feature_columns[:5]:
            assert col in panel.columns

    def test_split_purges_overlapping_labels(self, panel, data, charter):
        """Stage 6: no training label window may reach into the test period."""
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, data.calendar
        )
        assert not train.is_empty() and not test.is_empty()
        assert train["window_end"].max() < TEST_START, (
            "a training label window closes on or after the test start: purging failed"
        )
        assert train["as_of"].max() < test["as_of"].min()

    def test_models_train_and_predict(self, panel, data, charter):
        """Stage 7: every model produces predictions with valid ranks."""
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, data.calendar
        )
        for name, model in default_models(charter).items():
            preds = fit_and_predict(
                model, train, test, data.feature_columns, "excess_return_rank", charter
            )
            assert preds.height > 0, f"{name} produced no predictions"
            ranks = preds["cross_sectional_rank"]
            assert ranks.min() >= 0.0 and ranks.max() <= 1.0
            assert model.is_fitted
            assert model.metadata is not None
            # Predictions must only cover the test window.
            assert preds["as_of"].min() >= test["as_of"].min()

    def test_backtest_produces_a_valid_equity_curve(self, panel, data, charter):
        """Stages 8-10: portfolio construction, costs, and backtest."""
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, data.calendar
        )
        model = default_models(charter)["factor_composite"]
        preds = fit_and_predict(
            model, train, test, data.feature_columns, "excess_return_rank", charter
        )
        result = run_backtest_for_predictions(
            preds, data, charter, "equal_weight", "base", strategy_name="integration"
        )
        curve = result.equity_curve()
        assert curve.height > 0
        assert result.trades, "no trades were generated"
        assert (curve["total_value"] > 0).all(), "portfolio value went non-positive"
        # Execution delay must hold for every trade.
        assert all(t.fill_date > t.signal_date for t in result.trades)

    def test_evaluation_carries_provenance(self, panel, data, charter):
        """Stage 11: reports must state what produced the numbers."""
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, data.calendar
        )
        model = default_models(charter)["factor_composite"]
        preds = fit_and_predict(
            model, train, test, data.feature_columns, "excess_return_rank", charter
        )
        result = run_backtest_for_predictions(preds, data, charter, "equal_weight", "base")
        report = evaluate(
            result,
            charter,
            data,
            EvaluationStage.WALK_FORWARD_TEST,
            TEST_START,
            test["as_of"].max(),
        )
        assert report.is_synthetic_data is True
        assert "SYNTHETIC" in report.disclosure_note
        assert report.execution_delay_sessions >= 1
        assert "cagr" in report.metrics
        assert "gross_cagr" in report.metrics


class TestFullPipelineRun:
    def test_complete_flow_writes_all_report_artifacts(self, panel, data, charter, tmp_path):
        """The whole chain, ending in a written report set."""
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, data.calendar
        )
        test_end = test["as_of"].max()

        reports, curves = [], {}
        forecast_metrics = {}
        for name, model in default_models(charter).items():
            preds = fit_and_predict(
                model, train, test, data.feature_columns, "excess_return_rank", charter
            )
            forecast_metrics[name] = forecast_ic(preds, test, "excess_return_rank")
            result = run_backtest_for_predictions(
                preds, data, charter, "equal_weight", "base", strategy_name=name
            )
            reports.append(
                evaluate(
                    result, charter, data, EvaluationStage.WALK_FORWARD_TEST, TEST_START, test_end
                )
            )
            curves[name] = result.equity_curve()

        bundle = ReportBundle(
            title="Integration Test",
            reports=reports,
            equity_curves=curves,
            forecast_metrics=forecast_metrics,
            notes=["SYNTHETIC test data."],
        )
        paths = generate_report(bundle, tmp_path / "report")

        for key in ("json", "markdown", "html"):
            assert Path(paths[key]).exists(), f"{key} report missing"
        assert Path(paths["json"]).stat().st_size > 100

        payload = json.loads(Path(paths["json"]).read_text())
        assert payload["reports"], "report JSON has no results"
        assert all(r["is_synthetic_data"] for r in payload["reports"])
        assert "disclaimer" in payload

        markdown = Path(paths["markdown"]).read_text()
        assert "SYNTHETIC" in markdown, "the synthetic-data banner is missing from the report"
        assert "does not guarantee future results" in markdown

    def test_baselines_are_beaten_by_real_signal(self, panel, data, charter):
        """A sanity check on the whole chain: signal models must beat the no-skill control.

        This is what proves the pipeline transmits information end to end. If a constant
        predictor scored the same, some stage would be discarding the signal.
        """
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, data.calendar
        )
        models = default_models(charter)

        composite = fit_and_predict(
            models["factor_composite"],
            train,
            test,
            data.feature_columns,
            "excess_return_rank",
            charter,
        )
        composite_ic = forecast_ic(composite, test, "excess_return_rank")["mean_ic"]

        # The no-skill model predicts a constant, so its IC is undefined (all ties).
        no_skill = fit_and_predict(
            models["no_skill"], train, test, data.feature_columns, "excess_return_rank", charter
        )
        assert no_skill["prediction"].n_unique() == 1, "no_skill should predict a constant"
        assert composite_ic > 0.0, (
            f"factor composite IC is {composite_ic:.4f}; the pipeline is not transmitting the "
            f"planted signal"
        )


class TestDeterminism:
    def test_same_seed_reproduces_identical_data(self, charter):
        """Reproducibility: identical seeds must produce identical panels."""
        set_global_seeds(42)
        first = load_fixture_data(charter, n_securities=30)
        set_global_seeds(42)
        second = load_fixture_data(charter, n_securities=30)

        assert first.prices.shape == second.prices.shape
        assert first.prices["adjusted_close"].sum() == pytest.approx(
            second.prices["adjusted_close"].sum(), rel=1e-12
        )
        assert first.labels["excess_return"].sum() == pytest.approx(
            second.labels["excess_return"].sum(), rel=1e-12
        )

    def test_same_seed_reproduces_identical_predictions(self, panel, data, charter):
        train, test = split_train_test(
            panel, TEST_START, charter.validation.embargo_sessions, data.calendar
        )
        outputs = []
        for _ in range(2):
            set_global_seeds(42)
            model = default_models(charter)["factor_composite"]
            preds = fit_and_predict(
                model, train, test, data.feature_columns, "excess_return_rank", charter
            )
            outputs.append(preds["prediction"].sum())
        assert outputs[0] == pytest.approx(outputs[1], rel=1e-12)
