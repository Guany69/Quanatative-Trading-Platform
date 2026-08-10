from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_platform.research.results import ResultsReader, ResultsStore, ResultsStoreError


def _create_run(store: ResultsStore, run_id: str) -> None:
    store.create_run(
        run_id=run_id,
        snapshot_id="snapshot",
        charter_hash="charter",
        code_version="commit",
        code_dirty=False,
        seed=42,
        models=["factor_composite"],
        strategies=["equal_weight"],
        scenarios=["base"],
        fold_schedule_hash="folds",
    )


def test_duckdb_is_authoritative_across_runs_and_csv_is_export(tmp_path):
    path = tmp_path / "research.duckdb"
    with ResultsStore(path) as store:
        _create_run(store, "run-a")
        store.write_folds(
            "run-a",
            pl.DataFrame(
                {
                    "fold": [0],
                    "train_start": [date(2018, 1, 1)],
                    "train_end": [date(2019, 1, 1)],
                    "validation_start": [date(2019, 1, 2)],
                    "validation_end": [date(2019, 2, 1)],
                    "test_start": [date(2019, 2, 2)],
                    "test_end": [date(2019, 3, 1)],
                    "purge": [True],
                    "embargo": [5],
                }
            ),
        )
        store.write_predictions(
            pl.DataFrame(
                {
                    "run_id": ["run-a"],
                    "model": ["factor_composite"],
                    "fold": [0],
                    "as_of": [date(2019, 2, 4)],
                    "security_id": ["A"],
                    "score": [0.7],
                    "rank": [1.0],
                }
            )
        )
        store.write_fold_metrics(
            pl.DataFrame(
                {
                    "run_id": ["run-a"],
                    "model": ["factor_composite"],
                    "fold": [0],
                    "metric": ["mean_ic"],
                    "value": [0.1],
                }
            )
        )
        store.write_strategy_results(
            pl.DataFrame(
                {
                    "run_id": ["run-a"],
                    "model": ["factor_composite"],
                    "strategy": ["equal_weight"],
                    "cost_scenario": ["base"],
                    "metric": ["sharpe_ratio"],
                    "value": [0.5],
                }
            )
        )
        store.write_trades(
            pl.DataFrame(
                {
                    "run_id": ["run-a"],
                    "model": ["factor_composite"],
                    "strategy": ["equal_weight"],
                    "cost_scenario": ["base"],
                    "information_date": [date(2019, 2, 1)],
                    "signal_date": [date(2019, 2, 1)],
                    "order_date": [date(2019, 2, 4)],
                    "fill_date": [date(2019, 2, 4)],
                    "accounting_date": [date(2019, 2, 4)],
                    "security_id": ["A"],
                    "side": ["buy"],
                    "shares": [10.0],
                    "reference_price": [100.0],
                    "fill_price": [100.1],
                    "notional": [1001.0],
                    "commission": [1.0],
                    "spread_cost": [0.5],
                    "slippage_cost": [0.25],
                    "impact_cost": [0.1],
                }
            )
        )
        store.finish_run("run-a", "COMPLETED")
        _create_run(store, "run-b")
        store.finish_run("run-b", "FAILED", failure_stage="training", failure_cause="boom")

    with ResultsReader(path) as reader:
        runs = reader.query("research_run", limit=10)
        assert set(runs["status"]) == {"COMPLETED", "FAILED"}
        assert reader.query("prediction", run_id="run-a").height == 1
        assert (
            reader.query("trade", run_id="run-a")["fill_date"][0]
            > reader.query("trade", run_id="run-a")["signal_date"][0]
        )
        exported = reader.export_csv(
            reader.query("fold_metric", run_id="run-a"), tmp_path / "metrics.csv"
        )
        assert exported.exists()
        with pytest.raises(ResultsStoreError, match="unknown run_id"):
            reader.query("prediction", run_id="missing")


def test_stale_running_recovery_and_one_shot_holdout(tmp_path):
    path = tmp_path / "research.duckdb"
    with ResultsStore(path) as store:
        _create_run(store, "stale")
    with ResultsStore(path) as store:
        assert store.recover_stale_runs() == 1
        store.record_holdout_evaluation("stale", "factor_composite", "reviewer", {"mean_ic": 0.1})
        with pytest.raises(ResultsStoreError, match="already been evaluated"):
            store.record_holdout_evaluation(
                "stale", "factor_composite", "reviewer", {"mean_ic": 0.2}
            )
        assert (
            store.query("SELECT status FROM research_run WHERE run_id=?", ["stale"])["status"][0]
            == "FAILED"
        )


def test_results_writer_rejects_non_owner_process_identity(tmp_path):
    with ResultsStore(tmp_path / "research.duckdb") as store:
        store._owner_pid = -1
        with pytest.raises(ResultsStoreError, match="outside its owning orchestrator process"):
            store.recover_stale_runs()
