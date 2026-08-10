"""DuckDB research-results system of record.

The batch orchestrator is the sole writer.  Workers receive no store object or connection;
they return immutable task payloads to the parent, which flushes complete logical stages.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl


class ResultsStoreError(RuntimeError):
    """Raised for result ownership, lifecycle, or query errors."""


RUN_STATUSES = frozenset({"RUNNING", "COMPLETED", "FAILED"})


_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_run (
    run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    snapshot_id VARCHAR NOT NULL,
    charter_hash VARCHAR NOT NULL,
    code_version VARCHAR,
    code_dirty BOOLEAN,
    seed BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    failure_stage VARCHAR,
    failure_cause VARCHAR,
    models_json JSON NOT NULL,
    strategies_json JSON NOT NULL,
    scenarios_json JSON NOT NULL,
    fold_schedule_hash VARCHAR,
    report_path VARCHAR
);

CREATE TABLE IF NOT EXISTS fold (
    run_id VARCHAR NOT NULL,
    fold INTEGER NOT NULL,
    train_start DATE NOT NULL,
    train_end DATE NOT NULL,
    validation_start DATE NOT NULL,
    validation_end DATE NOT NULL,
    test_start DATE NOT NULL,
    test_end DATE NOT NULL,
    purge BOOLEAN NOT NULL,
    embargo INTEGER NOT NULL,
    PRIMARY KEY (run_id, fold)
);

CREATE TABLE IF NOT EXISTS prediction (
    run_id VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    fold INTEGER NOT NULL,
    as_of DATE NOT NULL,
    security_id VARCHAR NOT NULL,
    score DOUBLE NOT NULL,
    rank DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS fold_metric (
    run_id VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    fold INTEGER NOT NULL,
    metric VARCHAR NOT NULL,
    value DOUBLE,
    PRIMARY KEY (run_id, model, fold, metric)
);

CREATE TABLE IF NOT EXISTS artifact (
    run_id VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    fold INTEGER NOT NULL,
    artifact_path VARCHAR NOT NULL,
    metadata_path VARCHAR NOT NULL,
    PRIMARY KEY (run_id, model, fold)
);

CREATE TABLE IF NOT EXISTS strategy_execution (
    run_id VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    strategy VARCHAR NOT NULL,
    cost_scenario VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    failure_cause VARCHAR,
    PRIMARY KEY (run_id, model, strategy, cost_scenario)
);

CREATE TABLE IF NOT EXISTS strategy_result (
    run_id VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    strategy VARCHAR NOT NULL,
    cost_scenario VARCHAR NOT NULL,
    metric VARCHAR NOT NULL,
    value DOUBLE,
    PRIMARY KEY (run_id, model, strategy, cost_scenario, metric)
);

CREATE TABLE IF NOT EXISTS strategy_equity (
    run_id VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    strategy VARCHAR NOT NULL,
    cost_scenario VARCHAR NOT NULL,
    as_of DATE NOT NULL,
    total_value DOUBLE,
    gross_return DOUBLE,
    net_return DOUBLE,
    benchmark_return DOUBLE,
    turnover DOUBLE,
    costs DOUBLE,
    PRIMARY KEY (run_id, model, strategy, cost_scenario, as_of)
);

CREATE TABLE IF NOT EXISTS target_weight (
    run_id VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    strategy VARCHAR NOT NULL,
    as_of DATE NOT NULL,
    security_id VARCHAR NOT NULL,
    weight DOUBLE NOT NULL,
    PRIMARY KEY (run_id, model, strategy, as_of, security_id)
);

CREATE TABLE IF NOT EXISTS optimizer_diagnostic (
    run_id VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    strategy VARCHAR NOT NULL,
    as_of DATE NOT NULL,
    solver_status VARCHAR NOT NULL,
    solver_name VARCHAR,
    objective_value DOUBLE,
    expected_return DOUBLE,
    expected_risk DOUBLE,
    expected_turnover DOUBLE,
    expected_cost DOUBLE,
    relaxations_json JSON NOT NULL,
    message VARCHAR,
    PRIMARY KEY (run_id, model, strategy, as_of)
);

CREATE TABLE IF NOT EXISTS trade (
    run_id VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    strategy VARCHAR NOT NULL,
    cost_scenario VARCHAR NOT NULL,
    information_date DATE NOT NULL,
    signal_date DATE NOT NULL,
    order_date DATE NOT NULL,
    fill_date DATE NOT NULL,
    accounting_date DATE NOT NULL,
    security_id VARCHAR NOT NULL,
    side VARCHAR NOT NULL,
    shares DOUBLE NOT NULL,
    reference_price DOUBLE NOT NULL,
    fill_price DOUBLE NOT NULL,
    notional DOUBLE NOT NULL,
    commission DOUBLE NOT NULL,
    spread_cost DOUBLE NOT NULL,
    slippage_cost DOUBLE NOT NULL,
    impact_cost DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS overfitting_metric (
    run_id VARCHAR NOT NULL,
    metric VARCHAR NOT NULL,
    value DOUBLE,
    optimistic BOOLEAN NOT NULL DEFAULT FALSE,
    details_json JSON NOT NULL,
    PRIMARY KEY (run_id, metric)
);

CREATE TABLE IF NOT EXISTS holdout_evaluation (
    run_id VARCHAR PRIMARY KEY,
    model VARCHAR NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL,
    acknowledged_by VARCHAR NOT NULL,
    metrics_json JSON NOT NULL
);

CREATE INDEX IF NOT EXISTS prediction_lookup
    ON prediction (run_id, model, fold, as_of);
CREATE INDEX IF NOT EXISTS strategy_lookup
    ON strategy_result (run_id, model, strategy, cost_scenario);
"""


class ResultsStore:
    """Owned write interface used only by the experiment-run orchestrator."""

    def __init__(self, path: str | Path = "research.duckdb") -> None:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - dependency is mandatory in target
            raise ResultsStoreError("DuckDB is required for the research results store") from exc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._owner_pid = os.getpid()
        self._connection = duckdb.connect(str(self.path))
        self._connection.execute(_SCHEMA)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ResultsStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _assert_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise ResultsStoreError(
                "ResultsStore write attempted outside its owning orchestrator process"
            )

    def recover_stale_runs(
        self, cause: str = "orchestrator process exited before completion"
    ) -> int:
        """Mark prior RUNNING records failed before a new local writer starts."""
        self._assert_owner()
        rows = self._connection.execute(
            "SELECT count(*) FROM research_run WHERE status = 'RUNNING'"
        ).fetchone()
        count = int(rows[0]) if rows else 0
        if count:
            self._connection.execute(
                """UPDATE research_run
                   SET status='FAILED', completed_at=?, failure_stage='recovery', failure_cause=?
                   WHERE status='RUNNING'""",
                [datetime.now(UTC), cause],
            )
        return count

    def create_run(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        charter_hash: str,
        code_version: str | None,
        code_dirty: bool | None,
        seed: int,
        models: list[str],
        strategies: list[str],
        scenarios: list[str],
        fold_schedule_hash: str,
    ) -> None:
        self._assert_owner()
        self._connection.execute(
            """INSERT INTO research_run VALUES
               (?, ?, NULL, ?, ?, ?, ?, ?, 'RUNNING', NULL, NULL, ?, ?, ?, ?, NULL)""",
            [
                run_id,
                datetime.now(UTC),
                snapshot_id,
                charter_hash,
                code_version,
                code_dirty,
                seed,
                json.dumps(models),
                json.dumps(strategies),
                json.dumps(scenarios),
                fold_schedule_hash,
            ],
        )

    def finish_run(
        self,
        run_id: str,
        status: Literal["COMPLETED", "FAILED"],
        *,
        failure_stage: str | None = None,
        failure_cause: str | None = None,
        report_path: str | None = None,
    ) -> None:
        self._assert_owner()
        if status not in RUN_STATUSES - {"RUNNING"}:
            raise ResultsStoreError(f"invalid terminal run status {status}")
        row = self._connection.execute(
            "SELECT status FROM research_run WHERE run_id=?", [run_id]
        ).fetchone()
        if row is None:
            raise ResultsStoreError(f"unknown run_id '{run_id}'")
        if row[0] != "RUNNING":
            raise ResultsStoreError(f"run {run_id} is already terminal ({row[0]})")
        self._connection.execute(
            """UPDATE research_run SET status=?, completed_at=?, failure_stage=?,
               failure_cause=?, report_path=? WHERE run_id=?""",
            [status, datetime.now(UTC), failure_stage, failure_cause, report_path, run_id],
        )

    def write_folds(self, run_id: str, rows: pl.DataFrame) -> None:
        self._insert_frame(
            "fold",
            rows.with_columns(pl.lit(run_id).alias("run_id")),
            [
                "run_id",
                "fold",
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
                "purge",
                "embargo",
            ],
        )

    def write_predictions(self, rows: pl.DataFrame) -> None:
        self._insert_frame(
            "prediction",
            rows,
            [
                "run_id",
                "model",
                "fold",
                "as_of",
                "security_id",
                "score",
                "rank",
            ],
        )

    def write_fold_metrics(self, rows: pl.DataFrame) -> None:
        self._insert_frame("fold_metric", rows, ["run_id", "model", "fold", "metric", "value"])

    def write_artifacts(self, rows: pl.DataFrame) -> None:
        self._insert_frame(
            "artifact",
            rows,
            [
                "run_id",
                "model",
                "fold",
                "artifact_path",
                "metadata_path",
            ],
        )

    def write_strategy_execution(self, rows: pl.DataFrame) -> None:
        self._insert_frame(
            "strategy_execution",
            rows,
            [
                "run_id",
                "model",
                "strategy",
                "cost_scenario",
                "status",
                "failure_cause",
            ],
        )

    def write_strategy_results(self, rows: pl.DataFrame) -> None:
        self._insert_frame(
            "strategy_result",
            rows,
            [
                "run_id",
                "model",
                "strategy",
                "cost_scenario",
                "metric",
                "value",
            ],
        )

    def write_strategy_equity(self, rows: pl.DataFrame) -> None:
        self._insert_frame(
            "strategy_equity",
            rows,
            [
                "run_id",
                "model",
                "strategy",
                "cost_scenario",
                "as_of",
                "total_value",
                "gross_return",
                "net_return",
                "benchmark_return",
                "turnover",
                "costs",
            ],
        )

    def write_target_weights(self, rows: pl.DataFrame) -> None:
        self._insert_frame(
            "target_weight",
            rows,
            [
                "run_id",
                "model",
                "strategy",
                "as_of",
                "security_id",
                "weight",
            ],
        )

    def write_optimizer_diagnostics(self, rows: pl.DataFrame) -> None:
        self._insert_frame(
            "optimizer_diagnostic",
            rows,
            [
                "run_id",
                "model",
                "strategy",
                "as_of",
                "solver_status",
                "solver_name",
                "objective_value",
                "expected_return",
                "expected_risk",
                "expected_turnover",
                "expected_cost",
                "relaxations_json",
                "message",
            ],
        )

    def write_trades(self, rows: pl.DataFrame) -> None:
        self._insert_frame(
            "trade",
            rows,
            [
                "run_id",
                "model",
                "strategy",
                "cost_scenario",
                "information_date",
                "signal_date",
                "order_date",
                "fill_date",
                "accounting_date",
                "security_id",
                "side",
                "shares",
                "reference_price",
                "fill_price",
                "notional",
                "commission",
                "spread_cost",
                "slippage_cost",
                "impact_cost",
            ],
        )

    def write_overfitting_metrics(self, rows: pl.DataFrame) -> None:
        self._insert_frame(
            "overfitting_metric",
            rows,
            [
                "run_id",
                "metric",
                "value",
                "optimistic",
                "details_json",
            ],
        )

    def record_holdout_evaluation(
        self,
        run_id: str,
        model: str,
        acknowledged_by: str,
        metrics: dict[str, Any],
    ) -> None:
        self._assert_owner()
        existing = self._connection.execute(
            "SELECT 1 FROM holdout_evaluation WHERE run_id=?", [run_id]
        ).fetchone()
        if existing:
            raise ResultsStoreError(f"holdout for run {run_id} has already been evaluated")
        self._connection.execute(
            "INSERT INTO holdout_evaluation VALUES (?, ?, ?, ?, ?)",
            [run_id, model, datetime.now(UTC), acknowledged_by, json.dumps(metrics, default=str)],
        )

    def query(self, sql: str, parameters: list[Any] | None = None) -> pl.DataFrame:
        """Read analytical rows through the owning connection (used by reporting/evaluation)."""
        normalized = sql.strip().lower()
        if not normalized.startswith("select") or ";" in normalized:
            raise ResultsStoreError("owned result queries must be one read-only SELECT")
        return self._connection.execute(sql, parameters or []).pl()

    def _insert_frame(self, table: str, frame: pl.DataFrame, columns: list[str]) -> None:
        self._assert_owner()
        if frame.is_empty():
            return
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ResultsStoreError(f"{table} write is missing columns {sorted(missing)}")
        batch = frame.select(columns).to_arrow()
        self._connection.register("_qp_stage", batch)
        quoted = ", ".join(f'"{column}"' for column in columns)
        try:
            self._connection.execute("BEGIN")
            self._connection.execute(
                f'INSERT INTO "{table}" ({quoted}) SELECT {quoted} FROM _qp_stage'
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        finally:
            self._connection.unregister("_qp_stage")


_QUERY_TABLES = frozenset(
    {
        "research_run",
        "fold",
        "prediction",
        "fold_metric",
        "artifact",
        "strategy_execution",
        "strategy_result",
        "strategy_equity",
        "target_weight",
        "optimizer_diagnostic",
        "trade",
        "overfitting_metric",
        "holdout_evaluation",
    }
)


class ResultsReader:
    """Read-only cross-run analytical query interface."""

    def __init__(self, path: str | Path = "research.duckdb") -> None:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover
            raise ResultsStoreError("DuckDB is required to query research results") from exc
        self.path = Path(path)
        if not self.path.exists():
            raise ResultsStoreError(f"results database does not exist: {self.path}")
        self._connection = duckdb.connect(str(self.path), read_only=True)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ResultsReader:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def require_run(self, run_id: str) -> None:
        row = self._connection.execute(
            "SELECT status FROM research_run WHERE run_id=?", [run_id]
        ).fetchone()
        if row is None:
            raise ResultsStoreError(f"unknown run_id '{run_id}'")

    def query(
        self,
        table: str = "research_run",
        *,
        run_id: str | None = None,
        model: str | None = None,
        strategy: str | None = None,
        cost_scenario: str | None = None,
        limit: int = 1000,
    ) -> pl.DataFrame:
        if table not in _QUERY_TABLES:
            raise ResultsStoreError(f"unknown results table '{table}'")
        if run_id:
            self.require_run(run_id)
        columns = {
            row[1] for row in self._connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        }
        predicates: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("run_id", run_id),
            ("model", model),
            ("strategy", strategy),
            ("cost_scenario", cost_scenario),
        ):
            if value is not None and column in columns:
                predicates.append(f'"{column}" = ?')
                values.append(value)
        where = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        safe_limit = max(1, min(int(limit), 1_000_000))
        result = self._connection.execute(
            f'SELECT * FROM "{table}"{where} LIMIT {safe_limit}', values
        ).pl()
        if result.is_empty():
            raise ResultsStoreError("results query selected no rows")
        return result

    def query_sql(self, sql: str) -> pl.DataFrame:
        normalized = sql.strip().lower()
        if not normalized.startswith("select") or ";" in normalized:
            raise ResultsStoreError("only a single read-only SELECT statement is permitted")
        result = self._connection.execute(sql).pl()
        if result.is_empty():
            raise ResultsStoreError("results query selected no rows")
        return result

    def export_csv(self, frame: pl.DataFrame, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(target)
        return target
