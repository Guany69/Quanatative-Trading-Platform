"""Regenerable report bundle built from authoritative DuckDB rows."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl

from quant_platform.research.results import ResultsStore


def generate_persisted_run_report(
    store: ResultsStore, run_id: str, out_dir: str | Path
) -> dict[str, str]:
    """Read a completed stage from DuckDB and emit all HLD report formats."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = store.query("SELECT * FROM research_run WHERE run_id=?", [run_id])
    model_long = store.query(
        "SELECT model, fold, metric, value FROM fold_metric WHERE run_id=? "
        "ORDER BY model, fold, metric",
        [run_id],
    )
    strategy_long = store.query(
        "SELECT model, strategy, cost_scenario, metric, value FROM strategy_result "
        "WHERE run_id=? ORDER BY model, strategy, cost_scenario, metric",
        [run_id],
    )
    overfitting = store.query(
        "SELECT metric, value, optimistic, details_json FROM overfitting_metric "
        "WHERE run_id=? ORDER BY metric",
        [run_id],
    )

    model_csv = out / "model_comparison.csv"
    strategy_csv = out / "strategy_comparison.csv"
    overfit_csv = out / "overfitting.csv"
    model_long.write_csv(model_csv)
    strategy_long.write_csv(strategy_csv)
    overfitting.write_csv(overfit_csv)

    payload: dict[str, Any] = {
        "run": run.to_dicts()[0] if not run.is_empty() else {},
        "model_comparison": model_long.to_dicts(),
        "strategy_comparison": strategy_long.to_dicts(),
        "overfitting": overfitting.to_dicts(),
    }
    json_path = out / "results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    markdown = [
        f"# Research run `{run_id}`",
        "",
        "This report was regenerated from the DuckDB research system of record.",
        "",
        "## Model/fold metrics",
        "",
        _markdown_table(model_long),
        "",
        "## Strategy/scenario metrics",
        "",
        _markdown_table(strategy_long),
        "",
        "## Overfitting and robustness",
        "",
        _markdown_table(overfitting),
    ]
    markdown_path = out / "report.md"
    markdown_path.write_text("\n".join(markdown) + "\n")

    chart_path = out / "strategy_cagr.png"
    cagr = strategy_long.filter(pl.col("metric") == "cagr")
    fig, axis = plt.subplots(figsize=(10, 5))
    if cagr.is_empty():
        axis.text(0.5, 0.5, "No strategy CAGR rows", ha="center", va="center")
    else:
        labels = [
            f"{row['strategy']}\n{row['cost_scenario']}" for row in cagr.iter_rows(named=True)
        ]
        axis.bar(labels, cagr["value"].fill_null(float("nan")).to_list())
        axis.set_ylabel("CAGR")
        axis.tick_params(axis="x", rotation=45)
    axis.set_title(f"Strategy comparison — {run_id}")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=140)
    plt.close(fig)

    html_path = out / "report.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Research run "
        + html.escape(run_id)
        + "</title></head><body><h1>Research run <code>"
        + html.escape(run_id)
        + "</code></h1><p>Generated from authoritative DuckDB rows.</p>"
        + "<h2>Model/fold metrics</h2>"
        + _html_table(model_long)
        + "<h2>Strategy/scenario metrics</h2>"
        + _html_table(strategy_long)
        + "<h2>Overfitting and robustness</h2>"
        + _html_table(overfitting)
        + f"<p><img src='{html.escape(chart_path.name)}' alt='Strategy CAGR chart'></p>"
        + "</body></html>\n"
    )
    return {
        "html": str(html_path),
        "markdown": str(markdown_path),
        "json": str(json_path),
        "model_csv": str(model_csv),
        "strategy_csv": str(strategy_csv),
        "overfitting_csv": str(overfit_csv),
        "chart": str(chart_path),
    }


def _markdown_table(frame: pl.DataFrame, limit: int = 200) -> str:
    if frame.is_empty():
        return "_No rows._"
    columns = frame.columns
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.head(limit).iter_rows(named=True):
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _html_table(frame: pl.DataFrame, limit: int = 200) -> str:
    if frame.is_empty():
        return "<p><em>No rows.</em></p>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in frame.columns)
    rows = []
    for row in frame.head(limit).iter_rows(named=True):
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in frame.columns)
            + "</tr>"
        )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
