"""Report generation (spec section 30).

Reports carry provenance as prominently as they carry numbers. A Sharpe ratio without its
data source, cost assumptions, and evaluation stage is not a result -- it is a decoration.
Every artifact this module writes leads with those disclosures, and the synthetic-data banner
is impossible to miss by design.

Outputs: JSON metrics, CSV time series / trades / predictions, a Markdown summary, a
self-contained HTML report, and PNG charts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_platform.domain.portfolio import PerformanceReport
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("report")

DISCLAIMER = (
    "Simulated historical performance. Past or simulated performance does not guarantee "
    "future results and is not a prediction of future returns. This software is a research "
    "tool, not investment advice."
)


@dataclass
class ReportBundle:
    """Everything one report run needs."""

    title: str
    reports: list[PerformanceReport] = field(default_factory=list)
    equity_curves: dict[str, pl.DataFrame] = field(default_factory=dict)
    trades: dict[str, pl.DataFrame] = field(default_factory=dict)
    predictions: pl.DataFrame | None = None
    forecast_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_importance: pl.DataFrame | None = None
    stress_results: pl.DataFrame | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _fmt(value: float | None, kind: str = "float") -> str:
    """Format a metric for display, tolerating NaN/None rather than crashing a report."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.2f}%"
    if kind == "ratio":
        return f"{value:.3f}"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "money":
        return f"${value:,.0f}"
    return f"{value:.4f}"


def _provenance_lines(r: PerformanceReport) -> list[str]:
    """The disclosures that must accompany every set of numbers."""
    return [
        f"Data source: {r.data_source}",
        f"Benchmark: {r.benchmark_source}",
        f"Evaluation stage: {r.stage.value}",
        f"Cost scenario: {r.cost_scenario.value}",
        f"Execution delay: {r.execution_delay_sessions} session(s)",
        f"Period: {r.start_date} to {r.end_date} ({r.n_observations} sessions)",
        f"Historical constituents complete: {r.historical_constituents_complete}",
        f"Delisting returns complete: {r.delisting_returns_complete}",
        f"Point-in-time fundamentals complete: {r.point_in_time_fundamentals_complete}",
    ]


# --------------------------------------------------------------------------- writers
def write_json(bundle: ReportBundle, out_dir: Path) -> Path:
    """Machine-readable metrics."""
    payload = {
        "title": bundle.title,
        "generated_at": datetime.now(UTC).isoformat(),
        "disclaimer": DISCLAIMER,
        "run_metadata": bundle.run_metadata,
        "reports": [
            {
                "strategy": r.strategy_name,
                "stage": r.stage.value,
                "cost_scenario": r.cost_scenario.value,
                "start_date": str(r.start_date),
                "end_date": str(r.end_date),
                "is_synthetic_data": r.is_synthetic_data,
                "disclosure": r.disclosure_note,
                "data_source": r.data_source,
                "benchmark_source": r.benchmark_source,
                "execution_delay_sessions": r.execution_delay_sessions,
                "historical_constituents_complete": r.historical_constituents_complete,
                "delisting_returns_complete": r.delisting_returns_complete,
                "point_in_time_fundamentals_complete": r.point_in_time_fundamentals_complete,
                "metrics": {k: (None if not np.isfinite(v) else v) for k, v in r.metrics.items()},
                "benchmark_metrics": {
                    k: (None if not np.isfinite(v) else v) for k, v in r.benchmark_metrics.items()
                },
            }
            for r in bundle.reports
        ],
        "forecast_metrics": bundle.forecast_metrics,
        "notes": bundle.notes,
    }
    path = out_dir / "metrics.json"
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def write_csvs(bundle: ReportBundle, out_dir: Path) -> list[Path]:
    """CSV time series, trades, and predictions."""
    written: list[Path] = []
    for name, curve in bundle.equity_curves.items():
        p = out_dir / f"timeseries_{name}.csv"
        curve.write_csv(p)
        written.append(p)
    for name, trades in bundle.trades.items():
        if trades.is_empty():
            continue
        p = out_dir / f"trades_{name}.csv"
        trades.write_csv(p)
        written.append(p)
    if bundle.predictions is not None and not bundle.predictions.is_empty():
        p = out_dir / "predictions.csv"
        bundle.predictions.write_csv(p)
        written.append(p)
    if bundle.stress_results is not None and not bundle.stress_results.is_empty():
        p = out_dir / "stress_tests.csv"
        bundle.stress_results.write_csv(p)
        written.append(p)
    return written


def write_markdown(bundle: ReportBundle, out_dir: Path) -> Path:
    """Human-readable summary."""
    lines: list[str] = [f"# {bundle.title}", ""]

    synthetic = any(r.is_synthetic_data for r in bundle.reports)
    if synthetic:
        lines += [
            "> ## SYNTHETIC DEMONSTRATION DATA",
            "> These results describe **randomly generated fixture securities**. They carry",
            "> **no investment meaning whatsoever** and say nothing about any real market,",
            "> security, or strategy. The data was produced by a random number generator.",
            "",
        ]
    lines += [f"> {DISCLAIMER}", ""]

    if bundle.reports:
        lines += ["## Provenance", ""]
        lines += [f"- {line}" for line in _provenance_lines(bundle.reports[0])]
        lines += [""]

    lines += ["## Portfolio results", ""]
    lines += [
        "| Strategy | Cost | Net CAGR | Gross CAGR | Vol | Sharpe | MaxDD | TE | IR | "
        "Beta | Turnover | Trades |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in bundle.reports:
        m = r.metrics
        lines.append(
            f"| {r.strategy_name} | {r.cost_scenario.value} | "
            f"{_fmt(m.get('cagr'), 'pct')} | {_fmt(m.get('gross_cagr'), 'pct')} | "
            f"{_fmt(m.get('annualized_volatility'), 'pct')} | "
            f"{_fmt(m.get('sharpe_ratio'), 'ratio')} | "
            f"{_fmt(m.get('max_drawdown'), 'pct')} | "
            f"{_fmt(m.get('tracking_error'), 'pct')} | "
            f"{_fmt(m.get('information_ratio'), 'ratio')} | "
            f"{_fmt(m.get('beta'), 'ratio')} | "
            f"{_fmt(m.get('avg_turnover'), 'pct')} | "
            f"{_fmt(m.get('n_trades'), 'int')} |"
        )
    lines += [""]

    if bundle.reports:
        bm = bundle.reports[0].benchmark_metrics
        lines += [
            "## Benchmark",
            "",
            f"- CAGR: {_fmt(bm.get('cagr'), 'pct')}",
            f"- Volatility: {_fmt(bm.get('annualized_volatility'), 'pct')}",
            f"- Sharpe: {_fmt(bm.get('sharpe_ratio'), 'ratio')}",
            f"- Max drawdown: {_fmt(bm.get('max_drawdown'), 'pct')}",
            "",
        ]

    if bundle.forecast_metrics:
        lines += ["## Forecast quality (out-of-sample)", ""]
        lines += ["| Model | Mean IC | IC IR | Positive IC % | Periods |", "|---|---|---|---|---|"]
        for model, fm in bundle.forecast_metrics.items():
            lines.append(
                f"| {model} | {_fmt(fm.get('mean_ic'), 'ratio')} | "
                f"{_fmt(fm.get('ic_ir'), 'ratio')} | "
                f"{_fmt(fm.get('positive_ic_pct'), 'pct')} | "
                f"{_fmt(fm.get('n_periods'), 'int')} |"
            )
        lines += [""]

    if bundle.stress_results is not None and not bundle.stress_results.is_empty():
        lines += ["## Stress tests", "", "```", str(bundle.stress_results), "```", ""]

    if bundle.notes:
        lines += ["## Notes and limitations", ""]
        lines += [f"- {n}" for n in bundle.notes]
        lines += [""]

    if bundle.run_metadata:
        lines += ["## Reproducibility", ""]
        for k in ("run_id", "git_commit", "git_dirty", "config_hash", "data_snapshot_id"):
            if k in bundle.run_metadata:
                lines.append(f"- {k}: `{bundle.run_metadata[k]}`")
        lines += [""]

    path = out_dir / "report.md"
    path.write_text("\n".join(lines))
    return path


def write_charts(bundle: ReportBundle, out_dir: Path) -> list[Path]:
    """Render charts. Import of matplotlib is local so the core stays importable without it."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("charts skipped, matplotlib unavailable: %s", exc)
        return []

    written: list[Path] = []
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if not bundle.equity_curves:
        return written

    # 1. Equity curves vs benchmark.
    fig, ax = plt.subplots(figsize=(11, 5))
    bench_plotted = False
    for name, curve in bundle.equity_curves.items():
        d = curve["as_of"].to_list()
        net = np.nan_to_num(curve["net_return"].to_numpy(), nan=0.0)
        ax.plot(d, np.cumprod(1 + net), label=f"{name} (net)", linewidth=1.3)
        if not bench_plotted and "benchmark_return" in curve.columns:
            b = np.nan_to_num(curve["benchmark_return"].to_numpy(), nan=0.0)
            ax.plot(d, np.cumprod(1 + b), label="benchmark", color="black", ls="--", lw=1.5)
            bench_plotted = True
    ax.set_title("Cumulative growth of $1 (SIMULATED)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = charts_dir / "equity_curve.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    # 2. Drawdown.
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for name, curve in bundle.equity_curves.items():
        net = np.nan_to_num(curve["net_return"].to_numpy(), nan=0.0)
        eq = np.cumprod(1 + net)
        dd = eq / np.maximum.accumulate(eq) - 1.0
        ax.fill_between(curve["as_of"].to_list(), dd, 0, alpha=0.35, label=name)
    ax.set_title("Drawdown (net, SIMULATED)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = charts_dir / "drawdown.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    # 3. Gross vs net -- shows what costs actually removed.
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, curve in bundle.equity_curves.items():
        d = curve["as_of"].to_list()
        g = np.cumprod(1 + np.nan_to_num(curve["gross_return"].to_numpy(), nan=0.0))
        n = np.cumprod(1 + np.nan_to_num(curve["net_return"].to_numpy(), nan=0.0))
        ax.plot(d, g, ls=":", lw=1.2, label=f"{name} gross")
        ax.plot(d, n, lw=1.4, label=f"{name} net")
    ax.set_title("Gross vs net -- the gap is transaction cost (SIMULATED)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = charts_dir / "gross_vs_net.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    # 4. Feature importance.
    if bundle.feature_importance is not None and not bundle.feature_importance.is_empty():
        fi = bundle.feature_importance.head(20)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * fi.height)))
        ax.barh(fi["feature"].to_list()[::-1], fi["importance"].to_list()[::-1])
        ax.set_title("Feature importance")
        ax.grid(alpha=0.3, axis="x")
        p = charts_dir / "feature_importance.png"
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        written.append(p)

    # 5. Stress comparison.
    if bundle.stress_results is not None and not bundle.stress_results.is_empty():
        sr = bundle.stress_results
        if {"scenario", "net_cagr"}.issubset(sr.columns):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(sr["scenario"].to_list(), [v * 100 for v in sr["net_cagr"].to_list()])
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel("Net CAGR (%)")
            ax.set_title("Stress scenarios (SIMULATED)")
            plt.xticks(rotation=45, ha="right")
            ax.grid(alpha=0.3, axis="y")
            p = charts_dir / "stress_tests.png"
            fig.tight_layout()
            fig.savefig(p, dpi=110)
            plt.close(fig)
            written.append(p)

    return written


def write_html(bundle: ReportBundle, out_dir: Path, chart_paths: list[Path]) -> Path:
    """Self-contained HTML report."""
    synthetic = any(r.is_synthetic_data for r in bundle.reports)
    banner = (
        """<div class="banner synthetic">
        <h2>SYNTHETIC DEMONSTRATION DATA</h2>
        <p>These results describe <strong>randomly generated fixture securities</strong>.
        They carry <strong>no investment meaning whatsoever</strong> and say nothing about any
        real market, security, or strategy.</p></div>"""
        if synthetic
        else ""
    )

    rows = "".join(
        f"<tr><td>{r.strategy_name}</td><td>{r.cost_scenario.value}</td>"
        f"<td>{_fmt(r.metrics.get('cagr'), 'pct')}</td>"
        f"<td>{_fmt(r.metrics.get('gross_cagr'), 'pct')}</td>"
        f"<td>{_fmt(r.metrics.get('annualized_volatility'), 'pct')}</td>"
        f"<td>{_fmt(r.metrics.get('sharpe_ratio'), 'ratio')}</td>"
        f"<td>{_fmt(r.metrics.get('max_drawdown'), 'pct')}</td>"
        f"<td>{_fmt(r.metrics.get('information_ratio'), 'ratio')}</td>"
        f"<td>{_fmt(r.metrics.get('beta'), 'ratio')}</td></tr>"
        for r in bundle.reports
    )

    prov = ""
    if bundle.reports:
        prov = "".join(f"<li>{line}</li>" for line in _provenance_lines(bundle.reports[0]))

    ic_rows = "".join(
        f"<tr><td>{model}</td><td>{_fmt(fm.get('mean_ic'), 'ratio')}</td>"
        f"<td>{_fmt(fm.get('ic_ir'), 'ratio')}</td>"
        f"<td>{_fmt(fm.get('positive_ic_pct'), 'pct')}</td></tr>"
        for model, fm in bundle.forecast_metrics.items()
    )

    ic_section = (
        (
            "<h2>Forecast quality (out-of-sample)</h2><table><thead><tr><th>Model</th>"
            "<th>Mean IC</th><th>IC IR</th><th>Positive IC %</th></tr></thead><tbody>"
            + ic_rows
            + "</tbody></table>"
        )
        if ic_rows
        else ""
    )

    charts_html = "".join(
        f'<figure><img src="charts/{p.name}" alt="{p.stem}"/>'
        f"<figcaption>{p.stem.replace('_', ' ')}</figcaption></figure>"
        for p in chart_paths
    )

    notes_html = "".join(f"<li>{n}</li>" for n in bundle.notes)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{bundle.title}</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 .banner {{ padding: 1rem 1.25rem; border-radius: 8px; margin: 1rem 0; }}
 .synthetic {{ background: #fff3cd; border: 2px solid #e0a800; }}
 .disclaimer {{ background: #f0f0f0; border-left: 4px solid #666; padding: .75rem 1rem;
                font-size: .9rem; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }}
 th, td {{ border: 1px solid #ddd; padding: .5rem .6rem; text-align: right; }}
 th {{ background: #f7f7f7; }} td:first-child, th:first-child {{ text-align: left; }}
 figure {{ margin: 1.5rem 0; }} img {{ max-width: 100%; border: 1px solid #eee; }}
 figcaption {{ font-size: .85rem; color: #666; margin-top: .3rem; }}
 code {{ background: #f4f4f4; padding: .1rem .3rem; }}
</style></head><body>
<h1>{bundle.title}</h1>
{banner}
<div class="disclaimer">{DISCLAIMER}</div>
<h2>Provenance</h2><ul>{prov}</ul>
<h2>Portfolio results</h2>
<table><thead><tr><th>Strategy</th><th>Cost</th><th>Net CAGR</th><th>Gross CAGR</th>
<th>Vol</th><th>Sharpe</th><th>Max DD</th><th>IR</th><th>Beta</th></tr></thead>
<tbody>{rows}</tbody></table>
{ic_section}
<h2>Charts</h2>{charts_html}
{"<h2>Notes and limitations</h2><ul>" + notes_html + "</ul>" if notes_html else ""}
<hr><p style="font-size:.8rem;color:#666">Generated {datetime.now(UTC).isoformat()}
&middot; run <code>{bundle.run_metadata.get("run_id", "n/a")}</code>
&middot; commit <code>{bundle.run_metadata.get("git_commit", "n/a")}</code></p>
</body></html>"""

    path = out_dir / "report.html"
    path.write_text(html)
    return path


def generate_report(bundle: ReportBundle, out_dir: str | Path) -> dict[str, Any]:
    """Write the full report set and return the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = write_json(bundle, out)
    csvs = write_csvs(bundle, out)
    charts = write_charts(bundle, out)
    md_path = write_markdown(bundle, out)
    html_path = write_html(bundle, out, charts)

    logger.info("report written to %s", out)
    return {
        "dir": str(out),
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
        "csvs": [str(p) for p in csvs],
        "charts": [str(p) for p in charts],
    }
