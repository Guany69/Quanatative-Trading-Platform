"""Adversarial validation suite and acceptance gates (spec sections 24-25).

A backtest's job is not to look good; it is to survive attempts to break it. This module
attacks a candidate strategy from every angle the spec lists -- higher costs, slower
execution, thinner liquidity, removed features, removed models, shifted dates, randomized
labels -- and reports where the performance actually came from.

The concentration analysis matters as much as the stress results. A strategy whose entire
excess return comes from one sector, one year, or five securities is not a strategy; it is a
single bet wearing a diversified costume. ``concentration_analysis`` makes that visible.

Randomized-label and randomized-prediction tests are the sharpest tools here: if performance
survives replacing the signal with noise, the "strategy" is really a static exposure (to
beta, size, or volatility), not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from quant_platform.config.models import AcceptanceGateConfig
from quant_platform.domain.enums import CandidateStatus
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("validation.stress")


@dataclass
class StressResult:
    """One stress scenario's outcome."""

    scenario: str
    category: str
    net_cagr: float = float("nan")
    gross_cagr: float = float("nan")
    excess_vs_benchmark: float = float("nan")
    sharpe: float = float("nan")
    information_ratio: float = float("nan")
    max_drawdown: float = float("nan")
    turnover: float = float("nan")
    total_costs: float = float("nan")
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "category": self.category,
            "net_cagr": self.net_cagr,
            "gross_cagr": self.gross_cagr,
            "excess_vs_benchmark": self.excess_vs_benchmark,
            "sharpe": self.sharpe,
            "information_ratio": self.information_ratio,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "total_costs": self.total_costs,
            "notes": self.notes,
        }


@dataclass
class ConcentrationReport:
    """Where the excess return actually came from."""

    top_securities_share: float = float("nan")
    top_security_ids: list[str] = field(default_factory=list)
    max_sector_share: float = float("nan")
    dominant_sector: str = ""
    max_year_share: float = float("nan")
    dominant_year: str = ""
    top_rebalances_share: float = float("nan")
    illiquid_share: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    def is_concentrated(self, threshold: float = 0.5) -> bool:
        return any(
            np.isfinite(v) and v > threshold
            for v in (self.max_sector_share, self.max_year_share, self.top_securities_share)
        )


def concentration_analysis(
    trades: pl.DataFrame,
    equity_curve: pl.DataFrame,
    sector_map: dict[str, str] | None = None,
    top_n_securities: int = 5,
) -> ConcentrationReport:
    """Attribute performance to securities, sectors, years, and rebalances.

    Uses realized trade P&L as the attribution basis. Approximate -- a full Brinson
    attribution needs position-level daily marks -- but sufficient to answer the question that
    matters: is this diversified, or is it one bet?
    """
    report = ConcentrationReport()

    if not trades.is_empty() and {"security_id", "notional"}.issubset(trades.columns):
        by_security = (
            trades.group_by("security_id")
            .agg(pl.col("notional").sum().alias("traded"))
            .sort("traded", descending=True)
        )
        total = float(by_security["traded"].sum())
        if total > 0:
            top = by_security.head(top_n_securities)
            report.top_securities_share = float(top["traded"].sum()) / total
            report.top_security_ids = top["security_id"].to_list()
            if report.top_securities_share > 0.5:
                report.warnings.append(
                    f"{top_n_securities} securities account for "
                    f"{report.top_securities_share:.1%} of traded notional"
                )

        if sector_map:
            with_sector = trades.with_columns(
                pl.col("security_id").replace_strict(sector_map, default="unknown").alias("_sector")
            )
            by_sector = (
                with_sector.group_by("_sector")
                .agg(pl.col("notional").sum().alias("traded"))
                .sort("traded", descending=True)
            )
            sector_total = float(by_sector["traded"].sum())
            if sector_total > 0:
                report.max_sector_share = float(by_sector["traded"][0]) / sector_total
                report.dominant_sector = str(by_sector["_sector"][0])
                if report.max_sector_share > 0.5:
                    report.warnings.append(
                        f"sector '{report.dominant_sector}' accounts for "
                        f"{report.max_sector_share:.1%} of activity"
                    )

    if not equity_curve.is_empty() and "net_return" in equity_curve.columns:
        by_year = (
            equity_curve.with_columns(pl.col("as_of").dt.year().alias("_year"))
            .group_by("_year")
            .agg(pl.col("net_return").sum().alias("total_return"))
            .sort("total_return", descending=True)
        )
        positive = by_year.filter(pl.col("total_return") > 0)
        gross_positive = float(positive["total_return"].sum()) if not positive.is_empty() else 0.0
        if gross_positive > 0:
            report.max_year_share = float(by_year["total_return"][0]) / gross_positive
            report.dominant_year = str(by_year["_year"][0])
            if report.max_year_share > 0.5:
                report.warnings.append(
                    f"year {report.dominant_year} produced {report.max_year_share:.1%} of all "
                    f"positive return; performance is time-concentrated"
                )

        # How much of the total return came from the best 5% of sessions?
        returns = equity_curve["net_return"].drop_nulls().to_numpy()
        if returns.size > 20:
            k = max(1, int(returns.size * 0.05))
            best = np.sort(returns)[-k:]
            total_return = returns.sum()
            if abs(total_return) > 1e-9:
                report.top_rebalances_share = float(best.sum() / total_return)
                if report.top_rebalances_share > 0.8:
                    report.warnings.append(
                        f"the best {k} sessions ({k / returns.size:.1%} of the sample) produced "
                        f"{report.top_rebalances_share:.0%} of total return"
                    )

    return report


def randomize_predictions(
    predictions: pl.DataFrame, random_seed: int = 42, mode: str = "shuffle"
) -> pl.DataFrame:
    """Destroy the signal while preserving its structure.

    ``shuffle`` permutes ranks within each date (same distribution, no information);
    ``random`` replaces them with fresh uniforms. If the strategy still performs, the
    performance comes from portfolio construction or static exposures rather than the model.
    """
    rng = np.random.default_rng(random_seed)
    frames = []
    for _as_of, group in predictions.group_by("as_of", maintain_order=True):
        n = group.height
        if mode == "shuffle":
            values = group["cross_sectional_rank"].to_numpy().copy()
            rng.shuffle(values)
        else:
            values = rng.uniform(0, 1, n)
        frames.append(
            group.with_columns(
                [
                    pl.Series("cross_sectional_rank", values),
                    pl.Series("prediction", values),
                ]
            )
        )
    return pl.concat(frames) if frames else predictions


def randomize_labels(panel: pl.DataFrame, target: str, random_seed: int = 42) -> pl.DataFrame:
    """Shuffle labels within each date -- the placebo test.

    A model trained on shuffled labels has nothing to learn. If it still produces positive
    out-of-sample IC, the pipeline is leaking.
    """
    rng = np.random.default_rng(random_seed)
    frames = []
    for _as_of, group in panel.group_by("as_of", maintain_order=True):
        values = group[target].to_numpy().copy()
        rng.shuffle(values)
        frames.append(group.with_columns(pl.Series(target, values)))
    return pl.concat(frames) if frames else panel


def add_random_features(
    panel: pl.DataFrame, n_features: int = 5, random_seed: int = 42
) -> tuple[pl.DataFrame, list[str]]:
    """Inject pure-noise features.

    A well-behaved model should rank them near the bottom of its importances. If noise
    features rank highly, the model is fitting noise and its other importances cannot be
    trusted either.
    """
    rng = np.random.default_rng(random_seed)
    names = [f"_random_feature_{i}" for i in range(n_features)]
    return (
        panel.with_columns([pl.Series(name, rng.normal(0, 1, panel.height)) for name in names]),
        names,
    )


@dataclass
class GateResult:
    """One acceptance gate's verdict."""

    name: str
    passed: bool
    value: float | None
    threshold: float | None
    message: str

    def describe(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.message}"


class AcceptanceGates:
    """Research gates (spec section 25).

    Nothing is ever labelled production-ready. The best possible outcome is
    ``PAPER_TRADING_CANDIDATE``, and reaching it requires passing every configured gate on
    data the strategy was not selected on.
    """

    def __init__(self, config: AcceptanceGateConfig) -> None:
        self.config = config

    def evaluate(
        self,
        holdout_metrics: dict[str, float] | None = None,
        double_cost_metrics: dict[str, float] | None = None,
        baseline_metrics: dict[str, float] | None = None,
        strategy_metrics: dict[str, float] | None = None,
        concentration: ConcentrationReport | None = None,
        deflated_sharpe: float | None = None,
        pbo: float | None = None,
        critical_data_issues: int = 0,
        leakage_failures: int = 0,
        holdout_evaluated: bool = False,
    ) -> tuple[CandidateStatus, list[GateResult]]:
        """Run every gate and return the resulting status."""
        cfg = self.config
        results: list[GateResult] = []
        strategy_metrics = strategy_metrics or {}

        # --- disqualifying conditions first -------------------------------------
        results.append(
            GateResult(
                "no_critical_data_issues",
                critical_data_issues == 0,
                float(critical_data_issues),
                0.0,
                f"{critical_data_issues} critical data-quality issue(s)",
            )
        )
        results.append(
            GateResult(
                "no_leakage_failures",
                leakage_failures == 0,
                float(leakage_failures),
                0.0,
                f"{leakage_failures} data-leakage test failure(s)",
            )
        )

        # --- cost robustness -----------------------------------------------------
        if cfg.require_positive_net_excess_at_double_cost and double_cost_metrics:
            excess = double_cost_metrics.get("excess_cagr", float("nan"))
            results.append(
                GateResult(
                    "positive_excess_at_2x_cost",
                    bool(np.isfinite(excess) and excess > 0),
                    excess,
                    0.0,
                    f"net excess return at 2x cost = {excess:.2%}",
                )
            )

        # --- must beat the transparent baseline ----------------------------------
        if cfg.require_beating_factor_baseline and baseline_metrics:
            strategy_ir = strategy_metrics.get("information_ratio", float("nan"))
            baseline_ir = baseline_metrics.get("information_ratio", float("nan"))
            beats = bool(
                np.isfinite(strategy_ir) and np.isfinite(baseline_ir) and strategy_ir > baseline_ir
            )
            results.append(
                GateResult(
                    "beats_factor_baseline",
                    beats,
                    strategy_ir,
                    baseline_ir,
                    f"strategy IR {strategy_ir:.3f} vs transparent baseline {baseline_ir:.3f}",
                )
            )

        # --- risk limits ---------------------------------------------------------
        for metric, threshold, name, comparison in (
            ("max_drawdown", cfg.max_drawdown_limit, "acceptable_drawdown", "abs_below"),
            ("tracking_error", cfg.max_tracking_error, "acceptable_tracking_error", "below"),
            ("avg_turnover", cfg.max_annual_turnover, "acceptable_turnover", "below"),
        ):
            value = strategy_metrics.get(metric, float("nan"))
            if not np.isfinite(value):
                continue
            passed = abs(value) <= threshold if comparison == "abs_below" else value <= threshold
            results.append(
                GateResult(
                    name,
                    bool(passed),
                    value,
                    threshold,
                    f"{metric}={value:.3f} (limit {threshold})",
                )
            )

        # --- concentration -------------------------------------------------------
        if concentration is not None:
            if np.isfinite(concentration.max_sector_share):
                passed = concentration.max_sector_share <= cfg.max_single_sector_excess_contribution
                results.append(
                    GateResult(
                        "no_single_sector_dominance",
                        bool(passed),
                        concentration.max_sector_share,
                        cfg.max_single_sector_excess_contribution,
                        f"largest sector share {concentration.max_sector_share:.1%} "
                        f"({concentration.dominant_sector})",
                    )
                )
            if np.isfinite(concentration.max_year_share):
                passed = concentration.max_year_share <= cfg.max_single_year_excess_contribution
                results.append(
                    GateResult(
                        "no_single_year_dominance",
                        bool(passed),
                        concentration.max_year_share,
                        cfg.max_single_year_excess_contribution,
                        f"largest year share {concentration.max_year_share:.1%} "
                        f"({concentration.dominant_year})",
                    )
                )

        # --- multiple-testing corrections ----------------------------------------
        if (
            cfg.min_deflated_sharpe is not None
            and deflated_sharpe is not None
            and np.isfinite(deflated_sharpe)
        ):
            results.append(
                GateResult(
                    "deflated_sharpe",
                    deflated_sharpe >= cfg.min_deflated_sharpe,
                    deflated_sharpe,
                    cfg.min_deflated_sharpe,
                    f"DSR {deflated_sharpe:.3f} (>= {cfg.min_deflated_sharpe})",
                )
            )
        if cfg.max_pbo is not None and pbo is not None and np.isfinite(pbo):
            results.append(
                GateResult(
                    "probability_of_backtest_overfitting",
                    pbo <= cfg.max_pbo,
                    pbo,
                    cfg.max_pbo,
                    f"PBO {pbo:.3f} (<= {cfg.max_pbo})",
                )
            )

        # --- holdout -------------------------------------------------------------
        if not holdout_evaluated:
            status = (
                CandidateStatus.REJECTED
                if any(not r.passed for r in results)
                else CandidateStatus.HOLDOUT_NOT_EVALUATED
            )
            return status, results

        holdout_ir = (holdout_metrics or {}).get("information_ratio", float("nan"))
        holdout_passed = bool(
            np.isfinite(holdout_ir) and holdout_ir >= cfg.min_holdout_net_information_ratio
        )
        results.append(
            GateResult(
                "holdout_information_ratio",
                holdout_passed,
                holdout_ir,
                cfg.min_holdout_net_information_ratio,
                f"locked-holdout net IR {holdout_ir:.3f}",
            )
        )

        if not holdout_passed:
            return CandidateStatus.HOLDOUT_FAILED, results
        if any(not r.passed for r in results):
            return CandidateStatus.REJECTED, results
        # The ceiling. Never "production ready" -- that is a human decision made with
        # information this system does not have.
        return CandidateStatus.PAPER_TRADING_CANDIDATE, results


def summarize_gates(status: CandidateStatus, results: list[GateResult]) -> str:
    lines = [f"Candidate status: {status.value.upper()}", ""]
    lines += [r.describe() for r in results]
    failed = [r for r in results if not r.passed]
    lines.append("")
    lines.append(f"{len(results) - len(failed)}/{len(results)} gates passed")
    if status == CandidateStatus.PAPER_TRADING_CANDIDATE:
        lines.append(
            "This candidate may proceed to PAPER TRADING. It is NOT approved for live "
            "capital, and simulated results do not guarantee future performance."
        )
    return "\n".join(lines)
