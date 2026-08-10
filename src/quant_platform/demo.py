"""The end-to-end synthetic demonstration (spec section 36).

Runs every stage of the platform on deterministic fixture data with no credentials and no
network access, then writes a full report set. This is the command that proves the machinery
is wired together correctly.

What it demonstrates:
  1-2. deterministic fixture data + point-in-time construction
  3.   point-in-time universe (survivorship- and look-ahead-controlled)
  4.   features from strictly trailing windows
  5.   20-session benchmark-relative labels with delisting returns booked
  6.   purged + embargoed train/test split, preprocessing fitted on train only
  7.   predictions from three models, including two baselines that act as controls
  8.   three portfolio variants
  9-10. cost-aware backtest with execution delay
  11.  cost stress scenarios
  12.  benchmark comparison
  13-14. reports, labeled synthetic throughout

What it does NOT demonstrate: anything about real markets. The data is random.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from quant_platform.config import config_hash, load_charter, snapshot_config
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
from quant_platform.utilities.narrow import as_date, as_float
from quant_platform.utilities.reproducibility import (
    RunMetadata,
    dataset_snapshot_id,
    get_logger,
    set_global_seeds,
)

logger = get_logger("demo")

PORTFOLIO_VARIANTS = ["equal_weight", "score_weighted", "inverse_volatility"]
STRESS_SCENARIOS = ["zero", "base", "double", "triple"]


def _check_universe_is_selective(panel: pl.DataFrame, max_holdings: int) -> None:
    """Fail loudly if the universe is not meaningfully larger than the holding count.

    When the eligible universe is no bigger than ``max_holdings``, every constructor holds
    *everything* and the ranking is never consulted. Models then produce byte-identical
    portfolios -- including the no-skill baseline -- and the run silently demonstrates
    nothing while looking like it worked. A real S&P 500 mandate picks 50-100 names from
    ~500, so the demo must preserve that ratio to exercise selection at all.
    """
    median_size = as_float(
        panel.group_by("as_of").agg(pl.len().alias("n"))["n"].median(),
        context="median universe size",
    )
    if median_size <= max_holdings:
        raise RuntimeError(
            f"universe is not selective: median {median_size:.0f} eligible names per date vs "
            f"max_holdings={max_holdings}. Every portfolio would hold the entire universe and "
            f"the model ranking would be ignored, making all models look identical. Increase "
            f"--securities (a ratio of >= 3x max_holdings is realistic) or lower max_holdings."
        )
    ratio = median_size / max_holdings
    if ratio < 2.0:
        logger.warning(
            "universe is only %.1fx max_holdings; selection is weak and models will look "
            "similar. >= 3x is realistic.",
            ratio,
        )
    else:
        logger.info(
            "      universe is selective: picking %d of ~%.0f names (%.1fx)",
            max_holdings,
            median_size,
            ratio,
        )


def run_full_demo(
    config_path: str | Path = "configs/research_charter.yaml",
    out_dir: str | Path = "reports/demo",
    # Must comfortably exceed portfolio.max_holdings (100) or selection is a no-op --
    # see _check_universe_is_selective.
    n_securities: int = 400,
    test_start: date = date(2021, 1, 4),
) -> dict[str, Any]:
    """Run the demo and write reports. Returns the report paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    charter = load_charter(config_path)
    seeds = set_global_seeds(charter.random_seed)
    meta = RunMetadata(command="run-demo", config_hash=config_hash(charter), seeds=seeds)

    logger.info("=" * 70)
    logger.info("SYNTHETIC DEMONSTRATION -- randomly generated data, no investment meaning")
    logger.info("=" * 70)

    # --- 1-5. data, universe, features, labels -----------------------------------
    logger.info("[1/8] building fixture data + point-in-time panels")
    data = load_fixture_data(charter, n_securities=n_securities)
    meta.data_snapshot_id = dataset_snapshot_id(data.prices, data.features, data.labels)
    meta.feature_version = charter.features.feature_version
    meta.label_version = charter.label.label_version

    panel = build_training_frame(data)
    if panel.is_empty():
        raise RuntimeError("training panel is empty; check universe/feature/label alignment")
    logger.info(
        "      panel=%s securities=%d dates=%s..%s",
        panel.shape,
        panel["security_id"].n_unique(),
        panel["as_of"].min(),
        panel["as_of"].max(),
    )
    _check_universe_is_selective(panel, charter.portfolio.max_holdings)

    # --- 6. purged + embargoed split ---------------------------------------------
    logger.info("[2/8] purged + embargoed train/test split at %s", test_start)
    train, test = split_train_test(
        panel, test_start, charter.validation.embargo_sessions, data.calendar
    )
    if train.is_empty() or test.is_empty():
        raise RuntimeError(f"empty split: train={train.height} test={test.height}")

    max_train_window_end = as_date(train["window_end"].max(), "max train window_end")
    if max_train_window_end >= test_start:
        raise RuntimeError(
            f"PURGE FAILURE: a training label window ends {max_train_window_end}, on or after "
            f"test_start {test_start}. Training data overlaps the test period."
        )
    logger.info(
        "      train=%d rows (last label window closes %s) | test=%d rows (opens %s)",
        train.height,
        max_train_window_end,
        test.height,
        test["as_of"].min(),
    )

    # --- 7. models ----------------------------------------------------------------
    logger.info("[3/8] fitting models (preprocessing fitted on TRAIN only)")
    target = "excess_return_rank"
    models = default_models(charter)
    predictions: dict[str, pl.DataFrame] = {}
    forecast_metrics: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        preds = fit_and_predict(model, train, test, data.feature_columns, target, charter)
        predictions[name] = preds
        forecast_metrics[name] = forecast_ic(preds, test, target)
        logger.info(
            "      %-18s mean_IC=%+.4f  IC_IR=%+.3f",
            name,
            forecast_metrics[name].get("mean_ic", float("nan")),
            forecast_metrics[name].get("ic_ir", float("nan")),
        )

    primary_model_name = "factor_composite"
    primary_preds = predictions[primary_model_name]

    # --- 8. portfolio variants ------------------------------------------------------
    logger.info("[4/8] backtesting %d portfolio variants (base costs)", len(PORTFOLIO_VARIANTS))
    reports = []
    curves: dict[str, pl.DataFrame] = {}
    trades: dict[str, pl.DataFrame] = {}
    test_end = as_date(test["as_of"].max(), "test_end")

    for variant in PORTFOLIO_VARIANTS:
        result = run_backtest_for_predictions(
            primary_preds, data, charter, variant, "base", strategy_name=variant
        )
        rep = evaluate(
            result, charter, data, EvaluationStage.WALK_FORWARD_TEST, test_start, test_end
        )
        reports.append(rep)
        curves[variant] = result.equity_curve().filter(pl.col("as_of") >= test_start)
        trades[variant] = result.trade_log()
        logger.info(
            "      %-20s net CAGR=%+.2f%%  Sharpe=%.2f  IR=%+.2f  trades=%d",
            variant,
            rep.metrics.get("cagr", float("nan")) * 100,
            rep.metrics.get("sharpe_ratio", float("nan")),
            rep.metrics.get("information_ratio", float("nan")),
            len(result.trades),
        )

    # --- baseline model comparison (equal weight, same costs) -----------------------
    logger.info("[5/8] baseline model comparison (equal_weight, base costs)")
    for name in ("no_skill", "momentum_baseline"):
        result = run_backtest_for_predictions(
            predictions[name],
            data,
            charter,
            "equal_weight",
            "base",
            strategy_name=f"{name}_equal_weight",
        )
        rep = evaluate(
            result, charter, data, EvaluationStage.WALK_FORWARD_TEST, test_start, test_end
        )
        reports.append(rep)
        curves[f"{name}_equal_weight"] = result.equity_curve().filter(pl.col("as_of") >= test_start)
        logger.info(
            "      %-20s net CAGR=%+.2f%%  IR=%+.2f",
            name,
            rep.metrics.get("cagr", float("nan")) * 100,
            rep.metrics.get("information_ratio", float("nan")),
        )

    # --- 11. cost stress ------------------------------------------------------------
    logger.info("[6/8] cost stress scenarios")
    stress_rows = []
    for scenario in STRESS_SCENARIOS:
        result = run_backtest_for_predictions(
            primary_preds,
            data,
            charter,
            "equal_weight",
            scenario,
            strategy_name=f"equal_weight_{scenario}",
        )
        rep = evaluate(
            result, charter, data, EvaluationStage.WALK_FORWARD_TEST, test_start, test_end
        )
        if scenario != "base":
            reports.append(rep)
        bench_cagr = rep.benchmark_metrics.get("cagr", float("nan"))
        stress_rows.append(
            {
                "scenario": f"cost_{scenario}",
                "net_cagr": rep.metrics.get("cagr", float("nan")),
                "gross_cagr": rep.metrics.get("gross_cagr", float("nan")),
                "excess_vs_benchmark": rep.metrics.get("cagr", float("nan")) - bench_cagr,
                "sharpe": rep.metrics.get("sharpe_ratio", float("nan")),
                "total_costs": rep.metrics.get("total_costs", float("nan")),
            }
        )
        logger.info(
            "      cost=%-7s net CAGR=%+.2f%%  gross CAGR=%+.2f%%  costs=$%.0f",
            scenario,
            rep.metrics.get("cagr", float("nan")) * 100,
            rep.metrics.get("gross_cagr", float("nan")) * 100,
            rep.metrics.get("total_costs", 0.0),
        )
    stress = pl.DataFrame(stress_rows)

    # --- 13. reports -----------------------------------------------------------------
    logger.info("[7/8] generating reports")
    fi = models[primary_model_name].feature_importance()

    bench_cagr = reports[0].benchmark_metrics.get("cagr", float("nan"))
    notes = [
        "SYNTHETIC DATA: every security here is randomly generated. Results have no "
        "investment meaning and say nothing about real markets.",
        "The fixture deliberately plants a momentum effect (strength=0.02) so the pipeline "
        "has a known ground truth to detect. Measured IC reflects that planted effect, not "
        "a discovered market anomaly.",
        f"Benchmark CAGR over the test window: {bench_cagr * 100:.2f}%. Any strategy CAGR "
        f"above or below this is a property of randomly generated data.",
        "Execution delay of "
        f"{charter.backtest.rebalance.execution_delay_sessions} session(s) is applied: signals "
        "never fill at the close that produced them.",
        "Gross and net are reported separately. The zero-cost scenario is a diagnostic only "
        "and is not achievable.",
        "The locked holdout was NOT evaluated in this demo "
        f"(allow_holdout_evaluation={charter.validation.allow_holdout_evaluation}).",
        "Market-impact and slippage models are estimates, not calibrated measurements. See "
        "docs/limitations.md.",
    ]

    bundle = ReportBundle(
        title="Synthetic Demonstration -- Benchmark-Relative Equity Ranking",
        reports=reports,
        equity_curves=curves,
        trades=trades,
        predictions=primary_preds.head(5000),
        forecast_metrics=forecast_metrics,
        feature_importance=fi,
        stress_results=stress,
        run_metadata=meta.to_dict(),
        notes=notes,
    )
    paths = generate_report(bundle, out)

    logger.info("[8/8] persisting run metadata + resolved config")
    meta.artifacts = {k: str(v) for k, v in paths.items() if isinstance(v, str)}
    meta.save(out / "run_metadata.json")
    snapshot_config(charter, out / "resolved_config.json")

    logger.info("=" * 70)
    logger.info("DEMO COMPLETE -- reports in %s", out)
    logger.info("Results are SYNTHETIC. No investment meaning.")
    logger.info("=" * 70)
    return paths


# ---------------------------------------------------------------------------
# Stress suite and paper-trading entry points used by the CLI.
# ---------------------------------------------------------------------------
def run_stress_suite(
    config_path: str | Path = "configs/research_charter.yaml",
    model: str = "factor-composite",
    n_securities: int = 200,
    out_dir: str | Path = "reports/stress",
    test_start: date = date(2021, 1, 4),
) -> pl.DataFrame:
    """Run the adversarial validation suite (spec section 24).

    Attacks the candidate from every angle the spec lists and reports where performance
    actually came from. The randomization scenarios are the sharpest: if results survive
    replacing the signal with noise, the "strategy" is a static exposure, not a forecast.
    """
    from quant_platform.domain.enums import EvaluationStage
    from quant_platform.models.registry import create_model
    from quant_platform.utilities.narrow import as_date
    from quant_platform.validation.stress import (
        StressResult,
        concentration_analysis,
        randomize_predictions,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    charter = load_charter(config_path)
    set_global_seeds(charter.random_seed)
    data = load_fixture_data(charter, n_securities=n_securities)
    panel = build_training_frame(data)
    train, test = split_train_test(
        panel, test_start, charter.validation.embargo_sessions, data.calendar
    )
    test_end = as_date(test["as_of"].max(), "test_end")
    target = "excess_return_rank"

    base_model = create_model(model.replace("-", "_"), random_seed=charter.random_seed)
    preds = fit_and_predict(base_model, train, test, data.feature_columns, target, charter)

    results: list[StressResult] = []

    def evaluate_variant(
        predictions: pl.DataFrame,
        scenario: str,
        category: str,
        cost: str = "base",
        portfolio: str = "equal_weight",
        notes: str = "",
    ) -> None:
        result = run_backtest_for_predictions(
            predictions, data, charter, portfolio, cost, strategy_name=scenario
        )
        report = evaluate(
            result, charter, data, EvaluationStage.WALK_FORWARD_TEST, test_start, test_end
        )
        m = report.metrics
        bench = report.benchmark_metrics.get("cagr", float("nan"))
        results.append(
            StressResult(
                scenario=scenario,
                category=category,
                net_cagr=m.get("cagr", float("nan")),
                gross_cagr=m.get("gross_cagr", float("nan")),
                excess_vs_benchmark=m.get("cagr", float("nan")) - bench,
                sharpe=m.get("sharpe_ratio", float("nan")),
                information_ratio=m.get("information_ratio", float("nan")),
                max_drawdown=m.get("max_drawdown", float("nan")),
                turnover=m.get("avg_turnover", float("nan")),
                total_costs=m.get("total_costs", float("nan")),
                notes=notes,
            )
        )

    logger.info("stress: cost scenarios")
    for scenario in ("zero", "base", "double", "triple"):
        evaluate_variant(preds, f"cost_{scenario}", "cost", cost=scenario)

    logger.info("stress: portfolio variants")
    for variant in PORTFOLIO_VARIANTS:
        evaluate_variant(preds, f"portfolio_{variant}", "portfolio", portfolio=variant)

    logger.info("stress: execution delay")
    for delay in (1, 2):
        delayed_charter = charter.model_copy(deep=True)
        delayed_charter.backtest.rebalance.execution_delay_sessions = delay
        result = run_backtest_for_predictions(
            preds,
            data,
            delayed_charter,
            "equal_weight",
            "base",
            strategy_name=f"delay_{delay}",
        )
        report = evaluate(
            result, delayed_charter, data, EvaluationStage.WALK_FORWARD_TEST, test_start, test_end
        )
        bench = report.benchmark_metrics.get("cagr", float("nan"))
        results.append(
            StressResult(
                scenario=f"execution_delay_{delay}",
                category="execution",
                net_cagr=report.metrics.get("cagr", float("nan")),
                gross_cagr=report.metrics.get("gross_cagr", float("nan")),
                excess_vs_benchmark=report.metrics.get("cagr", float("nan")) - bench,
                sharpe=report.metrics.get("sharpe_ratio", float("nan")),
                information_ratio=report.metrics.get("information_ratio", float("nan")),
                notes=f"{delay}-session delay",
            )
        )

    logger.info("stress: feature-family ablation")
    families = charter.features.enabled_families
    for drop in families:
        remaining = [f for f in families if f != drop]
        if not remaining:
            continue
        ablated = charter.model_copy(deep=True)
        ablated.features.enabled_families = remaining
        try:
            ablated_data = load_fixture_data(ablated, n_securities=n_securities)
            ablated_panel = build_training_frame(ablated_data)
            a_train, a_test = split_train_test(
                ablated_panel,
                test_start,
                ablated.validation.embargo_sessions,
                ablated_data.calendar,
            )
            a_model = create_model(model.replace("-", "_"), random_seed=charter.random_seed)
            a_preds = fit_and_predict(
                a_model, a_train, a_test, ablated_data.feature_columns, target, ablated
            )
            a_result = run_backtest_for_predictions(
                a_preds,
                ablated_data,
                ablated,
                "equal_weight",
                "base",
                strategy_name=f"no_{drop}",
            )
            a_report = evaluate(
                a_result,
                ablated,
                ablated_data,
                EvaluationStage.WALK_FORWARD_TEST,
                test_start,
                test_end,
            )
            bench = a_report.benchmark_metrics.get("cagr", float("nan"))
            results.append(
                StressResult(
                    scenario=f"ablate_{drop}",
                    category="feature_ablation",
                    net_cagr=a_report.metrics.get("cagr", float("nan")),
                    gross_cagr=a_report.metrics.get("gross_cagr", float("nan")),
                    excess_vs_benchmark=a_report.metrics.get("cagr", float("nan")) - bench,
                    sharpe=a_report.metrics.get("sharpe_ratio", float("nan")),
                    information_ratio=a_report.metrics.get("information_ratio", float("nan")),
                    notes=f"removed the {drop} family",
                )
            )
        except Exception as exc:  # a family may be required by the model
            logger.warning("ablation of %s failed: %s", drop, exc)

    logger.info("stress: randomization (placebo) tests")
    evaluate_variant(
        randomize_predictions(preds, charter.random_seed, "shuffle"),
        "randomized_predictions_shuffled",
        "placebo",
        notes="ranks shuffled within each date; performance here is NOT from the signal",
    )
    evaluate_variant(
        randomize_predictions(preds, charter.random_seed, "random"),
        "randomized_predictions_uniform",
        "placebo",
        notes="ranks replaced with noise",
    )

    frame = pl.DataFrame([r.to_row() for r in results])
    frame.write_csv(out / "stress_results.csv")

    base_result = run_backtest_for_predictions(preds, data, charter, "equal_weight", "base")
    sector_map = dict(zip(data.securities["security_id"], data.securities["sector"], strict=True))
    concentration = concentration_analysis(
        base_result.trade_log(),
        base_result.equity_curve().filter(pl.col("as_of") >= test_start),
        sector_map,
    )
    with (out / "concentration.json").open("w") as fh:
        json.dump(
            {
                "top_securities_share": concentration.top_securities_share,
                "top_security_ids": concentration.top_security_ids,
                "max_sector_share": concentration.max_sector_share,
                "dominant_sector": concentration.dominant_sector,
                "max_year_share": concentration.max_year_share,
                "dominant_year": concentration.dominant_year,
                "top_rebalances_share": concentration.top_rebalances_share,
                "warnings": concentration.warnings,
            },
            fh,
            indent=2,
            default=str,
        )
    for warning in concentration.warnings:
        logger.warning("concentration: %s", warning)

    return frame


def run_paper_rebalance(
    config_path: str | Path = "configs/research_charter.yaml",
    approve_all: bool = False,
    n_securities: int = 200,
) -> dict[str, Any]:
    """Generate paper-trading proposals and execute only approved orders."""
    from quant_platform.costs.model import CompositeCostModel
    from quant_platform.execution.base import SimulatedBroker, approve
    from quant_platform.models.registry import create_model
    from quant_platform.paper.rebalance import (
        execute_approved,
        propose_orders,
        reconcile,
        record_rebalance,
    )
    from quant_platform.paper.state import PaperTradingState
    from quant_platform.portfolio.constructors import get_constructor
    from quant_platform.utilities.narrow import as_date

    charter = load_charter(config_path)
    set_global_seeds(charter.random_seed)

    state_path = Path(charter.paper_trading.state_dir) / "state.json"
    if state_path.exists():
        state = PaperTradingState.load(state_path)
    else:
        state = PaperTradingState.initialize(charter.paper_trading.initial_capital)

    data = load_fixture_data(charter, n_securities=n_securities)
    panel = build_training_frame(data)
    latest = as_date(panel["as_of"].max(), "latest signal date")
    split_at = data.calendar.shift(latest, -60) or latest
    train, recent = split_train_test(
        panel, split_at, charter.validation.embargo_sessions, data.calendar
    )
    model = create_model("factor_composite", random_seed=charter.random_seed)
    preds = fit_and_predict(
        model, train, recent, data.feature_columns, "excess_return_rank", charter
    )

    signal_date = as_date(preds["as_of"].max(), "signal date")
    latest_preds = preds.filter(pl.col("as_of") == signal_date)
    order_date = data.calendar.shift(
        signal_date, charter.backtest.rebalance.execution_delay_sessions
    )
    if order_date is None:
        # No future session in the fixture calendar: fall back to the signal date, and say so.
        order_date = signal_date
        logger.warning(
            "no session available after %s in the fixture calendar; using the signal date as "
            "the order date for this demonstration",
            signal_date,
        )

    constructor = get_constructor("equal_weight", charter.portfolio)
    targets, _diag = constructor.build(latest_preds, state.current_weights(), signal_date)

    price_rows = (
        data.prices.filter(pl.col("observation_date") <= order_date)
        .group_by("security_id")
        .agg(pl.col("adjusted_close").last().alias("px"))
    )
    prices = dict(zip(price_rows["security_id"], price_rows["px"], strict=True))

    cost_model = CompositeCostModel(charter.costs, charter.backtest.cost_scenario)
    proposals = propose_orders(
        state,
        targets,
        prices,
        cost_model,
        signal_date,
        order_date,
        min_trade_notional=charter.portfolio.min_trade_size,
    )
    state.stage_proposals(proposals)
    state.save(state_path)

    summary: dict[str, Any] = {
        "signal_date": str(signal_date),
        "order_date": str(order_date),
        "n_proposed": len(proposals),
        "proposals": [p.describe() for p in proposals],
        "n_filled": 0,
        "realized_cost": 0.0,
    }

    if not approve_all:
        # The default path stops here: orders stay proposals until a human approves them.
        return summary

    approve(proposals, approver="cli --approve-all")
    broker = SimulatedBroker(cost_model, cash=state.cash)
    fills, rejected = execute_approved(state, proposals, broker, prices)
    reconciliation = reconcile(state, targets, proposals, fills, prices, order_date)
    record_rebalance(
        state,
        proposals,
        fills,
        rejected,
        targets,
        reconciliation,
        signal_date,
        order_date,
        model_name=model.name,
        model_version=model.version,
        feature_version=charter.features.feature_version,
        approved_by="cli --approve-all",
    )
    state.record_equity(order_date, prices)
    state.save(state_path)

    summary["n_filled"] = len(fills)
    summary["realized_cost"] = reconciliation.realized_cost
    summary["warnings"] = reconciliation.warnings
    return summary
