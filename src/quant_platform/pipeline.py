"""End-to-end research pipeline (spec sections 35-36).

Wires the vertical slice together::

    fixture data -> PIT validation -> universe -> features -> labels
                 -> model -> predictions -> portfolios -> cost-aware backtest
                 -> benchmark comparison -> reports

The ordering here is the platform's contract with itself. Each stage consumes only what the
stage before it produced, and the walk-forward split is applied *before* preprocessing is
fitted, so training statistics can never see the test period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from quant_platform.backtest.engine import BacktestEngine, BacktestResult
from quant_platform.config.models import ResearchCharter
from quant_platform.data.adapters.fixture import FixtureDataProvider, FixtureSpec
from quant_platform.domain.enums import BenchmarkSource, CostScenario, EvaluationStage
from quant_platform.domain.portfolio import PerformanceReport
from quant_platform.evaluation.metrics import portfolio_metrics, spearman_ic
from quant_platform.features.compute import compute_price_features
from quant_platform.features.preprocessing import CrossSectionalPreprocessor
from quant_platform.labels.generator import LabelGenerator
from quant_platform.models.base import BaseModel
from quant_platform.models.baselines import FactorComposite, MomentumBaseline, NoSkillModel
from quant_platform.portfolio.constructors import get_constructor
from quant_platform.universe.builder import UniverseBuilder
from quant_platform.utilities.calendar import TradingCalendar
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("pipeline")


@dataclass
class PipelineData:
    """Everything the pipeline built, retained so the CLI and reports can inspect it."""

    prices: pl.DataFrame
    benchmark: pl.DataFrame
    securities: pl.DataFrame
    corporate_actions: pl.DataFrame
    features: pl.DataFrame
    labels: pl.DataFrame
    universe_panel: pl.DataFrame
    calendar: TradingCalendar
    is_synthetic: bool = True
    benchmark_source: BenchmarkSource = BenchmarkSource.FIXTURE_SYNTHETIC
    feature_columns: list[str] = field(default_factory=list)


def build_adjusted_prices(prices: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    """Apply split and dividend adjustments to produce ``adjusted_close``.

    Adjustment runs BACKWARD from the present: the current price is truth, and history is
    restated so that returns across a split are continuous. A 2-for-1 split halves every price
    before the ex-date; without it the split looks like a -50% return.
    """
    px = prices.sort(["security_id", "observation_date"])

    splits = actions.filter(pl.col("action_type").is_in(["stock_split", "reverse_split"])).select(
        ["security_id", "ex_date", "value"]
    )

    divs = actions.filter(
        pl.col("action_type").is_in(["cash_dividend", "special_dividend"])
    ).select(["security_id", "ex_date", pl.col("value").alias("dividend")])

    # Split factor: cumulative product of all splits STRICTLY AFTER each date. Prices before
    # a split must be divided by the ratio.
    if splits.is_empty():
        px = px.with_columns(pl.lit(1.0).alias("_split_factor"))
    else:
        frames = []
        for sec_id, group in px.group_by("security_id", maintain_order=True):
            sid = sec_id[0] if isinstance(sec_id, tuple) else sec_id
            sec_splits = splits.filter(pl.col("security_id") == sid)
            factor = np.ones(group.height)
            if not sec_splits.is_empty():
                dates = group["observation_date"].to_list()
                for row in sec_splits.iter_rows(named=True):
                    ex, ratio = row["ex_date"], float(row["value"])
                    # Every bar strictly before the ex-date gets divided by the ratio.
                    mask = np.array([d < ex for d in dates])
                    factor[mask] /= ratio
            frames.append(group.with_columns(pl.Series("_split_factor", factor)))
        px = pl.concat(frames)

    px = px.with_columns((pl.col("close") * pl.col("_split_factor")).alias("adjusted_close"))

    # Dividend adjustment: scale pre-ex-date prices by (1 - D/P) so the ex-date drop is not
    # counted as a loss. Applied cumulatively backward.
    if not divs.is_empty():
        frames = []
        for sec_id, group in px.group_by("security_id", maintain_order=True):
            sid = sec_id[0] if isinstance(sec_id, tuple) else sec_id
            sec_divs = divs.filter(pl.col("security_id") == sid)
            adj = group["adjusted_close"].to_numpy().astype(float).copy()
            if not sec_divs.is_empty():
                dates = group["observation_date"].to_list()
                closes = group["close"].to_numpy().astype(float)
                for row in sec_divs.iter_rows(named=True):
                    ex, amt = row["ex_date"], float(row["dividend"] or 0.0)
                    idx = [i for i, d in enumerate(dates) if d < ex]
                    if not idx or amt <= 0:
                        continue
                    ref_price = closes[idx[-1]]
                    if ref_price <= 0:
                        continue
                    ratio = max(1.0 - amt / ref_price, 1e-6)
                    adj[idx] *= ratio
            frames.append(group.with_columns(pl.Series("adjusted_close", adj)))
        px = pl.concat(frames)

    return px.drop("_split_factor").sort(["security_id", "observation_date"])


def load_fixture_data(charter: ResearchCharter, n_securities: int = 120) -> PipelineData:
    """Generate the deterministic fixture dataset and derive the panels."""
    spec = FixtureSpec(n_securities=n_securities, seed=charter.random_seed)
    ds = FixtureDataProvider(spec).generate()
    calendar = TradingCalendar(spec.start, spec.end)

    logger.info("applying corporate-action adjustments")
    prices = build_adjusted_prices(ds.prices, ds.corporate_actions)
    benchmark = ds.benchmark.with_columns(pl.col("close").alias("adjusted_close"))

    logger.info("building point-in-time universe")
    ub = UniverseBuilder(charter.universe, ds.securities, ds.membership, prices)
    signal_dates = calendar.rebalance_sessions(
        charter.backtest.rebalance.frequency, charter.backtest.rebalance.signal_day
    )
    universe_panel = ub.build_panel(signal_dates)

    logger.info("computing features")
    features = compute_price_features(prices, benchmark, charter.features.enabled_families)

    logger.info("generating labels")
    gen = LabelGenerator(
        charter.label, calendar, charter.backtest.rebalance.execution_delay_sessions
    )
    labels = gen.generate(prices, benchmark, ds.securities, signal_dates)

    feature_cols = [c for c in features.columns if c not in ("security_id", "as_of")]

    return PipelineData(
        prices=prices,
        benchmark=benchmark,
        securities=ds.securities,
        corporate_actions=ds.corporate_actions,
        features=features,
        labels=labels,
        universe_panel=universe_panel,
        calendar=calendar,
        is_synthetic=True,
        benchmark_source=BenchmarkSource.FIXTURE_SYNTHETIC,
        feature_columns=feature_cols,
    )


def build_training_frame(data: PipelineData) -> pl.DataFrame:
    """Join features, labels, and the eligible universe into a model-ready panel.

    The inner join on the universe panel is what enforces that the model only ever sees
    securities that were actually eligible on that date.
    """
    df = (
        data.universe_panel.join(data.features, on=["security_id", "as_of"], how="inner")
        .join(data.labels, on=["security_id", "as_of"], how="inner")
        .sort(["as_of", "security_id"])
    )
    sectors = data.securities.select(["security_id", "sector"])
    return df.join(sectors, on="security_id", how="left")


def split_train_test(
    panel: pl.DataFrame, test_start: date, embargo_sessions: int, calendar: TradingCalendar
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split with purging and an embargo (spec sections 16.1-16.2).

    Two separate protections:

    * **Purge** -- drop training rows whose forward label window reaches into the test
      period. Their ``as_of`` is in the past, but their *outcome* is measured in the test
      window, so training on them leaks the test period.
    * **Embargo** -- additionally drop training rows in the sessions immediately before the
      test starts, since serial correlation makes them near-duplicates of early test rows.
    """
    embargo_start = calendar.shift(test_start, -embargo_sessions) or test_start

    train = panel.filter(
        (pl.col("as_of") < embargo_start)  # embargo
        & (pl.col("window_end") < test_start)  # purge: window must close before test opens
    )
    test = panel.filter(pl.col("as_of") >= test_start)
    return train, test


def fit_and_predict(
    model: BaseModel,
    train: pl.DataFrame,
    test: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    charter: ResearchCharter,
) -> pl.DataFrame:
    """Fit preprocessing + model on train, then predict on test.

    The preprocessor is fitted on ``train`` only and then *applied* to test. This ordering is
    the whole ballgame; reversing it silently inflates every downstream metric.
    """
    pre = CrossSectionalPreprocessor(
        feature_columns,
        winsorize_quantile=charter.features.winsorize_quantile,
        method="rank",
        min_cross_section=charter.features.min_cross_section,
    )
    train_t = pre.fit_transform(train)  # fit on TRAIN
    test_t = pre.transform(test)  # apply to TEST (no refit)

    model.fit(train_t, feature_columns, target_column)
    return model.predict(test_t)


def run_backtest_for_predictions(
    predictions: pl.DataFrame,
    data: PipelineData,
    charter: ResearchCharter,
    constructor_name: str,
    cost_scenario: str = "base",
    strategy_name: str | None = None,
) -> BacktestResult:
    """Turn predictions into targets and simulate."""
    constructor = get_constructor(constructor_name, charter.portfolio)

    # Inverse-vol needs a volatility column alongside the rank.
    if constructor_name == "inverse_volatility":
        vol = data.features.select(["security_id", "as_of", "volatility_60d"])
        predictions = predictions.join(vol, on=["security_id", "as_of"], how="left")

    targets_by_date: dict[date, dict[str, float]] = {}
    previous: dict[str, float] = {}
    for as_of, group in predictions.group_by("as_of", maintain_order=True):
        d = as_of[0] if isinstance(as_of, tuple) else as_of
        weights, _diag = constructor.build(group, previous, d)
        if weights:
            targets_by_date[d] = weights
            previous = weights

    engine = BacktestEngine(
        charter.backtest, charter.costs, data.calendar, cost_scenario=cost_scenario
    )
    return engine.run(
        targets_by_date,
        data.prices,
        data.benchmark,
        data.securities,
        dividends=data.corporate_actions,
        strategy_name=strategy_name or f"{constructor_name}_{cost_scenario}",
    )


def evaluate(
    result: BacktestResult,
    charter: ResearchCharter,
    data: PipelineData,
    stage: EvaluationStage,
    start: date,
    end: date,
) -> PerformanceReport:
    """Build a PerformanceReport with provenance attached."""
    curve = result.equity_curve().filter((pl.col("as_of") >= start) & (pl.col("as_of") <= end))
    net = curve["net_return"].to_numpy()
    gross = curve["gross_return"].to_numpy()
    bench = curve["benchmark_return"].to_numpy()

    metrics = portfolio_metrics(net, bench)
    metrics["gross_cagr"] = portfolio_metrics(gross)["cagr"]
    metrics["total_costs"] = result.total_costs
    metrics["n_trades"] = float(len(result.trades))
    metrics["avg_turnover"] = float(np.nanmean(curve["turnover"].to_numpy()))

    return PerformanceReport(
        strategy_name=result.strategy_name,
        stage=stage,
        start_date=start,
        end_date=end,
        cost_scenario=CostScenario(result.cost_scenario),
        data_source="fixture_synthetic" if data.is_synthetic else "open_data",
        benchmark_source=data.benchmark_source.value,
        is_synthetic_data=data.is_synthetic,
        # The fixture has genuinely complete history by construction; real open data does not.
        historical_constituents_complete=data.is_synthetic,
        delisting_returns_complete=data.is_synthetic,
        point_in_time_fundamentals_complete=data.is_synthetic,
        execution_delay_sessions=result.execution_delay_sessions,
        metrics=metrics,
        benchmark_metrics=portfolio_metrics(bench),
        n_observations=curve.height,
    )


def forecast_ic(predictions: pl.DataFrame, panel: pl.DataFrame, target: str) -> dict[str, float]:
    """Per-date rank IC of predictions vs realized outcomes."""
    joined = predictions.join(
        panel.select(["security_id", "as_of", target]), on=["security_id", "as_of"], how="inner"
    )
    ics = []
    for _as_of, g in joined.group_by("as_of"):
        if g.height < 5:
            continue
        ic = spearman_ic(g["prediction"].to_numpy(), g[target].to_numpy())
        if np.isfinite(ic):
            ics.append(ic)
    if not ics:
        return {"mean_ic": float("nan"), "ic_ir": float("nan"), "positive_ic_pct": float("nan")}
    arr = np.array(ics)
    return {
        "mean_ic": float(arr.mean()),
        "ic_std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "ic_ir": float(arr.mean() / arr.std(ddof=1))
        if arr.size > 1 and arr.std(ddof=1) > 0
        else float("nan"),
        "positive_ic_pct": float((arr > 0).mean()),
        "n_periods": float(arr.size),
    }


def default_models(charter: ResearchCharter) -> dict[str, BaseModel]:
    """The models the demo compares. Baselines are not optional -- they are the control."""
    seed = charter.random_seed
    return {
        "no_skill": NoSkillModel(target_type=charter.label.target_type, random_seed=seed),
        "momentum_baseline": MomentumBaseline(
            momentum_column="momentum_252d", target_type=charter.label.target_type, random_seed=seed
        ),
        "factor_composite": FactorComposite(
            target_type=charter.label.target_type, random_seed=seed
        ),
    }
