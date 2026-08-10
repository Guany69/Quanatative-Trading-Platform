"""Composable transaction cost model (spec section 18).

Costs are decomposed into commission, spread, slippage, and market impact rather than a
single "5 bps" fudge, because the components scale differently with trade size and behave
very differently under stress:

* commission -- roughly linear in notional or shares;
* spread     -- linear in notional, widens when liquidity dries up;
* slippage   -- scales with volatility and urgency;
* impact     -- CONCAVE in size (square-root), and the binding constraint on capacity.

A strategy killed by impact has a capacity problem (trade smaller, or the alpha is not
scalable); a strategy killed by commission has a turnover problem. Collapsing them into one
number hides which.

IMPORTANT: the impact model is a widely-used *approximation*, not a law of nature. Real
impact depends on order flow, venue, time of day, and the trader's own footprint. Treat these
numbers as a plausible penalty, not a measurement. See docs/limitations.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_platform.config.models import CostConfig
from quant_platform.domain.portfolio import TransactionCost

BPS = 1e-4


@dataclass(frozen=True)
class TradeContext:
    """Everything the cost model needs about one trade.

    All fields are knowable *before* the fill, so costs never peek at the outcome.
    """

    notional: float  # absolute dollar value traded
    price: float
    shares: float
    adv_dollar: float | None = None  # trailing average daily dollar volume
    volatility: float | None = None  # annualized
    is_buy: bool = True

    @property
    def participation(self) -> float:
        """Fraction of a typical day's volume this trade represents."""
        if not self.adv_dollar or self.adv_dollar <= 0:
            # No liquidity data: assume a meaningful footprint rather than a free trade.
            # Assuming zero participation would make impact vanish exactly where we know
            # least, which is the optimistic direction.
            return 0.05
        return abs(self.notional) / self.adv_dollar


class CommissionModel:
    """Broker commission (spec section 18.1)."""

    def __init__(self, config: CostConfig) -> None:
        self.cfg = config.commission

    def compute(self, ctx: TradeContext) -> float:
        c = self.cfg
        if c.model == "zero":
            return 0.0
        if c.model == "per_share":
            fee = abs(ctx.shares) * c.per_share
        elif c.model == "per_order":
            fee = c.per_order
        else:  # basis_points
            fee = abs(ctx.notional) * c.basis_points * BPS
        return max(fee, c.minimum)


class SpreadModel:
    """Half-spread cost (spec section 18.2).

    A market order pays roughly half the quoted spread on entry. Only half, because the
    mid-price is the fair reference; paying the full spread would double-count.
    """

    def __init__(self, config: CostConfig) -> None:
        self.cfg = config.spread

    def compute(self, ctx: TradeContext) -> float:
        c = self.cfg
        if c.model == "fixed_bps":
            half_spread_bps = c.fixed_bps / 2.0
        else:  # liquidity_proxy: illiquid names quote wider
            base = c.fixed_bps / 2.0
            if ctx.adv_dollar and ctx.adv_dollar > 0:
                # Widen as ADV falls below a $10m reference; capped so it stays plausible.
                scale = float(np.clip((1e7 / ctx.adv_dollar) ** 0.5, 0.5, 10.0))
            else:
                scale = 3.0
            half_spread_bps = base * scale
        return abs(ctx.notional) * half_spread_bps * BPS * c.stress_multiplier


class SlippageModel:
    """Adverse price drift between decision and fill (spec section 18.3)."""

    def __init__(self, config: CostConfig) -> None:
        self.cfg = config.slippage

    def compute(self, ctx: TradeContext) -> float:
        c = self.cfg
        if c.model == "fixed_bps":
            bps = c.fixed_bps
        elif c.model == "volatility_scaled":
            # A volatile name can move a lot while the order works.
            daily_vol = (ctx.volatility or 0.25) / np.sqrt(252.0)
            bps = c.fixed_bps + c.volatility_coefficient * daily_vol * 1e4
        else:  # participation_scaled
            bps = c.fixed_bps + c.participation_coefficient * ctx.participation * 1e4
        return abs(ctx.notional) * max(bps, 0.0) * BPS


class MarketImpactModel:
    """Square-root market impact (spec section 18.4).

    cost = coefficient * sigma_daily * participation^exponent * notional

    The square-root form is the standard empirical approximation: impact grows sub-linearly
    with size, so doubling a trade less than doubles its impact per share. The concavity is
    what makes capacity finite but not cliff-edged.

    This is an estimate. It is not calibrated to any real venue or broker.
    """

    def __init__(self, config: CostConfig) -> None:
        self.cfg = config.impact

    def compute(self, ctx: TradeContext) -> float:
        c = self.cfg
        if c.model == "zero":
            return 0.0
        daily_vol = (ctx.volatility or 0.25) / np.sqrt(252.0)
        part = max(ctx.participation, 0.0)
        # linear impact grows proportionally with size; square_root is the standard
        # empirical approximation, concave so impact per share falls as size grows.
        factor = part if c.model == "linear" else part**c.exponent
        return abs(ctx.notional) * c.coefficient * daily_vol * factor


class CompositeCostModel:
    """Sums the components and applies the scenario multiplier (spec section 18.5)."""

    def __init__(self, config: CostConfig, scenario: str = "base") -> None:
        self.config = config
        if scenario not in config.scenario_multipliers:
            raise ValueError(
                f"unknown cost scenario '{scenario}'; defined: "
                f"{sorted(config.scenario_multipliers)}"
            )
        self.scenario = scenario
        self.multiplier = config.scenario_multipliers[scenario]
        self.commission = CommissionModel(config)
        self.spread = SpreadModel(config)
        self.slippage = SlippageModel(config)
        self.impact = MarketImpactModel(config)

    def compute(self, ctx: TradeContext) -> TransactionCost:
        """Full cost decomposition for one trade."""
        if ctx.notional == 0:
            return TransactionCost()
        cost = TransactionCost(
            commission=self.commission.compute(ctx),
            spread_cost=self.spread.compute(ctx),
            slippage_cost=self.slippage.compute(ctx),
            impact_cost=self.impact.compute(ctx),
        )
        return cost.scaled(self.multiplier) if self.multiplier != 1.0 else cost

    def cost_bps(self, ctx: TradeContext) -> float:
        """Total cost in basis points of notional -- the comparable unit across trade sizes."""
        if ctx.notional == 0:
            return 0.0
        return self.compute(ctx).total / abs(ctx.notional) / BPS

    def effective_price(self, ctx: TradeContext) -> float:
        """The price actually achieved, after costs push against the trader.

        Buys fill higher than the reference; sells fill lower. This is what makes the
        backtest's fills unachievable-in-reverse: cost always works against you.
        """
        if ctx.shares == 0:
            return ctx.price
        cost = self.compute(ctx)
        # Commission is booked as cash, not price; the rest moves the fill price.
        price_moving = cost.spread_cost + cost.slippage_cost + cost.impact_cost
        per_share = price_moving / abs(ctx.shares)
        return ctx.price + per_share if ctx.is_buy else ctx.price - per_share
