"""Enumerations shared across the domain model."""

from __future__ import annotations

from enum import StrEnum


class SecurityType(StrEnum):
    """Instrument classification. Only COMMON_STOCK is eligible for the default universe."""

    COMMON_STOCK = "common_stock"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"
    PREFERRED = "preferred"
    WARRANT = "warrant"
    RIGHT = "right"
    ADR = "adr"
    REIT = "reit"
    INDEX = "index"
    UNKNOWN = "unknown"


class Exchange(StrEnum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    OTC = "OTC"
    UNKNOWN = "UNKNOWN"


class Sector(StrEnum):
    """GICS-like sector labels. Kept as a closed set so sector constraints are well defined."""

    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTH_CARE = "health_care"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    UNKNOWN = "unknown"


class CorporateActionType(StrEnum):
    CASH_DIVIDEND = "cash_dividend"
    SPECIAL_DIVIDEND = "special_dividend"
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"
    STOCK_DIVIDEND = "stock_dividend"
    SPINOFF = "spinoff"
    MERGER = "merger"
    SYMBOL_CHANGE = "symbol_change"
    DELISTING = "delisting"


class DelistingReason(StrEnum):
    MERGER = "merger"
    ACQUISITION = "acquisition"
    BANKRUPTCY = "bankruptcy"
    LIQUIDATION = "liquidation"
    EXCHANGE_RULE = "exchange_rule"
    GOING_PRIVATE = "going_private"
    UNKNOWN = "unknown"


class DataQualityStatus(StrEnum):
    OK = "ok"
    IMPUTED = "imputed"
    STALE = "stale"
    SUSPECT = "suspect"
    MISSING = "missing"


class IssueSeverity(StrEnum):
    """CRITICAL issues fail the pipeline fast; WARNING/INFO are recorded and surfaced."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class TargetType(StrEnum):
    """The three supported prediction targets (spec section 3)."""

    FORWARD_RETURN = "forward_return"
    EXCESS_RETURN = "excess_return"
    EXCESS_RETURN_RANK = "excess_return_rank"
    OUTPERFORM_PROBABILITY = "outperform_probability"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SolverStatus(StrEnum):
    OPTIMAL = "optimal"
    OPTIMAL_INACCURATE = "optimal_inaccurate"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    SOLVER_ERROR = "solver_error"
    NOT_ATTEMPTED = "not_attempted"


class EvaluationStage(StrEnum):
    """Which slice of data produced a result. Reports must always state this."""

    IN_SAMPLE = "in_sample"
    VALIDATION = "validation"
    WALK_FORWARD_TEST = "walk_forward_test"
    LOCKED_HOLDOUT = "locked_holdout"
    PAPER = "paper"


class CandidateStatus(StrEnum):
    """Research gate outcomes (spec section 25). Nothing is ever auto-labeled production-ready."""

    RESEARCH_CANDIDATE = "research_candidate"
    REJECTED = "rejected"
    HOLDOUT_NOT_EVALUATED = "holdout_not_evaluated"
    HOLDOUT_FAILED = "holdout_failed"
    PAPER_TRADING_CANDIDATE = "paper_trading_candidate"


class BenchmarkSource(StrEnum):
    """Reports must identify which benchmark actually backed the results."""

    SP500_TOTAL_RETURN = "sp500_total_return"
    SPY_ADJUSTED = "spy_adjusted"
    FIXTURE_SYNTHETIC = "fixture_synthetic"


class CostScenario(StrEnum):
    ZERO = "zero"
    BASE = "base"
    DOUBLE = "double"
    TRIPLE = "triple"
