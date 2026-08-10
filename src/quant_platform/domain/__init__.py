"""Validated domain contracts for the platform.

Every time-dependent record carries point-in-time availability semantics; see
``quant_platform.domain.base`` for the observation/availability/ingestion distinction that
the rest of the platform depends on.
"""

from quant_platform.domain.base import FrozenModel, PointInTimeRecord
from quant_platform.domain.enums import (
    BenchmarkSource,
    CandidateStatus,
    CorporateActionType,
    CostScenario,
    DataQualityStatus,
    DelistingReason,
    EvaluationStage,
    Exchange,
    IssueSeverity,
    OrderSide,
    OrderStatus,
    OrderType,
    Sector,
    SecurityType,
    SolverStatus,
    TargetType,
)
from quant_platform.domain.observations import (
    FeatureObservation,
    FundamentalObservation,
    LabelObservation,
    MacroObservation,
    ModelPrediction,
)
from quant_platform.domain.portfolio import (
    BacktestSnapshot,
    DataQualityIssue,
    ExperimentRecord,
    Fill,
    OptimizationDiagnostics,
    Order,
    PerformanceReport,
    PortfolioTarget,
    Position,
    TransactionCost,
)
from quant_platform.domain.securities import (
    FAR_FUTURE,
    CorporateAction,
    DateInterval,
    PriceBar,
    Security,
    SecurityIdentifier,
    UniverseMembership,
)

__all__ = [
    "FAR_FUTURE",
    "BacktestSnapshot",
    "BenchmarkSource",
    "CandidateStatus",
    "CorporateAction",
    "CorporateActionType",
    "CostScenario",
    "DataQualityIssue",
    "DataQualityStatus",
    "DateInterval",
    "DelistingReason",
    "EvaluationStage",
    "Exchange",
    "ExperimentRecord",
    "FeatureObservation",
    "Fill",
    "FrozenModel",
    "FundamentalObservation",
    "IssueSeverity",
    "LabelObservation",
    "MacroObservation",
    "ModelPrediction",
    "OptimizationDiagnostics",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PerformanceReport",
    "PointInTimeRecord",
    "PortfolioTarget",
    "Position",
    "PriceBar",
    "Sector",
    "Security",
    "SecurityIdentifier",
    "SecurityType",
    "SolverStatus",
    "TargetType",
    "TransactionCost",
    "UniverseMembership",
]
