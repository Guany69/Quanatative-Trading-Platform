"""Walk-forward validation with purging and embargo (spec section 16).

Standard k-fold cross-validation is invalid for financial panels for two reasons, and this
module addresses both.

**Reason 1: time order.** Training on 2023 to predict 2015 is not a research result. Folds
here are strictly sequential with an expanding training window.

**Reason 2: overlapping labels.** This is the subtle one. A 20-session forward return
computed on 1 March is not resolved until roughly 29 March. A training sample dated 1 March
therefore *contains information about* the rest of March. If the test period begins 10 March,
that training sample encodes part of the test period's outcome even though its own timestamp
is safely in the past. Purging drops those samples; the embargo additionally removes a buffer
immediately before the test window, because serial correlation makes adjacent samples
near-duplicates.

The locked holdout is enforced structurally: ``WalkForwardSplitter`` never emits it as a
fold, and reaching it requires an explicit config flag plus an emitted warning, because a
holdout can only honestly be looked at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from quant_platform.config.models import ValidationConfig
from quant_platform.utilities.calendar import TradingCalendar
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("validation.walk_forward")


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold."""

    index: int
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date

    def describe(self) -> str:
        return (
            f"fold {self.index}: train {self.train_start}..{self.train_end} | "
            f"val {self.validation_start}..{self.validation_end} | "
            f"test {self.test_start}..{self.test_end}"
        )


class HoldoutViolationError(RuntimeError):
    """Raised when the locked holdout would be touched without explicit permission."""


class WalkForwardSplitter:
    """Generates expanding-window folds with purging and embargo."""

    def __init__(self, config: ValidationConfig, calendar: TradingCalendar) -> None:
        self.config = config
        self.calendar = calendar

    def holdout_start(self) -> date | None:
        return self.config.holdout.start if self.config.holdout else None

    def usable_end(self, panel_end: date) -> date:
        """The last date research may use. Everything from the holdout on is off-limits."""
        start = self.holdout_start()
        if start is None:
            return panel_end
        previous = self.calendar.previous_session(start)
        return min(panel_end, previous or start)

    def generate_folds(self, panel_start: date, panel_end: date) -> list[Fold]:
        """Build the fold schedule.

        Each fold extends the training window and steps validation/test forward, so later
        folds train on strictly more history -- mirroring how a strategy would actually have
        been developed over time.
        """
        cfg = self.config
        research_end = self.usable_end(panel_end)

        min_train_days = int(cfg.min_train_years * 365.25)
        val_days = cfg.validation_months * 30
        test_days = cfg.test_months * 30
        step_days = cfg.step_months * 30

        folds: list[Fold] = []
        index = 0
        train_end = _add_days(panel_start, min_train_days)

        while True:
            val_start = _add_days(train_end, 1)
            val_end = _add_days(val_start, val_days)
            test_start = _add_days(val_end, 1)
            test_end = _add_days(test_start, test_days)

            if test_end > research_end:
                # A partial final fold still counts if it has a usable amount of test data.
                if test_start < research_end and (research_end - test_start).days > test_days // 2:
                    test_end = research_end
                else:
                    break

            folds.append(
                Fold(
                    index=index,
                    train_start=panel_start,
                    train_end=train_end,
                    validation_start=val_start,
                    validation_end=val_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            index += 1
            train_end = _add_days(train_end, step_days)
            if train_end >= research_end:
                break

        if not folds:
            raise ValueError(
                f"no walk-forward folds fit between {panel_start} and {research_end}. "
                f"Need at least min_train_years ({cfg.min_train_years}) + validation "
                f"({cfg.validation_months}m) + test ({cfg.test_months}m) of data."
            )
        logger.info("generated %d walk-forward folds", len(folds))
        return folds

    def split(
        self, panel: pl.DataFrame, fold: Fold, as_of_col: str = "as_of"
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Materialize one fold's train/validation/test frames, purged and embargoed."""
        train = self._training_rows(panel, fold, as_of_col)
        validation = panel.filter(
            (pl.col(as_of_col) >= fold.validation_start)
            & (pl.col(as_of_col) <= fold.validation_end)
        )
        test = panel.filter(
            (pl.col(as_of_col) >= fold.test_start) & (pl.col(as_of_col) <= fold.test_end)
        )
        return train, validation, test

    def _training_rows(self, panel: pl.DataFrame, fold: Fold, as_of_col: str) -> pl.DataFrame:
        """Training rows after applying the embargo and purge."""
        cfg = self.config
        embargo_cutoff = fold.validation_start
        if cfg.embargo_sessions > 0:
            shifted = self.calendar.shift(fold.validation_start, -cfg.embargo_sessions)
            if shifted is not None:
                embargo_cutoff = shifted

        train = panel.filter(
            (pl.col(as_of_col) >= fold.train_start) & (pl.col(as_of_col) < embargo_cutoff)
        )

        if cfg.purge_enabled and "window_end" in panel.columns:
            # Drop any sample whose forward label window reaches into validation or test.
            before = train.height
            train = train.filter(pl.col("window_end") < fold.validation_start)
            purged = before - train.height
            if purged:
                logger.debug(
                    "fold %d: purged %d training rows with label windows overlapping "
                    "validation/test",
                    fold.index,
                    purged,
                )
        return train

    def holdout(
        self, panel: pl.DataFrame, as_of_col: str = "as_of", acknowledge: bool = False
    ) -> pl.DataFrame:
        """Return the locked holdout -- only with explicit acknowledgement.

        The gate is deliberately awkward. A holdout's statistical value comes from being
        looked at exactly once; every additional peek turns it into another validation set
        and quietly inflates whatever is reported from it.
        """
        if self.config.holdout is None:
            raise ValueError("no locked holdout is configured")
        if not (self.config.allow_holdout_evaluation and acknowledge):
            raise HoldoutViolationError(
                "Refusing to touch the locked holdout. It may not be used for feature "
                "selection, hyperparameter tuning, model selection, ensemble weighting, or "
                "cost calibration. To evaluate a FINAL, already-chosen candidate, set "
                "validation.allow_holdout_evaluation=true AND pass acknowledge=True. "
                "Evaluating it more than once invalidates the result."
            )
        logger.warning(
            "EVALUATING THE LOCKED HOLDOUT (%s..%s). This can only be done honestly once; "
            "any further model changes informed by this result make it in-sample.",
            self.config.holdout.start,
            self.config.holdout.end,
        )
        return panel.filter(
            (pl.col(as_of_col) >= self.config.holdout.start)
            & (pl.col(as_of_col) <= self.config.holdout.end)
        )

    def assert_holdout_untouched(self, frame: pl.DataFrame, as_of_col: str = "as_of") -> None:
        """Guard: fail if a frame contains holdout rows."""
        if self.config.holdout is None or frame.is_empty():
            return
        leaked = frame.filter(pl.col(as_of_col) >= self.config.holdout.start)
        if not leaked.is_empty():
            raise HoldoutViolationError(
                f"{leaked.height} rows fall inside the locked holdout "
                f"({self.config.holdout.start}..{self.config.holdout.end}). Training or "
                f"selection on these rows destroys the holdout's validity."
            )


def _add_days(d: date, days: int) -> date:
    from datetime import timedelta

    return d + timedelta(days=days)


def purge_overlapping_labels(
    train: pl.DataFrame, test_start: date, window_end_col: str = "window_end"
) -> pl.DataFrame:
    """Drop training rows whose label windows reach into the test period.

    Standalone form of the purge, for callers not using the full splitter.
    """
    if window_end_col not in train.columns:
        logger.warning(
            "cannot purge: no '%s' column. Label windows may overlap the test period.",
            window_end_col,
        )
        return train
    return train.filter(pl.col(window_end_col) < test_start)


def apply_embargo(
    train: pl.DataFrame,
    test_start: date,
    embargo_sessions: int,
    calendar: TradingCalendar,
    as_of_col: str = "as_of",
) -> pl.DataFrame:
    """Remove training rows in the embargo buffer immediately before the test window."""
    if embargo_sessions <= 0:
        return train
    cutoff = calendar.shift(test_start, -embargo_sessions) or test_start
    return train.filter(pl.col(as_of_col) < cutoff)
