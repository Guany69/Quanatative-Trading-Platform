"""Shallow MLP challenger (spec section 14.6).

Deliberately small: two or three narrow hidden layers with dropout and weight decay. Deep
networks need either abundant signal or abundant data, and cross-sectional equity prediction
offers neither -- a few hundred thousand noisy rows with R^2 near zero. A large network here
memorizes the training panel and produces confident nonsense out of sample.

Runs on CPU and seeds every RNG it touches. Full bit-level determinism is not guaranteed
across platforms (BLAS reductions reorder floating-point sums), which is documented in
docs/reproducibility.md rather than papered over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_platform.models.base import BaseModel, ModelError
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("models.nn")


class NeuralNetworkModel(BaseModel):
    """Small MLP regressor for cross-sectional ranking."""

    name = "neural_network"

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
        weight_decay: float = 1e-4,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 100,
        patience: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if len(hidden_sizes) > 3:
            raise ValueError(
                f"hidden_sizes has {len(hidden_sizes)} layers; the spec caps this at 3. Deeper "
                f"networks overfit financial panels rather than generalizing."
            )
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self._model: Any = None
        self._input_mean: np.ndarray | None = None
        self._input_std: np.ndarray | None = None
        self._validation: tuple[np.ndarray, np.ndarray] | None = None
        self._best_epoch: int | None = None
        self._history: list[dict[str, float]] = []

    def set_validation(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validation set for early stopping. Must not be test or holdout data."""
        self._validation = (X, y)

    def _build(self, n_features: int) -> Any:
        import torch.nn as nn

        layers: list[Any] = []
        prev = n_features
        for size in self.hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = size
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ModelError(
                "PyTorch is required for the neural network model. Install with: uv sync --extra ml"
            ) from exc

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.set_num_threads(1)  # deterministic reductions

        # Standardize inputs using TRAINING statistics only. Neural nets are far more
        # scale-sensitive than trees, and reusing these stats at predict time is what keeps
        # the test data from influencing its own normalization.
        self._input_mean = X.mean(axis=0)
        self._input_std = X.std(axis=0)
        self._input_std[self._input_std < 1e-8] = 1.0
        Xs = (X - self._input_mean) / self._input_std

        device = torch.device("cpu")
        model = self._build(X.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(2, self.patience // 3)
        )
        loss_fn = torch.nn.MSELoss()

        X_t = torch.tensor(Xs, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

        if self._validation is not None:
            vx, vy = self._validation
            vxs = (vx - self._input_mean) / self._input_std
            X_val = torch.tensor(vxs, dtype=torch.float32, device=device)
            y_val = torch.tensor(vy, dtype=torch.float32, device=device).unsqueeze(1)
        else:
            # Without an explicit validation fold, hold out the LAST slice of training rows.
            # Last, not random: a random split would let near-duplicate neighbouring dates
            # appear on both sides and make early stopping optimistic.
            split = int(len(X_t) * 0.85)
            X_val, y_val = X_t[split:], y_t[split:]
            X_t, y_t = X_t[:split], y_t[:split]

        n = len(X_t)
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        generator = torch.Generator().manual_seed(self.random_seed)

        for epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(n, generator=generator)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(X_t[idx]), y_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(idx)

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(X_val), y_val).item())
            scheduler.step(val_loss)
            self._history.append(
                {"epoch": epoch, "train_loss": epoch_loss / max(n, 1), "val_loss": val_loss}
            )

            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                self._best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.debug("early stopping at epoch %d (best %d)", epoch, self._best_epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)  # restore the checkpoint, not the last epoch
        model.eval()
        self._model = model

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelError(f"{self.name}: not fitted")
        import torch

        Xs = (X - self._input_mean) / self._input_std  # type: ignore[operator]
        with torch.no_grad():
            out = self._model(torch.tensor(Xs, dtype=torch.float32))
        return out.numpy().ravel()

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "best_epoch": self._best_epoch,
        }

    def feature_importance(self) -> pl.DataFrame | None:
        """Permutation-style proxy from first-layer weights.

        Honest caveat: this only reflects the input layer, so it is a rough attribution, not
        a true importance measure. SHAP or permutation importance would be more faithful.
        """
        if self._model is None or self.metadata is None:
            return None
        first = None
        for layer in self._model:
            if hasattr(layer, "weight") and layer.weight.dim() == 2:
                first = layer
                break
        if first is None:
            return None
        weights = np.abs(first.weight.detach().numpy()).mean(axis=0)
        total = weights.sum()
        return pl.DataFrame(
            {
                "feature": self.metadata.feature_names,
                "importance": weights / total if total > 0 else weights,
            }
        ).sort("importance", descending=True)

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist weights and the input normalization together.

        Saving weights alone would be useless: predictions depend on the training-set mean
        and std, so they are part of the model.
        """
        if self._model is None:
            raise ModelError(f"{self.name}: nothing to checkpoint")
        import torch

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "input_mean": self._input_mean,
                "input_std": self._input_std,
                "hidden_sizes": self.hidden_sizes,
                "feature_names": self.metadata.feature_names if self.metadata else [],
                "best_epoch": self._best_epoch,
            },
            p,
        )
        return p

    def training_history(self) -> pl.DataFrame:
        return pl.DataFrame(self._history) if self._history else pl.DataFrame()
