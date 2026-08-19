"""Small temporal convolutional challenger for flattened feature histories.

The caller supplies features in channel-major temporal order. They arrive through the common
two-dimensional model contract and are reshaped to ``(samples, channels, lookback)`` here.
The network is intentionally narrow because noisy equity panels punish excess capacity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_platform.models.base import BaseModel, ModelError
from quant_platform.utilities.reproducibility import get_logger

logger = get_logger("models.temporal_conv")


class TemporalConvModel(BaseModel):
    """Two-layer temporal Conv1d regressor, experimental by design."""

    name = "temporal_conv"
    is_production_candidate = False

    def __init__(
        self,
        channels: int = 1,
        lookback: int = 20,
        conv_channels: tuple[int, int] = (16, 8),
        kernel_size: int = 3,
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 100,
        patience: int = 10,
        device: str = "cpu",
        use_amp: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if channels < 1 or lookback < 1:
            raise ValueError("channels and lookback must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be either 'cpu' or 'cuda'")
        if use_amp and device != "cuda":
            raise ValueError("use_amp=True requires device='cuda'")
        self.channels = channels
        self.lookback = lookback
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.use_amp = use_amp
        self._model: Any = None
        self._input_mean: np.ndarray | None = None
        self._input_std: np.ndarray | None = None
        self._validation: tuple[np.ndarray, np.ndarray] | None = None
        self._best_epoch: int | None = None
        self._history: list[dict[str, float]] = []

    def set_validation(self, X: np.ndarray, y: np.ndarray) -> None:
        """Set a validation fold that is separate from test and holdout data."""
        self._validation = (X, y)

    def _validate_layout(self, n_features: int) -> None:
        expected = self.channels * self.lookback
        if n_features != expected:
            raise ModelError(
                f"{self.name}: expected {expected} flattened features in "
                f"(samples, channels={self.channels}, lookback={self.lookback}) layout, "
                f"got {n_features}"
            )

    def _build(self) -> Any:
        import torch.nn as nn

        padding = self.kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(self.channels, self.conv_channels[0], self.kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(
                self.conv_channels[0],
                self.conv_channels[1],
                self.kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.conv_channels[1], 1),
        )

    def _fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        del feature_names
        self._validate_layout(X.shape[1])
        try:
            import torch
        except ImportError as exc:
            raise ModelError(
                "PyTorch is required for the temporal convolution model. "
                "Install with: uv sync --extra ml"
            ) from exc

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise ModelError(
                    f"{self.name}: CUDA was requested but no CUDA device is available"
                )
            torch.cuda.manual_seed_all(self.random_seed)
        else:
            torch.set_num_threads(1)

        self._input_mean = X.mean(axis=0)
        self._input_std = X.std(axis=0)
        self._input_std[self._input_std < 1e-8] = 1.0
        Xs = (X - self._input_mean) / self._input_std

        device = torch.device(self.device)
        model = self._build().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(2, self.patience // 3)
        )
        loss_fn = torch.nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        X_t = torch.tensor(Xs, dtype=torch.float32, device=device).reshape(
            -1, self.channels, self.lookback
        )
        y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
        if self._validation is not None:
            vx, vy = self._validation
            self._validate_layout(vx.shape[1])
            vxs = (vx - self._input_mean) / self._input_std
            X_val = torch.tensor(vxs, dtype=torch.float32, device=device).reshape(
                -1, self.channels, self.lookback
            )
            y_val = torch.tensor(vy, dtype=torch.float32, device=device).unsqueeze(1)
        else:
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
                with torch.autocast("cuda", enabled=self.use_amp):
                    loss = loss_fn(model(X_t[idx]), y_t[idx])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
            model.load_state_dict(best_state)
        model.eval()
        self._model = model

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelError(f"{self.name}: not fitted")
        self._validate_layout(X.shape[1])
        import torch

        Xs = (X - self._input_mean) / self._input_std  # type: ignore[operator]
        tensor = torch.tensor(Xs, dtype=torch.float32, device=self.device).reshape(
            -1, self.channels, self.lookback
        )
        with torch.no_grad():
            out = self._model(tensor)
        return out.detach().cpu().numpy().ravel()

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "lookback": self.lookback,
            "conv_channels": list(self.conv_channels),
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "device": self.device,
            "use_amp": self.use_amp,
            "best_epoch": self._best_epoch,
        }

    def training_history(self) -> pl.DataFrame:
        return pl.DataFrame(self._history) if self._history else pl.DataFrame()
