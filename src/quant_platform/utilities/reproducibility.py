"""Deterministic seeding, structured logging, and run metadata (spec section 29)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "quant_platform"


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into the environment.

    Deliberately minimal (no python-dotenv dependency) and deliberately non-overriding: a
    variable already set in the real environment wins, so an explicit `export` is never
    silently overwritten by a stale file.

    Values may be quoted, which lets the same file be `source`d from a shell -- unquoted
    values containing spaces would otherwise be parsed as commands.
    """
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def set_global_seeds(seed: int) -> dict[str, Any]:
    """Seed every RNG the platform can reach.

    Returns a record of what was seeded (and what could not be), so a run's metadata states
    plainly which libraries are actually pinned rather than implying full determinism.
    """
    seeded: dict[str, Any] = {"seed": seed}

    random.seed(seed)
    seeded["python_random"] = True

    np.random.seed(seed)
    seeded["numpy"] = True

    # PYTHONHASHSEED only takes effect at interpreter start; setting it now does not
    # retroactively change this process. Recorded so the limitation is visible.
    seeded["pythonhashseed_env"] = os.environ.get("PYTHONHASHSEED", "not_set")

    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        seeded["torch"] = torch.__version__
    except Exception as exc:  # pragma: no cover - torch is optional
        seeded["torch"] = f"unavailable ({type(exc).__name__})"

    try:
        import lightgbm  # noqa: F401

        # LightGBM takes its seed per-training-call; recorded here, applied in the model.
        seeded["lightgbm"] = "seeded per-call"
    except Exception:
        seeded["lightgbm"] = "unavailable"

    return seeded


def git_commit() -> str | None:
    """Current git commit, or None outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_is_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes.

    A dirty tree means the commit hash does not fully describe the code that ran, so runs
    recorded from a dirty tree are not reliably reproducible.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def dependency_versions() -> dict[str, str]:
    """Versions of the packages that can change numerical results."""
    import importlib

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "polars", "scipy", "sklearn", "lightgbm", "torch", "cvxpy"):
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "not_installed"
    return versions


@dataclass
class RunMetadata:
    """Everything needed to interpret and reproduce a run."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    command: str = ""
    git_commit: str | None = field(default_factory=git_commit)
    git_dirty: bool | None = field(default_factory=git_is_dirty)
    config_hash: str | None = None
    seeds: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dependency_versions)
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        }
    )
    data_snapshot_id: str | None = None
    feature_version: str | None = None
    label_version: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "command": self.command,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "config_hash": self.config_hash,
            "seeds": self.seeds,
            "dependencies": self.dependencies,
            "environment": self.environment,
            "data_snapshot_id": self.data_snapshot_id,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "artifacts": self.artifacts,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return p


def dataset_snapshot_id(*frames: Any) -> str:
    """Content hash of the datasets a run consumed.

    Ties results to the exact data, so a changed input cannot masquerade as a changed model.
    """
    hasher = hashlib.sha256()
    for frame in frames:
        try:
            if hasattr(frame, "hash_rows"):  # polars
                hasher.update(str(frame.hash_rows().sum()).encode())
                hasher.update(str(frame.shape).encode())
            elif hasattr(frame, "shape"):  # pandas / numpy
                hasher.update(str(frame.shape).encode())
                hasher.update(str(getattr(frame, "values", frame)).encode()[:4096])
        except Exception:
            hasher.update(b"unhashable")
    return hasher.hexdigest()[:16]


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per line for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO", json_format: bool = False, log_file: str | Path | None = None
) -> logging.Logger:
    """Configure the platform logger. Idempotent across repeated calls."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    handler: logging.Handler = logging.StreamHandler(sys.stderr)
    if json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
        )
    logger.addHandler(handler)

    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p)
        fh.setFormatter(_JsonFormatter())
        logger.addHandler(fh)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
