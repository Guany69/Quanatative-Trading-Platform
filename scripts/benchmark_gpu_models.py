#!/usr/bin/env python3
"""Measure single-GPU training for the repository's neural models."""

from __future__ import annotations

import argparse
from datetime import date
from time import perf_counter

import numpy as np
import polars as pl

from quant_platform.models.neural_network import NeuralNetworkModel
from quant_platform.models.temporal_conv import TemporalConvModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("neural_network", "temporal_conv"), required=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--samples", type=int, default=262_144)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if min(args.samples, args.features, args.epochs, args.batch_size) < 1:
        parser.error("samples, features, epochs, and batch-size must be positive")
    if args.model == "temporal_conv" and args.features % args.channels != 0:
        parser.error("temporal_conv requires features to be divisible by channels")

    try:
        import torch
    except ImportError:
        raise SystemExit(
            "PyTorch is required. Install the ML dependencies with: pip install -e '.[ml]'"
        ) from None
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark; no CUDA device is available")

    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.samples, args.features)).astype(np.float32)
    coefficients = rng.normal(size=args.features).astype(np.float32)
    feature_names = [f"x{i}" for i in range(args.features)]
    frame = pl.DataFrame(
        {
            "security_id": [f"S{i}" for i in range(args.samples)],
            "as_of": [date(2024, 1, 2)] * args.samples,
            **{name: X[:, index] for index, name in enumerate(feature_names)},
            "target": X @ coefficients,
        }
    )

    common = {
        "device": "cuda",
        "use_amp": args.amp,
        "max_epochs": args.epochs,
        "patience": args.epochs,
        "batch_size": args.batch_size,
        "random_seed": args.seed,
    }
    if args.model == "neural_network":
        model = NeuralNetworkModel(**common)
    else:
        model = TemporalConvModel(
            channels=args.channels,
            lookback=args.features // args.channels,
            **common,
        )

    torch.cuda.synchronize()
    start = perf_counter()
    model.fit(frame, feature_names, "target")
    torch.cuda.synchronize()
    elapsed = perf_counter() - start

    print("=== single-GPU model benchmark summary ===")
    print(f"model={args.model}")
    print(f"amp={args.amp}")
    print(f"gpu_name={torch.cuda.get_device_name(0)}")
    print(f"torch_version={torch.__version__}")
    print(f"cuda_version={torch.version.cuda}")
    print(f"total_time_seconds={elapsed:.6f}")
    print(f"throughput_samples_per_second={args.samples * args.epochs / elapsed:.3f}")


if __name__ == "__main__":
    main()
