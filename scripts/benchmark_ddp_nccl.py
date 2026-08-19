#!/usr/bin/env python3
"""Benchmark the repository MLP with NCCL DistributedDataParallel.

Launch with ``torchrun`` for controlled one- versus multi-GPU comparisons. Running the script
directly uses one process per visible CUDA device as a convenient fallback.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    samples: int
    features: int
    hidden_sizes: tuple[int, ...]
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int


def _arguments() -> tuple[BenchmarkConfig, int | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=262_144)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="spawn fallback process count; defaults to all visible CUDA devices",
    )
    args = parser.parse_args()
    config = BenchmarkConfig(
        samples=args.samples,
        features=args.features,
        hidden_sizes=tuple(args.hidden_sizes),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    if min(config.samples, config.features, config.batch_size, config.epochs) < 1:
        parser.error("samples, features, batch-size, and epochs must be positive")
    return config, args.nproc_per_node


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _nccl_version(torch: Any) -> str:
    version = torch.cuda.nccl.version()
    if isinstance(version, tuple):
        return ".".join(str(part) for part in version)
    return str(version)


def _run(
    rank: int, world_size: int, config: BenchmarkConfig, local_rank: int | None = None
) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    from quant_platform.models.neural_network import NeuralNetworkModel

    device_index = rank if local_rank is None else local_rank
    torch.cuda.set_device(device_index)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        generator = torch.Generator().manual_seed(config.seed)
        features = torch.randn(config.samples, config.features, generator=generator)
        coefficients = torch.randn(config.features, 1, generator=generator)
        targets = features @ coefficients
        dataset = TensorDataset(features, targets)
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=0,
        )

        template = NeuralNetworkModel(hidden_sizes=config.hidden_sizes, dropout=0.0)
        model = template._build(config.features).to(device_index)
        model = DistributedDataParallel(model, device_ids=[device_index])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        loss_fn = torch.nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda")

        epoch_times: list[float] = []
        for epoch in range(config.epochs):
            sampler.set_epoch(epoch)
            model.train()
            torch.cuda.synchronize(device_index)
            start = perf_counter()
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device_index, non_blocking=True)
                batch_y = batch_y.to(device_index, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda"):
                    loss = loss_fn(model(batch_X), batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            torch.cuda.synchronize(device_index)
            end = perf_counter()
            elapsed = end - start
            timing = torch.tensor(elapsed, dtype=torch.float64, device=device_index)
            dist.reduce(timing, dst=0, op=dist.ReduceOp.MAX)
            if rank == 0:
                epoch_time = float(timing.item())
                epoch_times.append(epoch_time)
                print(f"epoch={epoch + 1} seconds={epoch_time:.6f}", flush=True)

        if rank == 0:
            total_time = sum(epoch_times)
            samples_processed = len(sampler) * world_size * config.epochs
            gpu_names = [torch.cuda.get_device_name(index) for index in range(world_size)]
            print("\n=== DDP NCCL benchmark summary ===")
            print(f"backend={dist.get_backend()}")
            print(f"world_size={world_size}")
            print(f"gpu_names={gpu_names}")
            print(f"torch_version={torch.__version__}")
            print(f"cuda_version={torch.version.cuda}")
            print(f"nccl_version={_nccl_version(torch)}")
            print(f"total_time_seconds={total_time:.6f}")
            print(f"throughput_samples_per_second={samples_processed / total_time:.3f}")
    finally:
        dist.destroy_process_group()


def main() -> None:
    try:
        import torch
        import torch.multiprocessing as mp
    except ImportError:
        raise SystemExit(
            "PyTorch is required. Install the ML dependencies with: pip install -e '.[ml]'"
        ) from None

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise SystemExit("CUDA is required for the NCCL DDP benchmark; no CUDA device is available")

    config, requested_processes = _arguments()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        _run(rank, int(os.environ["WORLD_SIZE"]), config, local_rank)
        return

    world_size = requested_processes or torch.cuda.device_count()
    if world_size > torch.cuda.device_count():
        raise SystemExit(
            f"requested {world_size} processes but only {torch.cuda.device_count()} CUDA devices "
            "are visible"
        )
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))
    mp.spawn(_run, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
