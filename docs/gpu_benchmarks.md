# GPU training benchmarks

This page is a runbook and an empty results record. No GPU measurement was available when it
was added, so every result remains the literal placeholder shown below. Replace a placeholder
only with output produced by the corresponding checked-in command and preserve the run log.

## What the benchmarks measure

The single-GPU harness measures end-to-end `fit()` time and training throughput for the
repository's shallow MLP and temporal Conv1d model. It includes dataframe conversion,
training-statistics-only standardization, device transfer, and optimization. Comparing the
default precision run with `--amp` isolates the effect of CUDA automatic mixed precision for
the same model and generated dataset.

The DDP harness measures optimization time and aggregate training throughput for the same MLP
architecture. It uses NCCL, one process per GPU, `DistributedDataParallel`, a
`DistributedSampler`, and AMP. Its total is the sum of synchronized per-epoch wall times; data
generation and process-group setup are outside that total.

These are training-system measurements, not evidence of predictive quality or trading
performance.

## Colab T4: single-GPU CUDA, AMP, and Conv1d

Select a T4 GPU runtime, then run:

```bash
git clone <YOUR-REPOSITORY-URL> quant-platform
cd quant-platform
pip install -e ".[ml]"
python scripts/benchmark_gpu_models.py --model neural_network
python scripts/benchmark_gpu_models.py --model neural_network --amp
python scripts/benchmark_gpu_models.py --model temporal_conv
python scripts/benchmark_gpu_models.py --model temporal_conv --amp
```

Use the same checkout and unchanged command defaults for every row. Save the executed notebook
before copying its summary output into the table.

## Kaggle T4 x2: NCCL DDP

Create a Kaggle notebook with two T4 accelerators, clone the same commit, and install it:

```bash
git clone <YOUR-REPOSITORY-URL> quant-platform
cd quant-platform
pip install -e ".[ml]"
```

Run the one-process baseline and then the two-process comparison:

```bash
torchrun --standalone --nproc_per_node=1 scripts/benchmark_ddp_nccl.py
torchrun --standalone --nproc_per_node=2 scripts/benchmark_ddp_nccl.py
```

Running `python scripts/benchmark_ddp_nccl.py` instead uses the spawn fallback and launches one
process for each visible CUDA device.

## Results

| Benchmark | Total time (seconds) | Throughput (samples/second) |
|---|---:|---:|
| MLP, CUDA full precision | `<MEASURED — pending GPU run>` | `<MEASURED — pending GPU run>` |
| MLP, CUDA AMP | `<MEASURED — pending GPU run>` | `<MEASURED — pending GPU run>` |
| Temporal Conv1d, CUDA full precision | `<MEASURED — pending GPU run>` | `<MEASURED — pending GPU run>` |
| Temporal Conv1d, CUDA AMP | `<MEASURED — pending GPU run>` | `<MEASURED — pending GPU run>` |
| MLP, NCCL DDP, one process | `<MEASURED — pending GPU run>` | `<MEASURED — pending GPU run>` |
| MLP, NCCL DDP, two processes | `<MEASURED — pending GPU run>` | `<MEASURED — pending GPU run>` |

## Verification artifacts

Commit the executed Colab notebook and the complete Kaggle run logs under `docs/benchmarks/`
after the runs. Record the commit hash used for each run in the artifact, and copy values into
this table verbatim from the corresponding summary block.
