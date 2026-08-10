"""Run the governed research workload cold and warm and persist measured timings."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from datetime import UTC, datetime
from pathlib import Path

from quant_platform.research.runner import ExperimentRunOrchestrator, ResearchRunRequest


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charter", default="configs/research_charter.yaml")
    parser.add_argument("--securities", type=int, default=120)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model subset. The default exercises the complete nine-model roster.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Fresh run root. Defaults to artifacts/benchmarks/<UTC timestamp>.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_root or Path("artifacts/benchmarks") / stamp
    if root.exists():
        raise FileExistsError(
            f"benchmark output root already exists: {root}; choose a fresh --output-root"
        )
    root.mkdir(parents=True)
    orchestrator = ExperimentRunOrchestrator(
        results_db=root / "research.duckdb",
        snapshot_root=root / "snapshots",
        cache_root=root / "pit_cache",
        artifact_root=root / "models",
        report_root=root / "reports",
        registry_path=root / "experiments.jsonl",
    )
    request_kwargs = {
        "charter_path": args.charter,
        "fixture_securities": args.securities,
        "max_workers": args.workers,
    }
    if args.models:
        request_kwargs["models"] = tuple(args.models)
    request = ResearchRunRequest(**request_kwargs)

    cold = orchestrator.run(request)
    warm = orchestrator.run(request)
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "measured_at": datetime.now(UTC).isoformat(),
        "workload": {
            "charter": str(args.charter),
            "fixture_securities": args.securities,
            "models": list(request.models),
            "strategies": list(request.strategies),
            "scenarios": list(request.scenarios),
            "workers": args.workers,
        },
        "cold": {
            "run_id": cold.run_id,
            "snapshot_id": cold.snapshot_id,
            "cache_hit": cold.cache_hit,
            "timings": cold.timings,
        },
        "warm": {
            "run_id": warm.run_id,
            "snapshot_id": warm.snapshot_id,
            "cache_hit": warm.cache_hit,
            "timings": warm.timings,
        },
        "peak_process_rss_platform_value": peak_kib,
        "peak_process_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "limitations": (
            "Measures this local synthetic fixture and process. It is not the HLD reference "
            "hardware/workload and makes no claim of reproducing the illustrative 42-to-12 "
            "minute result."
        ),
    }
    output = root / "benchmark.json"
    output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))
    print(f"benchmark written to {output}")


if __name__ == "__main__":
    main()
