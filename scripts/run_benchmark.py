"""Run the internal subset of a fixed TrajFlow-VSR benchmark matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.experiments import BenchmarkRunner  # noqa: E402
from trajflow_vsr.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a benchmark YAML/JSON/TOML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected internal run matrix only.")
    parser.add_argument("--output-dir", help="Directory for benchmark run artifacts.")
    parser.add_argument("--id", action="append", default=[], help="Run only the selected benchmark row id.")
    parser.add_argument("--method", action="append", default=[], help="Run only selected method names.")
    parser.add_argument("--dataset", action="append", default=[], help="Run only selected dataset names.")
    parser.add_argument("--degradation", action="append", default=[], help="Run only selected degradation names.")
    parser.add_argument("--scale", action="append", default=[], help="Run only selected scale names.")
    parser.add_argument("--protocol", action="append", default=[], help="Run only selected protocol names.")
    parser.add_argument("--max-runs", type=int, help="Limit the selected matrix after filtering.")
    parser.add_argument("--include-external", action="store_true", help="Include external baselines as skipped records.")
    parser.add_argument("--allow-missing-checkpoints", action="store_true", help="Run internal rows even if checkpoint paths are missing.")
    parser.add_argument("--keep-going", action="store_true", help="Record failures and continue with the remaining rows.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Extra dotted.path=value override applied to every executed evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = BenchmarkRunner(load_config(args.config))
    kwargs = {
        "row_ids": args.id,
        "methods": args.method,
        "datasets": args.dataset,
        "degradations": args.degradation,
        "scales": args.scale,
        "protocols": args.protocol,
        "max_runs": args.max_runs,
        "include_external": args.include_external,
        "output_dir": args.output_dir,
    }
    if args.dry_run:
        runner.summarize(**kwargs)
    else:
        runner.run(
            **kwargs,
            common_overrides=args.set,
            allow_missing_checkpoints=args.allow_missing_checkpoints,
            fail_fast=not args.keep_going,
        )


if __name__ == "__main__":
    main()
