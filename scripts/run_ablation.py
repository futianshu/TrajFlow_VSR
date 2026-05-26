"""Run an ablation grid from a reusable config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.experiments import AblationRunner  # noqa: E402
from trajflow_vsr.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an ablation YAML/JSON/TOML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print the expanded grid without running variants.")
    parser.add_argument("--output-dir", help="Root directory for ablation artifacts.")
    parser.add_argument("--variant", action="append", default=[], help="Run only a named variant. Can repeat.")
    parser.add_argument("--max-variants", type=int, help="Run only the first N selected variants.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override every variant with dotted.path=value. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = AblationRunner(load_config(args.config))
    if args.dry_run:
        runner.dry_run(
            variants=args.variant or None,
            max_variants=args.max_variants,
            output_dir=args.output_dir,
        )
    else:
        runner.run(
            common_overrides=args.set,
            variants=args.variant or None,
            max_variants=args.max_variants,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
