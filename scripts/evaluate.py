"""Evaluate a TrajFlow-VSR protocol on a configured benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.evaluation import EvaluationRunner  # noqa: E402
from trajflow_vsr.utils.config import apply_overrides, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an evaluation YAML/JSON/TOML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without running PyTorch.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config value with dotted.path=value. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    runner = EvaluationRunner(config)
    if args.dry_run or config.get("runtime", {}).get("dry_run", False):
        runner.dry_run()
    else:
        runner.run()


if __name__ == "__main__":
    main()
