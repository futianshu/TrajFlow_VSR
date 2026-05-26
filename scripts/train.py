"""Train or dry-run a staged TrajFlow-VSR experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.training.runner import TrainingRunner  # noqa: E402
from trajflow_vsr.utils.config import apply_overrides, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON/TOML experiment config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without importing PyTorch.")
    parser.add_argument("--resume", help="Optional training checkpoint to resume from.")
    parser.add_argument("--output-dir", help="Override project.output_dir for run artifacts.")
    parser.add_argument("--checkpoint-dir", help="Override checkpoint.output_dir for saved checkpoints.")
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
    if args.output_dir:
        config.setdefault("project", {})["output_dir"] = args.output_dir
    if args.resume:
        config.setdefault("checkpoint", {})["resume_path"] = args.resume
    if args.checkpoint_dir:
        config.setdefault("checkpoint", {})["output_dir"] = args.checkpoint_dir
    runner = TrainingRunner(config)
    if args.dry_run or config.get("runtime", {}).get("dry_run", False):
        runner.dry_run()
    else:
        runner.run()


if __name__ == "__main__":
    main()
