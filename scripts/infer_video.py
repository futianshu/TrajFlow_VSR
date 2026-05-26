"""Run TrajFlow-VSR inference on a synthetic clip, image sequence, or video file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.inference import InferenceRunner  # noqa: E402
from trajflow_vsr.utils.config import apply_overrides, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an inference YAML/JSON/TOML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without running PyTorch.")
    parser.add_argument("--input", help="Image sequence directory, image file, or video file.")
    parser.add_argument("--output-dir", help="Override inference.output_dir.")
    parser.add_argument("--checkpoint", help="Optional checkpoint path to load before inference.")
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
    inference = config.setdefault("inference", {})
    if args.input:
        inference["input_path"] = args.input
    if args.output_dir:
        inference["output_dir"] = args.output_dir
    if args.checkpoint:
        inference["checkpoint_path"] = args.checkpoint

    runner = InferenceRunner(config)
    if args.dry_run or config.get("runtime", {}).get("dry_run", False):
        runner.dry_run()
    else:
        runner.run()


if __name__ == "__main__":
    main()
