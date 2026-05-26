"""Export a benchmark run matrix without executing model evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.experiments import BenchmarkPlan  # noqa: E402
from trajflow_vsr.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a benchmark YAML/JSON/TOML config.")
    parser.add_argument("--output-dir", help="Directory for exported benchmark plan files.")
    parser.add_argument("--name", help="Prefix for exported plan filenames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export = BenchmarkPlan(load_config(args.config)).export(output_dir=args.output_dir, name=args.name)
    print(export.to_json())


if __name__ == "__main__":
    main()
