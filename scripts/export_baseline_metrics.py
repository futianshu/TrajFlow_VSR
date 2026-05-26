"""Collect baseline metric files into JSON/CSV/Markdown comparison tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.baselines import export_baseline_metrics, load_baseline_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a baseline registry YAML/JSON/TOML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported metric tables.")
    parser.add_argument("--name", default="baseline_metrics", help="Prefix for exported metric filenames.")
    parser.add_argument("--root", help="Root used to resolve relative metrics_path entries.")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric to include. Can repeat. Defaults to the standard VSR metric set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export = export_baseline_metrics(
        load_baseline_registry(args.config),
        args.output_dir,
        name=args.name,
        metrics=args.metric or None,
        root=args.root,
    )
    print(export.to_json())


if __name__ == "__main__":
    main()
