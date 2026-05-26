"""Export paper-facing tables from ablation comparison artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.experiments import export_paper_tables, load_comparison  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to comparison.json or ablation summary.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for Markdown/LaTeX table files.")
    parser.add_argument("--name", default="ablation", help="Prefix for exported table filenames.")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric to export as a matrix table. Can repeat. Defaults to comparison metrics.",
    )
    parser.add_argument(
        "--matrix-separator",
        default="_x_",
        help="Variant-name separator used to split matrix rows and columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export = export_paper_tables(
        load_comparison(args.input),
        args.output_dir,
        name=args.name,
        metrics=args.metric or None,
        matrix_separator=args.matrix_separator,
    )
    print(export.to_json())


if __name__ == "__main__":
    main()
