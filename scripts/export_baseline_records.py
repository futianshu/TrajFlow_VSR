"""Export reproducible baseline records from a registry config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.baselines import export_baseline_registry, load_baseline_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a baseline registry YAML/JSON/TOML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported baseline records.")
    parser.add_argument("--name", default="baseline_registry", help="Prefix for exported record filenames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export = export_baseline_registry(
        load_baseline_registry(args.config),
        args.output_dir,
        name=args.name,
    )
    print(export.to_json())


if __name__ == "__main__":
    main()
