"""Export a manifest inventory without reading image tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.experiments import build_data_inventory, export_data_inventory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root to inspect.")
    parser.add_argument("--output-dir", default="outputs/data_inventory", help="Directory for inventory artifacts.")
    parser.add_argument("--name", default="data_inventory", help="Artifact filename prefix.")
    parser.add_argument("--manifest", action="append", default=None, help="Additional manifest path to include.")
    parser.add_argument("--json-only", action="store_true", help="Print the inventory without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = build_data_inventory(args.root, manifest_paths=args.manifest)
    if args.json_only:
        import json

        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return
    export = export_data_inventory(inventory, args.output_dir, name=args.name)
    print(export.to_json())


if __name__ == "__main__":
    main()
