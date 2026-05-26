"""Audit TrajFlow-VSR against the CCF-A proposal readiness checklist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.experiments import audit_project_readiness, export_readiness_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root to audit.")
    parser.add_argument("--output-dir", default="outputs/readiness", help="Directory for audit artifacts.")
    parser.add_argument("--name", default="ccfa_readiness", help="Artifact filename prefix.")
    parser.add_argument("--json-only", action="store_true", help="Print the report without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_project_readiness(args.root)
    if args.json_only:
        import json

        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    export = export_readiness_audit(report, args.output_dir, name=args.name)
    print(export.to_json())


if __name__ == "__main__":
    main()
