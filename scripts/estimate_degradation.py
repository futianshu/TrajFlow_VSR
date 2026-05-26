"""Estimate offline degradation preprocessing time and storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.data.estimate import estimate_degradation_from_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Existing frame manifest to estimate from.")
    parser.add_argument("--profile", action="append", default=None, help="Degradation profile name. Repeat for multiple profiles.")
    parser.add_argument("--scale", type=float, default=4.0, help="LR scale factor.")
    parser.add_argument("--seconds-per-frame", type=float, default=0.06, help="Observed CPU seconds per frame.")
    parser.add_argument("--output-overhead", type=float, default=2.0, help="PNG/output overhead relative to area scaling.")
    parser.add_argument("--shards", type=int, default=1, help="Number of sequence shards to suggest.")
    parser.add_argument("--no-stat", action="store_true", help="Skip per-frame stat calls and omit size estimates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimate = estimate_degradation_from_manifest(
        args.manifest,
        profiles=args.profile,
        scale=args.scale,
        seconds_per_frame=args.seconds_per_frame,
        output_overhead=args.output_overhead,
        shards=args.shards,
        stat_input_bytes=not args.no_stat,
    )
    print(estimate.to_json())


if __name__ == "__main__":
    main()
