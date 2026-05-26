"""Prepare frame-sequence manifests for TrajFlow-VSR experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.data import build_frame_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root directory containing image sequences.")
    parser.add_argument("--output", required=True, help="Path to write the JSON manifest.")
    parser.add_argument("--dataset", default="custom", help="Dataset name stored in the manifest.")
    parser.add_argument("--split", default="train", help="Split name stored in the manifest.")
    parser.add_argument(
        "--layout",
        default="generic",
        choices=["generic", "vimeo90k", "reds", "vid4", "udm10"],
        help="Directory layout adapter.",
    )
    parser.add_argument("--split-file", help="Optional split file with sequence ids, such as Vimeo90K sep_trainlist.txt.")
    parser.add_argument("--sequence-glob", help="Optional glob relative to root for selecting sequence directories.")
    parser.add_argument("--hr-root", help="Optional paired HR/target frame root with matching sequence ids.")
    parser.add_argument(
        "--hr-layout",
        choices=["generic", "vimeo90k", "reds", "vid4", "udm10"],
        help="Optional HR directory layout adapter. Defaults to --layout when --hr-root is set.",
    )
    parser.add_argument("--hr-split-file", help="Optional split file for HR sequence ids.")
    parser.add_argument("--hr-sequence-glob", help="Optional glob relative to HR root for selecting target directories.")
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="Keep LR sequences even when --hr-root does not contain a matching target sequence.",
    )
    parser.add_argument("--clip-length", type=int, default=0, help="Frames per clip. 0 uses full sequences.")
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride between clips.")
    parser.add_argument("--min-frames", type=int, default=1, help="Minimum frames required for a sequence.")
    parser.add_argument("--flat", action="store_true", help="Only scan the root directory, not nested subdirectories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_frame_manifest(
        root=args.root,
        output_path=args.output,
        dataset=args.dataset,
        split=args.split,
        clip_length=args.clip_length,
        stride=args.stride,
        recursive=not args.flat,
        min_frames=args.min_frames,
        layout=args.layout,
        split_file=args.split_file,
        sequence_glob=args.sequence_glob,
        hr_root=args.hr_root,
        hr_layout=args.hr_layout,
        hr_split_file=args.hr_split_file,
        hr_sequence_glob=args.hr_sequence_glob,
        allow_unpaired=args.allow_unpaired,
    )
    summary = {
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "layout": manifest["layout"],
        "root": manifest["root"],
        "paired": manifest["paired"],
        "hr_root": manifest["hr_root"],
        "output": args.output,
        "sequences": len(manifest["sequences"]),
        "paired_sequences": sum(1 for sequence in manifest["sequences"] if sequence.get("hr_frames")),
        "clips": len(manifest["clips"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
