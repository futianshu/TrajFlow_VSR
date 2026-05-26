"""Generate degraded LR frame sequences from HR data and write paired manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.data.degradation import (  # noqa: E402
    DEGRADATION_PROFILE_NAMES,
    build_degraded_frame_dataset,
    degradation_profile_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr-root", required=True, help="Root directory containing clean HR image sequences.")
    parser.add_argument("--lr-output-root", required=True, help="Directory where degraded LR sequences will be written.")
    parser.add_argument("--manifest-output", help="Optional paired frame-manifest path to write.")
    parser.add_argument("--dataset", default="custom", help="Dataset name stored in the manifest and metadata.")
    parser.add_argument("--split", default="train", help="Split name stored in the manifest and metadata.")
    parser.add_argument(
        "--layout",
        default="generic",
        choices=["generic", "vimeo90k", "reds", "vid4", "udm10"],
        help="Directory layout adapter for the HR root and generated LR root.",
    )
    parser.add_argument("--split-file", help="Optional split file with HR sequence ids.")
    parser.add_argument("--sequence-glob", help="Optional glob relative to HR root for selecting sequence directories.")
    parser.add_argument(
        "--profile",
        default="mild_real",
        choices=sorted(DEGRADATION_PROFILE_NAMES),
        help="Built-in degradation profile.",
    )
    parser.add_argument("--scale", type=float, help="Override profile scale.")
    parser.add_argument("--blur-strength", type=float, help="Override profile blur strength in [0,1].")
    parser.add_argument("--noise-std", type=float, help="Override profile noise standard deviation.")
    parser.add_argument("--codec-strength", type=float, help="Override profile codec/block artifact strength in [0,1].")
    parser.add_argument("--motion-strength", type=float, help="Override profile temporal blend strength in [0,1].")
    parser.add_argument("--exposure", type=float, help="Override profile exposure multiplier.")
    parser.add_argument("--block-size", type=int, help="Override profile codec block size.")
    parser.add_argument("--clip-length", type=int, default=0, help="Frames per manifest clip. 0 uses full sequences.")
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride between manifest clips.")
    parser.add_argument("--min-frames", type=int, default=1, help="Minimum frames required for a sequence.")
    parser.add_argument("--flat", action="store_true", help="Only scan direct HR root sequence directory.")
    parser.add_argument("--image-format", default="png", help="Output image format, for example png or jpg.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated LR frames.")
    parser.add_argument("--seed", type=int, default=20260524, help="CPU RNG seed for noise generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_config = {"profile": args.profile}
    for cli_name, profile_name in [
        ("scale", "scale"),
        ("blur_strength", "blur_strength"),
        ("noise_std", "noise_std"),
        ("codec_strength", "codec_strength"),
        ("motion_strength", "motion_strength"),
        ("exposure", "exposure"),
        ("block_size", "block_size"),
    ]:
        value = getattr(args, cli_name)
        if value is not None:
            profile_config[profile_name] = value
    summary = build_degraded_frame_dataset(
        hr_root=args.hr_root,
        lr_output_root=args.lr_output_root,
        profile=degradation_profile_from_config(profile_config),
        manifest_output=args.manifest_output,
        dataset=args.dataset,
        split=args.split,
        layout=args.layout,
        split_file=args.split_file,
        sequence_glob=args.sequence_glob,
        clip_length=args.clip_length,
        stride=args.stride,
        recursive=not args.flat,
        min_frames=args.min_frames,
        image_format=args.image_format,
        overwrite=args.overwrite,
        seed=args.seed,
    )
    compact = {
        "dataset": summary["dataset"],
        "split": summary["split"],
        "layout": summary["layout"],
        "profile": summary["profile"],
        "hr_root": summary["hr_root"],
        "lr_output_root": summary["lr_output_root"],
        "sequences": summary["sequences"],
        "frames": summary["frames"],
        "manifest_output": summary["manifest_output"],
        "clips": summary["clips"],
        "metadata_path": summary["metadata_path"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
