"""Frame-sequence manifest utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff", ".webp"}
SUPPORTED_LAYOUTS = {"generic", "vimeo90k", "reds", "vid4", "udm10"}


@dataclass(frozen=True)
class FrameSequenceRecord:
    """One video sequence represented by an ordered frame list."""

    sequence_id: str
    frame_dir: str
    frames: list[str]
    frame_count: int
    height: int
    width: int
    hr_frame_dir: str | None = None
    hr_frames: list[str] | None = None
    hr_frame_count: int | None = None
    hr_height: int | None = None
    hr_width: int | None = None


@dataclass(frozen=True)
class ClipRecord:
    """One temporal clip sampled from a frame sequence."""

    clip_id: str
    sequence_id: str
    start: int
    length: int
    frame_indices: list[int]


def build_frame_manifest(
    root: str | Path,
    output_path: str | Path,
    dataset: str = "custom",
    split: str = "train",
    clip_length: int = 0,
    stride: int = 1,
    recursive: bool = True,
    min_frames: int = 1,
    layout: str = "generic",
    split_file: str | Path | None = None,
    sequence_glob: str | None = None,
    hr_root: str | Path | None = None,
    hr_layout: str | None = None,
    hr_split_file: str | Path | None = None,
    hr_sequence_glob: str | None = None,
    allow_unpaired: bool = False,
) -> dict[str, Any]:
    """Scan an image-sequence root and write a JSON manifest."""

    root = Path(root)
    layout = _normalize_layout(layout)
    sequences = discover_frame_sequences(
        root=root,
        recursive=recursive,
        min_frames=min_frames,
        layout=layout,
        split=split,
        split_file=split_file,
        sequence_glob=sequence_glob,
    )
    normalized_hr_layout = _normalize_layout(hr_layout or layout) if hr_root is not None else None
    if hr_root is not None:
        hr_sequences = discover_frame_sequences(
            root=hr_root,
            recursive=recursive,
            min_frames=min_frames,
            layout=normalized_hr_layout or layout,
            split=split,
            split_file=hr_split_file,
            sequence_glob=hr_sequence_glob,
        )
        sequences = attach_hr_sequences(sequences, hr_sequences, allow_unpaired=allow_unpaired)
    clips = build_clip_records(sequences, clip_length=clip_length, stride=stride)
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "layout": layout,
        "root": str(root),
        "paired": hr_root is not None,
        "hr_root": None if hr_root is None else str(hr_root),
        "hr_layout": normalized_hr_layout,
        "split_file": None if split_file is None else str(split_file),
        "sequence_glob": sequence_glob,
        "hr_split_file": None if hr_split_file is None else str(hr_split_file),
        "hr_sequence_glob": hr_sequence_glob,
        "allow_unpaired": bool(allow_unpaired),
        "clip_length": int(clip_length),
        "stride": int(stride),
        "sequences": [asdict(sequence) for sequence in sequences],
        "clips": [asdict(clip) for clip in clips],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def discover_frame_sequences(
    root: str | Path,
    recursive: bool = True,
    min_frames: int = 1,
    layout: str = "generic",
    split: str = "train",
    split_file: str | Path | None = None,
    sequence_glob: str | None = None,
) -> list[FrameSequenceRecord]:
    """Find directories that contain frame images."""

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Frame root does not exist: {root}")
    layout = _normalize_layout(layout)
    directories = _candidate_sequence_dirs(
        root=root,
        recursive=recursive,
        layout=layout,
        split=split,
        split_file=split_file,
        sequence_glob=sequence_glob,
    )
    sequences = []
    seen: set[tuple[str, ...]] = set()
    for directory in directories:
        frames = sorted(item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        if len(frames) < int(min_frames):
            continue
        frame_key = tuple(str(path) for path in frames)
        if frame_key in seen:
            continue
        seen.add(frame_key)
        height, width = _read_image_size(frames[0])
        sequence_id = _sequence_id(root, directory)
        sequences.append(
            FrameSequenceRecord(
                sequence_id=sequence_id,
                frame_dir=str(directory),
                frames=[str(path) for path in frames],
                frame_count=len(frames),
                height=height,
                width=width,
            )
        )
    return sequences


def attach_hr_sequences(
    lr_sequences: list[FrameSequenceRecord],
    hr_sequences: list[FrameSequenceRecord],
    allow_unpaired: bool = False,
) -> list[FrameSequenceRecord]:
    """Attach same-id HR target frames to LR sequence records."""

    hr_by_id = {sequence.sequence_id: sequence for sequence in hr_sequences}
    paired = []
    missing = []
    short = []
    for sequence in lr_sequences:
        hr_sequence = hr_by_id.get(sequence.sequence_id)
        if hr_sequence is None:
            missing.append(sequence.sequence_id)
            if allow_unpaired:
                paired.append(sequence)
            continue
        if hr_sequence.frame_count < sequence.frame_count:
            short.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "lr_frames": sequence.frame_count,
                    "hr_frames": hr_sequence.frame_count,
                }
            )
            if allow_unpaired:
                paired.append(sequence)
            continue
        paired.append(
            replace(
                sequence,
                hr_frame_dir=hr_sequence.frame_dir,
                hr_frames=hr_sequence.frames,
                hr_frame_count=hr_sequence.frame_count,
                hr_height=hr_sequence.height,
                hr_width=hr_sequence.width,
            )
        )

    if missing and not allow_unpaired:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise ValueError(f"Missing HR sequence(s) for LR manifest: {preview}{suffix}")
    if short and not allow_unpaired:
        preview = ", ".join(
            f"{item['sequence_id']} lr={item['lr_frames']} hr={item['hr_frames']}" for item in short[:5]
        )
        suffix = "" if len(short) <= 5 else f", ... ({len(short)} total)"
        raise ValueError(f"HR sequence(s) shorter than LR sequence(s): {preview}{suffix}")
    return paired


def _candidate_sequence_dirs(
    root: Path,
    recursive: bool,
    layout: str,
    split: str,
    split_file: str | Path | None,
    sequence_glob: str | None,
) -> list[Path]:
    if split_file is not None:
        return _directories_from_split_file(root=root, split_file=Path(split_file), layout=layout)
    if sequence_glob:
        return sorted(path for path in root.glob(sequence_glob) if path.is_dir())

    directories: list[Path] = []
    for scan_root in _layout_scan_roots(root=root, layout=layout, split=split):
        if not scan_root.exists() or not scan_root.is_dir():
            continue
        if recursive:
            directories.extend([scan_root, *sorted(item for item in scan_root.rglob("*") if item.is_dir())])
        else:
            directories.append(scan_root)
    return _deduplicate_paths(directories)


def _directories_from_split_file(root: Path, split_file: Path, layout: str) -> list[Path]:
    if not split_file.is_absolute():
        split_file = root / split_file
    if not split_file.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")

    directories = []
    for raw_line in split_file.read_text(encoding="utf-8").splitlines():
        entry = raw_line.split("#", 1)[0].strip()
        if not entry:
            continue
        entry = entry.replace("\\", "/")
        candidates = _split_entry_candidates(root=root, entry=entry, layout=layout)
        directory = next((candidate for candidate in candidates if candidate.exists() and candidate.is_dir()), None)
        if directory is None:
            raise FileNotFoundError(f"Split entry does not resolve to a frame directory: {entry}")
        directories.append(directory)
    return _deduplicate_paths(directories)


def _split_entry_candidates(root: Path, entry: str, layout: str) -> list[Path]:
    if layout == "vimeo90k":
        return [root / "sequences" / entry, root / entry]
    return [root / entry, root / "sequences" / entry]


def _layout_scan_roots(root: Path, layout: str, split: str) -> list[Path]:
    if layout == "vimeo90k":
        return [root / "sequences", root]
    if layout == "reds":
        return [root / f"{split}_sharp", root / split / "sharp", root / split, root]
    return [root]


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    deduped = []
    seen = set()
    for path in paths:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _normalize_layout(layout: str) -> str:
    normalized = str(layout or "generic").lower()
    if normalized not in SUPPORTED_LAYOUTS:
        raise ValueError(f"Unsupported frame manifest layout: {layout}")
    return normalized


def build_clip_records(
    sequences: list[FrameSequenceRecord],
    clip_length: int = 0,
    stride: int = 1,
) -> list[ClipRecord]:
    """Build temporal clips from sequence records."""

    clips = []
    stride = max(int(stride), 1)
    for sequence in sequences:
        length = sequence.frame_count if int(clip_length) <= 0 else int(clip_length)
        if sequence.frame_count < length:
            continue
        starts = range(0, sequence.frame_count - length + 1, stride)
        for start in starts:
            indices = list(range(start, start + length))
            clips.append(
                ClipRecord(
                    clip_id=f"{sequence.sequence_id}:{start:06d}:{length}",
                    sequence_id=sequence.sequence_id,
                    start=start,
                    length=length,
                    frame_indices=indices,
                )
            )
    return clips


def load_frame_manifest(path: str | Path) -> dict[str, Any]:
    """Load a frame-sequence manifest."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or "sequences" not in manifest or "clips" not in manifest:
        raise ValueError(f"Invalid frame manifest: {path}")
    return manifest


def describe_frame_manifest_config(config: dict[str, Any]) -> dict[str, Any]:
    """Describe a frame-manifest data config without loading tensors."""

    manifest_path = config.get("manifest_path")
    summary = {
        "name": "frame_manifest",
        "manifest_path": manifest_path,
        "layout": config.get("layout", "generic"),
        "paired": False,
        "require_paired": bool(config.get("require_paired", False)),
        "require_degradation": bool(config.get("require_degradation", False)),
        "batch_size": int(config.get("batch_size", 1)),
        "frames": int(config.get("frames", 0)),
        "scale": float(config.get("scale", 2.0)),
    }
    if manifest_path and Path(manifest_path).exists():
        manifest = load_frame_manifest(manifest_path)
        summary.update(
            {
                "dataset": manifest.get("dataset"),
                "split": manifest.get("split"),
                "layout": manifest.get("layout", summary["layout"]),
                "paired": bool(manifest.get("paired", False)),
                "hr_root": manifest.get("hr_root"),
                "degradation": manifest.get("degradation"),
                "sequences": len(manifest.get("sequences", [])),
                "paired_sequences": sum(1 for sequence in manifest.get("sequences", []) if sequence.get("hr_frames")),
                "clips": len(manifest.get("clips", [])),
            }
        )
    else:
        summary["status"] = "manifest not found yet"
    return summary


def make_frame_manifest_batch(config: dict[str, Any], device: str = "cpu") -> dict[str, Any]:
    """Create a B,T,C,H,W batch from a frame manifest."""

    torch = require_torch()
    manifest = load_frame_manifest(config["manifest_path"])
    clips = manifest.get("clips", [])
    if not clips:
        raise ValueError(f"Manifest has no clips: {config['manifest_path']}")
    sequence_by_id = {sequence["sequence_id"]: sequence for sequence in manifest.get("sequences", [])}
    batch_size = int(config.get("batch_size", 1))
    clip_index = int(config.get("clip_index", 0))
    requested_frames = int(config.get("frames", 0))
    scale = float(config.get("scale", manifest.get("scale", 2.0)))
    require_paired = bool(config.get("require_paired", False))

    videos = []
    targets = []
    metadata = []
    for batch_offset in range(batch_size):
        clip = clips[(clip_index + batch_offset) % len(clips)]
        sequence = sequence_by_id[clip["sequence_id"]]
        indices = list(clip["frame_indices"])
        if requested_frames > 0:
            indices = indices[:requested_frames]
        frames = [_read_frame_tensor(sequence["frames"][idx]) for idx in indices]
        video = torch.stack(frames, dim=0)
        video = _resize_video_if_needed(video, height=config.get("height"), width=config.get("width"))
        videos.append(video)
        paired_target = bool(sequence.get("hr_frames"))
        if paired_target:
            hr_frames = [_read_frame_tensor(sequence["hr_frames"][idx]) for idx in indices]
            target = torch.stack(hr_frames, dim=0)
            target_height, target_width = _resolve_target_size(config, video, scale=scale)
            target = _resize_video_if_needed(
                target,
                height=target_height,
                width=target_width,
            )
        else:
            if require_paired:
                raise ValueError(
                    "Frame manifest batch requires paired HR targets, but "
                    f"sequence '{sequence['sequence_id']}' has no hr_frames. "
                    "Use a paired LR/HR manifest for training, or set "
                    "data.require_paired=false only for smoke/no-reference runs."
                )
            target = torch.nn.functional.interpolate(
                video,
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
            )
        targets.append(target)
        metadata.append(
            {
                "clip_id": clip["clip_id"],
                "sequence_id": clip["sequence_id"],
                "frame_indices": indices,
                "paired_hr": paired_target,
                "hr_frame_dir": sequence.get("hr_frame_dir"),
            }
        )

    lr = torch.stack(videos, dim=0).to(device=device)
    hr = torch.stack(targets, dim=0).to(device=device)
    return {"lr": lr, "hr": hr, "scale": scale, "metadata": metadata}


def make_stage_a_frame_manifest_batch(config: dict[str, Any], device: str = "cpu") -> dict[str, Any]:
    """Create Stage A degradation-supervision tensors from a paired frame manifest."""

    torch = require_torch()
    manifest = load_frame_manifest(config["manifest_path"])
    batch_config = dict(config)
    batch_config.setdefault("require_paired", True)
    batch = make_frame_manifest_batch(batch_config, device=device)
    lr = batch["lr"]
    hr = batch["hr"]
    clean_lr = torch.nn.functional.interpolate(
        hr.flatten(0, 1),
        size=lr.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).unflatten(0, (lr.shape[0], lr.shape[1]))
    artifact = (lr - clean_lr).abs().mean(dim=2, keepdim=True)
    artifact = (artifact / artifact.amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)).clamp(0.0, 1.0)
    reliability = torch.exp(-4.0 * artifact).clamp(0.0, 1.0)
    profile = _stage_a_degradation_profile(manifest=manifest, config=config)
    if bool(config.get("require_degradation", False)) and not profile:
        raise ValueError(
            "Stage A manifest supervision requires degradation metadata. "
            "Generate the manifest with scripts/degrade_data.py or set data.require_degradation=false."
        )
    labels = _degradation_label(profile, scale=float(batch["scale"]), device=lr.device, dtype=lr.dtype)
    batch.update(
        {
            "clean_lr": clean_lr.clamp(0.0, 1.0),
            "artifact": artifact,
            "reliability": reliability,
            "degradation": labels.unsqueeze(0).expand(lr.shape[0], -1).contiguous(),
            "degradation_profile": profile,
        }
    )
    return batch


def _stage_a_degradation_profile(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    profile = manifest.get("degradation")
    if isinstance(profile, dict):
        return profile
    profile = config.get("degradation")
    if isinstance(profile, dict):
        return profile
    return {}


def _degradation_label(profile: dict[str, Any], scale: float, device: Any, dtype: Any) -> Any:
    torch = require_torch()
    blur = _unit_interval(profile.get("blur_strength", profile.get("blur", 0.0)))
    noise = _unit_interval(float(profile.get("noise_std", profile.get("noise", 0.0))) / 0.10)
    codec = _unit_interval(profile.get("codec_strength", profile.get("codec", 0.0)))
    motion = _unit_interval(profile.get("motion_strength", profile.get("motion", 0.0)))
    scale_label = _unit_interval(float(profile.get("scale", scale)) / 8.0)
    severity = (blur + noise + codec + motion) / 4.0
    exposure = _unit_interval((float(profile.get("exposure", 1.0)) - 0.5) / 1.5)
    rolling_shutter = _unit_interval(profile.get("rolling_shutter", 0.0))
    return torch.tensor(
        [blur, noise, codec, motion, scale_label, severity, exposure, rolling_shutter],
        device=device,
        dtype=dtype,
    )


def _unit_interval(value: Any) -> float:
    return min(max(float(value), 0.0), 1.0)


def _resolve_target_size(config: dict[str, Any], lr_video: Any, scale: float) -> tuple[int | None, int | None]:
    hr_height = config.get("hr_height", config.get("target_height"))
    hr_width = config.get("hr_width", config.get("target_width"))
    if hr_height is not None and hr_width is not None:
        return int(hr_height), int(hr_width)
    if config.get("height") is None or config.get("width") is None:
        return None, None
    return int(round(lr_video.shape[-2] * scale)), int(round(lr_video.shape[-1] * scale))


def _read_image_size(path: Path) -> tuple[int, int]:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("imageio is required to inspect frame sequences") from exc

    array = iio.imread(path)
    if getattr(array, "ndim", 0) < 2:
        raise ValueError(f"Expected image frame with at least 2 dimensions: {path}")
    return int(array.shape[0]), int(array.shape[1])


def _read_frame_tensor(path: str | Path) -> Any:
    torch = require_torch()
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("imageio is required to read frame sequences") from exc

    array = iio.imread(path)
    tensor = torch.as_tensor(array)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 3:
        raise ValueError(f"Expected H,W,C image, got {tuple(tensor.shape)} for {path}")
    if tensor.shape[-1] == 1:
        tensor = tensor.expand(-1, -1, 3)
    elif tensor.shape[-1] >= 3:
        tensor = tensor[..., :3]
    else:
        raise ValueError(f"Expected 1, 3, or 4 channels, got {tensor.shape[-1]} for {path}")
    tensor = tensor.permute(2, 0, 1).contiguous().float()
    if tensor.amax() > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _resize_video_if_needed(video: Any, height: Any = None, width: Any = None) -> Any:
    torch = require_torch()
    if height is None or width is None:
        return video
    target = (int(height), int(width))
    if tuple(video.shape[-2:]) == target:
        return video
    return torch.nn.functional.interpolate(video, size=target, mode="bilinear", align_corners=False)


def _sequence_id(root: Path, directory: Path) -> str:
    if directory == root:
        return root.name or "root"
    return directory.relative_to(root).as_posix()
