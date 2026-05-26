"""Realistic degradation profiles and offline LR synthesis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from trajflow_vsr.data.manifest import (
    FrameSequenceRecord,
    build_frame_manifest,
    discover_frame_sequences,
)
from trajflow_vsr.utils.torch_utils import require_torch


DEGRADATION_PROFILE_NAMES = {"bicubic", "mild_real", "strong_real", "codec_motion"}


@dataclass(frozen=True)
class RealisticDegradationProfile:
    """Deterministic settings for HR-to-LR degradation synthesis."""

    name: str = "mild_real"
    scale: float = 4.0
    blur_strength: float = 0.25
    noise_std: float = 0.015
    codec_strength: float = 0.20
    motion_strength: float = 0.10
    exposure: float = 1.0
    block_size: int = 4
    downsample_mode: str = "bicubic"


def degradation_profile_from_config(config: str | dict[str, Any] | RealisticDegradationProfile) -> RealisticDegradationProfile:
    """Build a degradation profile from a preset name or config mapping."""

    if isinstance(config, RealisticDegradationProfile):
        return config
    if isinstance(config, str):
        return preset_degradation_profile(config)
    preset_name = str(config.get("profile", config.get("name", "mild_real")))
    profile = preset_degradation_profile(preset_name)
    fields = {
        "name",
        "scale",
        "blur_strength",
        "noise_std",
        "codec_strength",
        "motion_strength",
        "exposure",
        "block_size",
        "downsample_mode",
    }
    overrides = {key: config[key] for key in fields if key in config}
    if "profile" in config and "name" not in overrides:
        overrides["name"] = str(config["profile"])
    return replace(profile, **overrides)


def preset_degradation_profile(name: str, scale: float | None = None) -> RealisticDegradationProfile:
    """Return one of the built-in real-world degradation profiles."""

    normalized = str(name or "mild_real").lower()
    if normalized not in DEGRADATION_PROFILE_NAMES:
        raise ValueError(f"Unsupported degradation profile: {name}")
    profile = {
        "bicubic": RealisticDegradationProfile(
            name="bicubic",
            blur_strength=0.0,
            noise_std=0.0,
            codec_strength=0.0,
            motion_strength=0.0,
            exposure=1.0,
            block_size=4,
        ),
        "mild_real": RealisticDegradationProfile(
            name="mild_real",
            blur_strength=0.25,
            noise_std=0.015,
            codec_strength=0.20,
            motion_strength=0.10,
            exposure=1.0,
            block_size=4,
        ),
        "strong_real": RealisticDegradationProfile(
            name="strong_real",
            blur_strength=0.55,
            noise_std=0.04,
            codec_strength=0.45,
            motion_strength=0.25,
            exposure=0.95,
            block_size=5,
        ),
        "codec_motion": RealisticDegradationProfile(
            name="codec_motion",
            blur_strength=0.35,
            noise_std=0.02,
            codec_strength=0.60,
            motion_strength=0.35,
            exposure=0.90,
            block_size=6,
        ),
    }[normalized]
    if scale is not None:
        return replace(profile, scale=float(scale))
    return profile


def apply_realistic_degradation(
    hr_video: Any,
    profile: RealisticDegradationProfile | dict[str, Any] | str = "mild_real",
    generator: Any | None = None,
) -> dict[str, Any]:
    """Degrade an HR video tensor into LR tensors and supervision maps."""

    torch = require_torch()
    profile = degradation_profile_from_config(profile)
    hr = torch.as_tensor(hr_video).float()
    if hr.ndim == 4:
        hr = hr.unsqueeze(0)
    if hr.ndim != 5:
        raise ValueError(f"Expected B,T,C,H,W video tensor, got {tuple(hr.shape)}")
    hr = hr.clamp(0.0, 1.0)

    degraded_hr = _apply_blur(hr, float(profile.blur_strength))
    clean_lr = _downsample_video(hr, scale=profile.scale, mode=profile.downsample_mode)
    lr = _downsample_video(degraded_hr, scale=profile.scale, mode=profile.downsample_mode)
    lr = _apply_motion_blend(lr, float(profile.motion_strength))
    lr = _apply_codec_proxy(lr, float(profile.codec_strength), int(profile.block_size))
    lr = (lr * float(profile.exposure)).clamp(0.0, 1.0)
    if float(profile.noise_std) > 0:
        noise = torch.randn(lr.shape, device=lr.device, dtype=lr.dtype, generator=generator) * float(profile.noise_std)
        lr = lr + noise
    lr = lr.clamp(0.0, 1.0)

    artifact = (lr - clean_lr).abs().mean(dim=2, keepdim=True)
    artifact = (artifact / artifact.amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)).clamp(0.0, 1.0)
    reliability = torch.exp(-4.0 * artifact).clamp(0.0, 1.0)

    batch_size = int(lr.shape[0])
    label = _degradation_label(profile, device=lr.device, dtype=lr.dtype)
    labels = label.unsqueeze(0).expand(batch_size, -1).contiguous()
    return {
        "lr": lr,
        "hr": hr,
        "clean_lr": clean_lr,
        "artifact": artifact,
        "reliability": reliability,
        "degradation": labels,
        "scale": float(profile.scale),
        "profile": asdict(profile),
    }


def build_degraded_frame_dataset(
    hr_root: str | Path,
    lr_output_root: str | Path,
    profile: RealisticDegradationProfile | dict[str, Any] | str = "mild_real",
    manifest_output: str | Path | None = None,
    dataset: str = "custom",
    split: str = "train",
    layout: str = "generic",
    split_file: str | Path | None = None,
    sequence_glob: str | None = None,
    clip_length: int = 0,
    stride: int = 1,
    recursive: bool = True,
    min_frames: int = 1,
    image_format: str = "png",
    overwrite: bool = False,
    seed: int = 20260524,
) -> dict[str, Any]:
    """Generate LR frames from HR sequences and optionally write a paired manifest."""

    torch = require_torch()
    hr_root = Path(hr_root)
    lr_output_root = Path(lr_output_root)
    profile = degradation_profile_from_config(profile)
    sequences = discover_frame_sequences(
        root=hr_root,
        recursive=recursive,
        min_frames=min_frames,
        layout=layout,
        split=split,
        split_file=split_file,
        sequence_glob=sequence_glob,
    )
    sequence_summaries = []
    for sequence_idx, sequence in enumerate(sequences):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + sequence_idx)
        sequence_summaries.append(
            degrade_frame_sequence(
                sequence=sequence,
                output_root=lr_output_root,
                profile=profile,
                image_format=image_format,
                overwrite=overwrite,
                generator=generator,
            )
        )

    manifest = None
    if manifest_output is not None:
        manifest = build_frame_manifest(
            root=lr_output_root,
            output_path=manifest_output,
            dataset=dataset,
            split=split,
            clip_length=clip_length,
            stride=stride,
            recursive=recursive,
            min_frames=min_frames,
            layout=layout,
            sequence_glob=sequence_glob,
            hr_root=hr_root,
            hr_layout=layout,
            hr_split_file=split_file,
            hr_sequence_glob=sequence_glob,
            allow_unpaired=False,
        )
        manifest["degradation"] = asdict(profile)
        Path(manifest_output).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "dataset": dataset,
        "split": split,
        "layout": layout,
        "profile": asdict(profile),
        "hr_root": str(hr_root),
        "lr_output_root": str(lr_output_root),
        "sequences": len(sequence_summaries),
        "frames": sum(item["frames"] for item in sequence_summaries),
        "manifest_output": None if manifest_output is None else str(manifest_output),
        "clips": 0 if manifest is None else len(manifest.get("clips", [])),
        "sequence_summaries": sequence_summaries,
    }
    metadata_path = lr_output_root / "degradation_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["metadata_path"] = str(metadata_path)
    return summary


def degrade_frame_sequence(
    sequence: FrameSequenceRecord,
    output_root: str | Path,
    profile: RealisticDegradationProfile | dict[str, Any] | str = "mild_real",
    image_format: str = "png",
    overwrite: bool = False,
    generator: Any | None = None,
) -> dict[str, Any]:
    """Write one degraded LR sequence while preserving the input sequence id."""

    profile = degradation_profile_from_config(profile)
    output_dir = Path(output_root) / sequence.sequence_id
    output_dir.mkdir(parents=True, exist_ok=True)
    hr_video = torch_stack_frames(sequence.frames)
    result = apply_realistic_degradation(hr_video, profile=profile, generator=generator)
    lr_video = result["lr"][0]
    output_frames = []
    for frame_idx, source_path in enumerate(sequence.frames):
        source = Path(source_path)
        frame_path = output_dir / f"{source.stem}.{image_format.lower().lstrip('.')}"
        if frame_path.exists() and not overwrite:
            output_frames.append(str(frame_path))
            continue
        write_training_frame(frame_path, lr_video[frame_idx])
        output_frames.append(str(frame_path))
    return {
        "sequence_id": sequence.sequence_id,
        "input_dir": sequence.frame_dir,
        "output_dir": str(output_dir),
        "frames": len(output_frames),
        "height": int(lr_video.shape[-2]),
        "width": int(lr_video.shape[-1]),
        "output_frames": output_frames,
    }


def torch_stack_frames(paths: list[str] | list[Path]) -> Any:
    """Read image paths into a 1,T,C,H,W float tensor."""

    torch = require_torch()
    frames = [_read_training_frame(path) for path in paths]
    return torch.stack(frames, dim=0).unsqueeze(0)


def write_training_frame(path: str | Path, image: Any) -> Path:
    """Write a training frame without per-image min/max normalization."""

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("imageio is required to write degraded training frames") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_hwc = _to_uint8_hwc(image)
    iio.imwrite(path, image_hwc.numpy())
    return path


def _apply_blur(video: Any, strength: float) -> Any:
    torch = require_torch()
    strength = float(max(strength, 0.0))
    if strength <= 0:
        return video
    kernel = max(3, min(9, int(round(3 + 6 * min(strength, 1.0)))))
    if kernel % 2 == 0:
        kernel += 1
    blurred = torch.nn.functional.avg_pool2d(
        video.flatten(0, 1),
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    ).unflatten(0, (video.shape[0], video.shape[1]))
    blend = min(strength, 1.0)
    return ((1.0 - blend) * video + blend * blurred).clamp(0.0, 1.0)


def _downsample_video(video: Any, scale: float, mode: str = "bicubic") -> Any:
    torch = require_torch()
    height = max(1, int(round(video.shape[-2] / float(scale))))
    width = max(1, int(round(video.shape[-1] / float(scale))))
    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    lr = torch.nn.functional.interpolate(video.flatten(0, 1), size=(height, width), **kwargs)
    return lr.unflatten(0, (video.shape[0], video.shape[1])).clamp(0.0, 1.0)


def _apply_motion_blend(video: Any, strength: float) -> Any:
    strength = float(max(min(strength, 1.0), 0.0))
    if strength <= 0:
        return video
    previous = torch_cat_previous(video)
    return ((1.0 - strength) * video + strength * previous).clamp(0.0, 1.0)


def _apply_codec_proxy(video: Any, strength: float, block_size: int) -> Any:
    torch = require_torch()
    strength = float(max(min(strength, 1.0), 0.0))
    if strength <= 0:
        return video
    height, width = int(video.shape[-2]), int(video.shape[-1])
    block = max(2, min(int(block_size), height, width))
    pooled = torch.nn.functional.interpolate(
        video.flatten(0, 1),
        size=(max(1, height // block), max(1, width // block)),
        mode="area",
    )
    pixelated = torch.nn.functional.interpolate(
        pooled,
        size=(height, width),
        mode="nearest",
    ).unflatten(0, (video.shape[0], video.shape[1]))
    return ((1.0 - strength) * video + strength * pixelated).clamp(0.0, 1.0)


def torch_cat_previous(video: Any) -> Any:
    torch = require_torch()
    return torch.cat([video[:, :1], video[:, :-1]], dim=1)


def _degradation_label(profile: RealisticDegradationProfile, device: Any, dtype: Any) -> Any:
    torch = require_torch()
    blur = min(max(float(profile.blur_strength), 0.0), 1.0)
    noise = min(max(float(profile.noise_std) / 0.10, 0.0), 1.0)
    codec = min(max(float(profile.codec_strength), 0.0), 1.0)
    motion = min(max(float(profile.motion_strength), 0.0), 1.0)
    scale = min(max(float(profile.scale) / 8.0, 0.0), 1.0)
    severity = (blur + noise + codec + motion) / 4.0
    exposure = min(max((float(profile.exposure) - 0.5) / 1.5, 0.0), 1.0)
    return torch.tensor([blur, noise, codec, motion, scale, severity, exposure, 0.0], device=device, dtype=dtype)


def _read_training_frame(path: str | Path) -> Any:
    torch = require_torch()
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("imageio is required to read training frames") from exc

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


def _to_uint8_hwc(image: Any) -> Any:
    torch = require_torch()
    tensor = torch.as_tensor(image).detach().float().cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected C,H,W or H,W image, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    elif tensor.shape[0] >= 3:
        tensor = tensor[:3]
    else:
        raise ValueError(f"Expected 1 or 3+ channels, got {tensor.shape[0]}")
    return (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()
