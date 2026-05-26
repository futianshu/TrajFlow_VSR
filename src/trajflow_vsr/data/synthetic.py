"""Synthetic data utilities for smoke tests and interface checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


@dataclass(frozen=True)
class SyntheticVideoSpec:
    """Shape and scale settings for a synthetic VSR batch."""

    batch_size: int = 1
    frames: int = 3
    channels: int = 3
    height: int = 24
    width: int = 24
    scale: float = 2.0


@dataclass(frozen=True)
class SyntheticDegradationSpec:
    """Synthetic real-world degradation ranges for Stage A pretraining."""

    blur_max: float = 0.75
    noise_max: float = 0.08
    codec_max: float = 0.65
    motion_max: float = 0.45
    exposure_min: float = 0.75
    exposure_max: float = 1.25
    block_size: int = 4


def spec_from_config(config: dict[str, Any]) -> SyntheticVideoSpec:
    """Create a synthetic batch spec from a config mapping."""

    return SyntheticVideoSpec(
        batch_size=int(config.get("batch_size", 1)),
        frames=int(config.get("frames", 3)),
        channels=int(config.get("channels", 3)),
        height=int(config.get("height", 24)),
        width=int(config.get("width", 24)),
        scale=float(config.get("scale", 2.0)),
    )


def degradation_spec_from_config(config: dict[str, Any]) -> SyntheticDegradationSpec:
    """Create a synthetic degradation spec from a config mapping."""

    degradation = config.get("degradation", {})
    return SyntheticDegradationSpec(
        blur_max=float(degradation.get("blur_max", 0.75)),
        noise_max=float(degradation.get("noise_max", 0.08)),
        codec_max=float(degradation.get("codec_max", 0.65)),
        motion_max=float(degradation.get("motion_max", 0.45)),
        exposure_min=float(degradation.get("exposure_min", 0.75)),
        exposure_max=float(degradation.get("exposure_max", 1.25)),
        block_size=int(degradation.get("block_size", 4)),
    )


def make_synthetic_batch(spec: SyntheticVideoSpec, device: str = "cpu") -> dict[str, Any]:
    """Create a tiny paired LR/HR video batch.

    Tensor layout follows the whole project convention: B, T, C, H, W.
    """

    torch = require_torch()
    lr = torch.rand(
        spec.batch_size,
        spec.frames,
        spec.channels,
        spec.height,
        spec.width,
        device=device,
    )
    hr = torch.nn.functional.interpolate(
        lr.flatten(0, 1),
        scale_factor=spec.scale,
        mode="bilinear",
        align_corners=False,
    ).unflatten(0, (spec.batch_size, spec.frames))
    return {"lr": lr, "hr": hr, "scale": spec.scale}


def make_controlled_motion_batch(
    spec: SyntheticVideoSpec,
    motion: dict[str, Any] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Create a paired LR/HR video with a known integer translation trajectory."""

    torch = require_torch()
    motion = motion or {}
    shift_x = int(motion.get("shift_x", 1))
    shift_y = int(motion.get("shift_y", 0))
    noise = float(motion.get("noise", 0.0))
    occlusion_size = max(int(motion.get("occlusion_size", 0)), 0)
    wrap = bool(motion.get("wrap", True))

    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, spec.height, device=device),
        torch.linspace(0.0, 1.0, spec.width, device=device),
        indexing="ij",
    )
    videos = []
    for batch_idx in range(spec.batch_size):
        phase = 0.37 * float(batch_idx + 1)
        base_channels = []
        for channel_idx in range(spec.channels):
            frequency = float(channel_idx + 2)
            wave = torch.sin(6.283185307179586 * (frequency * xx + 0.71 * frequency * yy + phase))
            cross = torch.cos(6.283185307179586 * (0.43 * frequency * xx - frequency * yy - phase))
            blob = torch.exp(
                -((xx - 0.28 - 0.08 * channel_idx) ** 2 + (yy - 0.62 + 0.04 * channel_idx) ** 2) / 0.015
            )
            checker = (((xx * spec.width).floor() + (yy * spec.height).floor()) % 2.0) * 0.2
            channel = 0.45 + 0.2 * wave + 0.15 * cross + 0.25 * blob + checker
            base_channels.append(channel)
        base = torch.stack(base_channels, dim=0).clamp(0.0, 1.0)

        frames = []
        for frame_idx in range(spec.frames):
            dy = int(frame_idx * shift_y)
            dx = int(frame_idx * shift_x)
            if wrap:
                frame = torch.roll(base, shifts=(dy, dx), dims=(-2, -1))
            else:
                frame = _shift_with_padding(base, shift_y=dy, shift_x=dx)
            if occlusion_size > 0:
                frame = _apply_moving_occlusion(frame, frame_idx, occlusion_size=occlusion_size)
            if noise > 0.0:
                frame = frame + torch.randn_like(frame) * noise
            frames.append(frame.clamp(0.0, 1.0))
        videos.append(torch.stack(frames, dim=0))

    lr = torch.stack(videos, dim=0)
    hr = torch.nn.functional.interpolate(
        lr.flatten(0, 1),
        scale_factor=spec.scale,
        mode="bilinear",
        align_corners=False,
    ).unflatten(0, (spec.batch_size, spec.frames))
    metadata = [
        {
            "controlled_motion": True,
            "shift_x": shift_x,
            "shift_y": shift_y,
            "wrap": wrap,
            "noise": noise,
            "occlusion_size": occlusion_size,
        }
        for _ in range(spec.batch_size)
    ]
    return {
        "lr": lr,
        "hr": hr,
        "scale": spec.scale,
        "metadata": metadata,
        "controlled_motion": {
            "shift_x": shift_x,
            "shift_y": shift_y,
            "wrap": wrap,
        },
    }


def make_stage_a_batch(
    spec: SyntheticVideoSpec,
    degradation: SyntheticDegradationSpec | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Create a paired clean/degraded batch with synthetic degradation labels."""

    torch = require_torch()
    degradation = degradation or SyntheticDegradationSpec()
    clean_hr = _make_clean_hr_video(spec, device=device)
    clean_lr = torch.nn.functional.interpolate(
        clean_hr.flatten(0, 1),
        size=(spec.height, spec.width),
        mode="bilinear",
        align_corners=False,
    ).unflatten(0, (spec.batch_size, spec.frames))

    degraded_lr, labels = _apply_synthetic_degradation(clean_lr, spec, degradation)
    artifact = (degraded_lr - clean_lr).abs().mean(dim=2, keepdim=True)
    artifact = (artifact / artifact.amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)).clamp(0.0, 1.0)
    reliability = torch.exp(-4.0 * artifact).clamp(0.0, 1.0)

    return {
        "lr": degraded_lr.clamp(0.0, 1.0),
        "hr": clean_hr.clamp(0.0, 1.0),
        "clean_lr": clean_lr.clamp(0.0, 1.0),
        "artifact": artifact,
        "reliability": reliability,
        "degradation": labels,
        "scale": spec.scale,
    }


def _shift_with_padding(frame, shift_y: int, shift_x: int):
    shifted = frame.new_zeros(frame.shape)
    height, width = frame.shape[-2:]
    source_y0 = max(-int(shift_y), 0)
    source_y1 = min(height - int(shift_y), height)
    source_x0 = max(-int(shift_x), 0)
    source_x1 = min(width - int(shift_x), width)
    target_y0 = source_y0 + int(shift_y)
    target_y1 = source_y1 + int(shift_y)
    target_x0 = source_x0 + int(shift_x)
    target_x1 = source_x1 + int(shift_x)
    if source_y1 > source_y0 and source_x1 > source_x0:
        shifted[..., target_y0:target_y1, target_x0:target_x1] = frame[..., source_y0:source_y1, source_x0:source_x1]
    return shifted.to(dtype=frame.dtype, device=frame.device)


def _apply_moving_occlusion(frame, frame_idx: int, occlusion_size: int):
    height, width = frame.shape[-2:]
    size = min(int(occlusion_size), height, width)
    if size <= 0:
        return frame
    y0 = min(max((height // 3) + frame_idx, 0), height - size)
    x0 = min(max((width // 4) + 2 * frame_idx, 0), width - size)
    occluded = frame.clone()
    occluded[..., y0 : y0 + size, x0 : x0 + size] = 0.0
    return occluded


def _make_clean_hr_video(spec: SyntheticVideoSpec, device: str):
    torch = require_torch()
    height = int(round(spec.height * spec.scale))
    width = int(round(spec.width * spec.scale))
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device),
        torch.linspace(0.0, 1.0, width, device=device),
        indexing="ij",
    )

    videos = []
    for _batch_idx in range(spec.batch_size):
        channel_phase = torch.rand(spec.channels, device=device) * 6.283185307179586
        channel_freq = 1.0 + 3.0 * torch.rand(spec.channels, device=device)
        frames = []
        for frame_idx in range(spec.frames):
            t = frame_idx / max(spec.frames - 1, 1)
            drift = 0.08 * t
            channels = []
            for channel_idx in range(spec.channels):
                wave = torch.sin(
                    6.283185307179586
                    * (
                        channel_freq[channel_idx] * (xx + drift)
                        + (channel_freq[channel_idx] * 0.67) * (yy - drift)
                    )
                    + channel_phase[channel_idx]
                )
                blob = torch.exp(-((xx - 0.35 - drift) ** 2 + (yy - 0.55 + 0.5 * drift) ** 2) / 0.025)
                channel = 0.5 + 0.25 * wave + 0.25 * blob
                channels.append(channel)
            frames.append(torch.stack(channels, dim=0))
        videos.append(torch.stack(frames, dim=0))
    return torch.stack(videos, dim=0).clamp(0.0, 1.0)


def _apply_synthetic_degradation(
    clean_lr,
    spec: SyntheticVideoSpec,
    degradation: SyntheticDegradationSpec,
):
    torch = require_torch()
    batch_size = clean_lr.shape[0]
    blur = torch.rand(batch_size, device=clean_lr.device) * degradation.blur_max
    noise = torch.rand(batch_size, device=clean_lr.device) * degradation.noise_max
    codec = torch.rand(batch_size, device=clean_lr.device) * degradation.codec_max
    motion = torch.rand(batch_size, device=clean_lr.device) * degradation.motion_max
    exposure = degradation.exposure_min + torch.rand(batch_size, device=clean_lr.device) * (
        degradation.exposure_max - degradation.exposure_min
    )

    x = clean_lr
    blur_view = blur.view(batch_size, 1, 1, 1, 1)
    blurred = torch.nn.functional.avg_pool2d(
        x.flatten(0, 1),
        kernel_size=3,
        stride=1,
        padding=1,
    ).unflatten(0, (spec.batch_size, spec.frames))
    x = (1.0 - blur_view) * x + blur_view * blurred

    previous = torch.cat([x[:, :1], x[:, :-1]], dim=1)
    motion_view = motion.view(batch_size, 1, 1, 1, 1)
    x = (1.0 - motion_view) * x + motion_view * previous

    block = max(2, min(degradation.block_size, spec.height, spec.width))
    pooled = torch.nn.functional.interpolate(
        x.flatten(0, 1),
        size=(max(1, spec.height // block), max(1, spec.width // block)),
        mode="area",
    )
    pixelated = torch.nn.functional.interpolate(
        pooled,
        size=(spec.height, spec.width),
        mode="nearest",
    ).unflatten(0, (spec.batch_size, spec.frames))
    codec_view = codec.view(batch_size, 1, 1, 1, 1)
    x = (1.0 - codec_view) * x + codec_view * pixelated

    x = x * exposure.view(batch_size, 1, 1, 1, 1)
    x = x + torch.randn_like(x) * noise.view(batch_size, 1, 1, 1, 1)
    blur_label = blur / max(degradation.blur_max, 1e-6)
    noise_label = noise / max(degradation.noise_max, 1e-6)
    codec_label = codec / max(degradation.codec_max, 1e-6)
    motion_label = motion / max(degradation.motion_max, 1e-6)
    severity = (blur_label + noise_label + codec_label + motion_label) / 4.0
    labels = torch.stack(
        [
            blur_label,
            noise_label,
            codec_label,
            motion_label,
            torch.full_like(blur, float(spec.scale) / 8.0),
            severity.clamp(0.0, 1.0),
            ((exposure - degradation.exposure_min) / max(degradation.exposure_max - degradation.exposure_min, 1e-6)),
            torch.zeros_like(blur),
        ],
        dim=1,
    ).clamp(0.0, 1.0)
    return x, labels
