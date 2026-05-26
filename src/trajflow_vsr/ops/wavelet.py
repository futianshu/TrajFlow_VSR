"""Lightweight wavelet-style low/high-frequency operators."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def split_low_high_2d(image: Any, min_low_size: int = 1) -> tuple[Any, Any]:
    """Split an image batch into smooth low band and residual high band."""

    torch = require_torch()
    if image.ndim != 4:
        raise ValueError("Expected image shape N,C,H,W")

    height, width = image.shape[-2:]
    low_height = max(int(min_low_size), height // 2)
    low_width = max(int(min_low_size), width // 2)
    low = torch.nn.functional.interpolate(
        image,
        size=(low_height, low_width),
        mode="area",
    )
    low = torch.nn.functional.interpolate(
        low,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    high = image - low
    return low, high


def split_video_low_high(video: Any) -> tuple[Any, Any]:
    """Apply low/high split to B,T,C,H,W video tensors."""

    if video.ndim != 5:
        raise ValueError("Expected video shape B,T,C,H,W")
    batch, frames = video.shape[:2]
    low, high = split_low_high_2d(video.flatten(0, 1))
    return low.unflatten(0, (batch, frames)), high.unflatten(0, (batch, frames))
