"""Temporal consistency objectives for video restoration."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def temporal_consistency_loss(outputs: dict[str, Any], batch: dict[str, Any], epsilon: float = 1e-3):
    """Match adjacent-frame HR changes to upsampled LR evidence changes."""

    torch = require_torch()
    prediction = outputs["hr"]
    lr_video = batch["lr"]
    if prediction.shape[1] < 2:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)

    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    lr_delta = lr_video[:, 1:] - lr_video[:, :-1]
    target_delta = torch.nn.functional.interpolate(
        lr_delta.flatten(0, 1),
        size=prediction.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).unflatten(0, lr_delta.shape[:2])
    residual = torch.sqrt((pred_delta - target_delta) ** 2 + epsilon**2)

    uncertainty = outputs.get("uncertainty", {})
    reliability = uncertainty.get("reliability") if isinstance(uncertainty, dict) else None
    if reliability is None:
        return residual.mean()

    adjacent_reliability = 0.5 * (reliability[:, 1:] + reliability[:, :-1])
    weights = torch.nn.functional.interpolate(
        adjacent_reliability.flatten(0, 1),
        size=prediction.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).unflatten(0, adjacent_reliability.shape[:2])
    weights = weights.to(dtype=residual.dtype).expand_as(residual)
    return (residual * weights).sum() / weights.sum().clamp_min(1.0)
