"""Wavelet frequency and anti-aliasing losses."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.ops.wavelet import split_video_low_high


def wavelet_frequency_loss(outputs: dict[str, Any], batch: dict[str, Any]):
    """Match low and high wavelet-style bands of predicted and target HR video."""

    prediction = outputs["hr"]
    target = batch["hr"]
    pred_low, pred_high = split_video_low_high(prediction)
    target_low, target_high = split_video_low_high(target)
    low_loss = (pred_low - target_low).abs().mean()
    high_loss = (pred_high - target_high).abs().mean()
    return low_loss + 0.5 * high_loss


def anti_aliasing_loss(outputs: dict[str, Any]):
    """Penalize unsuppressed high-frequency energy in the decoder residual band."""

    decoded = outputs["decoded"]
    high = decoded["wavelet_high"]
    gate = decoded["anti_alias_gate"].to(dtype=high.dtype)
    return (high.abs() * gate).mean()
