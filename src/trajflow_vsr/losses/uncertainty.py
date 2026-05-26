"""Uncertainty and reliability calibration losses."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def uncertainty_calibration_loss(outputs: dict[str, Any], batch: dict[str, Any]):
    """Encourage predicted reliability to match reconstruction confidence."""

    torch = require_torch()
    prediction = outputs["hr"]
    target = batch["hr"]
    uncertainty = outputs.get("uncertainty", {})
    reliability = uncertainty.get("reliability") if isinstance(uncertainty, dict) else None
    if reliability is None:
        return prediction.new_zeros(())
    error = (prediction - target).abs().mean(dim=2, keepdim=True).detach().clamp(0.0, 1.0)
    confidence_target = (1.0 - error).clamp(0.0, 1.0)
    confidence = torch.nn.functional.interpolate(
        reliability.flatten(0, 1),
        size=prediction.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).unflatten(0, prediction.shape[:2])
    return (confidence.clamp(0.0, 1.0) - confidence_target).square().mean()
