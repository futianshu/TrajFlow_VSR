"""Basic reconstruction and regularization losses."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def charbonnier_loss(prediction: Any, target: Any, epsilon: float = 1e-3):
    """Robust reconstruction loss."""

    torch = require_torch()
    return torch.sqrt((prediction - target) ** 2 + epsilon**2).mean()


def koopman_dynamics_loss(memory: dict[str, Any]):
    """MSE loss for the current placeholder Koopman prediction head."""

    torch = require_torch()
    prediction = memory.get("koopman_prediction")
    target = memory.get("koopman_target")
    if prediction is None or target is None or prediction.numel() == 0:
        return torch.zeros((), device=next(iter(memory.values())).device)
    return torch.nn.functional.mse_loss(prediction, target.detach())


def data_consistency_loss(outputs: dict[str, Any], batch: dict[str, Any], epsilon: float = 1e-3):
    """Downsample HR predictions and compare them with LR evidence."""

    torch = require_torch()
    prediction = outputs["hr"]
    target_lr = batch["lr"]
    downsampled = torch.nn.functional.interpolate(
        prediction.flatten(0, 1),
        size=target_lr.shape[-2:],
        mode="area",
    ).unflatten(0, target_lr.shape[:2])
    residual = torch.sqrt((downsampled - target_lr) ** 2 + epsilon**2)

    uncertainty = outputs.get("uncertainty", {})
    reliability = uncertainty.get("reliability") if isinstance(uncertainty, dict) else None
    if reliability is None:
        return residual.mean()

    weights = reliability.to(dtype=residual.dtype).expand_as(residual)
    return (residual * weights).sum() / weights.sum().clamp_min(1.0)
