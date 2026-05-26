"""Conditional rectified-flow matching objectives."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def rectified_flow_matching_loss(outputs: dict[str, Any]):
    """Match predicted latent vector fields to bridge-conditioned residual transport."""

    velocity = outputs["residual"]["flow_velocity"]
    target_velocity = outputs["residual"]["flow_target_velocity"].detach()
    residual = (velocity - target_velocity).square()
    gate = outputs["residual"].get("residual_gate")
    if gate is not None:
        weights = gate.to(dtype=residual.dtype).expand_as(residual)
        return (residual * weights).sum() / weights.sum().clamp_min(1.0)

    uncertainty = outputs.get("uncertainty", {})
    reliability = uncertainty.get("reliability") if isinstance(uncertainty, dict) else None
    if reliability is None:
        return residual.mean()

    weights = reliability.permute(0, 1, 3, 4, 2).to(dtype=residual.dtype).expand_as(residual)
    return (residual * weights).sum() / weights.sum().clamp_min(1.0)


def bridge_residual_consistency_loss(outputs: dict[str, Any]):
    """Keep the generated residual close to the SB target residual at tau=1."""

    torch = require_torch()
    residual = outputs["residual"]["residual_grid"]
    target = outputs["residual"]["flow_target_residual"].detach()
    tau = outputs["residual"].get("tau")
    if tau is None:
        weight = torch.ones_like(residual[..., :1])
    else:
        weight = tau.view(residual.shape[0], 1, 1, 1, 1).to(dtype=residual.dtype)
    gate = outputs["residual"].get("residual_gate")
    if gate is not None:
        weight = weight * gate.to(dtype=residual.dtype)
    weight_sum = weight.expand_as(residual).sum().clamp_min(1.0)
    return ((residual - target).abs() * weight).sum() / weight_sum


def residual_amplitude_loss(outputs: dict[str, Any]):
    """Penalize residual energy after reliability/uncertainty gating."""

    residual = outputs["residual"].get("residual_grid")
    if residual is None:
        return require_torch().zeros(())
    return residual.abs().mean()


def residual_low_frequency_loss(outputs: dict[str, Any]):
    """Penalize low-frequency content removed from the flow residual path."""

    residual_low = outputs["residual"].get("residual_low_band")
    if residual_low is None:
        return require_torch().zeros(())
    return residual_low.abs().mean()
