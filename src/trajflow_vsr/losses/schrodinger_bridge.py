"""Schrodinger bridge path regularization."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def schrodinger_bridge_loss(
    transport: dict[str, Any],
    curvature_weight: float = 0.25,
    diffusion_weight: float = 0.05,
):
    """Regularize bridge paths between degraded and transported token states."""

    torch = require_torch()
    source = transport["source_grid"]
    target = transport["target_grid"]
    states = transport["bridge_states"]
    times = transport["bridge_times"]
    drift = target - source

    kinetic = drift.square().mean()
    time_view = times.view(1, -1, 1, 1, 1, 1)
    linear_states = (1.0 - time_view) * source.unsqueeze(1) + time_view * target.unsqueeze(1)
    curvature = (states - linear_states.detach()).square().mean()

    expected_diffusion = torch.sqrt((times * (1.0 - times)).clamp_min(0.0))
    observed_step = torch.zeros((), device=states.device, dtype=states.dtype)
    if states.shape[1] > 1:
        observed_step = (states[:, 1:] - states[:, :-1]).square().mean().sqrt()
    diffusion_scale = expected_diffusion.mean().to(dtype=states.dtype)
    diffusion = (observed_step - diffusion_scale * drift.square().mean().sqrt().detach()).square()

    return kinetic + float(curvature_weight) * curvature + float(diffusion_weight) * diffusion
