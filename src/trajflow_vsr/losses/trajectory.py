"""Regularizers for soft token trajectories."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def trajectory_regularization_loss(
    transport: dict[str, Any],
    temporal_window: int = 1,
    max_same_frame_mass: float = 0.6,
    entropy_weight: float = 0.01,
):
    """Encourage local, non-degenerate soft trajectories over the video grid."""

    torch = require_torch()
    plan = transport["transport_plan"]
    grid = transport["bridge_grid"]
    _, frames, height, width, _ = grid.shape
    if frames < 2:
        return torch.zeros((), device=plan.device, dtype=plan.dtype)

    frame_index, y_index, x_index = _token_coordinates(frames, height, width, device=plan.device, dtype=plan.dtype)
    frame_distance = (frame_index[:, None] - frame_index[None, :]).abs()
    y_distance = y_index[:, None] - y_index[None, :]
    x_distance = x_index[:, None] - x_index[None, :]

    outside_window = (frame_distance - float(temporal_window)).clamp_min(0.0)
    outside_window = outside_window / max(float(frames - 1), 1.0)
    spatial_distance = y_distance.square() + x_distance.square()
    locality = outside_window + 0.25 * spatial_distance

    plan_mass = plan.sum(dim=(-1, -2)).clamp_min(1e-8)
    locality_loss = (plan * locality).sum(dim=(-1, -2)) / plan_mass

    same_frame = (frame_distance == 0).to(dtype=plan.dtype)
    same_frame_mass = (plan * same_frame).sum(dim=(-1, -2)) / plan_mass
    same_frame_loss = (same_frame_mass - float(max_same_frame_mass)).clamp_min(0.0)

    normalized_entropy = -(plan.clamp_min(1e-8) * plan.clamp_min(1e-8).log()).sum(dim=-1)
    normalized_entropy = normalized_entropy / max(float(plan.shape[-1]), 1.0)
    entropy_loss = normalized_entropy.mean(dim=-1)

    return (locality_loss + same_frame_loss + float(entropy_weight) * entropy_loss).mean()


def _token_coordinates(frames: int, height: int, width: int, device: Any, dtype: Any):
    torch = require_torch()
    frame = torch.arange(frames, device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    tt, yy, xx = torch.meshgrid(frame, y, x, indexing="ij")
    return tt.flatten(), yy.flatten(), xx.flatten()
