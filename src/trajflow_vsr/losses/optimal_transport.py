"""Optimal-transport regularization losses."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def optimal_transport_loss(transport: dict[str, Any]):
    """Penalize high-cost transport and residual marginal mismatch."""

    torch = require_torch()
    plan = transport["ot_plan"]
    cost = transport["cost"]
    transport_cost = (plan * cost).sum(dim=(-1, -2)).mean()
    row_error = transport.get("row_marginal_error")
    column_error = transport.get("column_marginal_error")
    marginal_error = torch.zeros((), device=plan.device, dtype=plan.dtype)
    if row_error is not None:
        marginal_error = marginal_error + row_error
    if column_error is not None:
        marginal_error = marginal_error + column_error
    return transport_cost + marginal_error


def transport_entropy_loss(transport: dict[str, Any]):
    """Encourage row-wise transport concentration without changing marginals."""

    torch = require_torch()
    plan = transport["transport_plan"].clamp_min(1e-12)
    entropy = -(plan * plan.log()).sum(dim=-1)
    candidate_mask = transport.get("candidate_mask")
    if candidate_mask is not None:
        candidate_count = candidate_mask.to(device=plan.device).sum(dim=-1).to(dtype=plan.dtype).clamp_min(2.0)
        normalizer = candidate_count.log().view(1, -1)
    else:
        normalizer = torch.log(plan.new_tensor(float(plan.shape[-1])).clamp_min(2.0))
    return (entropy / normalizer.clamp_min(1e-8)).mean()


def motion_supervised_transport_loss(transport: dict[str, Any], batch: dict[str, Any]):
    """Cross-entropy to a known controlled-motion oracle trajectory, when available."""

    motion = batch.get("controlled_motion")
    if not motion:
        plan = transport["transport_plan"]
        return plan.new_zeros(())
    source_grid = transport.get("source_grid")
    if source_grid is None:
        plan = transport["transport_plan"]
        return plan.new_zeros(())
    _, frames, height, width, _ = source_grid.shape
    oracle = _controlled_motion_oracle_mask(
        frames=int(frames),
        height=int(height),
        width=int(width),
        shift_x=int(motion.get("shift_x", 0)),
        shift_y=int(motion.get("shift_y", 0)),
        device=source_grid.device,
    )
    candidate_mask = transport.get("candidate_mask")
    if candidate_mask is not None:
        oracle = oracle & candidate_mask.to(device=oracle.device)
    oracle = _ensure_nonempty_oracle_rows(oracle, frames=int(frames), height=int(height), width=int(width))
    target = oracle.to(dtype=transport["transport_plan"].dtype)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    plan = transport["transport_plan"].clamp_min(1e-12)
    return -(target.unsqueeze(0) * plan.log()).sum(dim=-1).mean()


def _controlled_motion_oracle_mask(
    *,
    frames: int,
    height: int,
    width: int,
    shift_x: int,
    shift_y: int,
    device: Any,
):
    frame, y_index, x_index = _token_coordinates(frames, height, width, device=device)
    dt = frame.view(1, -1) - frame.view(-1, 1)
    expected_y = (y_index.view(-1, 1) + dt * int(shift_y)).remainder(int(height))
    expected_x = (x_index.view(-1, 1) + dt * int(shift_x)).remainder(int(width))
    return (y_index.view(1, -1) == expected_y) & (x_index.view(1, -1) == expected_x)


def _ensure_nonempty_oracle_rows(oracle, *, frames: int, height: int, width: int):
    torch = require_torch()
    row_has_target = oracle.any(dim=-1)
    if bool(row_has_target.all().detach().cpu()):
        return oracle
    token_count = int(frames) * int(height) * int(width)
    identity = torch.eye(token_count, dtype=torch.bool, device=oracle.device)
    return oracle | ((~row_has_target).unsqueeze(-1) & identity)


def _token_coordinates(frames: int, height: int, width: int, device: Any):
    torch = require_torch()
    frame = torch.arange(frames, device=device).repeat_interleave(height * width)
    y_index = torch.arange(height, device=device).repeat_interleave(width).repeat(frames)
    x_index = torch.arange(width, device=device).repeat(frames * height)
    return frame, y_index, x_index
