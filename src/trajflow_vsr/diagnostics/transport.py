"""Transport-plan diagnostics for trajectory aggregation."""

from __future__ import annotations

import math
from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def transport_plan_diagnostics(
    transport: dict[str, Any],
    *,
    frames: int,
    height: int,
    width: int,
    shift_x: int = 0,
    shift_y: int = 0,
    reliability: Any | None = None,
) -> dict[str, float]:
    """Summarize whether an OT/SB plan behaves like a useful trajectory map."""

    torch = require_torch()
    plan = transport.get("transport_plan")
    if plan is None:
        raise ValueError("transport must include transport_plan")
    plan = plan.detach().float()
    batch, source_count, target_count = plan.shape
    token_count = int(frames) * int(height) * int(width)
    if source_count != token_count or target_count != token_count:
        raise ValueError("transport_plan shape does not match frames*height*width")

    device = plan.device
    source_frame, source_y, source_x = _token_coordinates(frames, height, width, device=device)
    target_frame = source_frame
    target_y = source_y
    target_x = source_x

    dt = target_frame.view(1, 1, target_count) - source_frame.view(1, source_count, 1)
    dy = target_y.view(1, 1, target_count) - source_y.view(1, source_count, 1)
    dx = target_x.view(1, 1, target_count) - source_x.view(1, source_count, 1)
    same_frame = dt == 0
    future = dt > 0
    past = dt < 0
    diagonal = torch.eye(token_count, dtype=torch.bool, device=device).view(1, token_count, token_count)

    entropy = -(plan.clamp_min(1e-12) * plan.clamp_min(1e-12).log()).sum(dim=-1)
    candidate_mask = transport.get("candidate_mask")
    if candidate_mask is not None:
        candidate_count = candidate_mask.to(device=device).sum(dim=-1).float().clamp_min(2.0)
        normalized_entropy = entropy / candidate_count.log().view(1, token_count)
    else:
        normalized_entropy = entropy / math.log(float(token_count))

    row_top_mass, row_top_index = plan.max(dim=-1)
    top_dt = target_frame[row_top_index] - source_frame.view(1, token_count)
    top_dy = target_y[row_top_index] - source_y.view(1, token_count)
    top_dx = target_x[row_top_index] - source_x.view(1, token_count)
    oracle_mask = _oracle_mask(
        frames=frames,
        height=height,
        width=width,
        shift_x=int(shift_x),
        shift_y=int(shift_y),
        device=device,
    ).view(1, token_count, token_count)
    oracle_cross = oracle_mask & ~same_frame
    top_oracle = oracle_mask.expand(batch, -1, -1).gather(dim=-1, index=row_top_index.unsqueeze(-1)).squeeze(-1)

    diagnostics = {
        "plan_entropy": _mean(entropy),
        "plan_entropy_normalized": _mean(normalized_entropy),
        "plan_top1_mass": _mean(row_top_mass),
        "plan_diagonal_mass": _mean((plan * diagonal).sum(dim=-1)),
        "plan_same_frame_mass": _mean((plan * same_frame).sum(dim=-1)),
        "plan_cross_frame_mass": _mean((plan * (~same_frame)).sum(dim=-1)),
        "plan_future_mass": _mean((plan * future).sum(dim=-1)),
        "plan_past_mass": _mean((plan * past).sum(dim=-1)),
        "plan_oracle_mass": _mean((plan * oracle_mask).sum(dim=-1)),
        "plan_oracle_cross_frame_mass": _mean((plan * oracle_cross).sum(dim=-1)),
        "plan_top1_oracle_accuracy": _mean(top_oracle.float()),
        "plan_expected_dt": _mean((plan * dt.float()).sum(dim=-1)),
        "plan_expected_dx": _mean((plan * dx.float()).sum(dim=-1)),
        "plan_expected_dy": _mean((plan * dy.float()).sum(dim=-1)),
        "plan_abs_expected_dx": _mean((plan * dx.float()).sum(dim=-1).abs()),
        "plan_abs_expected_dy": _mean((plan * dy.float()).sum(dim=-1).abs()),
        "plan_top1_abs_dt": _mean(top_dt.float().abs()),
        "plan_top1_abs_dx": _mean(top_dx.float().abs()),
        "plan_top1_abs_dy": _mean(top_dy.float().abs()),
        "batch_size": float(batch),
        "token_count": float(token_count),
    }
    diagnostics.update(_grid_change_diagnostics(transport))
    diagnostics.update(_reliability_diagnostics(reliability, entropy, transport))
    return diagnostics


def _token_coordinates(frames: int, height: int, width: int, device: Any):
    torch = require_torch()
    frame = torch.arange(frames, device=device).repeat_interleave(height * width)
    y = torch.arange(height, device=device).repeat_interleave(width).repeat(frames)
    x = torch.arange(width, device=device).repeat(frames * height)
    return frame, y, x


def _oracle_mask(
    *,
    frames: int,
    height: int,
    width: int,
    shift_x: int,
    shift_y: int,
    device: Any,
):
    source_frame, source_y, source_x = _token_coordinates(frames, height, width, device=device)
    target_frame, target_y, target_x = source_frame, source_y, source_x
    dt = target_frame.view(1, -1) - source_frame.view(-1, 1)
    expected_y = (source_y.view(-1, 1) + dt * int(shift_y)).remainder(int(height))
    expected_x = (source_x.view(-1, 1) + dt * int(shift_x)).remainder(int(width))
    return (target_y.view(1, -1) == expected_y) & (target_x.view(1, -1) == expected_x)


def _grid_change_diagnostics(transport: dict[str, Any]) -> dict[str, float]:
    source = transport.get("source_grid")
    transported = transport.get("transported_grid")
    bridge = transport.get("bridge_grid")
    output: dict[str, float] = {}
    if source is not None and transported is not None:
        output["transported_l2_ratio"] = _relative_l2(transported.detach().float(), source.detach().float())
    if source is not None and bridge is not None:
        output["bridge_l2_ratio"] = _relative_l2(bridge.detach().float(), source.detach().float())
    bridge_weight = transport.get("bridge_weight")
    if bridge_weight is not None:
        output["bridge_weight_mean"] = _mean(bridge_weight.detach().float())
        output["bridge_weight_max"] = float(bridge_weight.detach().float().max().cpu())
    return output


def _reliability_diagnostics(
    reliability: Any | None,
    entropy: Any,
    transport: dict[str, Any],
) -> dict[str, float]:
    if reliability is None:
        return {}
    reliability_flat = reliability.detach().float().flatten(1)
    output = {
        "reliability_mean": _mean(reliability_flat),
        "reliability_entropy_corr": _corr(reliability_flat, entropy.detach().float()),
    }
    bridge_weight = transport.get("bridge_weight")
    if bridge_weight is not None:
        output["reliability_bridge_weight_corr"] = _corr(
            reliability_flat,
            bridge_weight.detach().float().flatten(1),
        )
    return output


def _relative_l2(value: Any, reference: Any) -> float:
    return float((value - reference).square().mean().sqrt().div(reference.square().mean().sqrt().clamp_min(1e-8)).cpu())


def _corr(left: Any, right: Any) -> float:
    left = left.reshape(-1)
    right = right.reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denom = left.square().mean().sqrt() * right.square().mean().sqrt()
    if float(denom.cpu()) <= 1e-12:
        return 0.0
    return float((left * right).mean().div(denom).cpu())


def _mean(value: Any) -> float:
    return float(value.detach().float().mean().cpu())
