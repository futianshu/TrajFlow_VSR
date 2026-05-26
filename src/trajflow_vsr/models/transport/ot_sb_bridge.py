"""OT/SB soft trajectory bridge implementation."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.ops.sinkhorn import normalize_mass, pairwise_squared_distance, sinkhorn_plan
from trajflow_vsr.utils.torch_utils import require_torch_nn


class OTSBTrajectoryBridge:
    """Build a reliability-calibrated soft transport plan over evidence tokens."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch, nn = require_torch_nn()

        class _Bridge(nn.Module):
            def __init__(
                self,
                hidden_channels: int = 64,
                temperature: float = 0.2,
                sinkhorn_iterations: int = 12,
                unbalanced_floor: float = 0.05,
                bridge_steps: int = 3,
                spatial_radius: int = 4,
                temporal_radius: int = 2,
                use_unbalanced: bool = True,
                use_reliability: bool = True,
            ):
                super().__init__()
                self.query = nn.Linear(hidden_channels, hidden_channels)
                self.key = nn.Linear(hidden_channels, hidden_channels)
                self.value = nn.Linear(hidden_channels, hidden_channels)
                self.temperature = temperature
                self.sinkhorn_iterations = sinkhorn_iterations
                self.unbalanced_floor = unbalanced_floor
                self.bridge_steps = bridge_steps
                self.spatial_radius = spatial_radius
                self.temporal_radius = temporal_radius
                self.use_unbalanced = bool(use_unbalanced)
                self.use_reliability = bool(use_reliability)

            def forward(
                self,
                token_bundle: dict[str, Any],
                uncertainty: dict[str, Any],
                causal: bool = False,
            ) -> dict[str, Any]:
                tokens = token_bundle["tokens"]
                grid = token_bundle["feature_grid"]
                q = torch.nn.functional.normalize(self.query(tokens), dim=-1)
                k = torch.nn.functional.normalize(self.key(tokens), dim=-1)
                v = self.value(tokens)
                cost = pairwise_squared_distance(q, k)
                candidate_mask = _local_token_mask(
                    frames=grid.shape[1],
                    height=grid.shape[2],
                    width=grid.shape[3],
                    spatial_radius=self.spatial_radius,
                    temporal_radius=self.temporal_radius,
                    device=grid.device,
                )
                causal_mask = None
                if causal:
                    causal_mask = _causal_token_mask(
                        frames=grid.shape[1],
                        height=grid.shape[2],
                        width=grid.shape[3],
                        device=grid.device,
                    )
                    candidate_mask = candidate_mask & causal_mask
                cost = cost.masked_fill(~candidate_mask.unsqueeze(0), cost.detach().amax() + 1e4)

                reliability = uncertainty.get("reliability")
                reliability_grid = None
                if reliability is not None and self.use_reliability:
                    mass = reliability.flatten(1).to(dtype=tokens.dtype)
                    reliability_grid = reliability.permute(0, 1, 3, 4, 2).to(dtype=grid.dtype)
                    if self.use_unbalanced:
                        unmatched_mass = normalize_mass((1.0 - mass).clamp_min(0.0) + float(self.unbalanced_floor))
                        mass = mass + float(self.unbalanced_floor)
                    else:
                        unmatched_mass = mass.new_zeros(mass.shape)
                    source_mass = normalize_mass(mass)
                    target_mass = normalize_mass(mass)
                else:
                    batch, token_count, _ = tokens.shape
                    source_mass = tokens.new_full((batch, token_count), 1.0 / token_count)
                    target_mass = source_mass
                    unmatched_mass = source_mass.new_zeros(source_mass.shape)

                ot_plan = sinkhorn_plan(
                    cost,
                    source_mass=source_mass,
                    target_mass=target_mass,
                    epsilon=self.temperature,
                    iterations=self.sinkhorn_iterations,
                )
                causal_violation = ot_plan.new_zeros(())
                if causal_mask is not None:
                    future_mass = ot_plan.masked_fill(causal_mask.unsqueeze(0), 0.0).sum()
                    causal_violation = future_mass / ot_plan.sum().clamp_min(1e-8)
                    ot_plan = ot_plan * causal_mask.unsqueeze(0).to(dtype=ot_plan.dtype)

                plan = ot_plan / ot_plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                transported = plan @ v
                transported_grid = transported.reshape(*grid.shape)
                bridge_times = torch.linspace(
                    0.0,
                    1.0,
                    steps=max(int(self.bridge_steps), 1) + 2,
                    device=grid.device,
                    dtype=grid.dtype,
                )[1:-1]
                time_view = bridge_times.view(1, -1, 1, 1, 1, 1)
                source_grid = grid
                target_grid = transported_grid
                bridge_states = (1.0 - time_view) * source_grid.unsqueeze(1) + time_view * target_grid.unsqueeze(1)
                bridge_diffusion = torch.sqrt((bridge_times * (1.0 - bridge_times)).clamp_min(0.0))
                if reliability_grid is None:
                    bridge_weight = grid.new_full((*grid.shape[:-1], 1), 0.5)
                else:
                    bridge_weight = (1.0 - reliability_grid.clamp(0.0, 1.0)).to(dtype=grid.dtype)
                bridge_grid = (1.0 - bridge_weight) * grid + bridge_weight * transported_grid
                return {
                    "cost": cost,
                    "ot_plan": ot_plan,
                    "transport_plan": plan,
                    "causal": causal,
                    "causal_mask": causal_mask,
                    "candidate_mask": candidate_mask,
                    "causal_violation": causal_violation,
                    "source_mass": source_mass,
                    "target_mass": target_mass,
                    "unmatched_mass": unmatched_mass,
                    "occlusion_mass": unmatched_mass.mean(),
                    "row_marginal_error": (ot_plan.sum(dim=-1) - source_mass).abs().mean(),
                    "column_marginal_error": (ot_plan.sum(dim=-2) - target_mass).abs().mean(),
                    "transported_tokens": transported,
                    "transported_grid": transported_grid,
                    "source_grid": source_grid,
                    "target_grid": target_grid,
                    "bridge_grid": bridge_grid,
                    "bridge_weight": bridge_weight,
                    "bridge_states": bridge_states,
                    "bridge_times": bridge_times,
                    "bridge_drift": target_grid - source_grid,
                    "bridge_diffusion": bridge_diffusion,
                }

        return _Bridge(*args, **kwargs)


def _causal_token_mask(frames: int, height: int, width: int, device: Any):
    torch, _ = require_torch_nn()
    frame_index = torch.arange(frames, device=device).repeat_interleave(height * width)
    return frame_index[None, :] <= frame_index[:, None]


def _local_token_mask(
    frames: int,
    height: int,
    width: int,
    spatial_radius: int,
    temporal_radius: int,
    device: Any,
):
    torch, _ = require_torch_nn()
    frame_index = torch.arange(frames, device=device).repeat_interleave(height * width)
    y_index = torch.arange(height, device=device).repeat_interleave(width).repeat(frames)
    x_index = torch.arange(width, device=device).repeat(frames * height)
    dt = (frame_index[:, None] - frame_index[None, :]).abs()
    dy = (y_index[:, None] - y_index[None, :]).abs()
    dx = (x_index[:, None] - x_index[None, :]).abs()
    return (dt <= int(temporal_radius)) & (dy <= int(spatial_radius)) & (dx <= int(spatial_radius))
