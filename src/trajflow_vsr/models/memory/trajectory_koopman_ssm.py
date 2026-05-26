"""Trajectory-conditioned selective state-space and Koopman memory."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch_nn


class TrajectoryKoopmanSSMMemory:
    """Aggregate long-context evidence along soft trajectory-conditioned sequences."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch, nn = require_torch_nn()

        class _Memory(nn.Module):
            def __init__(
                self,
                hidden_channels: int = 64,
                scan_policy: str = "ot_sb",
                use_koopman: bool = True,
                trajectory_topk: int = 4,
            ):
                super().__init__()
                self.scan_policy = _normalize_scan_policy(scan_policy)
                self.use_koopman = bool(use_koopman)
                self.trajectory_topk = max(int(trajectory_topk), 1)
                self.input_projection = nn.Linear(hidden_channels, hidden_channels)
                self.state_projection = nn.Linear(hidden_channels, hidden_channels, bias=False)
                self.gate_projection = nn.Linear(hidden_channels * 2, hidden_channels)
                self.evidence_projection = nn.Linear(hidden_channels, hidden_channels)
                self.cross_trajectory_mixer = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.GELU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.output_norm = nn.LayerNorm(hidden_channels)
                self.koopman_predictor = nn.Linear(hidden_channels, hidden_channels)

            def forward(
                self,
                transport: dict[str, Any],
                uncertainty: dict[str, Any] | None = None,
            ) -> dict[str, Any]:
                del uncertainty
                trajectory_diagnostics = {}
                if self.scan_policy in {"ot_sb", "ot_sb_topk", "ot_sb_hard"}:
                    seq, restore, trajectory_diagnostics = self._make_soft_trajectory_sequence(transport)
                else:
                    grid = self._select_scan_grid(transport)
                    seq, restore = self._make_scan_sequence(grid)
                memory_seq = self._selective_scan(seq)
                memory_grid = restore(memory_seq)
                if self.use_koopman and memory_seq.shape[1] > 1:
                    koopman_next = self.koopman_predictor(memory_seq[:, :-1])
                    koopman_target = memory_seq[:, 1:]
                else:
                    koopman_next = memory_seq.new_zeros((memory_seq.shape[0], 0, memory_seq.shape[-1]))
                    koopman_target = memory_seq.new_zeros((memory_seq.shape[0], 0, memory_seq.shape[-1]))
                return {
                    "memory_grid": memory_grid,
                    "memory_tokens": memory_grid.flatten(1, 3),
                    "koopman_prediction": koopman_next,
                    "koopman_target": koopman_target,
                    "scan_policy": self.scan_policy,
                    "scan_sequence_length": memory_seq.shape[1],
                    "bridge_states": transport.get("bridge_states"),
                    "bridge_times": transport.get("bridge_times"),
                    "bridge_drift": transport.get("bridge_drift"),
                    **trajectory_diagnostics,
                }

            def _selective_scan(self, seq):
                state = seq.new_zeros((seq.shape[0], seq.shape[-1]))
                outputs = []
                for index in range(seq.shape[1]):
                    evidence = self.input_projection(seq[:, index])
                    proposal = torch.tanh(evidence + self.state_projection(state))
                    gate = torch.sigmoid(self.gate_projection(torch.cat([evidence, state], dim=-1)))
                    reliability_gate = torch.sigmoid(self.evidence_projection(seq[:, index]))
                    state = gate * state + (1.0 - gate) * proposal
                    outputs.append(self.output_norm(state + reliability_gate * evidence))
                memory_seq = torch.stack(outputs, dim=1)
                context = memory_seq.mean(dim=1, keepdim=True)
                return memory_seq + 0.1 * self.cross_trajectory_mixer(context)

            def _select_scan_grid(self, transport: dict[str, Any]):
                if self.scan_policy == "bridge_temporal":
                    return transport["bridge_grid"]
                return transport.get("source_grid", transport["bridge_grid"])

            def _make_scan_sequence(self, grid):
                batch, frames, height, width, channels = grid.shape
                if self.scan_policy == "temporal" or self.scan_policy == "bridge_temporal":
                    return self._make_temporal_sequence(grid)

                if self.scan_policy == "raster":
                    seq = grid.reshape(batch, frames * height * width, channels)

                    def restore(memory_seq):
                        return memory_seq.reshape(batch, frames, height, width, channels).contiguous()

                    return seq, restore

                if self.scan_policy == "hilbert":
                    order = _hilbert_order(height, width, device=grid.device)
                    inverse = torch.argsort(order)
                    frame_tokens = grid.reshape(batch, frames, height * width, channels)
                    seq = frame_tokens[:, :, order, :].reshape(batch, frames * height * width, channels)

                    def restore(memory_seq):
                        restored = memory_seq.reshape(batch, frames, height * width, channels)
                        restored = restored[:, :, inverse, :]
                        return restored.reshape(batch, frames, height, width, channels).contiguous()

                    return seq, restore

                if self.scan_policy == "content":
                    flat = grid.reshape(batch, frames * height * width, channels)
                    order = torch.argsort(flat.norm(dim=-1), dim=-1, descending=True)
                    gather_index = order.unsqueeze(-1).expand(-1, -1, channels)
                    seq = flat.gather(dim=1, index=gather_index)

                    def restore(memory_seq):
                        restored = torch.empty_like(memory_seq)
                        restored.scatter_(dim=1, index=gather_index, src=memory_seq)
                        return restored.reshape(batch, frames, height, width, channels).contiguous()

                    return seq, restore

                raise ValueError(f"Unsupported scan policy: {self.scan_policy}")

            def _make_temporal_sequence(self, grid):
                batch, frames, height, width, channels = grid.shape
                seq = grid.permute(0, 2, 3, 1, 4).reshape(batch * height * width, frames, channels)

                def restore(memory_seq):
                    restored = memory_seq.reshape(batch, height, width, frames, channels)
                    return restored.permute(0, 3, 1, 2, 4).contiguous()

                return seq, restore

            def _make_soft_trajectory_sequence(self, transport: dict[str, Any]):
                grid = transport.get("source_grid", transport["bridge_grid"])
                plan = transport.get("transport_plan")
                if plan is None:
                    seq, restore = self._make_temporal_sequence(transport["bridge_grid"])
                    return seq, restore, {
                        "soft_trajectory_used": False,
                        "trajectory_fallback": "missing_transport_plan",
                    }

                batch, frames, height, width, channels = grid.shape
                spatial_count = height * width
                token_count = frames * spatial_count
                if tuple(plan.shape[:2]) != (batch, token_count) or plan.shape[-1] != token_count:
                    raise ValueError(
                        "transport_plan must have shape B,N,N matching source_grid tokens"
                    )

                source_tokens = grid.reshape(batch, token_count, channels)
                fixed_by_spatial = grid.reshape(batch, frames, spatial_count, channels).permute(0, 2, 1, 3)
                anchor_spatial = torch.arange(token_count, device=grid.device) % spatial_count
                fixed_sequence = fixed_by_spatial[:, anchor_spatial, :, :]

                frame_expectations = []
                frame_masses = []
                for frame_index in range(frames):
                    start = frame_index * spatial_count
                    end = start + spatial_count
                    frame_plan = plan[:, :, start:end]
                    frame_plan = self._trajectory_frame_plan(frame_plan)
                    frame_mass = frame_plan.sum(dim=-1, keepdim=True)
                    expected = frame_plan @ source_tokens[:, start:end, :]
                    expected = expected / frame_mass.clamp_min(1e-8)
                    fallback = fixed_sequence[:, :, frame_index, :]
                    expected = torch.where(frame_mass > 1e-8, expected, fallback)
                    frame_expectations.append(expected)
                    frame_masses.append(frame_mass.squeeze(-1))

                trajectory_sequence = torch.stack(frame_expectations, dim=2)
                frame_mass_grid = torch.stack(frame_masses, dim=-1)
                bridge_weight = transport.get("bridge_weight")
                if bridge_weight is None:
                    trajectory_weight = grid.new_ones((batch, token_count, 1, 1))
                else:
                    trajectory_weight = bridge_weight.reshape(batch, token_count, -1)
                    trajectory_weight = trajectory_weight.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
                    trajectory_weight = trajectory_weight.unsqueeze(-1)
                blended_sequence = fixed_sequence + trajectory_weight * (trajectory_sequence - fixed_sequence)
                seq = blended_sequence.reshape(batch * token_count, frames, channels)
                anchor_frame = torch.arange(frames, device=grid.device).repeat_interleave(spatial_count)

                def restore(memory_seq):
                    restored_sequence = memory_seq.reshape(batch, token_count, frames, channels)
                    gather_index = anchor_frame.view(1, token_count, 1, 1).expand(batch, token_count, 1, channels)
                    selected = restored_sequence.gather(dim=2, index=gather_index).squeeze(2)
                    return selected.reshape(batch, frames, height, width, channels).contiguous()

                return seq, restore, {
                    "soft_trajectory_used": True,
                    "trajectory_scan_mode": self._trajectory_scan_mode(),
                    "trajectory_sequence_length": frames,
                    "trajectory_frame_mass_mean": frame_mass_grid.mean(),
                    "trajectory_frame_mass_min": frame_mass_grid.min(),
                    "trajectory_weight_mean": trajectory_weight.mean(),
                }

            def _trajectory_frame_plan(self, frame_plan):
                if self.scan_policy == "ot_sb":
                    return frame_plan
                if self.scan_policy == "ot_sb_topk":
                    topk = min(self.trajectory_topk, frame_plan.shape[-1])
                    _, top_index = frame_plan.topk(topk, dim=-1)
                    mask = frame_plan.new_zeros(frame_plan.shape)
                    mask.scatter_(dim=-1, index=top_index, value=1.0)
                    return frame_plan * mask
                if self.scan_policy == "ot_sb_hard":
                    soft = frame_plan / frame_plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    top_index = frame_plan.argmax(dim=-1, keepdim=True)
                    hard = frame_plan.new_zeros(frame_plan.shape)
                    hard.scatter_(dim=-1, index=top_index, value=1.0)
                    return hard + soft - soft.detach()
                return frame_plan

            def _trajectory_scan_mode(self) -> str:
                if self.scan_policy == "ot_sb_topk":
                    return f"top{self.trajectory_topk}_expectation"
                if self.scan_policy == "ot_sb_hard":
                    return "hard_top1_straight_through"
                return "soft_expectation"

        return _Memory(*args, **kwargs)


def _normalize_scan_policy(policy: str) -> str:
    normalized = str(policy).lower().replace("-", "_")
    aliases = {
        "trajectory": "ot_sb",
        "soft_trajectory": "ot_sb",
        "otsb": "ot_sb",
        "ot": "ot_sb",
        "topk_trajectory": "ot_sb_topk",
        "soft_topk": "ot_sb_topk",
        "hard_trajectory": "ot_sb_hard",
        "hard_ot_sb": "ot_sb_hard",
        "bridge": "bridge_temporal",
        "bridge_scan": "bridge_temporal",
        "bridge_temporal_scan": "bridge_temporal",
        "ot_sb_bridge": "bridge_temporal",
        "otsb_bridge": "bridge_temporal",
        "z_order": "hilbert",
        "morton": "hilbert",
        "coordinate": "temporal",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {
        "ot_sb",
        "ot_sb_topk",
        "ot_sb_hard",
        "bridge_temporal",
        "temporal",
        "raster",
        "hilbert",
        "content",
    }:
        raise ValueError(
            "scan_policy must be one of: ot_sb, ot_sb_topk, ot_sb_hard, bridge_temporal, temporal, raster, hilbert, content"
        )
    return normalized


def _hilbert_order(height: int, width: int, device: Any):
    torch, _ = require_torch_nn()
    side = 1
    while side < max(int(height), int(width), 1):
        side *= 2
    order = []
    for y in range(int(height)):
        for x in range(int(width)):
            order.append((_hilbert_xy_to_index(x, y, side), y * int(width) + x))
    order.sort(key=lambda item: item[0])
    return torch.tensor([index for _, index in order], dtype=torch.long, device=device)


def _hilbert_xy_to_index(x: int, y: int, side: int) -> int:
    distance = 0
    step = side // 2
    while step > 0:
        rx = 1 if x & step else 0
        ry = 1 if y & step else 0
        distance += step * step * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = side - 1 - x
                y = side - 1 - y
            x, y = y, x
        step //= 2
    return distance
