"""Conditional rectified flow residual generator skeleton."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch_nn


class ConditionalRectifiedFlowResidualGenerator:
    """Generate high-frequency residual latents conditioned on trajectory memory."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch, nn = require_torch_nn()

        class _ResidualFlow(nn.Module):
            def __init__(
                self,
                hidden_channels: int = 64,
                gate_max: float = 0.25,
                reliability_weight: float = 1.0,
                texture_uncertainty_weight: float = 1.0,
                amplitude_limit: float = 0.25,
                bandlimit_kernel_size: int = 3,
            ):
                super().__init__()
                self.gate_max = float(gate_max)
                self.reliability_weight = float(reliability_weight)
                self.texture_uncertainty_weight = float(texture_uncertainty_weight)
                self.amplitude_limit = float(amplitude_limit)
                self.bandlimit_kernel_size = max(int(bandlimit_kernel_size), 1)
                if self.bandlimit_kernel_size % 2 == 0:
                    self.bandlimit_kernel_size += 1
                self.timestep_embed = nn.Sequential(
                    nn.Linear(1, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.vector_field = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_channels * 2, hidden_channels),
                )

            def forward(
                self,
                memory: dict[str, Any],
                uncertainty: dict[str, Any],
                tau=None,
                noise=None,
                sample_noise: bool = False,
                distill: bool = False,
                teacher_steps: int = 4,
            ) -> dict[str, Any]:
                grid = memory["memory_grid"]
                target_residual = memory.get("bridge_drift")
                if target_residual is None:
                    target_residual = torch.zeros_like(grid)
                residual_gate = self._residual_gate(target_residual, uncertainty)
                target_low_band, target_residual = self._safe_residual(target_residual, residual_gate)
                if tau is None:
                    if sample_noise:
                        tau = torch.rand((grid.shape[0],), device=grid.device, dtype=grid.dtype)
                    else:
                        tau = torch.full((grid.shape[0],), 0.5, device=grid.device, dtype=grid.dtype)
                if noise is None:
                    if sample_noise:
                        noise = torch.randn_like(target_residual)
                    else:
                        noise = torch.zeros_like(target_residual)
                noise_low_band, noise = self._safe_residual(noise, residual_gate)

                tau_view = tau.view(grid.shape[0], 1, 1, 1, 1)
                interpolant = (1.0 - tau_view) * noise + tau_view * target_residual
                velocity_low_band, velocity = self._safe_residual(
                    self._velocity(grid, target_residual, interpolant, tau),
                    residual_gate,
                )
                target_velocity = target_residual - noise
                residual_low_band, residual = self._safe_residual(
                    interpolant + (1.0 - tau_view) * velocity,
                    residual_gate,
                )
                output = {
                    "residual_grid": residual,
                    "ungated_residual_grid": interpolant + (1.0 - tau_view) * velocity,
                    "residual_gate": residual_gate,
                    "residual_low_band": residual_low_band,
                    "target_residual_low_band": target_low_band,
                    "noise_low_band": noise_low_band,
                    "velocity_low_band": velocity_low_band,
                    "flow_velocity": velocity,
                    "flow_target_velocity": target_velocity,
                    "flow_interpolant": interpolant,
                    "flow_noise": noise,
                    "flow_target_residual": target_residual,
                    "tau": tau,
                }
                if distill:
                    teacher_residual = self._teacher_sample(
                        grid=grid,
                        target_residual=target_residual,
                        noise=noise,
                        residual_gate=residual_gate,
                        steps=teacher_steps,
                    )
                    student_tau = torch.zeros_like(tau)
                    student_velocity = self._safe_residual(
                        self._velocity(grid, target_residual, noise, student_tau),
                        residual_gate,
                    )[1]
                    student_residual = self._safe_residual(noise + student_velocity, residual_gate)[1]
                    output.update(
                        {
                            "residual_grid": student_residual,
                            "flow_velocity": student_velocity,
                            "student_residual": student_residual,
                            "teacher_residual": teacher_residual.detach(),
                            "teacher_steps": int(teacher_steps),
                        }
                    )
                return output

            def _velocity(self, grid, target_residual, state, tau):
                tau_embedding = self.timestep_embed(tau[:, None]).view(grid.shape[0], 1, 1, 1, -1)
                conditioned = grid + 0.25 * target_residual + state + tau_embedding
                return self.vector_field(conditioned)

            def _teacher_sample(self, grid, target_residual, noise, residual_gate, steps: int):
                state = noise
                num_steps = max(int(steps), 1)
                dt = 1.0 / float(num_steps)
                for step in range(num_steps):
                    tau = torch.full((grid.shape[0],), step * dt, device=grid.device, dtype=grid.dtype)
                    velocity = self._safe_residual(
                        self._velocity(grid, target_residual, state, tau),
                        residual_gate,
                    )[1]
                    state = self._safe_residual(state + dt * velocity, residual_gate)[1]
                return state

            def _safe_residual(self, residual, residual_gate):
                low_band, high_band = self._split_low_high(residual)
                gated = high_band * residual_gate
                if self.amplitude_limit > 0:
                    gated = gated.clamp(min=-self.amplitude_limit, max=self.amplitude_limit)
                return low_band, gated

            def _split_low_high(self, residual):
                if self.bandlimit_kernel_size <= 1:
                    return torch.zeros_like(residual), residual
                batch, frames, height, width, channels = residual.shape
                x = residual.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, height, width)
                low = torch.nn.functional.avg_pool2d(
                    x,
                    kernel_size=self.bandlimit_kernel_size,
                    stride=1,
                    padding=self.bandlimit_kernel_size // 2,
                )
                low = low.reshape(batch, frames, channels, height, width).permute(0, 1, 3, 4, 2)
                return low, residual - low

            def _residual_gate(self, reference, uncertainty: dict[str, Any]):
                batch, frames, height, width, _ = reference.shape
                gate = reference.new_zeros((batch, frames, 1, height, width))
                total_weight = 0.0
                if isinstance(uncertainty, dict):
                    reliability = uncertainty.get("reliability")
                    if reliability is not None and self.reliability_weight > 0:
                        reliability = _resize_map(reliability, height=height, width=width)
                        gate = gate + self.reliability_weight * (1.0 - reliability.clamp(0.0, 1.0))
                        total_weight += self.reliability_weight
                    texture_uncertainty = uncertainty.get("texture_uncertainty")
                    if texture_uncertainty is not None and self.texture_uncertainty_weight > 0:
                        texture_uncertainty = _resize_map(texture_uncertainty, height=height, width=width)
                        gate = gate + self.texture_uncertainty_weight * texture_uncertainty.clamp(0.0, 1.0)
                        total_weight += self.texture_uncertainty_weight
                if total_weight <= 0.0:
                    gate = torch.ones_like(gate)
                else:
                    gate = gate / total_weight
                gate = gate.clamp(0.0, 1.0) * self.gate_max
                return gate.permute(0, 1, 3, 4, 2).to(dtype=reference.dtype)

        return _ResidualFlow(*args, **kwargs)


def _resize_map(map_tensor, *, height: int, width: int):
    if map_tensor.shape[-2:] == (height, width):
        return map_tensor
    torch, _ = require_torch_nn()
    batch, frames = map_tensor.shape[:2]
    resized = torch.nn.functional.interpolate(
        map_tensor.flatten(0, 1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.unflatten(0, (batch, frames))
