"""Spacetime neural operator plus wavelet anti-aliasing decoder."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.ops.wavelet import split_low_high_2d
from trajflow_vsr.utils.torch_utils import require_torch_nn


class SpacetimeWaveletOperatorDecoder:
    """Decode latent video fields into arbitrary-scale HR RGB frames."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch, nn = require_torch_nn()
        functional = torch.nn.functional

        class _Decoder(nn.Module):
            def __init__(
                self,
                hidden_channels: int = 64,
                out_channels: int = 3,
                use_operator: bool = True,
                use_fourier: bool = True,
                use_coordinate: bool = True,
                use_wavelet: bool = True,
                use_anti_aliasing: bool = True,
            ):
                super().__init__()
                self.use_operator = bool(use_operator)
                self.use_fourier = bool(use_fourier)
                self.use_coordinate = bool(use_coordinate)
                self.use_wavelet = bool(use_wavelet)
                self.use_anti_aliasing = bool(use_anti_aliasing)
                self.operator = nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                self.fourier_mixer = nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                )
                self.rgb_head = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
                self.coordinate_mlp = nn.Sequential(
                    nn.Linear(hidden_channels + 4, hidden_channels),
                    nn.GELU(),
                    nn.Linear(hidden_channels, out_channels),
                )
                self.anti_alias_gate = nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
                    nn.Sigmoid(),
                )

            def forward(
                self,
                lr_video,
                memory: dict[str, Any],
                residual: dict[str, Any],
                scale: float,
            ) -> dict[str, Any]:
                del lr_video
                field = memory["memory_grid"] + residual["residual_grid"]
                batch, frames, height, width, channels = field.shape
                x = field.reshape(batch * frames, height, width, channels).permute(0, 3, 1, 2)
                hr_height = max(1, int(round(height * float(scale))))
                hr_width = max(1, int(round(width * float(scale))))
                x_up = functional.interpolate(x, size=(hr_height, hr_width), mode="bilinear", align_corners=False)
                operator_field = self.operator(x_up) if self.use_operator else x_up
                if self.use_fourier:
                    operator_field = operator_field + 0.1 * self.fourier_mixer(operator_field)
                rgb_raw = self.rgb_head(operator_field)
                query = _query_grid(
                    batch=batch,
                    frames=frames,
                    height=hr_height,
                    width=hr_width,
                    scale=float(scale),
                    device=x_up.device,
                    dtype=x_up.dtype,
                )
                if self.use_coordinate:
                    coordinate_rgb = self.coordinate_mlp(
                        torch.cat([operator_field.permute(0, 2, 3, 1), query], dim=-1)
                    ).permute(0, 3, 1, 2)
                    rgb_raw = rgb_raw + coordinate_rgb
                gate = self.anti_alias_gate(operator_field) if self.use_anti_aliasing else torch.ones_like(rgb_raw)
                if self.use_wavelet:
                    low_band, high_band = split_low_high_2d(rgb_raw)
                    rgb = low_band + gate * high_band
                else:
                    low_band = rgb_raw
                    high_band = torch.zeros_like(rgb_raw)
                    rgb = rgb_raw
                _, _, hr_h, hr_w = rgb.shape
                hr = rgb.reshape(batch, frames, -1, hr_h, hr_w)
                return {
                    "hr_raw": hr,
                    "operator_rgb": rgb_raw.reshape(batch, frames, -1, hr_h, hr_w),
                    "wavelet_low": low_band.reshape(batch, frames, -1, hr_h, hr_w),
                    "wavelet_high": high_band.reshape(batch, frames, -1, hr_h, hr_w),
                    "anti_alias_gate": gate.reshape(batch, frames, -1, hr_h, hr_w),
                    "query_coordinates": query[..., :3].reshape(batch, frames, hr_h, hr_w, 3),
                    "query_footprint": query[..., 3:].reshape(batch, frames, hr_h, hr_w, 1),
                }

        return _Decoder(*args, **kwargs)


def _query_grid(
    *,
    batch: int,
    frames: int,
    height: int,
    width: int,
    scale: float,
    device: Any,
    dtype: Any,
):
    torch, _ = require_torch_nn()
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, int(height), device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, int(width), device=device, dtype=dtype),
        indexing="ij",
    )
    tt = torch.linspace(0.0, 1.0, int(frames), device=device, dtype=dtype)
    x_grid = xx.view(1, 1, height, width).expand(batch, frames, height, width)
    y_grid = yy.view(1, 1, height, width).expand(batch, frames, height, width)
    t_grid = tt.view(1, frames, 1, 1).expand(batch, frames, height, width)
    footprint = torch.full_like(x_grid, 1.0 / max(float(scale), 1e-6))
    return torch.stack([x_grid, y_grid, t_grid, footprint], dim=-1).reshape(batch * frames, height, width, 4)
