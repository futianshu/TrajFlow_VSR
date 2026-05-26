"""Multi-scale evidence tokenizer."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch_nn


class MultiScaleEvidenceTokenizer:
    """Tokenize LR video into multi-scale evidence grids and token sequences."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch, nn = require_torch_nn()
        functional = torch.nn.functional

        class _Tokenizer(nn.Module):
            def __init__(self, in_channels: int = 3, hidden_channels: int = 64):
                super().__init__()
                self.hidden_channels = hidden_channels
                augmented_channels = in_channels * 4 + 6
                self.stem = nn.Sequential(
                    nn.Conv3d(augmented_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                self.texture_head = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1)
                self.artifact_head = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1)

            def forward(self, lr_video, scale: float | None = None) -> dict[str, Any]:
                if lr_video.ndim != 5:
                    raise ValueError("Expected LR video shape B,T,C,H,W")

                low_band = functional.avg_pool2d(
                    lr_video.flatten(0, 1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ).unflatten(0, lr_video.shape[:2])
                high_band = lr_video - low_band
                temporal_delta = torch.cat([lr_video[:, :1].new_zeros(lr_video[:, :1].shape), lr_video[:, 1:] - lr_video[:, :-1]], dim=1)
                coordinate_features, coordinates, footprints, scale_grid = _coordinate_features(
                    batch=lr_video.shape[0],
                    frames=lr_video.shape[1],
                    height=lr_video.shape[-2],
                    width=lr_video.shape[-1],
                    scale=float(scale or 1.0),
                    device=lr_video.device,
                    dtype=lr_video.dtype,
                )
                evidence = torch.cat([lr_video, low_band, high_band, temporal_delta, coordinate_features], dim=2)

                x = evidence.permute(0, 2, 1, 3, 4)
                structure = self.stem(x)
                texture = self.texture_head(structure)
                artifact = self.artifact_head(structure)

                grid = structure.permute(0, 2, 3, 4, 1).contiguous()
                tokens = grid.flatten(1, 3)
                return {
                    "feature_grid": grid,
                    "tokens": tokens,
                    "structure_grid": grid,
                    "texture_grid": texture.permute(0, 2, 3, 4, 1).contiguous(),
                    "artifact_grid": artifact.permute(0, 2, 3, 4, 1).contiguous(),
                    "low_band": low_band,
                    "high_band": high_band,
                    "temporal_delta": temporal_delta,
                    "coordinates": coordinates,
                    "footprints": footprints,
                    "scale_grid": scale_grid,
                    "token_coordinates": coordinates.flatten(1, 3),
                    "token_footprints": footprints.flatten(1, 3),
                }

        return _Tokenizer(*args, **kwargs)


def _coordinate_features(
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
    tt = torch.linspace(0.0, 1.0, int(frames), device=device, dtype=dtype).view(1, frames, 1, 1)
    x_grid = xx.view(1, 1, height, width).expand(batch, frames, height, width)
    y_grid = yy.view(1, 1, height, width).expand(batch, frames, height, width)
    t_grid = tt.expand(batch, frames, height, width)
    normalized_scale = torch.full_like(x_grid, float(scale) / 8.0)
    footprint = torch.full_like(x_grid, 1.0 / max(float(scale), 1e-6))
    features = torch.stack([x_grid, y_grid, t_grid, normalized_scale, footprint, footprint], dim=2)
    coordinates = torch.stack([x_grid, y_grid, t_grid], dim=-1).contiguous()
    footprints = torch.stack([footprint, footprint], dim=-1).contiguous()
    scale_grid = normalized_scale.unsqueeze(-1).contiguous()
    return features, coordinates, footprints, scale_grid
