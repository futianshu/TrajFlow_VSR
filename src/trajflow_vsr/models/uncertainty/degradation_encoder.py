"""Degradation-causal uncertainty encoder."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch_nn


class DegradationCausalUncertaintyEncoder:
    """Estimate reliability, artifacts, motion uncertainty, and texture uncertainty."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch, nn = require_torch_nn()

        class _Encoder(nn.Module):
            def __init__(
                self,
                in_channels: int = 3,
                hidden_channels: int = 64,
                degradation_dim: int = 8,
            ):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                self.artifact_head = nn.Conv3d(hidden_channels, 1, kernel_size=1)
                self.motion_uncertainty_head = nn.Conv3d(hidden_channels, 1, kernel_size=1)
                self.texture_uncertainty_head = nn.Conv3d(hidden_channels, 1, kernel_size=1)
                self.global_head = nn.Linear(hidden_channels, degradation_dim)

            def forward(self, lr_video) -> dict[str, Any]:
                if lr_video.ndim != 5:
                    raise ValueError("Expected LR video shape B,T,C,H,W")

                x = lr_video.permute(0, 2, 1, 3, 4)
                feat = self.backbone(x)
                artifact = torch.sigmoid(self.artifact_head(feat))
                reliability = 1.0 - artifact
                motion_u = torch.sigmoid(self.motion_uncertainty_head(feat))
                texture_u = torch.sigmoid(self.texture_uncertainty_head(feat))
                pooled = feat.mean(dim=(2, 3, 4))
                d_global = self.global_head(pooled)

                def to_btchw(tensor):
                    return tensor.permute(0, 2, 1, 3, 4).contiguous()

                return {
                    "d_global": d_global,
                    "artifact": to_btchw(artifact),
                    "reliability": to_btchw(reliability),
                    "motion_uncertainty": to_btchw(motion_u),
                    "texture_uncertainty": to_btchw(texture_u),
                }

        return _Encoder(*args, **kwargs)
