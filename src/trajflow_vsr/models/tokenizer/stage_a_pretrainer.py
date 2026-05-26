"""Stage A tokenizer and uncertainty pretraining model."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.models.tokenizer.evidence_tokenizer import MultiScaleEvidenceTokenizer
from trajflow_vsr.models.uncertainty.degradation_encoder import DegradationCausalUncertaintyEncoder
from trajflow_vsr.utils.torch_utils import require_torch_nn


class StageATokenizerPretrainer:
    """Pretrain tokenizer and uncertainty encoder on synthetic degradations."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch, nn = require_torch_nn()

        class _StageATokenizerPretrainer(nn.Module):
            def __init__(
                self,
                in_channels: int = 3,
                hidden_channels: int = 64,
                degradation_dim: int = 8,
                mask_ratio: float = 0.25,
            ):
                super().__init__()
                self.mask_ratio = mask_ratio
                self.tokenizer = MultiScaleEvidenceTokenizer(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                )
                self.uncertainty = DegradationCausalUncertaintyEncoder(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    degradation_dim=degradation_dim,
                )
                self.reconstruction_head = nn.Sequential(
                    nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv3d(hidden_channels, in_channels, kernel_size=3, padding=1),
                    nn.Sigmoid(),
                )
                self.degradation_head = nn.Sequential(
                    nn.LayerNorm(hidden_channels),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.GELU(),
                    nn.Linear(hidden_channels, degradation_dim),
                    nn.Sigmoid(),
                )

            def forward(self, lr_video, scale: float | None = None) -> dict[str, Any]:
                if lr_video.ndim != 5:
                    raise ValueError("Expected LR video shape B,T,C,H,W")

                mask = (torch.rand_like(lr_video[:, :, :1]) < self.mask_ratio).to(lr_video.dtype)
                masked_lr = lr_video * (1.0 - mask)
                token_bundle = self.tokenizer(masked_lr, scale=scale)
                uncertainty = self.uncertainty(masked_lr)
                feature_grid = token_bundle["feature_grid"]
                features = feature_grid.permute(0, 4, 1, 2, 3).contiguous()
                reconstruction = self.reconstruction_head(features).permute(0, 2, 1, 3, 4).contiguous()
                pooled_tokens = token_bundle["tokens"].mean(dim=1)
                degradation = self.degradation_head(pooled_tokens)
                return {
                    "masked_lr": masked_lr,
                    "mask": mask,
                    "token_bundle": token_bundle,
                    "uncertainty": uncertainty,
                    "reconstruction": reconstruction,
                    "degradation": degradation,
                }

        return _StageATokenizerPretrainer(*args, **kwargs)


def build_stage_a_tokenizer_pretrainer(config: dict[str, Any]) -> Any:
    """Build the Stage A pretraining model."""

    uncertainty_cfg = config.get("uncertainty", {})
    return StageATokenizerPretrainer(
        in_channels=int(config.get("in_channels", 3)),
        hidden_channels=int(config.get("hidden_channels", config.get("tokenizer", {}).get("hidden_channels", 64))),
        degradation_dim=int(uncertainty_cfg.get("degradation_dim", 8)),
        mask_ratio=float(config.get("mask_ratio", 0.25)),
    )
