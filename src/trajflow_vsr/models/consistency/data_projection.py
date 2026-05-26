"""Reliability-calibrated data consistency projection."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch_nn


class ReliabilityCalibratedDataConsistency:
    """Blend generated HR output with LR-consistent low-frequency evidence."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch, nn = require_torch_nn()
        functional = torch.nn.functional

        class _Projection(nn.Module):
            def __init__(self, strength: float = 0.2):
                super().__init__()
                self.strength = strength

            def forward(
                self,
                lr_video,
                decoded: dict[str, Any],
                uncertainty: dict[str, Any],
                scale: float,
            ) -> dict[str, Any]:
                hr_raw = decoded["hr_raw"]
                batch, frames, channels, _, _ = hr_raw.shape
                base = functional.interpolate(
                    lr_video.flatten(0, 1),
                    scale_factor=float(scale),
                    mode="bilinear",
                    align_corners=False,
                ).reshape_as(hr_raw)

                reliability = uncertainty["reliability"].flatten(0, 1)
                reliability = functional.interpolate(
                    reliability,
                    size=hr_raw.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).reshape(batch, frames, 1, *hr_raw.shape[-2:])
                alpha = (self.strength * reliability).expand(batch, frames, channels, *hr_raw.shape[-2:])
                hr = (1.0 - alpha) * hr_raw + alpha * base
                return {"hr": hr, "low_frequency_base": base, "data_consistency_alpha": alpha}

        return _Projection(*args, **kwargs)
