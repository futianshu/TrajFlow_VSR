"""Stage A tokenizer and uncertainty pretraining losses."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def compute_stage_a_loss(outputs: dict[str, Any], batch: dict[str, Any], config: dict[str, Any]):
    """Compute Stage A losses from synthetic degradation supervision."""

    torch = require_torch()
    functional = torch.nn.functional
    total = torch.zeros((), device=batch["lr"].device)
    parts: dict[str, Any] = {}

    mask = outputs["mask"]
    target = batch.get("clean_lr", batch["lr"])
    recon_weight = float(config.get("masked_reconstruction", config.get("charbonnier", 1.0)))
    if recon_weight:
        denom = (mask.sum() * target.shape[2]).clamp_min(1.0)
        parts["masked_reconstruction"] = ((outputs["reconstruction"] - target).abs() * mask).sum() / denom
        total = total + recon_weight * parts["masked_reconstruction"]

    degradation_weight = float(config.get("degradation", 1.0))
    if degradation_weight:
        parts["degradation"] = functional.mse_loss(outputs["degradation"], batch["degradation"])
        total = total + degradation_weight * parts["degradation"]

    artifact_weight = float(config.get("artifact", 0.5))
    if artifact_weight:
        artifact_pred = outputs["uncertainty"]["artifact"].clamp(1e-5, 1.0 - 1e-5).float()
        artifact_target = batch["artifact"].clamp(0.0, 1.0).float()
        with torch.autocast(device_type=artifact_pred.device.type, enabled=False):
            parts["artifact"] = functional.binary_cross_entropy(artifact_pred, artifact_target)
        total = total + artifact_weight * parts["artifact"]

    reliability_weight = float(config.get("reliability", 0.25))
    if reliability_weight:
        parts["reliability"] = functional.l1_loss(outputs["uncertainty"]["reliability"], batch["reliability"])
        total = total + reliability_weight * parts["reliability"]

    parts["total"] = total
    return total, parts
