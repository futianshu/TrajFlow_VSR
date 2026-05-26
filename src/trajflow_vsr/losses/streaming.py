"""Streaming and online-mode regularization losses."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def streaming_causality_loss(outputs: dict[str, Any]):
    """Penalize future-token transport mass in streaming mode."""

    torch = require_torch()
    transport = outputs["transport"]
    violation = transport.get("causal_violation")
    if violation is None:
        return torch.zeros((), device=outputs["hr"].device, dtype=outputs["hr"].dtype)
    return violation.to(device=outputs["hr"].device, dtype=outputs["hr"].dtype)
