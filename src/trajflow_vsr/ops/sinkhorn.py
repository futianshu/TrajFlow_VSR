"""Sinkhorn operators for reliability-calibrated token transport."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def pairwise_squared_distance(source: Any, target: Any):
    """Return batched pairwise squared distances for B,N,C and B,M,C tensors."""

    source_norm = (source * source).sum(dim=-1, keepdim=True)
    target_norm = (target * target).sum(dim=-1).unsqueeze(-2)
    distance = source_norm + target_norm - 2.0 * (source @ target.transpose(-1, -2))
    return distance.clamp_min(0.0)


def normalize_mass(mass: Any, minimum: float = 1e-6):
    """Clamp and normalize a batched non-negative mass vector."""

    clamped = mass.clamp_min(minimum)
    return clamped / clamped.sum(dim=-1, keepdim=True).clamp_min(minimum)


def sinkhorn_plan(
    cost: Any,
    source_mass: Any | None = None,
    target_mass: Any | None = None,
    epsilon: float = 0.2,
    iterations: int = 12,
):
    """Compute an entropic OT plan with prescribed source and target marginals."""

    torch = require_torch()
    if cost.ndim != 3:
        raise ValueError("Expected cost shape B,N,M")

    batch, source_count, target_count = cost.shape
    dtype = cost.dtype
    device = cost.device
    if source_mass is None:
        source_mass = torch.full((batch, source_count), 1.0 / source_count, dtype=dtype, device=device)
    if target_mass is None:
        target_mass = torch.full((batch, target_count), 1.0 / target_count, dtype=dtype, device=device)

    source_mass = normalize_mass(source_mass).to(dtype=dtype, device=device)
    target_mass = normalize_mass(target_mass).to(dtype=dtype, device=device)
    log_source = source_mass.clamp_min(1e-8).log()
    log_target = target_mass.clamp_min(1e-8).log()
    log_kernel = -cost / max(float(epsilon), 1e-6)
    log_u = torch.zeros_like(log_source)
    log_v = torch.zeros_like(log_target)

    for _ in range(max(int(iterations), 1)):
        log_u = log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        log_v = log_target - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)

    plan = torch.exp(log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2))
    return plan
