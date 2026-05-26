"""Consistency distillation objectives for one-step flow students."""

from __future__ import annotations

from typing import Any


def consistency_distillation_loss(outputs: dict[str, Any]):
    """Match one-step student residuals to a detached multi-step teacher sample."""

    residual = outputs["residual"]
    student = residual["student_residual"]
    teacher = residual["teacher_residual"].detach()
    loss = (student - teacher).square()
    return _gate_weighted_mean(loss, residual.get("residual_gate"))


def teacher_target_reconstruction_loss(outputs: dict[str, Any]):
    """Keep the teacher residual close to the SB target residual."""

    residual = outputs["residual"]
    teacher = residual["teacher_residual"]
    target = residual["flow_target_residual"].detach()
    loss = (teacher - target).abs()
    return _gate_weighted_mean(loss, residual.get("residual_gate"))


def _gate_weighted_mean(loss, gate):
    if gate is None:
        return loss.mean()
    weights = gate.to(dtype=loss.dtype).expand_as(loss)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)
