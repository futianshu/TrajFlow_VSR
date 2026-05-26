"""Loss factory helpers."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.losses.distillation import consistency_distillation_loss, teacher_target_reconstruction_loss
from trajflow_vsr.losses.flow_matching import (
    bridge_residual_consistency_loss,
    rectified_flow_matching_loss,
    residual_amplitude_loss,
    residual_low_frequency_loss,
)
from trajflow_vsr.losses.optimal_transport import (
    motion_supervised_transport_loss,
    optimal_transport_loss,
    transport_entropy_loss,
)
from trajflow_vsr.losses.reconstruction import charbonnier_loss, data_consistency_loss, koopman_dynamics_loss
from trajflow_vsr.losses.schrodinger_bridge import schrodinger_bridge_loss
from trajflow_vsr.losses.streaming import streaming_causality_loss
from trajflow_vsr.losses.temporal import temporal_consistency_loss
from trajflow_vsr.losses.trajectory import trajectory_regularization_loss
from trajflow_vsr.losses.uncertainty import uncertainty_calibration_loss
from trajflow_vsr.losses.wavelet_anti_aliasing import anti_aliasing_loss, wavelet_frequency_loss
from trajflow_vsr.utils.torch_utils import require_torch


def describe_losses(config: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight loss summary."""

    return {
        "charbonnier": float(config.get("charbonnier", 1.0)),
        "masked_reconstruction": float(config.get("masked_reconstruction", 0.0)),
        "degradation": float(config.get("degradation", 0.0)),
        "artifact": float(config.get("artifact", 0.0)),
        "reliability": float(config.get("reliability", 0.0)),
        "optimal_transport": float(config.get("optimal_transport", 0.0)),
        "motion_transport": float(config.get("motion_transport", 0.0)),
        "transport_entropy": float(config.get("transport_entropy", 0.0)),
        "schrodinger_bridge": float(config.get("schrodinger_bridge", 0.0)),
        "koopman": float(config.get("koopman", 0.0)),
        "flow_matching": float(config.get("flow_matching", 0.0)),
        "bridge_consistency": float(config.get("bridge_consistency", 0.0)),
        "residual_amplitude": float(config.get("residual_amplitude", 0.0)),
        "residual_low_frequency": float(config.get("residual_low_frequency", 0.0)),
        "distillation": float(config.get("distillation", 0.0)),
        "teacher_target": float(config.get("teacher_target", 0.0)),
        "streaming_causality": float(config.get("streaming_causality", 0.0)),
        "data_consistency": float(config.get("data_consistency", 0.0)),
        "temporal": float(config.get("temporal", 0.0)),
        "wavelet_frequency": float(config.get("wavelet_frequency", config.get("frequency", 0.0))),
        "anti_aliasing": float(config.get("anti_aliasing", 0.0)),
        "trajectory": float(config.get("trajectory", 0.0)),
        "uncertainty": float(config.get("uncertainty", 0.0)),
    }


def compute_training_loss(outputs: dict[str, Any], batch: dict[str, Any], config: dict[str, Any]):
    """Compute the currently implemented subset of training losses."""

    torch = require_torch()
    total = torch.zeros((), device=batch["hr"].device)
    parts: dict[str, Any] = {}

    char_weight = float(config.get("charbonnier", 1.0))
    if char_weight:
        parts["charbonnier"] = charbonnier_loss(outputs["hr"], batch["hr"])
        total = total + char_weight * parts["charbonnier"]

    ot_weight = float(config.get("optimal_transport", 0.0))
    if ot_weight:
        parts["optimal_transport"] = optimal_transport_loss(outputs["transport"])
        total = total + ot_weight * parts["optimal_transport"]

    motion_transport_weight = float(config.get("motion_transport", 0.0))
    if motion_transport_weight:
        parts["motion_transport"] = motion_supervised_transport_loss(outputs["transport"], batch)
        total = total + motion_transport_weight * parts["motion_transport"]

    transport_entropy_weight = float(config.get("transport_entropy", 0.0))
    if transport_entropy_weight:
        parts["transport_entropy"] = transport_entropy_loss(outputs["transport"])
        total = total + transport_entropy_weight * parts["transport_entropy"]

    sb_weight = float(config.get("schrodinger_bridge", 0.0))
    if sb_weight:
        parts["schrodinger_bridge"] = schrodinger_bridge_loss(outputs["transport"])
        total = total + sb_weight * parts["schrodinger_bridge"]

    koopman_weight = float(config.get("koopman", 0.0))
    if koopman_weight:
        parts["koopman"] = koopman_dynamics_loss(outputs["memory"])
        total = total + koopman_weight * parts["koopman"]

    flow_matching_weight = float(config.get("flow_matching", 0.0))
    if flow_matching_weight:
        parts["flow_matching"] = rectified_flow_matching_loss(outputs)
        total = total + flow_matching_weight * parts["flow_matching"]

    bridge_consistency_weight = float(config.get("bridge_consistency", 0.0))
    if bridge_consistency_weight:
        parts["bridge_consistency"] = bridge_residual_consistency_loss(outputs)
        total = total + bridge_consistency_weight * parts["bridge_consistency"]

    residual_amplitude_weight = float(config.get("residual_amplitude", 0.0))
    if residual_amplitude_weight:
        parts["residual_amplitude"] = residual_amplitude_loss(outputs)
        total = total + residual_amplitude_weight * parts["residual_amplitude"]

    residual_low_frequency_weight = float(config.get("residual_low_frequency", 0.0))
    if residual_low_frequency_weight:
        parts["residual_low_frequency"] = residual_low_frequency_loss(outputs)
        total = total + residual_low_frequency_weight * parts["residual_low_frequency"]

    distillation_weight = float(config.get("distillation", 0.0))
    if distillation_weight:
        parts["distillation"] = consistency_distillation_loss(outputs)
        total = total + distillation_weight * parts["distillation"]

    teacher_target_weight = float(config.get("teacher_target", 0.0))
    if teacher_target_weight:
        parts["teacher_target"] = teacher_target_reconstruction_loss(outputs)
        total = total + teacher_target_weight * parts["teacher_target"]

    streaming_causality_weight = float(config.get("streaming_causality", 0.0))
    if streaming_causality_weight:
        parts["streaming_causality"] = streaming_causality_loss(outputs)
        total = total + streaming_causality_weight * parts["streaming_causality"]

    data_consistency_weight = float(config.get("data_consistency", 0.0))
    if data_consistency_weight:
        parts["data_consistency"] = data_consistency_loss(outputs, batch)
        total = total + data_consistency_weight * parts["data_consistency"]

    temporal_weight = float(config.get("temporal", 0.0))
    if temporal_weight:
        parts["temporal"] = temporal_consistency_loss(outputs, batch)
        total = total + temporal_weight * parts["temporal"]

    wavelet_weight = float(config.get("wavelet_frequency", config.get("frequency", 0.0)))
    if wavelet_weight:
        parts["wavelet_frequency"] = wavelet_frequency_loss(outputs, batch)
        total = total + wavelet_weight * parts["wavelet_frequency"]

    anti_aliasing_weight = float(config.get("anti_aliasing", 0.0))
    if anti_aliasing_weight:
        parts["anti_aliasing"] = anti_aliasing_loss(outputs)
        total = total + anti_aliasing_weight * parts["anti_aliasing"]

    trajectory_weight = float(config.get("trajectory", 0.0))
    if trajectory_weight:
        parts["trajectory"] = trajectory_regularization_loss(outputs["transport"])
        total = total + trajectory_weight * parts["trajectory"]

    uncertainty_weight = float(config.get("uncertainty", 0.0))
    if uncertainty_weight:
        parts["uncertainty"] = uncertainty_calibration_loss(outputs, batch)
        total = total + uncertainty_weight * parts["uncertainty"]

    parts["total"] = total
    return total, parts
