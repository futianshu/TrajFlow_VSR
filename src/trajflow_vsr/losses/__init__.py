"""Training objectives for transport, flow, dynamics, and consistency."""

from trajflow_vsr.losses.distillation import consistency_distillation_loss, teacher_target_reconstruction_loss
from trajflow_vsr.losses.factory import compute_training_loss, describe_losses
from trajflow_vsr.losses.flow_matching import (
    bridge_residual_consistency_loss,
    rectified_flow_matching_loss,
    residual_amplitude_loss,
    residual_low_frequency_loss,
)
from trajflow_vsr.losses.optimal_transport import optimal_transport_loss
from trajflow_vsr.losses.schrodinger_bridge import schrodinger_bridge_loss
from trajflow_vsr.losses.streaming import streaming_causality_loss
from trajflow_vsr.losses.stage_a import compute_stage_a_loss
from trajflow_vsr.losses.temporal import temporal_consistency_loss
from trajflow_vsr.losses.trajectory import trajectory_regularization_loss
from trajflow_vsr.losses.uncertainty import uncertainty_calibration_loss
from trajflow_vsr.losses.wavelet_anti_aliasing import anti_aliasing_loss, wavelet_frequency_loss

__all__ = [
    "compute_stage_a_loss",
    "compute_training_loss",
    "bridge_residual_consistency_loss",
    "consistency_distillation_loss",
    "describe_losses",
    "optimal_transport_loss",
    "rectified_flow_matching_loss",
    "residual_amplitude_loss",
    "residual_low_frequency_loss",
    "schrodinger_bridge_loss",
    "anti_aliasing_loss",
    "streaming_causality_loss",
    "temporal_consistency_loss",
    "teacher_target_reconstruction_loss",
    "trajectory_regularization_loss",
    "uncertainty_calibration_loss",
    "wavelet_frequency_loss",
]
