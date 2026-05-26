"""Data loading, degradation, and sampling utilities."""

from trajflow_vsr.data.degradation import (
    DEGRADATION_PROFILE_NAMES,
    RealisticDegradationProfile,
    apply_realistic_degradation,
    build_degraded_frame_dataset,
    degradation_profile_from_config,
    degrade_frame_sequence,
    preset_degradation_profile,
    write_training_frame,
)
from trajflow_vsr.data.estimate import DegradationEstimate, estimate_degradation_from_manifest
from trajflow_vsr.data.manifest import (
    attach_hr_sequences,
    build_clip_records,
    build_frame_manifest,
    discover_frame_sequences,
    load_frame_manifest,
    make_frame_manifest_batch,
    make_stage_a_frame_manifest_batch,
)
from trajflow_vsr.data.mixed import (
    STAGE_A_MIXED_NAME,
    describe_stage_a_mixed_config,
    make_stage_a_mixed_batch,
    resolve_stage_a_mixed_source,
)

__all__ = [
    "DEGRADATION_PROFILE_NAMES",
    "DegradationEstimate",
    "RealisticDegradationProfile",
    "STAGE_A_MIXED_NAME",
    "apply_realistic_degradation",
    "attach_hr_sequences",
    "build_clip_records",
    "build_degraded_frame_dataset",
    "build_frame_manifest",
    "degradation_profile_from_config",
    "describe_stage_a_mixed_config",
    "degrade_frame_sequence",
    "discover_frame_sequences",
    "estimate_degradation_from_manifest",
    "load_frame_manifest",
    "make_frame_manifest_batch",
    "make_stage_a_frame_manifest_batch",
    "make_stage_a_mixed_batch",
    "preset_degradation_profile",
    "resolve_stage_a_mixed_source",
    "write_training_frame",
]

from trajflow_vsr.data.factory import (
    CONTROLLED_MOTION_NAME,
    SYNTHETIC_NAME,
    build_synthetic_degradation_spec,
    build_synthetic_spec,
    describe_data,
)
from trajflow_vsr.data.synthetic import (
    SyntheticDegradationSpec,
    SyntheticVideoSpec,
    make_controlled_motion_batch,
    make_stage_a_batch,
    make_synthetic_batch,
)

__all__ += [
    "CONTROLLED_MOTION_NAME",
    "SYNTHETIC_NAME",
    "SyntheticDegradationSpec",
    "SyntheticVideoSpec",
    "build_synthetic_degradation_spec",
    "build_synthetic_spec",
    "describe_data",
    "make_controlled_motion_batch",
    "make_stage_a_batch",
    "make_synthetic_batch",
]
