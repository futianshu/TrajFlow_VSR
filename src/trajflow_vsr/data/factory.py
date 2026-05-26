"""Dataset and batch factory helpers."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.data.manifest import describe_frame_manifest_config
from trajflow_vsr.data.mixed import STAGE_A_MIXED_NAME, describe_stage_a_mixed_config
from trajflow_vsr.data.synthetic import (
    SyntheticDegradationSpec,
    SyntheticVideoSpec,
    degradation_spec_from_config,
    spec_from_config,
)

CONTROLLED_MOTION_NAME = "controlled_motion"
SYNTHETIC_NAME = "synthetic"


def describe_data(config: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight data summary without importing heavy dependencies."""

    name = config.get("name", "synthetic")
    if name == "frame_manifest":
        return describe_frame_manifest_config(config)
    if name == STAGE_A_MIXED_NAME:
        return describe_stage_a_mixed_config(config)
    if name == CONTROLLED_MOTION_NAME:
        spec = spec_from_config(config)
        motion = config.get("motion", {})
        return {
            "name": CONTROLLED_MOTION_NAME,
            "batch_size": spec.batch_size,
            "frames": spec.frames,
            "channels": spec.channels,
            "height": spec.height,
            "width": spec.width,
            "scale": spec.scale,
            "motion": {
                "shift_x": int(motion.get("shift_x", 1)),
                "shift_y": int(motion.get("shift_y", 0)),
                "wrap": bool(motion.get("wrap", True)),
                "noise": float(motion.get("noise", 0.0)),
                "occlusion_size": int(motion.get("occlusion_size", 0)),
            },
        }
    if name != SYNTHETIC_NAME:
        return {"name": name, "status": "registered path-based dataset placeholder"}

    spec = spec_from_config(config)
    return {
        "name": SYNTHETIC_NAME,
        "batch_size": spec.batch_size,
        "frames": spec.frames,
        "channels": spec.channels,
        "height": spec.height,
        "width": spec.width,
        "scale": spec.scale,
        "degradation": degradation_spec_from_config(config).__dict__,
    }


def build_synthetic_spec(config: dict[str, Any]) -> SyntheticVideoSpec:
    """Build a synthetic batch spec for smoke training."""

    if config.get("name", SYNTHETIC_NAME) not in {SYNTHETIC_NAME, CONTROLLED_MOTION_NAME}:
        raise ValueError("Only synthetic and controlled_motion datasets are implemented in Phase 0")
    return spec_from_config(config)


def build_synthetic_degradation_spec(config: dict[str, Any]) -> SyntheticDegradationSpec:
    """Build a synthetic degradation spec for Stage A smoke training."""

    if config.get("name", SYNTHETIC_NAME) != SYNTHETIC_NAME:
        raise ValueError("Only the synthetic smoke dataset is implemented in Phase 0")
    return degradation_spec_from_config(config)
