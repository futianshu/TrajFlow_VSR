"""Mixed Stage A sampling between synthetic and real degraded manifests."""

from __future__ import annotations

import copy
from typing import Any

from trajflow_vsr.data.manifest import describe_frame_manifest_config, make_stage_a_frame_manifest_batch
from trajflow_vsr.data.synthetic import degradation_spec_from_config, make_stage_a_batch, spec_from_config


STAGE_A_MIXED_NAME = "stage_a_mixed"


def describe_stage_a_mixed_config(config: dict[str, Any]) -> dict[str, Any]:
    """Describe a Stage A mixed synthetic/real data config."""

    synthetic = _synthetic_config(config)
    spec = spec_from_config(synthetic)
    return {
        "name": STAGE_A_MIXED_NAME,
        "schedule": str(config.get("schedule", "alternating")),
        "synthetic_steps": int(config.get("synthetic_steps", 1)),
        "real_steps": int(config.get("real_steps", 1)),
        "synthetic_warmup_steps": int(config.get("synthetic_warmup_steps", 0)),
        "synthetic": {
            "name": "synthetic",
            "batch_size": spec.batch_size,
            "frames": spec.frames,
            "channels": spec.channels,
            "height": spec.height,
            "width": spec.width,
            "scale": spec.scale,
            "degradation": degradation_spec_from_config(synthetic).__dict__,
        },
        "real": describe_frame_manifest_config(_real_config(config)),
    }


def make_stage_a_mixed_batch(config: dict[str, Any], step: int = 0, device: str = "cpu") -> dict[str, Any]:
    """Create one Stage A batch from either synthetic or real manifest data."""

    source = resolve_stage_a_mixed_source(config, step=step)
    if source == "synthetic":
        synthetic = _synthetic_config(config)
        batch = make_stage_a_batch(
            spec_from_config(synthetic),
            degradation=degradation_spec_from_config(synthetic),
            device=device,
        )
    elif source == "real":
        real = _real_config(config)
        if "clip_index" not in real:
            real["clip_index"] = int(config.get("real_clip_offset", 0)) + int(step)
        batch = make_stage_a_frame_manifest_batch(real, device=device)
    else:
        raise ValueError(f"Unsupported Stage A mixed data source: {source}")

    batch["source"] = source
    metadata = batch.get("metadata")
    if isinstance(metadata, list):
        for item in metadata:
            if isinstance(item, dict):
                item["source"] = source
                item["mixed_step"] = int(step)
    else:
        batch["metadata"] = [{"source": source, "mixed_step": int(step)}]
    return batch


def resolve_stage_a_mixed_source(config: dict[str, Any], step: int = 0) -> str:
    """Resolve which Stage A source should be sampled at a training step."""

    schedule = str(config.get("schedule", "alternating")).lower()
    step = int(step)
    warmup = max(int(config.get("synthetic_warmup_steps", 0)), 0)
    if step < warmup:
        return "synthetic"
    if schedule in {"synthetic", "synthetic_only"}:
        return "synthetic"
    if schedule in {"real", "real_only", "manifest"}:
        return "real"
    if schedule != "alternating":
        raise ValueError(f"Unsupported Stage A mixed schedule: {schedule}")

    synthetic_steps = max(int(config.get("synthetic_steps", 1)), 0)
    real_steps = max(int(config.get("real_steps", 1)), 0)
    cycle = synthetic_steps + real_steps
    if cycle <= 0:
        raise ValueError("Stage A mixed schedule needs synthetic_steps or real_steps > 0")
    position = (step - warmup) % cycle
    return "synthetic" if position < synthetic_steps else "real"


def _synthetic_config(config: dict[str, Any]) -> dict[str, Any]:
    synthetic = copy.deepcopy(config.get("synthetic", {}))
    synthetic.setdefault("name", "synthetic")
    return synthetic


def _real_config(config: dict[str, Any]) -> dict[str, Any]:
    real = copy.deepcopy(config.get("real", config.get("manifest", {})))
    real.setdefault("name", "frame_manifest")
    return real
