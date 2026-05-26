"""Offline degradation preprocessing cost estimates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajflow_vsr.data.manifest import load_frame_manifest


@dataclass(frozen=True)
class DegradationEstimate:
    """Estimated cost for generating degraded LR frame sequences."""

    manifest_path: str
    dataset: str
    split: str
    sequences: int
    frames: int
    profiles: list[str]
    scale: float
    seconds_per_frame: float
    estimated_seconds: float
    estimated_hours: float
    input_gb: float | None
    estimated_output_gb: float | None
    suggested_shards: list[dict[str, int]]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


def estimate_degradation_from_manifest(
    manifest_path: str | Path,
    *,
    profiles: list[str] | None = None,
    scale: float = 4.0,
    seconds_per_frame: float = 0.06,
    output_overhead: float = 2.0,
    shards: int = 1,
    stat_input_bytes: bool = True,
) -> DegradationEstimate:
    """Estimate preprocessing time and output size from an existing manifest."""

    manifest_path = Path(manifest_path)
    manifest = load_frame_manifest(manifest_path)
    sequences = manifest.get("sequences", [])
    frame_count = sum(len(sequence.get("frames", [])) for sequence in sequences)
    profile_names = list(profiles or ["mild_real"])
    estimated_seconds = float(frame_count) * float(seconds_per_frame) * max(len(profile_names), 1)
    input_bytes = _input_bytes(sequences) if stat_input_bytes else None
    output_gb = None
    if input_bytes is not None:
        output_bytes = input_bytes * max(len(profile_names), 1) * float(output_overhead) / max(float(scale) ** 2, 1.0)
        output_gb = output_bytes / (1024**3)
    return DegradationEstimate(
        manifest_path=str(manifest_path),
        dataset=str(manifest.get("dataset", "")),
        split=str(manifest.get("split", "")),
        sequences=len(sequences),
        frames=frame_count,
        profiles=profile_names,
        scale=float(scale),
        seconds_per_frame=float(seconds_per_frame),
        estimated_seconds=estimated_seconds,
        estimated_hours=estimated_seconds / 3600.0,
        input_gb=None if input_bytes is None else input_bytes / (1024**3),
        estimated_output_gb=output_gb,
        suggested_shards=_shards(len(sequences), max(int(shards), 1)),
    )


def _input_bytes(sequences: list[dict[str, Any]]) -> int:
    total = 0
    for sequence in sequences:
        for frame in sequence.get("frames", []):
            path = Path(frame)
            if path.exists():
                total += path.stat().st_size
    return total


def _shards(sequence_count: int, shard_count: int) -> list[dict[str, int]]:
    shards = []
    if shard_count <= 1:
        return [{"shard": 0, "sequence_start": 0, "sequence_end": max(sequence_count - 1, 0)}]
    for shard in range(shard_count):
        start = int(round(sequence_count * shard / shard_count))
        end = int(round(sequence_count * (shard + 1) / shard_count)) - 1
        shards.append({"shard": shard, "sequence_start": start, "sequence_end": max(end, start)})
    return shards
