"""Curriculum scheduling utilities for staged training."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CurriculumState:
    """Resolved per-step curriculum settings."""

    enabled: bool
    phase: str | None
    progress: float
    loss_config: dict[str, Any]
    freeze_components: tuple[str, ...]

    @property
    def numeric_loss_weights(self) -> dict[str, float]:
        """Return numeric loss weights for compact history logging."""

        weights = {}
        for key, value in self.loss_config.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                weights[str(key)] = float(value)
        return weights


class TrainingCurriculum:
    """Resolve phase-specific loss weights and component freezes."""

    def __init__(self, config: dict[str, Any] | None, base_losses: dict[str, Any]):
        self.config = config if isinstance(config, dict) else {}
        self.base_losses = copy.deepcopy(base_losses)
        self.enabled = bool(self.config.get("enabled", False))
        self.phases = _normalize_phases(self.config.get("phases", {}))

    def describe(self) -> dict[str, Any]:
        """Return a serializable summary for run manifests."""

        return {
            "enabled": self.enabled,
            "phase_count": len(self.phases),
            "phases": [
                {
                    "name": phase["name"],
                    "start_step": phase["start_step"],
                    "end_step": phase["end_step"],
                    "freeze_components": list(phase.get("freeze_components", ())),
                }
                for phase in self.phases
            ],
        }

    def state_for_step(self, step: int) -> CurriculumState:
        """Return effective losses and frozen components for one step."""

        base = copy.deepcopy(self.base_losses)
        if not self.enabled:
            return CurriculumState(
                enabled=False,
                phase=None,
                progress=0.0,
                loss_config=base,
                freeze_components=(),
            )

        phase = self._phase_for_step(step)
        if phase is None:
            return CurriculumState(
                enabled=True,
                phase="base",
                progress=1.0,
                loss_config=base,
                freeze_components=(),
            )

        progress = _phase_progress(step, phase)
        losses = _apply_phase_losses(base, phase, progress=progress)
        return CurriculumState(
            enabled=True,
            phase=str(phase["name"]),
            progress=progress,
            loss_config=losses,
            freeze_components=tuple(str(item) for item in phase.get("freeze_components", ())),
        )

    def _phase_for_step(self, step: int) -> dict[str, Any] | None:
        for phase in self.phases:
            start = int(phase["start_step"])
            end = phase["end_step"]
            if step < start:
                continue
            if end is None or step < int(end):
                return phase
        return None


def _normalize_phases(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for index, phase in enumerate(raw):
            if not isinstance(phase, dict):
                raise ValueError(f"curriculum phase must be a mapping: {index}")
            name = phase.get("name", f"phase_{index}")
            items.append((name, phase))
    else:
        raise ValueError("curriculum.phases must be a mapping or list")

    phases = []
    for name, phase in items:
        if not isinstance(phase, dict):
            raise ValueError(f"curriculum phase must be a mapping: {name}")
        normalized = copy.deepcopy(phase)
        normalized["name"] = str(normalized.get("name", name))
        start = int(normalized.get("start_step", normalized.get("start", 0)))
        end = normalized.get("end_step", normalized.get("until_step"))
        if end is None and "steps" in normalized:
            end = start + int(normalized["steps"])
        end = None if end is None else int(end)
        if end is not None and end < start:
            raise ValueError(f"curriculum phase end_step must be >= start_step: {name}")
        normalized["start_step"] = start
        normalized["end_step"] = end
        normalized["freeze_components"] = _string_tuple(
            normalized.get("freeze_components", normalized.get("frozen_components", ()))
        )
        phases.append(normalized)
    return sorted(phases, key=lambda phase: (int(phase["start_step"]), str(phase["name"])))


def _phase_progress(step: int, phase: dict[str, Any]) -> float:
    start = int(phase["start_step"])
    end = phase["end_step"]
    if end is None or int(end) <= start + 1:
        return 0.0
    denominator = max(int(end) - start - 1, 1)
    return min(max((int(step) - start) / float(denominator), 0.0), 1.0)


def _apply_phase_losses(base: dict[str, Any], phase: dict[str, Any], *, progress: float) -> dict[str, Any]:
    losses = copy.deepcopy(base)

    for key, multiplier in _mapping(phase.get("loss_multipliers", {})).items():
        losses[str(key)] = float(losses.get(str(key), 0.0)) * float(multiplier)

    for key, value in _mapping(phase.get("loss_overrides", {})).items():
        losses[str(key)] = value

    for key, schedule in _mapping(phase.get("loss_schedules", {})).items():
        losses[str(key)] = _scheduled_value(
            schedule,
            progress=progress,
            fallback=float(losses.get(str(key), 0.0)),
        )

    return losses


def _scheduled_value(schedule: Any, *, progress: float, fallback: float) -> float:
    if isinstance(schedule, int | float) and not isinstance(schedule, bool):
        return float(schedule)
    if not isinstance(schedule, dict):
        raise ValueError(f"loss schedule must be a number or mapping: {schedule}")
    start = float(schedule.get("start", fallback))
    end = float(schedule.get("end", start))
    mode = str(schedule.get("mode", "linear")).lower()
    if mode in {"constant", "none"}:
        return start
    if mode in {"linear", "lin"}:
        factor = progress
    elif mode in {"cosine", "cos"}:
        factor = 0.5 - 0.5 * math.cos(math.pi * progress)
    else:
        raise ValueError(f"Unsupported curriculum schedule mode: {mode}")
    return start + (end - start) * factor


def _mapping(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping, got: {raw}")
    return raw


def _string_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None or raw == "":
        return ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list | tuple):
        return tuple(str(item) for item in raw)
    raise ValueError(f"Expected string or list of strings, got: {raw}")
