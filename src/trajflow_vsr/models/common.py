"""Common model dataclasses and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelComponentSpec:
    """A serializable description of a model component."""

    name: str
    role: str
    status: str
    config: dict[str, Any]


def component_spec(name: str, role: str, config: dict[str, Any]) -> ModelComponentSpec:
    """Create a standard component description."""

    return ModelComponentSpec(name=name, role=role, status="interface-ready", config=dict(config))
