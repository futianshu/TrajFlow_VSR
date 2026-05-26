"""Pretrained component loading for staged training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


DEFAULT_STAGE_A_COMPONENTS = ["tokenizer", "uncertainty"]


def load_pretrained_components(
    model: Any,
    path: str | Path,
    components: list[str] | tuple[str, ...] | None = None,
    freeze_components: list[str] | tuple[str, ...] | None = None,
    device: str = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    """Load selected same-name components from a checkpoint into a model."""

    torch = require_torch()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported pretrained checkpoint payload: {path}")
    source_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(source_state, dict):
        raise ValueError(f"Pretrained checkpoint does not contain a model state dict: {path}")

    components = _normalize_components(components)
    target_state = model.state_dict()
    selected_state = {}
    skipped_shape_mismatch = []
    for key, value in source_state.items():
        component = _component_for_key(key, components)
        if component is None:
            continue
        if key not in target_state:
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            skipped_shape_mismatch.append(
                {
                    "key": key,
                    "source_shape": list(value.shape),
                    "target_shape": list(target_state[key].shape),
                }
            )
            continue
        selected_state[key] = value

    if strict and skipped_shape_mismatch:
        keys = ", ".join(item["key"] for item in skipped_shape_mismatch[:5])
        raise ValueError(f"Pretrained component shape mismatch: {keys}")
    if not selected_state:
        raise ValueError(f"No compatible pretrained keys found for components {components} in {path}")

    missing, unexpected = model.load_state_dict(selected_state, strict=False)
    frozen = freeze_model_components(model, freeze_components or [])
    return {
        "loaded": True,
        "path": str(path),
        "components": components,
        "loaded_keys": sorted(selected_state),
        "loaded_key_count": len(selected_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "skipped_shape_mismatch": skipped_shape_mismatch,
        "frozen_components": list(freeze_components or []),
        "frozen_parameters": frozen,
    }


def freeze_model_components(model: Any, components: list[str] | tuple[str, ...]) -> list[str]:
    """Freeze parameters whose names belong to selected top-level components."""

    normalized = _normalize_components(components, default=[])
    frozen = []
    for name, parameter in model.named_parameters():
        if _component_for_key(name, normalized) is None:
            continue
        parameter.requires_grad_(False)
        frozen.append(name)
    return frozen


def trainable_parameters(model: Any) -> list[Any]:
    """Return model parameters that still require gradients."""

    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def pretrained_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight summary of model.pretrained config."""

    pretrained = config.get("pretrained", {})
    if not isinstance(pretrained, dict) or not pretrained.get("path"):
        return {"enabled": False}
    return {
        "enabled": True,
        "path": pretrained.get("path"),
        "components": _normalize_components(pretrained.get("components")),
        "freeze_components": _normalize_components(pretrained.get("freeze_components"), default=[]),
        "strict": bool(pretrained.get("strict", False)),
    }


def _normalize_components(components: Any, default: list[str] | None = None) -> list[str]:
    if default is None:
        default = DEFAULT_STAGE_A_COMPONENTS
    if components is None:
        return list(default)
    if isinstance(components, str):
        return [components]
    return [str(component) for component in components]


def _component_for_key(key: str, components: list[str]) -> str | None:
    for component in components:
        if key == component or key.startswith(f"{component}."):
            return component
    return None
