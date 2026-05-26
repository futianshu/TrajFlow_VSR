"""Optional PyTorch helpers."""

from __future__ import annotations


class OptionalDependencyError(ImportError):
    """Raised when an optional runtime dependency is required."""


def is_torch_available() -> bool:
    """Return whether PyTorch can be imported."""

    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def require_torch():
    """Import and return torch, or raise a clear project-specific error."""

    try:
        import torch

        return torch
    except ImportError as exc:
        raise OptionalDependencyError(
            "PyTorch is required for real training/inference. Install it with uv, "
            "for example: uv add torch torchvision"
        ) from exc


def require_torch_nn():
    """Import and return torch plus torch.nn."""

    torch = require_torch()
    from torch import nn

    return torch, nn
