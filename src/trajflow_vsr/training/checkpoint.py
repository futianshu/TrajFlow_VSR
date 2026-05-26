"""Checkpoint and experiment artifact helpers for training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def save_training_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any,
    step: int,
    summary: dict[str, Any],
    history: list[dict[str, Any]],
    scheduler: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a resumable training checkpoint."""

    torch = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "step": int(step),
        "summary": summary,
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": metadata or {},
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
    return str(path)


def load_training_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any,
    device: str,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Load model/optimizer state and return resume metadata."""

    torch = require_torch()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint payload: {path}")

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain a model state dict: {path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    optimizer_loaded = False
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
        optimizer_loaded = True
    scheduler_state = checkpoint.get("scheduler_state_dict")
    scheduler_loaded = False
    if scheduler is not None and isinstance(scheduler_state, dict):
        scheduler.load_state_dict(scheduler_state)
        scheduler_loaded = True

    step = int(checkpoint.get("step", -1))
    history = checkpoint.get("history", [])
    if not isinstance(history, list):
        history = []

    return {
        "loaded": True,
        "path": str(path),
        "step": step,
        "next_step": step + 1,
        "history": history,
        "optimizer_loaded": optimizer_loaded,
        "scheduler_loaded": scheduler_loaded,
        "metadata": checkpoint.get("metadata", {}),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def checkpoint_step_path(output_dir: str | Path, step: int) -> Path:
    """Return a stable per-step checkpoint path."""

    return Path(output_dir) / f"step_{int(step):06d}.pt"


def checkpoint_final_path(output_dir: str | Path) -> Path:
    """Return the final checkpoint path for a run."""

    return Path(output_dir) / "final.pt"


def write_json_artifact(path: str | Path, payload: dict[str, Any]) -> str:
    """Write a JSON artifact with UTF-8 encoding."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
