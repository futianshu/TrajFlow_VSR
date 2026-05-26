"""Optional official metric adapters with explicit fallback reporting."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from functools import lru_cache
from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


OFFICIAL_METRIC_BACKENDS = {
    "lpips": "lpips",
    "dists": "piq",
    "niqe": ("pyiqa", "piq"),
    "vmaf": "ffmpeg",
    "musiq": "pyiqa",
    "clipiqa": "pyiqa",
    "fvd": None,
}

PROXY_METRICS = {"lpips", "dists", "tof"}


def metric_backend_status(metric_names: list[str], requested_backend: str = "proxy") -> dict[str, dict[str, Any]]:
    """Return availability and fallback status for paper-facing metrics."""

    backend = str(requested_backend or "proxy").lower()
    status = {}
    for metric in metric_names:
        name = str(metric).lower()
        official_backend = OFFICIAL_METRIC_BACKENDS.get(name)
        if official_backend is None and name not in OFFICIAL_METRIC_BACKENDS:
            continue
        available_backend = _available_backend(official_backend)
        available = available_backend is not None
        use_official = backend == "official" and bool(available)
        status[name] = {
            "requested_backend": backend,
            "official_backend": _backend_label(official_backend),
            "official_available": bool(available),
            "used_backend": available_backend if use_official else ("proxy" if name in PROXY_METRICS else "unavailable"),
            "value_kind": "official" if use_official else ("proxy" if name in PROXY_METRICS else "missing"),
        }
    return status


def compute_official_metric(metric_name: str, prediction: Any, target: Any):
    """Compute an optional official metric when its dependency is installed."""

    name = str(metric_name).lower()
    if name == "lpips":
        return _compute_lpips(prediction, target)
    if name == "dists":
        return _compute_piq_metric("DISTS", prediction, target)
    if name == "niqe":
        if _backend_available("pyiqa"):
            return _compute_pyiqa_metric("niqe", prediction)
        return _compute_piq_function("niqe", prediction)
    if name in {"musiq", "clipiqa"}:
        return _compute_pyiqa_metric(name, prediction)
    raise RuntimeError(f"No in-process official adapter is available for metric: {metric_name}")


def _available_backend(backend: str | tuple[str, ...] | None) -> str | None:
    if isinstance(backend, tuple):
        for item in backend:
            if _backend_available(item):
                return item
        return None
    if _backend_available(backend):
        return backend
    return None


def _backend_available(backend: str | None) -> bool:
    if backend is None:
        return False
    if backend == "ffmpeg":
        return _ffmpeg_has_libvmaf()
    return importlib.util.find_spec(backend) is not None


def _backend_label(backend: str | tuple[str, ...] | None) -> str | None:
    if isinstance(backend, tuple):
        return "|".join(backend)
    return backend


@lru_cache(maxsize=1)
def _ffmpeg_has_libvmaf() -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "libvmaf" in f"{result.stdout}\n{result.stderr}"


def _flatten_video(video: Any):
    torch = require_torch()
    tensor = torch.as_tensor(video).float().clamp(0.0, 1.0)
    if tensor.ndim != 5:
        raise ValueError("Expected B,T,C,H,W tensor")
    return tensor.flatten(0, 1)


@lru_cache(maxsize=2)
def _lpips_model(device_type: str):
    import lpips  # type: ignore[import-not-found]

    torch = require_torch()
    model = lpips.LPIPS(net="alex")
    device = torch.device(device_type)
    model = model.to(device)
    model.eval()
    return model


def _compute_lpips(prediction: Any, target: Any):
    torch = require_torch()
    pred = _flatten_video(prediction)
    ref = _flatten_video(target).to(device=pred.device, dtype=pred.dtype)
    model = _lpips_model(str(pred.device))
    with torch.no_grad():
        value = model(pred * 2.0 - 1.0, ref * 2.0 - 1.0)
    return value.mean()


@lru_cache(maxsize=8)
def _piq_metric(metric_class: str, device_type: str):
    import piq  # type: ignore[import-not-found]

    torch = require_torch()
    metric = getattr(piq, metric_class)()
    metric = metric.to(torch.device(device_type))
    metric.eval()
    return metric


def _compute_piq_metric(metric_class: str, prediction: Any, target: Any):
    torch = require_torch()
    pred = _flatten_video(prediction)
    ref = _flatten_video(target).to(device=pred.device, dtype=pred.dtype)
    metric = _piq_metric(metric_class, str(pred.device))
    with torch.no_grad():
        return metric(pred, ref).mean()


def _compute_piq_function(function_name: str, prediction: Any):
    import piq  # type: ignore[import-not-found]

    torch = require_torch()
    pred = _flatten_video(prediction)
    fn = getattr(piq, function_name)
    with torch.no_grad():
        return fn(pred).mean()


@lru_cache(maxsize=8)
def _pyiqa_metric(metric_name: str, device_type: str):
    import pyiqa  # type: ignore[import-not-found]

    torch = require_torch()
    metric = pyiqa.create_metric(metric_name, device=torch.device(device_type))
    metric.eval()
    return metric


def _compute_pyiqa_metric(metric_name: str, prediction: Any):
    torch = require_torch()
    pred = _flatten_video(prediction)
    metric = _pyiqa_metric(metric_name, str(pred.device))
    with torch.no_grad():
        return metric(pred).mean()
