"""Export lightweight visualization artifacts for model diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def write_ppm(path: str | Path, image: Any) -> Path:
    """Write an RGB image tensor to a dependency-light PPM file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_hwc = to_uint8_hwc(image)
    height = int(image_hwc.shape[0])
    width = int(image_hwc.shape[1])
    payload = bytes(image_hwc.flatten().tolist())
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)
    return path


def write_image(path: str | Path, image: Any) -> Path:
    """Write an RGB image tensor, using imageio for common formats."""

    path = Path(path)
    if path.suffix.lower() == ".ppm":
        return write_ppm(path, image)

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("imageio is required for non-PPM visualization export") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, to_uint8_hwc(image).numpy())
    return path


def to_uint8_hwc(image: Any, min_value: float | None = None, max_value: float | None = None) -> Any:
    """Normalize a tensor-like image to uint8 HWC RGB layout."""

    torch = require_torch()
    tensor = torch.as_tensor(image).detach().float().cpu()
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

    while tensor.ndim > 3:
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got {tuple(tensor.shape)}")

    if tensor.shape[0] in {1, 2, 3, 4}:
        chw = tensor[:3]
    elif tensor.shape[-1] in {1, 2, 3, 4}:
        chw = tensor[..., :3].permute(2, 0, 1)
    else:
        chw = tensor[:1]

    if chw.shape[0] == 1:
        chw = chw.expand(3, -1, -1)
    elif chw.shape[0] == 2:
        chw = torch.cat([chw, chw[:1]], dim=0)

    low = float(chw.amin()) if min_value is None else float(min_value)
    high = float(chw.amax()) if max_value is None else float(max_value)
    denom = max(high - low, 1e-8)
    chw = ((chw - low) / denom).clamp(0.0, 1.0)
    return (chw * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()


def scalar_heatmap(map_tensor: Any, kind: str = "uncertainty", normalize: bool = True) -> Any:
    """Convert a scalar map to an RGB heatmap tensor in CHW layout."""

    torch = require_torch()
    value = torch.as_tensor(map_tensor).detach().float().cpu()
    while value.ndim > 2:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"Expected scalar map with 2 dimensions, got {tuple(value.shape)}")

    value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
    if normalize:
        value = (value - value.amin()) / (value.amax() - value.amin()).clamp_min(1e-8)
    else:
        value = value.clamp(0.0, 1.0)
    if kind == "reliability":
        red = 1.0 - value
        green = value
        blue = 0.15 * (1.0 - (2.0 * value - 1.0).abs())
    elif kind == "signed":
        red = value
        green = 1.0 - (2.0 * value - 1.0).abs()
        blue = 1.0 - value
    else:
        red = value
        green = 0.35 * (1.0 - (2.0 * value - 1.0).abs())
        blue = 1.0 - value
    return torch.stack([red, green, blue], dim=0).clamp(0.0, 1.0)


def export_uncertainty_maps(
    outputs: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "uncertainty",
    batch_index: int = 0,
    max_frames: int | None = None,
    image_format: str = "ppm",
) -> dict[str, list[str]]:
    """Export artifact, reliability, and uncertainty maps as images."""

    output_dir = Path(output_dir)
    uncertainty = outputs.get("uncertainty", {})
    if not isinstance(uncertainty, dict):
        raise ValueError("outputs['uncertainty'] must be a mapping")

    files: dict[str, list[str]] = {}
    summary: dict[str, dict[str, float]] = {}
    for name in ["artifact", "reliability", "motion_uncertainty", "texture_uncertainty"]:
        tensor = uncertainty.get(name)
        if tensor is None:
            continue

        selected = _select_batch_video(tensor, batch_index=batch_index)
        frame_count = selected.shape[0]
        if max_frames is not None:
            frame_count = min(frame_count, max(int(max_frames), 0))

        files[name] = []
        for frame_idx in range(frame_count):
            scalar = selected[frame_idx].mean(dim=0)
            kind = "reliability" if name == "reliability" else "uncertainty"
            path = _image_path(output_dir, f"{prefix}_{name}_t{frame_idx:03d}", image_format)
            files[name].append(str(write_image(path, scalar_heatmap(scalar, kind=kind, normalize=False))))

        summary[name] = _tensor_stats(selected)

    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files["summary"] = [str(summary_path)]
    return files


def export_trajectory_maps(
    outputs: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "trajectory",
    batch_index: int = 0,
    max_frames: int | None = None,
    top_edges: int = 32,
    image_format: str = "ppm",
) -> dict[str, list[str]]:
    """Export soft-trajectory maps plus top transport edges."""

    torch = require_torch()
    output_dir = Path(output_dir)
    transport = outputs.get("transport", {})
    if not isinstance(transport, dict):
        raise ValueError("outputs['transport'] must be a mapping")

    plan = transport.get("transport_plan")
    bridge_grid = transport.get("bridge_grid")
    if plan is None or bridge_grid is None:
        raise ValueError("transport outputs must include transport_plan and bridge_grid")

    plan = torch.as_tensor(plan).detach().float().cpu()[batch_index]
    bridge_grid = torch.as_tensor(bridge_grid).detach().float().cpu()[batch_index]
    frames, height, width = [int(item) for item in bridge_grid.shape[:3]]
    token_count = frames * height * width
    if plan.shape != (token_count, token_count):
        raise ValueError(f"Expected plan shape {(token_count, token_count)}, got {tuple(plan.shape)}")

    frame_coord, y_coord, x_coord = _token_coordinates(frames, height, width, dtype=plan.dtype)
    target_coords = torch.stack([frame_coord, y_coord, x_coord], dim=-1)
    expected = plan @ target_coords
    source_coords = target_coords

    expected_frame = expected[:, 0].reshape(frames, height, width)
    dx = (expected[:, 2] - source_coords[:, 2]).reshape(frames, height, width)
    dy = (expected[:, 1] - source_coords[:, 1]).reshape(frames, height, width)
    magnitude = torch.sqrt(dx.square() + dy.square())

    frame_count = frames if max_frames is None else min(frames, max(int(max_frames), 0))
    files: dict[str, list[str]] = {
        "target_frame": [],
        "dx": [],
        "dy": [],
        "motion_magnitude": [],
    }
    for frame_idx in range(frame_count):
        files["target_frame"].append(
            str(
                write_image(
                    _image_path(output_dir, f"{prefix}_target_frame_t{frame_idx:03d}", image_format),
                    scalar_heatmap(expected_frame[frame_idx], kind="uncertainty"),
                )
            )
        )
        files["dx"].append(
            str(
                write_image(
                    _image_path(output_dir, f"{prefix}_dx_t{frame_idx:03d}", image_format),
                    _signed_map(dx[frame_idx]),
                )
            )
        )
        files["dy"].append(
            str(
                write_image(
                    _image_path(output_dir, f"{prefix}_dy_t{frame_idx:03d}", image_format),
                    _signed_map(dy[frame_idx]),
                )
            )
        )
        files["motion_magnitude"].append(
            str(
                write_image(
                    _image_path(output_dir, f"{prefix}_motion_magnitude_t{frame_idx:03d}", image_format),
                    scalar_heatmap(magnitude[frame_idx], kind="uncertainty"),
                )
            )
        )

    graph = _trajectory_graph_summary(
        plan=plan,
        frames=frames,
        height=height,
        width=width,
        transport=transport,
        top_edges=top_edges,
    )
    graph_path = output_dir / f"{prefix}_graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    files["graph"] = [str(graph_path)]
    return files


def export_sample_frames(
    outputs: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "sample",
    batch_index: int = 0,
    max_frames: int | None = None,
    image_format: str = "ppm",
) -> dict[str, list[str]]:
    """Export model output frames for quick posterior/sample inspection."""

    output_dir = Path(output_dir)
    files: dict[str, list[str]] = {}
    for name in ["hr", "hr_raw"]:
        video = outputs.get(name)
        if video is None:
            continue
        selected = _select_batch_video(video, batch_index=batch_index)
        frame_count = selected.shape[0]
        if max_frames is not None:
            frame_count = min(frame_count, max(int(max_frames), 0))
        files[name] = []
        for frame_idx in range(frame_count):
            path = _image_path(output_dir, f"{prefix}_{name}_t{frame_idx:03d}", image_format)
            files[name].append(str(write_image(path, selected[frame_idx])))
    return files


def export_visualization_bundle(
    outputs: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "viz",
    batch_index: int = 0,
    max_frames: int | None = None,
    top_edges: int = 32,
    image_format: str = "ppm",
) -> dict[str, list[str]]:
    """Export uncertainty, trajectory, and sample artifacts together."""

    files: dict[str, list[str]] = {}
    files.update(
        {
            f"uncertainty_{key}": value
            for key, value in export_uncertainty_maps(
                outputs,
                output_dir,
                prefix=f"{prefix}_uncertainty",
                batch_index=batch_index,
                max_frames=max_frames,
                image_format=image_format,
            ).items()
        }
    )
    files.update(
        {
            f"trajectory_{key}": value
            for key, value in export_trajectory_maps(
                outputs,
                output_dir,
                prefix=f"{prefix}_trajectory",
                batch_index=batch_index,
                max_frames=max_frames,
                top_edges=top_edges,
                image_format=image_format,
            ).items()
        }
    )
    files.update(
        {
            f"sample_{key}": value
            for key, value in export_sample_frames(
                outputs,
                output_dir,
                prefix=f"{prefix}_sample",
                batch_index=batch_index,
                max_frames=max_frames,
                image_format=image_format,
            ).items()
        }
    )
    return files


def _image_path(output_dir: Path, stem: str, image_format: str) -> Path:
    suffix = str(image_format or "ppm").lower().lstrip(".")
    return output_dir / f"{stem}.{suffix}"


def _select_batch_video(tensor: Any, batch_index: int) -> Any:
    torch = require_torch()
    video = torch.as_tensor(tensor).detach().float().cpu()
    if video.ndim == 5:
        return video[batch_index]
    if video.ndim == 4:
        return video
    if video.ndim == 3:
        return video.unsqueeze(0)
    raise ValueError(f"Expected video tensor with 3 to 5 dimensions, got {tuple(video.shape)}")


def _tensor_stats(tensor: Any) -> dict[str, float]:
    torch = require_torch()
    value = torch.as_tensor(tensor).detach().float().cpu()
    return {
        "min": float(value.amin()),
        "max": float(value.amax()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
    }


def _token_coordinates(frames: int, height: int, width: int, dtype: Any) -> tuple[Any, Any, Any]:
    torch = require_torch()
    frame_idx = torch.arange(frames, dtype=dtype).repeat_interleave(height * width)
    y_idx = torch.arange(height, dtype=dtype).repeat_interleave(width).repeat(frames)
    x_idx = torch.arange(width, dtype=dtype).repeat(frames * height)
    return frame_idx, y_idx, x_idx


def _signed_map(value: Any) -> Any:
    torch = require_torch()
    tensor = torch.as_tensor(value).detach().float().cpu()
    denom = tensor.abs().amax().clamp_min(1e-8)
    return scalar_heatmap(0.5 + 0.5 * tensor / denom, kind="signed", normalize=False)


def _trajectory_graph_summary(
    plan: Any,
    frames: int,
    height: int,
    width: int,
    transport: dict[str, Any],
    top_edges: int,
) -> dict[str, Any]:
    torch = require_torch()
    flattened = plan.flatten()
    edge_count = min(max(int(top_edges), 0), int(flattened.numel()))
    values, indices = torch.topk(flattened, k=edge_count) if edge_count else ([], [])

    edges = []
    for probability, flat_index in zip(values, indices, strict=False):
        source = int(flat_index) // int(plan.shape[1])
        target = int(flat_index) % int(plan.shape[1])
        edges.append(
            {
                "probability": float(probability),
                "source": _token_index_to_coord(source, height=height, width=width),
                "target": _token_index_to_coord(target, height=height, width=width),
            }
        )

    row_entropy = -(plan.clamp_min(1e-12) * plan.clamp_min(1e-12).log()).sum(dim=-1)
    causal_violation = transport.get("causal_violation")
    if causal_violation is not None:
        causal_violation = float(torch.as_tensor(causal_violation).detach().cpu())
    return {
        "frames": frames,
        "height": height,
        "width": width,
        "token_count": frames * height * width,
        "causal": bool(transport.get("causal", False)),
        "causal_violation": causal_violation,
        "row_entropy_mean": float(row_entropy.mean()),
        "top_edges": edges,
    }


def _token_index_to_coord(index: int, height: int, width: int) -> dict[str, int]:
    frame = index // (height * width)
    remainder = index % (height * width)
    return {
        "frame": int(frame),
        "y": int(remainder // width),
        "x": int(remainder % width),
    }
