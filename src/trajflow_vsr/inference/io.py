"""Image and video IO helpers for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch
from trajflow_vsr.visualization import write_image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff", ".webp"}


def read_video_source(path: str | Path, device: str = "cpu", max_frames: int | None = None) -> Any:
    """Read an image sequence directory or video/image file into B,T,C,H,W."""

    torch = require_torch()
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("imageio is required for inference input loading") from exc

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Inference input does not exist: {source}")

    arrays = []
    if source.is_dir():
        frame_paths = [item for item in sorted(source.iterdir()) if item.suffix.lower() in IMAGE_SUFFIXES]
        if max_frames is not None:
            frame_paths = frame_paths[: max(int(max_frames), 0)]
        arrays = [iio.imread(frame_path) for frame_path in frame_paths]
    else:
        try:
            iterator = iio.imiter(source)
            for idx, frame in enumerate(iterator):
                if max_frames is not None and idx >= int(max_frames):
                    break
                arrays.append(frame)
        except Exception:
            array = iio.imread(source)
            if getattr(array, "ndim", 0) == 4:
                arrays = [array[idx] for idx in range(array.shape[0])]
            else:
                arrays = [array]
            if max_frames is not None:
                arrays = arrays[: max(int(max_frames), 0)]

    if not arrays:
        raise ValueError(f"No readable frames found in inference input: {source}")

    frames = [_array_to_chw_float(array) for array in arrays]
    return torch.stack(frames, dim=0).unsqueeze(0).to(device=device)


def write_video_frames(
    video: Any,
    output_dir: str | Path,
    prefix: str = "frame",
    image_format: str = "png",
    batch_index: int = 0,
) -> list[str]:
    """Write B,T,C,H,W or T,C,H,W video frames to an image directory."""

    output_dir = Path(output_dir)
    selected = _select_video(video, batch_index=batch_index)
    paths = []
    for frame_idx in range(selected.shape[0]):
        path = output_dir / f"{prefix}_{frame_idx:04d}.{str(image_format).lstrip('.')}"
        paths.append(str(write_image(path, selected[frame_idx])))
    return paths


def write_video_file(video: Any, path: str | Path, fps: int = 24, batch_index: int = 0) -> str:
    """Write a video tensor to a single video file when imageio-ffmpeg is available."""

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("imageio is required for video file export") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = _select_video(video, batch_index=batch_index)
    frames = [writeable_uint8_frame(frame) for frame in selected]
    iio.imwrite(path, frames, fps=fps)
    return str(path)


def writeable_uint8_frame(frame: Any) -> Any:
    """Convert one tensor frame to uint8 HWC numpy layout."""

    return write_image_uint8(frame).numpy()


def write_image_uint8(image: Any) -> Any:
    """Return a uint8 HWC tensor using the visualization conversion path."""

    from trajflow_vsr.visualization import to_uint8_hwc

    return to_uint8_hwc(image)


def _array_to_chw_float(array: Any) -> Any:
    torch = require_torch()
    tensor = torch.as_tensor(array)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image frame with H,W,C layout, got {tuple(tensor.shape)}")

    if tensor.shape[-1] == 1:
        tensor = tensor.expand(-1, -1, 3)
    elif tensor.shape[-1] >= 3:
        tensor = tensor[..., :3]
    else:
        raise ValueError(f"Expected 1, 3, or 4 channels, got {tensor.shape[-1]}")

    tensor = tensor.permute(2, 0, 1).contiguous().float()
    if tensor.amax() > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _select_video(video: Any, batch_index: int) -> Any:
    torch = require_torch()
    tensor = torch.as_tensor(video).detach().float().cpu()
    if tensor.ndim == 5:
        return tensor[batch_index]
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    raise ValueError(f"Expected video tensor with 3 to 5 dimensions, got {tuple(tensor.shape)}")
