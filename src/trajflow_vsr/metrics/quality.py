"""Lightweight image/video quality metrics for smoke evaluation."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch


def psnr(prediction: Any, target: Any, max_value: float = 1.0):
    """Compute PSNR over a batched video tensor."""

    torch = require_torch()
    mse = (prediction - target).square().mean().clamp_min(1e-12)
    return 20.0 * torch.log10(torch.tensor(float(max_value), device=prediction.device, dtype=prediction.dtype)) - 10.0 * torch.log10(mse)


def ssim(prediction: Any, target: Any, max_value: float = 1.0, window_size: int = 3):
    """Compute a compact SSIM approximation for B,T,C,H,W tensors."""

    torch = require_torch()
    if prediction.ndim != 5 or target.ndim != 5:
        raise ValueError("Expected prediction and target shapes B,T,C,H,W")

    pred = prediction.flatten(0, 1)
    ref = target.flatten(0, 1)
    padding = window_size // 2
    mu_x = torch.nn.functional.avg_pool2d(pred, kernel_size=window_size, stride=1, padding=padding)
    mu_y = torch.nn.functional.avg_pool2d(ref, kernel_size=window_size, stride=1, padding=padding)
    sigma_x = torch.nn.functional.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu_x.square()
    sigma_y = torch.nn.functional.avg_pool2d(ref * ref, window_size, stride=1, padding=padding) - mu_y.square()
    sigma_xy = torch.nn.functional.avg_pool2d(pred * ref, window_size, stride=1, padding=padding) - mu_x * mu_y
    c1 = (0.01 * float(max_value)) ** 2
    c2 = (0.03 * float(max_value)) ** 2
    score = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return score.mean()


def temporal_delta_error(prediction: Any, target: Any):
    """Compare adjacent-frame changes in predicted and target videos."""

    torch = require_torch()
    if prediction.shape[1] < 2:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    return (pred_delta - target_delta).abs().mean()


def tof_proxy(prediction: Any, target: Any):
    """Proxy for tOF/warping error using temporal gradient disagreement."""

    return temporal_delta_error(prediction, target)


def temporal_activity(video: Any):
    """Mean adjacent-frame change without requiring a reference video."""

    torch = require_torch()
    if video.shape[1] < 2:
        return torch.zeros((), device=video.device, dtype=video.dtype)
    return (video[:, 1:] - video[:, :-1]).abs().mean()


def spatial_sharpness(video: Any):
    """Reference-free sharpness proxy from mean absolute spatial gradients."""

    return _spatial_gradients(video).mean()


def blockiness_proxy(video: Any, block_size: int = 8):
    """Reference-free block artifact proxy at regular block boundaries."""

    require_torch()
    block = max(int(block_size), 2)
    vertical = video.new_zeros(())
    horizontal = video.new_zeros(())
    if video.shape[-1] > block:
        cols = list(range(block, int(video.shape[-1]), block))
        if cols:
            right = video[..., :, cols]
            left = video[..., :, [col - 1 for col in cols]]
            vertical = (right - left).abs().mean()
    if video.shape[-2] > block:
        rows = list(range(block, int(video.shape[-2]), block))
        if rows:
            below = video[..., rows, :]
            above = video[..., [row - 1 for row in rows], :]
            horizontal = (below - above).abs().mean()
    return (vertical + horizontal) * 0.5


def lpips_proxy(prediction: Any, target: Any):
    """No-download perceptual proxy based on image and gradient differences."""

    torch = require_torch()
    image_error = (prediction - target).abs().mean()
    pred_grad = _spatial_gradients(prediction)
    target_grad = _spatial_gradients(target)
    gradient_error = (pred_grad - target_grad).abs().mean()
    pooled_pred = torch.nn.functional.avg_pool2d(prediction.flatten(0, 1), kernel_size=4, stride=4).unflatten(
        0, prediction.shape[:2]
    )
    pooled_target = torch.nn.functional.avg_pool2d(target.flatten(0, 1), kernel_size=4, stride=4).unflatten(
        0, target.shape[:2]
    )
    coarse_error = (pooled_pred - pooled_target).abs().mean()
    return 0.5 * image_error + 0.35 * gradient_error + 0.15 * coarse_error


def dists_proxy(prediction: Any, target: Any):
    """No-download DISTS-style proxy mixing structure and texture errors."""

    structure = (1.0 - ssim(prediction, target)).clamp_min(0.0)
    texture = (_spatial_gradients(prediction) - _spatial_gradients(target)).square().mean().sqrt()
    return 0.5 * structure + 0.5 * texture


def reliability_ece(outputs: dict[str, Any], target: Any, bins: int = 10):
    """Expected calibration error between reliability confidence and HR accuracy."""

    torch = require_torch()
    prediction = outputs["hr"]
    reliability = _hr_reliability(outputs, prediction)
    if reliability is None:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    error = (prediction - target).abs().mean(dim=2, keepdim=True).clamp(0.0, 1.0)
    accuracy = (1.0 - error).clamp(0.0, 1.0)
    confidence = reliability.clamp(0.0, 1.0)
    ece = prediction.new_zeros(())
    for index in range(int(bins)):
        lower = index / float(bins)
        upper = (index + 1) / float(bins)
        mask = (confidence >= lower) & (confidence < upper if index + 1 < bins else confidence <= upper)
        if mask.any():
            weight = mask.to(dtype=prediction.dtype).mean()
            ece = ece + weight * (confidence[mask].mean() - accuracy[mask].mean()).abs()
    return ece


def selective_reconstruction_auc(outputs: dict[str, Any], target: Any):
    """Mean selective reconstruction error over reliable-pixel coverages."""

    torch = require_torch()
    prediction = outputs["hr"]
    reliability = _hr_reliability(outputs, prediction)
    error = (prediction - target).abs().mean(dim=2, keepdim=True).flatten()
    if reliability is None:
        return error.mean()
    confidence = reliability.flatten()
    order = torch.argsort(confidence, descending=True)
    sorted_error = error[order]
    values = []
    for coverage in [0.25, 0.5, 0.75, 1.0]:
        count = max(1, int(round(sorted_error.numel() * coverage)))
        values.append(sorted_error[:count].mean())
    return torch.stack(values).mean()


def uncertainty_error_correlation(outputs: dict[str, Any], target: Any):
    """Measure Pearson correlation between predicted uncertainty and absolute HR error."""

    torch = require_torch()
    prediction = outputs["hr"]
    uncertainty = outputs.get("uncertainty", {})
    reliability = uncertainty.get("reliability") if isinstance(uncertainty, dict) else None
    if reliability is None:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)

    error = (prediction - target).abs().mean(dim=2, keepdim=True)
    confidence = torch.nn.functional.interpolate(
        reliability.flatten(0, 1),
        size=prediction.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).unflatten(0, prediction.shape[:2])
    uncertainty_map = 1.0 - confidence.clamp(0.0, 1.0)
    err = error.flatten()
    unc = uncertainty_map.flatten().to(dtype=err.dtype)
    err = err - err.mean()
    unc = unc - unc.mean()
    denom = err.square().mean().sqrt() * unc.square().mean().sqrt()
    return (err * unc).mean() / denom.clamp_min(1e-8)


def _spatial_gradients(video: Any):
    dx = video[..., :, 1:] - video[..., :, :-1]
    dy = video[..., 1:, :] - video[..., :-1, :]
    dx = torch_pad_last(dx)
    dy = torch_pad_height(dy)
    return dx.abs() + dy.abs()


def _hr_reliability(outputs: dict[str, Any], prediction: Any):
    uncertainty = outputs.get("uncertainty", {})
    reliability = uncertainty.get("reliability") if isinstance(uncertainty, dict) else None
    if reliability is None:
        return None
    return require_torch().nn.functional.interpolate(
        reliability.flatten(0, 1),
        size=prediction.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).unflatten(0, prediction.shape[:2])


def torch_pad_last(tensor: Any):
    torch = require_torch()
    return torch.nn.functional.pad(tensor, (0, 1, 0, 0))


def torch_pad_height(tensor: Any):
    torch = require_torch()
    return torch.nn.functional.pad(tensor, (0, 0, 0, 1))
