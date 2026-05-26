"""Model efficiency profiling helpers."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.utils.torch_utils import require_torch_nn


def profile_model_macs(model: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Estimate multiply-accumulate counts for common PyTorch layers.

    The profiler intentionally stays dependency-free. It covers Conv2d, Conv3d,
    and Linear modules, which dominate the current TrajFlow-VSR prototype. The
    result is an estimate for one forward pass with the provided inputs.
    """

    torch, nn = require_torch_nn()
    module_macs: dict[str, int] = {}
    hooks = []
    was_training = bool(getattr(model, "training", False))

    def register(name: str, module: Any) -> None:
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(_conv2d_hook(name, module, module_macs)))
        elif isinstance(module, nn.Conv3d):
            hooks.append(module.register_forward_hook(_conv3d_hook(name, module, module_macs)))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_linear_hook(name, module, module_macs)))

    for module_name, module in model.named_modules():
        register(module_name, module)

    try:
        model.eval()
        with torch.no_grad():
            model(*args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()

    total = int(sum(module_macs.values()))
    return {
        "macs": total,
        "gmacs": float(total / 1_000_000_000.0),
        "profiled_modules": int(len(module_macs)),
        "macs_by_module": module_macs,
        "macs_note": "estimated_forward_macs_conv_linear_only",
    }


def _conv2d_hook(name: str, module: Any, module_macs: dict[str, int]):
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not hasattr(output, "shape") or len(output.shape) != 4:
            return
        batch, out_channels, out_height, out_width = output.shape
        kernel_h, kernel_w = _pair(module.kernel_size)
        groups = int(getattr(module, "groups", 1))
        in_channels = int(getattr(module, "in_channels", 0))
        macs_per_output = (in_channels // max(groups, 1)) * kernel_h * kernel_w
        _accumulate_macs(
            module_macs,
            name,
            int(batch * out_channels * out_height * out_width * macs_per_output),
        )

    return hook


def _conv3d_hook(name: str, module: Any, module_macs: dict[str, int]):
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not hasattr(output, "shape") or len(output.shape) != 5:
            return
        batch, out_channels, out_depth, out_height, out_width = output.shape
        kernel_d, kernel_h, kernel_w = _triple(module.kernel_size)
        groups = int(getattr(module, "groups", 1))
        in_channels = int(getattr(module, "in_channels", 0))
        macs_per_output = (in_channels // max(groups, 1)) * kernel_d * kernel_h * kernel_w
        _accumulate_macs(
            module_macs,
            name,
            int(batch * out_channels * out_depth * out_height * out_width * macs_per_output),
        )

    return hook


def _linear_hook(name: str, module: Any, module_macs: dict[str, int]):
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not hasattr(output, "numel"):
            return
        _accumulate_macs(module_macs, name, int(output.numel() * int(getattr(module, "in_features", 0))))

    return hook


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _triple(value: Any) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1]), int(value[2])
    return int(value), int(value), int(value)


def _accumulate_macs(module_macs: dict[str, int], name: str, value: int) -> None:
    module_macs[name] = int(module_macs.get(name, 0)) + int(value)
