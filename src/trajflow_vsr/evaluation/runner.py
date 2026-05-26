"""Evaluation runner for offline and streaming protocols."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajflow_vsr.data.factory import CONTROLLED_MOTION_NAME, SYNTHETIC_NAME, build_synthetic_spec, describe_data
from trajflow_vsr.data.manifest import make_frame_manifest_batch
from trajflow_vsr.data.synthetic import make_controlled_motion_batch, make_synthetic_batch
from trajflow_vsr.metrics import (
    blockiness_proxy,
    compute_official_metric,
    dists_proxy,
    lpips_proxy,
    metric_backend_status,
    profile_model_macs,
    psnr,
    reliability_ece,
    selective_reconstruction_auc,
    spatial_sharpness,
    ssim,
    temporal_activity,
    temporal_delta_error,
    tof_proxy,
    uncertainty_error_correlation,
)
from trajflow_vsr.models.factory import build_model, describe_model
from trajflow_vsr.utils.seed import seed_everything
from trajflow_vsr.utils.torch_utils import is_torch_available, require_torch


@dataclass(frozen=True)
class EvalSummary:
    """Serializable summary emitted by evaluation dry-runs."""

    project: dict[str, Any]
    runtime: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]
    evaluation: dict[str, Any]
    torch_available: bool

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


class EvaluationRunner:
    """Run lightweight synthetic benchmark protocols."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.project = config.get("project", {})
        self.runtime = config.get("runtime", {})
        self.evaluation = config.get("evaluation", {})

    def summarize(self) -> EvalSummary:
        return EvalSummary(
            project=self.project,
            runtime=self.runtime,
            data=describe_data(self.config.get("data", {})),
            model=describe_model(self.config.get("model", {})),
            evaluation=self.evaluation,
            torch_available=is_torch_available(),
        )

    def dry_run(self) -> EvalSummary:
        summary = self.summarize()
        print(summary.to_json())
        return summary

    def run(self) -> dict[str, Any]:
        torch = require_torch()
        seed_everything(int(self.runtime.get("seed", 20260524)))
        device = str(self.runtime.get("device", "cpu"))
        data_config = self.config.get("data", {})
        model = build_model(self.config.get("model", {})).to(device)
        checkpoint_info = self._load_checkpoint(model, device=device)
        model_profile = self._model_profile(model)
        model.eval()
        modes = self._modes()
        clip_count = self._clip_count(data_config)
        if bool(self.evaluation.get("profile_macs", True)):
            profile_batch = self._make_batch(data_config, device=device, clip_offset=0)
            model_profile.update(
                profile_model_macs(
                    model,
                    profile_batch["lr"],
                    scale=profile_batch["scale"],
                    mode=modes[0],
                )
            )

        with torch.no_grad():
            mode_metrics = {}
            per_clip_metrics = {}
            for mode in modes:
                records = []
                for clip_offset in range(clip_count):
                    batch = self._make_batch(data_config, device=device, clip_offset=clip_offset)
                    _synchronize_if_needed(torch, device)
                    started = time.perf_counter()
                    outputs = model(batch["lr"], scale=batch["scale"], mode=mode)
                    posterior_outputs = self._posterior_outputs(
                        model,
                        batch=batch,
                        mode=mode,
                        sample_count=int(self.evaluation.get("posterior_samples", 0)),
                    )
                    _synchronize_if_needed(torch, device)
                    elapsed = time.perf_counter() - started
                    records.append(
                        {
                            "clip_offset": clip_offset,
                            "metadata": batch.get("metadata", []),
                            "metrics": self._compute_metrics(
                                outputs,
                                batch,
                                elapsed_seconds=elapsed,
                                posterior_outputs=posterior_outputs,
                            ),
                        }
                    )
                mode_metrics[mode] = self._aggregate_metrics(records)
                per_clip_metrics[mode] = records

        result = {
            "summary": self.summarize().__dict__,
            "checkpoint": checkpoint_info,
            "profile": model_profile,
            "metric_status": self._metric_status(),
            "reference_free": self._reference_free(),
            "metrics": mode_metrics,
            "per_clip_metrics": per_clip_metrics,
        }
        output_path = self._output_path()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            result["output_path"] = str(output_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    def _modes(self) -> list[str]:
        mode = self.evaluation.get("mode", "offline")
        if mode in {"mixed", "joint"}:
            return ["offline", "streaming"]
        if mode in {"online", "online_causal"}:
            return ["streaming"]
        modes = self.evaluation.get("modes")
        if isinstance(modes, list) and modes:
            return [str(item) for item in modes]
        return [str(mode)]

    def _make_batch(self, data_config: dict[str, Any], device: str, clip_offset: int = 0) -> dict[str, Any]:
        if data_config.get("name", SYNTHETIC_NAME) == "frame_manifest":
            manifest_config = dict(data_config)
            base_index = int(manifest_config.get("clip_index", 0))
            manifest_config["clip_index"] = base_index + int(clip_offset)
            return make_frame_manifest_batch(manifest_config, device=device)

        data_spec = build_synthetic_spec(data_config)
        if data_config.get("name", SYNTHETIC_NAME) == CONTROLLED_MOTION_NAME:
            return make_controlled_motion_batch(data_spec, motion=data_config.get("motion", {}), device=device)
        return make_synthetic_batch(data_spec, device=device)

    def _clip_count(self, data_config: dict[str, Any]) -> int:
        if "clip_count" in self.evaluation:
            return max(int(self.evaluation.get("clip_count", 1)), 1)
        if data_config.get("name", SYNTHETIC_NAME) == "frame_manifest":
            return max(int(self.evaluation.get("max_clips", 1)), 1)
        return 1

    def _reference_free(self) -> bool:
        protocol = str(self.evaluation.get("protocol", "") or self.evaluation.get("mode", "")).lower()
        if bool(self.evaluation.get("reference_free", False)):
            return True
        if protocol in {"no_reference", "reference_free", "noref"}:
            return True
        data_config = self.config.get("data", {})
        return str(data_config.get("protocol", "")).lower() in {"no_reference", "no_reference_qualitative"}

    def _posterior_outputs(self, model: Any, *, batch: dict[str, Any], mode: str, sample_count: int) -> list[dict[str, Any]]:
        if sample_count <= 0:
            return []
        return [
            model(
                batch["lr"],
                scale=batch["scale"],
                mode=mode,
                sample_flow_noise=True,
            )
            for _ in range(sample_count)
        ]

    def _compute_metrics(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        elapsed_seconds: float,
        posterior_outputs: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        prediction = outputs["hr"]
        frame_count = float(prediction.shape[0] * prediction.shape[1])
        pixel_count = float(prediction.shape[0] * prediction.shape[1] * prediction.shape[-2] * prediction.shape[-1])
        elapsed = max(float(elapsed_seconds), 1e-12)
        if self._reference_free():
            metrics = {
                "temporal_activity": float(temporal_activity(prediction).detach().cpu()),
                "spatial_sharpness": float(spatial_sharpness(prediction).detach().cpu()),
                "blockiness": float(blockiness_proxy(prediction).detach().cpu()),
                "causal_violation": float(outputs["transport"].get("causal_violation", prediction.new_zeros(())).detach().cpu()),
                "latency_seconds": elapsed,
                "fps": frame_count / elapsed,
                "megapixels_per_second": pixel_count / elapsed / 1_000_000.0,
                "vram_gb": _vram_gb(prediction.device),
            }
            metrics.update(self._official_no_reference_metrics(prediction))
            requested = self._metric_names()
            if requested:
                return {key: value for key, value in metrics.items() if key in requested}
            return metrics

        target = batch["hr"]
        metrics = {
            "psnr": float(psnr(prediction, target).detach().cpu()),
            "ssim": float(ssim(prediction, target).detach().cpu()),
            "temporal_delta_error": float(temporal_delta_error(prediction, target).detach().cpu()),
            "tof": float(tof_proxy(prediction, target).detach().cpu()),
            "uncertainty_error_correlation": float(uncertainty_error_correlation(outputs, target).detach().cpu()),
            "reliability_ece": float(reliability_ece(outputs, target).detach().cpu()),
            "selective_reconstruction_auc": float(selective_reconstruction_auc(outputs, target).detach().cpu()),
            "causal_violation": float(outputs["transport"].get("causal_violation", prediction.new_zeros(())).detach().cpu()),
            "latency_seconds": elapsed,
            "fps": frame_count / elapsed,
            "megapixels_per_second": pixel_count / elapsed / 1_000_000.0,
            "vram_gb": _vram_gb(prediction.device),
        }
        metrics.update(self._perceptual_metrics(prediction, target))
        metrics.update(_posterior_metrics(posterior_outputs or [], target))
        requested = self._metric_names()
        if requested:
            return {key: value for key, value in metrics.items() if key in requested or key.startswith("posterior_")}
        return metrics

    def _aggregate_metrics(self, records: list[dict[str, Any]]) -> dict[str, float]:
        if not records:
            return {}
        keys = records[0]["metrics"].keys()
        return {
            key: float(sum(float(record["metrics"][key]) for record in records) / len(records))
            for key in keys
        }

    def _load_checkpoint(self, model: Any, device: str) -> dict[str, Any]:
        checkpoint_path = self.evaluation.get("checkpoint_path")
        if not checkpoint_path:
            return {"loaded": False, "path": None}

        torch = require_torch()
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")

        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ["model_state_dict", "state_dict", "model"]:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        return {
            "loaded": True,
            "path": str(path),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }

    def _model_profile(self, model: Any) -> dict[str, Any]:
        parameters = list(model.parameters())
        buffers = list(model.buffers())
        return {
            "parameters": int(sum(parameter.numel() for parameter in parameters)),
            "trainable_parameters": int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad)),
            "buffers": int(sum(buffer.numel() for buffer in buffers)),
        }

    def _output_path(self) -> Path | None:
        output_path = self.evaluation.get("output_path")
        if output_path:
            return Path(output_path)
        if bool(self.evaluation.get("save_results", False)):
            output_dir = self.project.get("output_dir") or "outputs/evaluation"
            return Path(output_dir) / "metrics.json"
        return None

    def _metric_names(self) -> list[str]:
        metrics = self.evaluation.get("metrics")
        if isinstance(metrics, list):
            return [str(item) for item in metrics]
        return []

    def _metric_backend(self) -> str:
        return str(self.evaluation.get("metric_backend", "proxy")).lower()

    def _metric_status(self) -> dict[str, dict[str, Any]]:
        candidates = self.evaluation.get("official_metric_candidates", [])
        metric_names = [*self._metric_names()]
        if isinstance(candidates, list):
            metric_names.extend(str(item) for item in candidates)
        return metric_backend_status(sorted(set(metric_names)), requested_backend=self._metric_backend())

    def _perceptual_metrics(self, prediction: Any, target: Any) -> dict[str, float]:
        backend = self._metric_backend()
        status = self._metric_status()
        metrics = {}
        for name, proxy_fn in {"lpips": lpips_proxy, "dists": dists_proxy}.items():
            if backend == "official" and status.get(name, {}).get("official_available"):
                try:
                    metrics[name] = float(compute_official_metric(name, prediction, target).detach().cpu())
                    continue
                except Exception:
                    pass
            metrics[name] = float(proxy_fn(prediction, target).detach().cpu())
        return metrics

    def _official_no_reference_metrics(self, prediction: Any) -> dict[str, float]:
        backend = self._metric_backend()
        if backend != "official":
            return {}
        status = self._metric_status()
        metrics = {}
        for name in ["niqe", "musiq", "clipiqa"]:
            if status.get(name, {}).get("official_available"):
                try:
                    metrics[name] = float(compute_official_metric(name, prediction, None).detach().cpu())
                except Exception:
                    continue
        return metrics


def _synchronize_if_needed(torch: Any, device: str) -> None:
    if str(device).startswith("cuda") and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _posterior_metrics(posterior_outputs: list[dict[str, Any]], target: Any) -> dict[str, float]:
    if not posterior_outputs:
        return {}
    torch = require_torch()
    samples = torch.stack([item["hr"] for item in posterior_outputs], dim=0)
    sample_psnr = torch.stack([psnr(sample, target) for sample in samples], dim=0)
    sample_error = torch.stack([(sample - target).abs().mean() for sample in samples], dim=0)
    return {
        "posterior_samples": float(samples.shape[0]),
        "posterior_variance": float(samples.var(dim=0, unbiased=False).mean().detach().cpu()),
        "posterior_psnr_mean": float(sample_psnr.mean().detach().cpu()),
        "posterior_psnr_std": float(sample_psnr.std(unbiased=False).detach().cpu()),
        "posterior_error_mean": float(sample_error.mean().detach().cpu()),
    }


def _vram_gb(device: Any) -> float:
    torch = require_torch()
    if str(device).startswith("cuda") and hasattr(torch, "cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device) / (1024**3))
    return 0.0
