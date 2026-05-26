"""Inference runner for offline and streaming VSR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajflow_vsr.data.factory import build_synthetic_spec, describe_data
from trajflow_vsr.data.synthetic import make_synthetic_batch
from trajflow_vsr.inference.io import read_video_source, write_video_file, write_video_frames
from trajflow_vsr.models.factory import build_model, describe_model
from trajflow_vsr.utils.seed import seed_everything
from trajflow_vsr.utils.torch_utils import is_torch_available, require_torch
from trajflow_vsr.visualization import export_visualization_bundle


@dataclass(frozen=True)
class InferenceSummary:
    """Serializable summary emitted by inference dry-runs."""

    project: dict[str, Any]
    runtime: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]
    inference: dict[str, Any]
    torch_available: bool

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


class InferenceRunner:
    """Run TrajFlow-VSR on a synthetic clip, image sequence, or video file."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.project = config.get("project", {})
        self.runtime = config.get("runtime", {})
        self.inference = config.get("inference", {})

    def summarize(self) -> InferenceSummary:
        return InferenceSummary(
            project=self.project,
            runtime=self.runtime,
            data=describe_data(self.config.get("data", {})),
            model=describe_model(self.config.get("model", {})),
            inference=self.inference,
            torch_available=is_torch_available(),
        )

    def dry_run(self) -> InferenceSummary:
        summary = self.summarize()
        print(summary.to_json())
        return summary

    def run(self) -> dict[str, Any]:
        torch = require_torch()
        seed_everything(int(self.runtime.get("seed", 20260524)))
        device = str(self.runtime.get("device", "cpu"))
        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        batch = self._load_input(device=device)
        lr_video = batch["lr"]
        scale = float(self.inference.get("scale", batch.get("scale", self.config.get("data", {}).get("scale", 2.0))))
        mode = self._mode()

        model = build_model(self.config.get("model", {})).to(device)
        checkpoint_info = self._load_checkpoint(model, device=device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                lr_video,
                scale=scale,
                mode=mode,
                sample_flow_noise=bool(self.inference.get("sample_flow_noise", False)),
            )
            posterior_count = int(self.inference.get("posterior_samples", 0))
            if posterior_count > 0:
                outputs["posterior_samples"] = [
                    model(lr_video, scale=scale, mode=mode, sample_flow_noise=True)["hr"]
                    for _ in range(posterior_count)
                ]

        image_format = str(self.inference.get("image_format", "png"))
        files = {
            "hr_frames": write_video_frames(
                outputs["hr"],
                output_dir / "frames",
                prefix="hr",
                image_format=image_format,
            ),
            "lr_frames": write_video_frames(
                lr_video,
                output_dir / "lr_frames",
                prefix="lr",
                image_format=image_format,
            ),
        }
        if bool(self.inference.get("save_raw", False)) and "hr_raw" in outputs:
            files["hr_raw_frames"] = write_video_frames(
                outputs["hr_raw"],
                output_dir / "raw_frames",
                prefix="hr_raw",
                image_format=image_format,
            )

        video_path = self.inference.get("video_output")
        if video_path:
            files["video"] = [
                write_video_file(
                    outputs["hr"],
                    output_dir / str(video_path),
                    fps=int(self.inference.get("fps", 24)),
                )
            ]

        if bool(self.inference.get("export_visualization", False)):
            files.update(
                {
                    f"diagnostic_{key}": value
                    for key, value in export_visualization_bundle(
                        outputs,
                        output_dir / "diagnostics",
                        prefix="infer",
                        max_frames=self._optional_int("max_visualization_frames"),
                        top_edges=int(self.inference.get("top_edges", 32)),
                        image_format=image_format,
                    ).items()
                }
            )
        if outputs.get("posterior_samples"):
            posterior_dir = output_dir / "posterior_samples"
            files["posterior_sample_frames"] = []
            for sample_idx, sample in enumerate(outputs["posterior_samples"]):
                files["posterior_sample_frames"].extend(
                    write_video_frames(
                        sample,
                        posterior_dir / f"sample_{sample_idx:03d}",
                        prefix="hr",
                        image_format=image_format,
                    )
                )

        manifest = {
            "summary": self.summarize().__dict__,
            "mode": mode,
            "scale": scale,
            "input_shape": list(lr_video.shape),
            "output_shape": list(outputs["hr"].shape),
            "posterior_samples": int(len(outputs.get("posterior_samples", []))),
            "checkpoint": checkpoint_info,
            "output_dir": str(output_dir),
            "files": files,
        }
        manifest_path = output_dir / "manifest.json"
        manifest["files"]["manifest"] = [str(manifest_path)]
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return manifest

    def _load_input(self, device: str) -> dict[str, Any]:
        input_path = self.inference.get("input_path")
        if input_path:
            lr = read_video_source(
                input_path,
                device=device,
                max_frames=self._optional_int("max_frames"),
            )
            return {"lr": lr, "scale": float(self.inference.get("scale", self.config.get("data", {}).get("scale", 2.0)))}

        spec = build_synthetic_spec(self.config.get("data", {}))
        return make_synthetic_batch(spec, device=device)

    def _load_checkpoint(self, model: Any, device: str) -> dict[str, Any]:
        checkpoint_path = self.inference.get("checkpoint_path")
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

    def _mode(self) -> str:
        mode = self.inference.get("mode", "offline")
        if mode in {"online", "online_causal"}:
            return "streaming"
        return str(mode)

    def _output_dir(self) -> Path:
        output_dir = self.inference.get("output_dir") or self.project.get("output_dir")
        return Path(output_dir or "outputs/inference")

    def _optional_int(self, key: str) -> int | None:
        value = self.inference.get(key)
        return None if value is None else int(value)
