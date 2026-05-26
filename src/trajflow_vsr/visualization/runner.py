"""Visualization runner for diagnostic artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajflow_vsr.data.factory import build_synthetic_spec, describe_data
from trajflow_vsr.data.synthetic import make_synthetic_batch
from trajflow_vsr.models.factory import build_model, describe_model
from trajflow_vsr.utils.seed import seed_everything
from trajflow_vsr.utils.torch_utils import is_torch_available, require_torch
from trajflow_vsr.visualization.export import (
    export_sample_frames,
    export_trajectory_maps,
    export_uncertainty_maps,
    export_visualization_bundle,
)


@dataclass(frozen=True)
class VisualizationSummary:
    """Serializable summary for visualization runs."""

    project: dict[str, Any]
    runtime: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]
    visualization: dict[str, Any]
    torch_available: bool

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


class VisualizationRunner:
    """Run the model on a synthetic clip and export diagnostic artifacts."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.project = config.get("project", {})
        self.runtime = config.get("runtime", {})
        self.visualization = config.get("visualization", {})
        self.evaluation = config.get("evaluation", {})

    def summarize(self) -> VisualizationSummary:
        return VisualizationSummary(
            project=self.project,
            runtime=self.runtime,
            data=describe_data(self.config.get("data", {})),
            model=describe_model(self.config.get("model", {})),
            visualization=self.visualization,
            torch_available=is_torch_available(),
        )

    def dry_run(self) -> VisualizationSummary:
        summary = self.summarize()
        print(summary.to_json())
        return summary

    def run(self, kind: str = "bundle") -> dict[str, Any]:
        torch = require_torch()
        seed_everything(int(self.runtime.get("seed", 20260524)))
        device = str(self.runtime.get("device", "cpu"))
        output_dir = self._output_dir()
        mode = self._mode()

        data_spec = build_synthetic_spec(self.config.get("data", {}))
        model = build_model(self.config.get("model", {})).to(device)
        model.eval()
        batch = make_synthetic_batch(data_spec, device=device)

        with torch.no_grad():
            outputs = model(
                batch["lr"],
                scale=batch["scale"],
                mode=mode,
                sample_flow_noise=bool(self.visualization.get("sample_flow_noise", False)),
            )
            files = self._export(kind=kind, outputs=outputs, output_dir=output_dir)
            files.update(self._export_posterior_samples(model, batch, mode, output_dir))

        manifest = {
            "summary": self.summarize().__dict__,
            "mode": mode,
            "kind": kind,
            "output_dir": str(output_dir),
            "files": files,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest["files"]["manifest"] = [str(manifest_path)]
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return manifest

    def _mode(self) -> str:
        mode = self.visualization.get("mode", self.evaluation.get("mode", "offline"))
        if mode in {"online", "online_causal"}:
            return "streaming"
        if mode in {"mixed", "joint"}:
            return "offline"
        return str(mode)

    def _output_dir(self) -> Path:
        output_dir = self.visualization.get("output_dir") or self.project.get("output_dir")
        return Path(output_dir or "outputs/visualization")

    def _export(self, kind: str, outputs: dict[str, Any], output_dir: Path) -> dict[str, list[str]]:
        max_frames = self._optional_int("max_frames")
        batch_index = int(self.visualization.get("batch_index", 0))
        top_edges = int(self.visualization.get("top_edges", 32))
        image_format = str(self.visualization.get("image_format", "png"))

        if kind == "uncertainty":
            return export_uncertainty_maps(
                outputs,
                output_dir,
                prefix="uncertainty",
                batch_index=batch_index,
                max_frames=max_frames,
                image_format=image_format,
            )
        if kind == "trajectory":
            return export_trajectory_maps(
                outputs,
                output_dir,
                prefix="trajectory",
                batch_index=batch_index,
                max_frames=max_frames,
                top_edges=top_edges,
                image_format=image_format,
            )
        if kind == "samples":
            return export_sample_frames(
                outputs,
                output_dir,
                prefix="sample",
                batch_index=batch_index,
                max_frames=max_frames,
                image_format=image_format,
            )
        if kind != "bundle":
            raise ValueError(f"Unsupported visualization kind: {kind}")
        return export_visualization_bundle(
            outputs,
            output_dir,
            prefix="bundle",
            batch_index=batch_index,
            max_frames=max_frames,
            top_edges=top_edges,
            image_format=image_format,
        )

    def _export_posterior_samples(self, model: Any, batch: dict[str, Any], mode: str, output_dir: Path) -> dict[str, list[str]]:
        torch = require_torch()
        sample_count = int(self.visualization.get("posterior_samples", 0))
        if sample_count <= 0:
            return {}

        max_frames = self._optional_int("max_frames")
        batch_index = int(self.visualization.get("batch_index", 0))
        image_format = str(self.visualization.get("image_format", "png"))
        files: dict[str, list[str]] = {}
        for sample_idx in range(sample_count):
            with torch.no_grad():
                sample_outputs = model(
                    batch["lr"],
                    scale=batch["scale"],
                    mode=mode,
                    sample_flow_noise=True,
                )
            sample_files = export_sample_frames(
                sample_outputs,
                output_dir,
                prefix=f"posterior_{sample_idx:03d}",
                batch_index=batch_index,
                max_frames=max_frames,
                image_format=image_format,
            )
            for key, value in sample_files.items():
                files.setdefault(f"posterior_{key}", []).extend(value)
        return files

    def _optional_int(self, key: str) -> int | None:
        value = self.visualization.get(key)
        return None if value is None else int(value)
