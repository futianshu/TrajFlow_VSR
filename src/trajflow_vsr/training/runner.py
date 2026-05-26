"""Training runner for staged TrajFlow-VSR experiments."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajflow_vsr.data.factory import (
    CONTROLLED_MOTION_NAME,
    SYNTHETIC_NAME,
    build_synthetic_degradation_spec,
    build_synthetic_spec,
    describe_data,
)
from trajflow_vsr.data.manifest import make_frame_manifest_batch, make_stage_a_frame_manifest_batch
from trajflow_vsr.data.mixed import STAGE_A_MIXED_NAME, make_stage_a_mixed_batch
from trajflow_vsr.data.synthetic import make_controlled_motion_batch, make_stage_a_batch, make_synthetic_batch
from trajflow_vsr.evaluation import EvaluationRunner
from trajflow_vsr.losses.factory import compute_training_loss, describe_losses
from trajflow_vsr.losses.stage_a import compute_stage_a_loss
from trajflow_vsr.models.factory import build_stage_model, describe_model
from trajflow_vsr.training.checkpoint import (
    checkpoint_final_path,
    checkpoint_step_path,
    load_training_checkpoint,
    save_training_checkpoint,
    write_json_artifact,
)
from trajflow_vsr.training.curriculum import CurriculumState, TrainingCurriculum
from trajflow_vsr.training.pretrained import load_pretrained_components, trainable_parameters
from trajflow_vsr.utils.config import apply_overrides, load_config
from trajflow_vsr.utils.seed import seed_everything
from trajflow_vsr.utils.torch_utils import is_torch_available, require_torch


@dataclass(frozen=True)
class RunSummary:
    """Serializable summary emitted by dry-runs and smoke runs."""

    project: dict[str, Any]
    stage: dict[str, Any]
    runtime: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]
    losses: dict[str, Any]
    curriculum: dict[str, Any]
    torch_available: bool

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


class TrainingRunner:
    """Coordinate config, data, model, and training stage execution."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.project = config.get("project", {})
        self.stage = config.get("stage", {})
        self.runtime = config.get("runtime", {})
        self.checkpoint = config.get("checkpoint", {})

    def summarize(self) -> RunSummary:
        """Return a dependency-light summary of the planned run."""

        return RunSummary(
            project=self.project,
            stage=self.stage,
            runtime=self.runtime,
            data=describe_data(self.config.get("data", {})),
            model=describe_model(self.config.get("model", {})),
            losses=describe_losses(self.config.get("losses", {})),
            curriculum=TrainingCurriculum(
                self.config.get("curriculum", {}),
                self.config.get("losses", {}),
            ).describe(),
            torch_available=is_torch_available(),
        )

    def dry_run(self) -> RunSummary:
        """Validate the config and print the component plan."""

        summary = self.summarize()
        print(summary.to_json())
        return summary

    def run(self) -> dict[str, Any]:
        """Run staged training with optional validation and resumable artifacts."""

        torch = require_torch()
        seed_everything(int(self.runtime.get("seed", 20260524)))
        device = str(self.runtime.get("device", "cpu"))
        optimizer_config = self.config.get("optimizer", {})
        steps = int(optimizer_config.get("max_steps", 1))
        accumulation_steps = max(int(optimizer_config.get("gradient_accumulation_steps", 1)), 1)
        max_grad_norm = float(optimizer_config.get("max_grad_norm", 0.0))
        use_amp = bool(optimizer_config.get("amp", False)) and device.startswith("cuda")

        data_config = self.config.get("data", {})
        data_spec = None
        degradation_spec = None
        if data_config.get("name", SYNTHETIC_NAME) in {SYNTHETIC_NAME, CONTROLLED_MOTION_NAME}:
            data_spec = build_synthetic_spec(data_config)
            if data_config.get("name", SYNTHETIC_NAME) == SYNTHETIC_NAME:
                degradation_spec = build_synthetic_degradation_spec(data_config)
        model = build_stage_model(self.stage, self.config.get("model", {})).to(device)
        pretrained = self._maybe_load_pretrained(model, device=device)
        base_trainable = _parameter_trainability(model)
        curriculum = TrainingCurriculum(self.config.get("curriculum", {}), self.config.get("losses", {}))
        parameters = trainable_parameters(model)
        if not parameters:
            raise ValueError("No trainable parameters remain after applying pretrained freeze settings")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(self.config.get("optimizer", {}).get("lr", 1e-4)),
            weight_decay=float(self.config.get("optimizer", {}).get("weight_decay", 0.0)),
        )
        scheduler = self._build_scheduler(optimizer, steps=steps)
        scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp and hasattr(torch, "cuda") else None

        output_dir = self._output_dir()
        checkpoint_dir = self._checkpoint_dir()
        validation_dir = output_dir / "validation"
        resume = self._maybe_resume(model, optimizer, scheduler=scheduler, device=device)
        history = list(resume.get("history", []))
        start_step = int(resume.get("next_step", 0))
        checkpoint_files: list[str] = []
        best_metric = _resume_best_metric(resume)
        validation_history = _resume_validation_history(resume)
        best_checkpoint = _best_checkpoint_path(checkpoint_dir, best_metric)
        summary = self.summarize().__dict__
        summary["pretrained"] = pretrained

        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step, steps):
            batch = self._make_batch(data_config, data_spec, degradation_spec, device, step=step)
            mode = self._resolve_mode(step)
            curriculum_state = curriculum.state_for_step(step)
            _apply_curriculum_trainability(model, base_trainable, curriculum_state.freeze_components)
            with _autocast_context(torch, device=device, enabled=use_amp):
                outputs = self._forward(model, batch, mode=mode)
                total_loss, loss_parts = self._compute_loss(
                    outputs,
                    batch,
                    loss_config=curriculum_state.loss_config,
                )
                scaled_loss = total_loss / float(accumulation_steps)
            if scaler is not None and use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step_optimizer = (step + 1) % accumulation_steps == 0 or step + 1 >= steps
            if should_step_optimizer:
                if scaler is not None and use_amp:
                    scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                if scaler is not None and use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            record = {
                "step": step,
                "mode": mode,
                "total": float(total_loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "optimizer_step": bool(should_step_optimizer),
                **{
                    key: float(value.detach().cpu())
                    for key, value in loss_parts.items()
                    if hasattr(value, "detach")
                },
            }
            _append_curriculum_record(record, curriculum_state)
            if "source" in batch:
                record["data_source"] = str(batch["source"])
            history.append(record)
            if self._should_save_step(step):
                checkpoint_files.append(
                    save_training_checkpoint(
                        checkpoint_step_path(checkpoint_dir, step),
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        summary=summary,
                        history=history,
                        metadata={"validation": validation_history, "best_metric": best_metric},
                    )
                )
            validation_record = self._maybe_validate(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                summary=summary,
                history=history,
                validation_dir=validation_dir,
                device=device,
            )
            if validation_record:
                validation_history.append(validation_record)
                best_metric, best_checkpoint = self._maybe_save_best_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    summary=summary,
                    history=history,
                    validation_history=validation_history,
                    best_metric=best_metric,
                    checkpoint_dir=checkpoint_dir,
                    validation_record=validation_record,
                )

        final_checkpoint = None
        last_step = history[-1]["step"] if history else start_step - 1
        if bool(self.checkpoint.get("save_final", False)):
            final_checkpoint = save_training_checkpoint(
                checkpoint_final_path(checkpoint_dir),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=int(last_step),
                summary=summary,
                history=history,
                metadata={"validation": validation_history, "best_metric": best_metric},
            )

        result = {
            "summary": summary,
            "history": history,
            "resume": resume,
            "pretrained": pretrained,
            "artifacts": {
                "output_dir": str(output_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "step_checkpoints": checkpoint_files,
                "final_checkpoint": final_checkpoint,
                "best_checkpoint": best_checkpoint,
            },
            "validation": validation_history,
            "best_metric": best_metric,
        }
        if bool(self.checkpoint.get("save_history", True)):
            result["artifacts"]["history"] = write_json_artifact(output_dir / "history.json", {"history": history})
            result["artifacts"]["manifest"] = write_json_artifact(output_dir / "manifest.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    def _make_batch(
        self,
        data_config: dict[str, Any],
        data_spec,
        degradation_spec,
        device: str,
        step: int = 0,
    ) -> dict[str, Any]:
        if self.stage.get("name") == "stage_a_tokenizer":
            if data_config.get("name") == STAGE_A_MIXED_NAME:
                return make_stage_a_mixed_batch(data_config, step=step, device=device)
            if data_config.get("name") == "frame_manifest":
                return make_stage_a_frame_manifest_batch(_step_manifest_config(data_config, step), device=device)
            return make_stage_a_batch(data_spec, degradation=degradation_spec, device=device)
        if data_config.get("name") == "frame_manifest":
            return make_frame_manifest_batch(_step_manifest_config(data_config, step), device=device)
        if data_config.get("name") == CONTROLLED_MOTION_NAME:
            return make_controlled_motion_batch(data_spec, motion=data_config.get("motion", {}), device=device)
        return make_synthetic_batch(data_spec, device=device)

    def _resolve_mode(self, step: int) -> str:
        stage_name = self.stage.get("name")
        mode = self.stage.get("mode", "offline")
        if stage_name == "stage_e_streaming" and mode in {"mixed", "joint"}:
            return "offline" if step % 2 == 0 else "streaming"
        return "streaming" if mode in {"online", "online_causal"} else mode

    def _forward(self, model, batch: dict[str, Any], mode: str) -> dict[str, Any]:
        if self.stage.get("name") == "stage_a_tokenizer":
            return model(batch["lr"], scale=batch["scale"])
        stage_name = self.stage.get("name")
        flow_config = self.config.get("model", {}).get("flow", {})
        return model(
            batch["lr"],
            scale=batch["scale"],
            mode=mode,
            sample_flow_noise=stage_name in {"stage_c_rectified_flow", "stage_d_distill", "stage_e_streaming"},
            distill_flow=stage_name in {"stage_d_distill", "stage_e_streaming"},
            flow_teacher_steps=int(flow_config.get("teacher_steps", 4)),
        )

    def _compute_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        loss_config: dict[str, Any] | None = None,
    ):
        losses = loss_config if loss_config is not None else self.config.get("losses", {})
        if self.stage.get("name") == "stage_a_tokenizer":
            return compute_stage_a_loss(outputs, batch, losses)
        return compute_training_loss(outputs, batch, losses)

    def _output_dir(self) -> Path:
        return Path(self.project.get("output_dir") or "outputs/training")

    def _checkpoint_dir(self) -> Path:
        output_dir = self.checkpoint.get("output_dir")
        if output_dir:
            return Path(output_dir)
        stage_name = str(self.stage.get("name", "stage"))
        experiment = str(self.project.get("experiment", stage_name))
        return Path("checkpoints") / stage_name / experiment

    def _build_scheduler(self, optimizer, steps: int):
        torch = require_torch()
        scheduler_config = self.config.get("scheduler", {})
        if not isinstance(scheduler_config, dict) or not scheduler_config:
            return None

        name = str(scheduler_config.get("name", "none")).lower()
        if name in {"none", "null", "off", ""}:
            return None
        if name in {"cosine", "cosine_annealing"}:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(scheduler_config.get("t_max", steps)), 1),
                eta_min=float(scheduler_config.get("eta_min", 0.0)),
            )
        if name in {"step", "step_lr"}:
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(int(scheduler_config.get("step_size", max(steps, 1))), 1),
                gamma=float(scheduler_config.get("gamma", 0.1)),
            )
        raise ValueError(f"Unsupported scheduler: {name}")

    def _maybe_resume(self, model, optimizer, scheduler, device: str) -> dict[str, Any]:
        resume_path = self.checkpoint.get("resume_path")
        if not resume_path:
            return {"loaded": False, "path": None, "next_step": 0, "history": []}
        return load_training_checkpoint(resume_path, model=model, optimizer=optimizer, scheduler=scheduler, device=device)

    def _maybe_validate(
        self,
        *,
        model,
        optimizer,
        scheduler,
        step: int,
        summary: dict[str, Any],
        history: list[dict[str, Any]],
        validation_dir: Path,
        device: str,
    ) -> dict[str, Any] | None:
        validation = self.config.get("validation", {})
        if not isinstance(validation, dict) or not bool(validation.get("enabled", False)):
            return None
        every_steps = int(validation.get("every_steps", 0))
        if every_steps <= 0 or (step + 1) % every_steps != 0:
            return None

        validation_dir.mkdir(parents=True, exist_ok=True)
        validation_checkpoint = validation_dir / f"step_{step:06d}.pt"
        save_training_checkpoint(
            validation_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            summary=summary,
            history=history,
            metadata={"purpose": "validation_checkpoint"},
        )
        eval_config = self._validation_eval_config(
            validation=validation,
            validation_checkpoint=validation_checkpoint,
            validation_dir=validation_dir,
            step=step,
            device=device,
        )
        result = EvaluationRunner(eval_config).run()
        mode = str(validation.get("mode", eval_config.get("evaluation", {}).get("mode", "offline")))
        metrics = _extract_mode_metrics(result.get("metrics", {}), mode=mode)
        metric = str(validation.get("metric", "psnr"))
        _validate_checkpoint_selection_metric(metric, validation)
        metric_value = metrics.get(metric)
        return {
            "step": int(step),
            "status": "ok",
            "config": str(validation.get("config", "configs/eval/offline.yaml")),
            "checkpoint": str(validation_checkpoint),
            "mode": mode,
            "metric": metric,
            "metric_value": float(metric_value) if _is_number(metric_value) else None,
            "metrics": metrics,
            "output_path": result.get("output_path"),
            "profile": result.get("profile", {}),
        }

    def _validation_eval_config(
        self,
        *,
        validation: dict[str, Any],
        validation_checkpoint: Path,
        validation_dir: Path,
        step: int,
        device: str,
    ) -> dict[str, Any]:
        eval_config_path = str(validation.get("config", "configs/eval/offline.yaml"))
        eval_config = load_config(eval_config_path)
        eval_config["model"] = dict(self.config.get("model", eval_config.get("model", {})))
        if isinstance(validation.get("data"), dict):
            eval_config["data"] = dict(validation["data"])
        overrides = [
            f"runtime.device={device}",
            "runtime.dry_run=false",
            f"project.output_dir={validation_dir / f'step_{step:06d}'}",
            f"evaluation.checkpoint_path={validation_checkpoint}",
            f"evaluation.output_path={validation_dir / f'step_{step:06d}_metrics.json'}",
            *_override_list(validation.get("overrides", [])),
        ]
        return apply_overrides(eval_config, overrides)

    def _maybe_save_best_checkpoint(
        self,
        *,
        model,
        optimizer,
        scheduler,
        step: int,
        summary: dict[str, Any],
        history: list[dict[str, Any]],
        validation_history: list[dict[str, Any]],
        best_metric: dict[str, Any] | None,
        checkpoint_dir: Path,
        validation_record: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        value = validation_record.get("metric_value")
        if not _is_number(value):
            return best_metric, _best_checkpoint_path(checkpoint_dir, best_metric)

        validation = self.config.get("validation", {})
        direction = _metric_direction(validation.get("direction"), metric=str(validation_record.get("metric", "")))
        candidate = {
            "metric": validation_record.get("metric"),
            "value": float(value),
            "direction": direction,
            "step": int(step),
            "mode": validation_record.get("mode"),
        }
        if not _is_better_metric(candidate, best_metric):
            return best_metric, _best_checkpoint_path(checkpoint_dir, best_metric)

        best_path = checkpoint_dir / "best.pt"
        saved = save_training_checkpoint(
            best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            summary=summary,
            history=history,
            metadata={
                "validation": validation_history,
                "best_metric": candidate,
                "validation_record": validation_record,
            },
        )
        candidate["checkpoint"] = saved
        return candidate, saved

    def _maybe_load_pretrained(self, model, device: str) -> dict[str, Any]:
        pretrained = self.config.get("model", {}).get("pretrained", {})
        if not isinstance(pretrained, dict) or not pretrained.get("path"):
            return {"loaded": False, "path": None}
        return load_pretrained_components(
            model=model,
            path=pretrained["path"],
            components=pretrained.get("components"),
            freeze_components=pretrained.get("freeze_components", []),
            strict=bool(pretrained.get("strict", False)),
            device=device,
        )

    def _should_save_step(self, step: int) -> bool:
        save_every = int(self.checkpoint.get("save_every_steps", 0))
        if save_every <= 0:
            return False
        return (step + 1) % save_every == 0


def _step_manifest_config(config: dict[str, Any], step: int) -> dict[str, Any]:
    manifest_config = dict(config)
    if bool(manifest_config.get("sample_sequential_clips", True)):
        base_index = int(manifest_config.get("clip_index", 0))
        stride = int(manifest_config.get("clip_stride", 1))
        manifest_config["clip_index"] = base_index + int(step) * max(stride, 1)
    return manifest_config


def _parameter_trainability(model) -> dict[str, bool]:
    return {name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()}


def _apply_curriculum_trainability(
    model,
    base_trainable: dict[str, bool],
    freeze_components: tuple[str, ...],
) -> None:
    frozen = tuple(component for component in freeze_components if component)
    for name, parameter in model.named_parameters():
        originally_trainable = bool(base_trainable.get(name, parameter.requires_grad))
        parameter.requires_grad = originally_trainable and not _is_frozen_parameter(name, frozen)


def _is_frozen_parameter(parameter_name: str, freeze_components: tuple[str, ...]) -> bool:
    for component in freeze_components:
        if parameter_name == component or parameter_name.startswith(f"{component}."):
            return True
    return False


def _append_curriculum_record(record: dict[str, Any], state: CurriculumState) -> None:
    if not state.enabled:
        return
    record["curriculum_phase"] = str(state.phase or "base")
    record["curriculum_progress"] = float(state.progress)
    if state.freeze_components:
        record["curriculum_frozen"] = ",".join(state.freeze_components)
    for key, value in state.numeric_loss_weights.items():
        record[f"weight.{key}"] = value


def _autocast_context(torch: Any, *, device: str, enabled: bool):
    if not enabled:
        return nullcontext()
    device_type = str(device).split(":", 1)[0]
    return torch.autocast(device_type=device_type, enabled=True)


def _resume_best_metric(resume: dict[str, Any]) -> dict[str, Any] | None:
    metadata = resume.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    best_metric = metadata.get("best_metric")
    if isinstance(best_metric, dict) and _is_number(best_metric.get("value")):
        return dict(best_metric)
    return None


def _resume_validation_history(resume: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = resume.get("metadata", {})
    if not isinstance(metadata, dict):
        return []
    validation = metadata.get("validation", [])
    if not isinstance(validation, list):
        return []
    return [dict(item) for item in validation if isinstance(item, dict)]


def _extract_mode_metrics(metrics: dict[str, Any], *, mode: str) -> dict[str, Any]:
    if not isinstance(metrics, dict) or not metrics:
        return {}
    selected = metrics.get(mode)
    if isinstance(selected, dict):
        return selected
    first = next(iter(metrics.values()))
    return first if isinstance(first, dict) else {}


def _metric_direction(raw: Any, *, metric: str) -> str:
    if raw is not None:
        direction = str(raw).lower()
        if direction in {"min", "lower", "lower_is_better"}:
            return "min"
        if direction in {"max", "higher", "higher_is_better"}:
            return "max"
        raise ValueError(f"Validation metric direction must be 'min' or 'max': {raw}")
    if metric in {"loss", "total", "latency_seconds", "temporal_delta_error", "tof", "lpips", "dists"}:
        return "min"
    return "max"


def _validate_checkpoint_selection_metric(metric: str, validation: dict[str, Any]) -> None:
    no_reference_metrics = {"niqe", "musiq", "clipiqa"}
    if metric.lower() in no_reference_metrics and not bool(validation.get("allow_no_reference_selection", False)):
        raise ValueError(
            "No-reference metrics may not be used as the primary checkpoint selection metric. "
            "Use PSNR/SSIM/tOF/LPIPS/DISTS/temporal metrics, or set "
            "validation.allow_no_reference_selection=true for a deliberate diagnostic run."
        )


def _is_better_metric(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None or not _is_number(current.get("value")):
        return True
    direction = str(candidate.get("direction", "max"))
    candidate_value = float(candidate["value"])
    current_value = float(current["value"])
    if direction == "min":
        return candidate_value < current_value
    return candidate_value > current_value


def _best_checkpoint_path(checkpoint_dir: Path, best_metric: dict[str, Any] | None) -> str | None:
    if isinstance(best_metric, dict) and best_metric.get("checkpoint"):
        return str(best_metric["checkpoint"])
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        return str(best_path)
    return None


def _override_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if raw is None or raw == "":
        return []
    raise ValueError(f"Overrides must be a string or list: {raw}")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
