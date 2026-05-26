"""Reusable ablation runner for config variant grids."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from trajflow_vsr.evaluation import EvaluationRunner
from trajflow_vsr.training import TrainingRunner
from trajflow_vsr.utils.config import apply_overrides, load_config


@dataclass(frozen=True)
class AblationSummary:
    """Serializable ablation plan summary."""

    name: str
    purpose: str | None
    base_config: str
    runner_type: str
    evaluation_config: str | None
    output_dir: str
    metric_focus: list[str]
    variants: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


class AblationRunner:
    """Expand ablation variants and run each one with isolated artifacts."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.ablation = config.get("ablation", {})
        if not isinstance(self.ablation, dict):
            raise ValueError("Ablation config must contain an 'ablation' mapping")

    def summarize(
        self,
        *,
        variants: list[str] | None = None,
        max_variants: int | None = None,
        output_dir: str | Path | None = None,
    ) -> AblationSummary:
        selected = self._selected_variants(variants=variants, max_variants=max_variants)
        summary = AblationSummary(
            name=self._name(),
            purpose=self.ablation.get("purpose"),
            base_config=str(self.config.get("base_config", "")),
            runner_type=self._runner_type(self._load_base_config()),
            evaluation_config=self._evaluation_config_path(),
            output_dir=str(self._output_dir(output_dir)),
            metric_focus=[str(item) for item in self.ablation.get("metric_focus", [])],
            variants=[
                {
                    "name": name,
                    "overrides": overrides,
                    "output_dir": str(self._variant_dir(self._output_dir(output_dir), name)),
                }
                for name, overrides in selected
            ],
        )
        print(summary.to_json())
        return summary

    def dry_run(
        self,
        *,
        variants: list[str] | None = None,
        max_variants: int | None = None,
        output_dir: str | Path | None = None,
    ) -> AblationSummary:
        return self.summarize(variants=variants, max_variants=max_variants, output_dir=output_dir)

    def run(
        self,
        *,
        common_overrides: list[str] | None = None,
        variants: list[str] | None = None,
        max_variants: int | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        base_config = self._load_base_config()
        runner_type = self._runner_type(base_config)
        root = self._output_dir(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        records = []
        for name, variant_overrides in self._selected_variants(variants=variants, max_variants=max_variants):
            variant_dir = self._variant_dir(root, name)
            run_config = self._variant_config(
                base_config=base_config,
                name=name,
                variant_dir=variant_dir,
                overrides=[
                    *self._common_overrides(),
                    *variant_overrides,
                    *(common_overrides or []),
                ],
            )
            started = time.perf_counter()
            if runner_type == "training":
                result = TrainingRunner(run_config).run()
                record = self._training_record(name, variant_overrides, variant_dir, result)
                evaluation = self._evaluate_training_variant(
                    name=name,
                    variant_dir=variant_dir,
                    training_result=result,
                    overrides=[
                        *self._common_overrides(),
                        *variant_overrides,
                        *(common_overrides or []),
                    ],
                )
                if evaluation is not None:
                    record["evaluation"] = evaluation
                    if "metrics" in evaluation:
                        record["metrics"] = evaluation["metrics"]
            elif runner_type == "evaluation":
                result = EvaluationRunner(run_config).run()
                record = self._evaluation_record(name, variant_overrides, variant_dir, result)
            else:
                raise ValueError(f"Unsupported ablation runner type: {runner_type}")
            record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            records.append(record)

        summary = {
            "name": self._name(),
            "purpose": self.ablation.get("purpose"),
            "base_config": str(self.config.get("base_config", "")),
            "runner_type": runner_type,
            "evaluation_config": self._evaluation_config_path(),
            "output_dir": str(root),
            "variant_count": len(records),
            "records": records,
        }
        summary["comparison"] = self._write_comparison(root, records)
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    def _load_base_config(self) -> dict[str, Any]:
        base_config = self.config.get("base_config")
        if not base_config:
            raise ValueError("Ablation config must define base_config")
        return load_config(str(base_config))

    def _runner_type(self, config: dict[str, Any]) -> str:
        if "stage" in config:
            return "training"
        if "evaluation" in config:
            return "evaluation"
        raise ValueError("Base config must be a training or evaluation config")

    def _variant_config(
        self,
        *,
        base_config: dict[str, Any],
        name: str,
        variant_dir: Path,
        overrides: list[str],
    ) -> dict[str, Any]:
        config = apply_overrides(base_config, overrides)
        project = config.setdefault("project", {})
        project["experiment"] = f"{project.get('experiment', self._name())}_{_slug(name)}"
        project["output_dir"] = str(variant_dir / "outputs")
        if "checkpoint" in config:
            checkpoint = config.setdefault("checkpoint", {})
            checkpoint["output_dir"] = str(variant_dir / "checkpoints")
        if "evaluation" in config and bool(config.get("evaluation", {}).get("save_results", False)):
            config["evaluation"]["output_path"] = str(variant_dir / "metrics.json")
        return config

    def _evaluate_training_variant(
        self,
        *,
        name: str,
        variant_dir: Path,
        training_result: dict[str, Any],
        overrides: list[str],
    ) -> dict[str, Any] | None:
        evaluation_config = self._evaluation_config_path()
        if evaluation_config is None:
            return None

        final_checkpoint = training_result.get("artifacts", {}).get("final_checkpoint")
        if not final_checkpoint:
            return {
                "status": "skipped",
                "reason": "final_checkpoint_not_available",
            }

        eval_dir = variant_dir / "evaluation"
        eval_overrides = [
            *overrides,
            *self._evaluation_overrides(),
            f"project.experiment={self._name()}_{_slug(name)}_eval",
            f"project.output_dir={eval_dir}",
            f"evaluation.checkpoint_path={final_checkpoint}",
            f"evaluation.output_path={eval_dir / 'metrics.json'}",
        ]
        eval_config = apply_overrides(load_config(evaluation_config), eval_overrides)
        result = EvaluationRunner(eval_config).run()
        return {
            "status": "ok",
            "config": evaluation_config,
            "checkpoint": result.get("checkpoint", {}),
            "profile": result.get("profile", {}),
            "metrics": result.get("metrics", {}),
            "output_path": result.get("output_path"),
        }

    def _training_record(
        self,
        name: str,
        overrides: list[str],
        variant_dir: Path,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        history = result.get("history", [])
        last = history[-1] if history else {}
        return {
            "name": name,
            "overrides": overrides,
            "output_dir": str(variant_dir),
            "status": "ok",
            "final_step": last.get("step"),
            "final_mode": last.get("mode"),
            "final_total": last.get("total"),
            "final_losses": {
                key: value
                for key, value in last.items()
                if key not in {"step", "mode", "total", "data_source"}
            },
            "artifacts": result.get("artifacts", {}),
        }

    def _evaluation_record(
        self,
        name: str,
        overrides: list[str],
        variant_dir: Path,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "name": name,
            "overrides": overrides,
            "output_dir": str(variant_dir),
            "status": "ok",
            "metrics": result.get("metrics", {}),
            "checkpoint": result.get("checkpoint", {}),
            "profile": result.get("profile", {}),
            "output_path": result.get("output_path"),
        }

    def _name(self) -> str:
        return str(self.ablation.get("name") or "ablation")

    def _output_dir(self, output_dir: str | Path | None = None) -> Path:
        if output_dir is not None:
            return Path(output_dir)
        configured = self.ablation.get("output_dir")
        if configured:
            return Path(configured)
        return Path("outputs") / "ablations" / _slug(self._name())

    def _variant_dir(self, root: Path, name: str) -> Path:
        return root / _slug(name)

    def _selected_variants(
        self,
        *,
        variants: list[str] | None = None,
        max_variants: int | None = None,
    ) -> list[tuple[str, list[str]]]:
        all_variants = self._variants()
        if variants:
            unknown = [name for name in variants if name not in all_variants]
            if unknown:
                raise ValueError(f"Unknown ablation variant(s): {', '.join(unknown)}")
            selected = [(name, all_variants[name]) for name in variants]
        else:
            selected = list(all_variants.items())
        if max_variants is not None:
            selected = selected[: max(int(max_variants), 0)]
        return selected

    def _variants(self) -> dict[str, list[str]]:
        variants: dict[str, list[str]] = {}
        raw = self.ablation.get("variants")
        if raw is not None:
            if not isinstance(raw, dict):
                raise ValueError("ablation.variants must be a mapping when provided")
            if raw:
                variants.update(self._named_variants(raw))
        for name, overrides in self._matrix_variants().items():
            if name in variants:
                raise ValueError(f"Duplicate ablation variant name: {name}")
            variants[name] = overrides
        if not variants:
            raise ValueError("Ablation config must define non-empty ablation.variants or ablation.matrix")
        return variants

    def _named_variants(self, raw: dict[str, Any]) -> dict[str, list[str]]:
        variants: dict[str, list[str]] = {}
        for name, overrides in raw.items():
            variants[str(name)] = _override_list(f"variant {name}", overrides)
        return variants

    def _matrix_variants(self) -> dict[str, list[str]]:
        raw = self.ablation.get("matrix")
        if not raw:
            return {}
        if not isinstance(raw, dict):
            raise ValueError("ablation.matrix must be a mapping when provided")

        axes = raw.get("axes")
        if not isinstance(axes, dict) or not axes:
            raise ValueError("ablation.matrix.axes must be a non-empty mapping")

        joiner = str(raw.get("joiner", "_x_"))
        expanded_axes = [_matrix_axis_values(axis_name, axis) for axis_name, axis in axes.items()]
        variants: dict[str, list[str]] = {}
        for combo in product(*expanded_axes):
            name = joiner.join(axis_value for axis_value, _ in combo)
            overrides = [override for _, axis_overrides in combo for override in axis_overrides]
            if name in variants:
                raise ValueError(f"Duplicate matrix variant name: {name}")
            variants[name] = overrides
        return variants

    def _common_overrides(self) -> list[str]:
        overrides = self.ablation.get("common_overrides", [])
        if isinstance(overrides, str):
            return [overrides]
        if isinstance(overrides, list):
            return [str(item) for item in overrides]
        raise ValueError("ablation.common_overrides must be a string or list when provided")

    def _evaluation_config_path(self) -> str | None:
        path = self.ablation.get("evaluation_config")
        if not path:
            return None
        return str(path)

    def _evaluation_overrides(self) -> list[str]:
        overrides = self.ablation.get("evaluation_overrides", [])
        if isinstance(overrides, str):
            return [overrides]
        if isinstance(overrides, list):
            return [str(item) for item in overrides]
        raise ValueError("ablation.evaluation_overrides must be a string or list when provided")

    def _write_comparison(self, root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [self._comparison_row(record) for record in records]
        ranking = self._rank_rows(rows)
        best_by_metric = self._best_by_metric(rows)
        comparison = {
            "mode": self._selection_mode(),
            "metrics": self._selection_metrics(),
            "directions": self._metric_directions(),
            "weights": self._metric_weights(),
            "rows": rows,
            "ranking": ranking,
            "best_by_metric": best_by_metric,
        }
        comparison_json = root / "comparison.json"
        comparison_csv = root / "comparison.csv"
        comparison_md = root / "comparison.md"
        comparison_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        comparison_csv.write_text(_rows_to_csv(rows), encoding="utf-8")
        comparison_md.write_text(_rows_to_markdown(rows, ranking), encoding="utf-8")
        return {
            **comparison,
            "json_path": str(comparison_json),
            "csv_path": str(comparison_csv),
            "markdown_path": str(comparison_md),
        }

    def _comparison_row(self, record: dict[str, Any]) -> dict[str, Any]:
        mode = self._selection_mode()
        metrics = record.get("metrics", {})
        mode_metrics = metrics.get(mode, {}) if isinstance(metrics, dict) else {}
        row = {
            "name": record.get("name"),
            "status": record.get("status"),
            "final_total": record.get("final_total"),
            "elapsed_seconds": record.get("elapsed_seconds"),
        }
        final_losses = record.get("final_losses", {})
        for key, value in final_losses.items():
            row[f"loss.{key}"] = value
        if isinstance(mode_metrics, dict):
            row.update(mode_metrics)
        profile = record.get("evaluation", {}).get("profile") or record.get("profile", {})
        if isinstance(profile, dict):
            for key, value in profile.items():
                row[f"profile.{key}"] = value
        return row

    def _rank_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        metrics = self._selection_metrics()
        weights = self._metric_weights()
        directions = self._metric_directions()
        values_by_metric = {
            metric: [float(row[metric]) for row in rows if _is_number(row.get(metric))]
            for metric in metrics
        }
        ranked = []
        for row in rows:
            score = 0.0
            weight_sum = 0.0
            parts: dict[str, float] = {}
            for metric in metrics:
                if not _is_number(row.get(metric)) or not values_by_metric.get(metric):
                    continue
                values = values_by_metric[metric]
                normalized = _normalize_value(
                    float(row[metric]),
                    min(values),
                    max(values),
                    direction=directions.get(metric, "max"),
                )
                weight = float(weights.get(metric, 1.0))
                parts[metric] = normalized
                score += weight * normalized
                weight_sum += abs(weight)
            score = score / weight_sum if weight_sum else 0.0
            ranked.append(
                {
                    "name": row.get("name"),
                    "selection_score": round(float(score), 6),
                    "normalized": parts,
                }
            )
        ranked.sort(key=lambda item: item["selection_score"], reverse=True)
        for index, item in enumerate(ranked, start=1):
            item["rank"] = index
        return ranked

    def _best_by_metric(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        best = {}
        directions = self._metric_directions()
        for metric in self._selection_metrics():
            candidates = [row for row in rows if _is_number(row.get(metric))]
            if not candidates:
                continue
            reverse = directions.get(metric, "max") == "max"
            winner = sorted(candidates, key=lambda row: float(row[metric]), reverse=reverse)[0]
            best[metric] = {
                "name": winner.get("name"),
                "value": winner.get(metric),
                "direction": directions.get(metric, "max"),
            }
        return best

    def _selection_mode(self) -> str:
        selection = self.ablation.get("selection", {})
        if isinstance(selection, dict) and selection.get("mode"):
            return str(selection["mode"])
        return "offline"

    def _selection_metrics(self) -> list[str]:
        selection = self.ablation.get("selection", {})
        metrics = selection.get("metrics") if isinstance(selection, dict) else None
        if isinstance(metrics, list) and metrics:
            return [str(metric) for metric in metrics]
        focus = [str(metric) for metric in self.ablation.get("metric_focus", [])]
        return [metric for metric in focus if metric not in {"runtime", "memory"}] or ["final_total"]

    def _metric_directions(self) -> dict[str, str]:
        selection = self.ablation.get("selection", {})
        raw = selection.get("directions", {}) if isinstance(selection, dict) else {}
        directions = {
            "final_total": "min",
            "elapsed_seconds": "min",
            "latency_seconds": "min",
            "temporal_delta_error": "min",
            "causal_violation": "min",
            "fps": "max",
            "megapixels_per_second": "max",
            "psnr": "max",
            "ssim": "max",
            "uncertainty_error_correlation": "max",
        }
        if isinstance(raw, dict):
            directions.update({str(key): _normalize_direction(value) for key, value in raw.items()})
        return directions

    def _metric_weights(self) -> dict[str, float]:
        selection = self.ablation.get("selection", {})
        raw = selection.get("weights", {}) if isinstance(selection, dict) else {}
        if not isinstance(raw, dict):
            return {}
        return {str(key): float(value) for key, value in raw.items()}


def _slug(value: str) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    slug = "_".join(part for part in "".join(chars).split("_") if part)
    return slug or "variant"


def _normalize_value(value: float, minimum: float, maximum: float, direction: str) -> float:
    if maximum <= minimum:
        return 1.0
    if direction == "min":
        return (maximum - value) / (maximum - minimum)
    return (value - minimum) / (maximum - minimum)


def _normalize_direction(value: Any) -> str:
    direction = str(value).lower()
    if direction in {"min", "lower", "lower_is_better"}:
        return "min"
    if direction in {"max", "higher", "higher_is_better"}:
        return "max"
    raise ValueError(f"Metric direction must be 'min' or 'max': {value}")


def _matrix_axis_values(axis_name: str, axis: Any) -> list[tuple[str, list[str]]]:
    if not isinstance(axis, dict):
        raise ValueError(f"Matrix axis must be a mapping: {axis_name}")
    values = axis.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError(f"Matrix axis must define non-empty values: {axis_name}")
    overrides = axis.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"Matrix axis overrides must be a mapping: {axis_name}")

    axis_values = []
    for value in values:
        value_name = str(value)
        if value_name not in overrides:
            raise ValueError(f"Matrix axis value is missing overrides: {axis_name}.{value_name}")
        axis_values.append(
            (
                value_name,
                _override_list(f"matrix axis {axis_name}.{value_name}", overrides[value_name]),
            )
        )
    return axis_values


def _override_list(label: str, raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    raise ValueError(f"Overrides must be a string or list: {label}")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    columns = _row_columns(rows)
    lines = [",".join(_csv_cell(column) for column in columns)]
    for row in rows:
        lines.append(",".join(_csv_cell(row.get(column, "")) for column in columns))
    return "\n".join(lines) + "\n"


def _rows_to_markdown(rows: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> str:
    rank_by_name = {item["name"]: item for item in ranking}
    columns = ["rank", *_row_columns(rows), "selection_score"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        rank = rank_by_name.get(row.get("name"), {})
        values = {
            **row,
            "rank": rank.get("rank", ""),
            "selection_score": rank.get("selection_score", ""),
        }
        lines.append("| " + " | ".join(_markdown_cell(values.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _row_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["name", "status", "final_total", "elapsed_seconds", "psnr", "ssim", "temporal_delta_error"]
    columns = []
    for column in preferred:
        if any(column in row for row in rows):
            columns.append(column)
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    return columns


def _csv_cell(value: Any) -> str:
    text = _format_cell(value)
    if any(char in text for char in [",", "\"", "\n"]):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text


def _markdown_cell(value: Any) -> str:
    return _format_cell(value).replace("|", "\\|")


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
