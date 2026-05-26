"""Benchmark protocol planning utilities."""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from trajflow_vsr.evaluation import EvaluationRunner
from trajflow_vsr.utils.config import apply_overrides, load_config


PLAN_COLUMNS = [
    "id",
    "method",
    "kind",
    "dataset",
    "degradation",
    "scale",
    "protocol",
    "status",
    "metrics_path",
    "command",
]

RUN_COLUMNS = [
    "id",
    "method",
    "dataset",
    "degradation",
    "scale",
    "protocol",
    "status",
    "metrics_path",
    "psnr",
    "ssim",
    "temporal_delta_error",
    "tof",
    "lpips",
    "dists",
    "reliability_ece",
    "selective_reconstruction_auc",
    "fps",
    "latency_seconds",
    "vram_gb",
    "parameters",
    "macs",
    "gmacs",
    "elapsed_seconds",
    "error",
]


@dataclass(frozen=True)
class BenchmarkPlanExport:
    """Files written by a benchmark plan export."""

    output_dir: str
    files: dict[str, str]
    run_count: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class BenchmarkRunExport:
    """Files written by an executable benchmark run."""

    output_dir: str
    files: dict[str, str]
    run_count: int
    ok_count: int
    skipped_count: int
    failed_count: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


class BenchmarkPlan:
    """Expand benchmark methods, data, degradations, scales, and protocols."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.benchmark = config.get("benchmark", {})
        if not isinstance(self.benchmark, dict):
            raise ValueError("Benchmark config must contain a 'benchmark' mapping")

    def rows(self) -> list[dict[str, Any]]:
        methods = _required_mapping(self.benchmark, "methods")
        datasets = _required_mapping(self.benchmark, "datasets")
        degradations = _required_mapping(self.benchmark, "degradations")
        scales = _required_mapping(self.benchmark, "scales")
        protocols = _required_mapping(self.benchmark, "protocols")
        rows = []
        for method, dataset, degradation, scale, protocol in product(
            methods.items(),
            datasets.items(),
            degradations.items(),
            scales.items(),
            protocols.items(),
        ):
            rows.append(
                self._row(
                    method_name=method[0],
                    method_cfg=method[1],
                    dataset_name=dataset[0],
                    dataset_cfg=dataset[1],
                    degradation_name=degradation[0],
                    degradation_cfg=degradation[1],
                    scale_name=scale[0],
                    scale_cfg=scale[1],
                    protocol_name=protocol[0],
                    protocol_cfg=protocol[1],
                )
            )
        return rows

    def export(self, output_dir: str | Path | None = None, *, name: str | None = None) -> BenchmarkPlanExport:
        out = Path(output_dir) if output_dir is not None else Path(self.benchmark.get("output_dir", "outputs/benchmark"))
        out.mkdir(parents=True, exist_ok=True)
        plan_name = name or str(self.benchmark.get("name", "benchmark_plan"))
        rows = self.rows()
        payload = {
            "name": self.benchmark.get("name", plan_name),
            "purpose": self.benchmark.get("purpose"),
            "metrics": self.benchmark.get("metrics", []),
            "run_count": len(rows),
            "rows": rows,
        }

        json_path = out / f"{plan_name}.json"
        csv_path = out / f"{plan_name}.csv"
        md_path = out / f"{plan_name}.md"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        csv_path.write_text(_rows_to_csv(rows), encoding="utf-8")
        md_path.write_text(_rows_to_markdown(rows), encoding="utf-8")

        files = {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
        }
        manifest = out / f"{plan_name}_export.json"
        export = BenchmarkPlanExport(output_dir=str(out), files=files, run_count=len(rows))
        manifest.write_text(export.to_json(), encoding="utf-8")
        files["manifest"] = str(manifest)
        return BenchmarkPlanExport(output_dir=str(out), files=files, run_count=len(rows))

    def _row(
        self,
        *,
        method_name: str,
        method_cfg: dict[str, Any],
        dataset_name: str,
        dataset_cfg: dict[str, Any],
        degradation_name: str,
        degradation_cfg: dict[str, Any],
        scale_name: str,
        scale_cfg: dict[str, Any],
        protocol_name: str,
        protocol_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        run_id = _slug("_".join([method_name, dataset_name, degradation_name, scale_name, protocol_name]))
        root = Path(str(self.benchmark.get("metrics_root", "experiments/benchmark_metrics")))
        metrics_path = root / method_name / f"{dataset_name}_{degradation_name}_{scale_name}_{protocol_name}.json"
        overrides = [
            *_override_list(method_cfg.get("overrides", [])),
            *_override_list(dataset_cfg.get("overrides", [])),
            *_override_list(degradation_cfg.get("overrides", [])),
            *_override_list(scale_cfg.get("overrides", [])),
            *_override_list(protocol_cfg.get("overrides", [])),
        ]
        kind = str(method_cfg.get("kind", "internal"))
        row = {
            "id": run_id,
            "method": method_name,
            "display_name": method_cfg.get("display_name", method_name),
            "kind": kind,
            "dataset": dataset_name,
            "degradation": degradation_name,
            "scale": scale_name,
            "protocol": protocol_name,
            "metrics_path": str(metrics_path),
            "overrides": overrides,
            "status": "planned_internal" if kind == "internal" else "planned_external",
        }
        row["command"] = self._command(
            method_cfg=method_cfg,
            protocol_cfg=protocol_cfg,
            kind=kind,
            metrics_path=metrics_path,
            overrides=overrides,
        )
        return row

    def _command(
        self,
        *,
        method_cfg: dict[str, Any],
        protocol_cfg: dict[str, Any],
        kind: str,
        metrics_path: Path,
        overrides: list[str],
    ) -> str:
        if kind != "internal":
            registry_id = method_cfg.get("registry_id", "")
            return f"write_external_baseline_metrics registry_id={registry_id} metrics_path={metrics_path}"
        eval_config = str(protocol_cfg.get("eval_config") or method_cfg.get("eval_config") or "configs/eval/offline.yaml")
        checkpoint = str(method_cfg.get("checkpoint_path", "")).strip()
        command_parts = [
            'CUDA_VISIBLE_DEVICES=""',
            "uv",
            "run",
            "python",
            "scripts/evaluate.py",
            "--config",
            eval_config,
            "--set",
            "runtime.device=cpu",
            "--set",
            f"evaluation.output_path={metrics_path}",
        ]
        if checkpoint and "TODO" not in checkpoint.upper():
            command_parts.extend(["--set", f"evaluation.checkpoint_path={checkpoint}"])
        for override in overrides:
            command_parts.extend(["--set", override])
        return " ".join(command_parts)


class BenchmarkRunner:
    """Execute the internal subset of a fixed benchmark matrix."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.plan = BenchmarkPlan(config)
        self.benchmark = self.plan.benchmark

    def summarize(
        self,
        *,
        row_ids: list[str] | None = None,
        methods: list[str] | None = None,
        datasets: list[str] | None = None,
        degradations: list[str] | None = None,
        scales: list[str] | None = None,
        protocols: list[str] | None = None,
        max_runs: int | None = None,
        include_external: bool = False,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        rows = self._selected_rows(
            row_ids=row_ids,
            methods=methods,
            datasets=datasets,
            degradations=degradations,
            scales=scales,
            protocols=protocols,
            max_runs=max_runs,
            include_external=include_external,
        )
        summary = {
            "name": self._name(),
            "purpose": self.benchmark.get("purpose"),
            "output_dir": str(self._output_dir(output_dir)),
            "run_count": len(rows),
            "rows": rows,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    def run(
        self,
        *,
        row_ids: list[str] | None = None,
        methods: list[str] | None = None,
        datasets: list[str] | None = None,
        degradations: list[str] | None = None,
        scales: list[str] | None = None,
        protocols: list[str] | None = None,
        max_runs: int | None = None,
        include_external: bool = False,
        output_dir: str | Path | None = None,
        common_overrides: list[str] | None = None,
        allow_missing_checkpoints: bool = False,
        fail_fast: bool = True,
    ) -> dict[str, Any]:
        root = self._output_dir(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        rows = self._selected_rows(
            row_ids=row_ids,
            methods=methods,
            datasets=datasets,
            degradations=degradations,
            scales=scales,
            protocols=protocols,
            max_runs=max_runs,
            include_external=include_external,
        )
        records = []
        for row in rows:
            started = time.perf_counter()
            try:
                if row.get("kind") != "internal":
                    record = self._skipped_record(row, status="skipped_external_baseline")
                else:
                    record = self._run_internal_row(
                        row,
                        root=root,
                        common_overrides=common_overrides or [],
                        allow_missing_checkpoints=allow_missing_checkpoints,
                    )
            except Exception as exc:
                if fail_fast:
                    raise
                record = self._skipped_record(row, status="failed", error=str(exc))
            record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            records.append(record)

        export = self._write_run_artifacts(root, rows=rows, records=records)
        summary = {
            "name": self._name(),
            "purpose": self.benchmark.get("purpose"),
            "output_dir": str(root),
            "run_count": len(records),
            "ok_count": sum(1 for record in records if record.get("status") == "ok"),
            "skipped_count": sum(1 for record in records if str(record.get("status", "")).startswith("skipped")),
            "failed_count": sum(1 for record in records if record.get("status") == "failed"),
            "records": records,
            "export": export.__dict__,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    def _run_internal_row(
        self,
        row: dict[str, Any],
        *,
        root: Path,
        common_overrides: list[str],
        allow_missing_checkpoints: bool,
    ) -> dict[str, Any]:
        method_cfg = _required_mapping(self.benchmark, "methods")[str(row["method"])]
        protocol_cfg = _required_mapping(self.benchmark, "protocols")[str(row["protocol"])]
        eval_config_path = str(protocol_cfg.get("eval_config") or method_cfg.get("eval_config") or "configs/eval/offline.yaml")
        metrics_path = self._metrics_path(root, row)
        overrides = [
            *list(row.get("overrides", [])),
            *common_overrides,
            "runtime.device=cpu",
            "runtime.dry_run=false",
            f"evaluation.output_path={metrics_path}",
        ]
        checkpoint_override = self._checkpoint_override(method_cfg, allow_missing_checkpoints=allow_missing_checkpoints)
        if checkpoint_override is None and _has_checkpoint_path(method_cfg) and not allow_missing_checkpoints:
            return self._skipped_record(
                row,
                status="skipped_missing_checkpoint",
                metrics_path=str(metrics_path),
                error=f"Missing checkpoint: {method_cfg.get('checkpoint_path')}",
            )
        if checkpoint_override:
            overrides.append(checkpoint_override)

        eval_config = apply_overrides(load_config(eval_config_path), overrides)
        result = EvaluationRunner(eval_config).run()
        metrics = result.get("metrics", {})
        return {
            **self._record_identity(row),
            "status": "ok",
            "eval_config": eval_config_path,
            "metrics_path": result.get("output_path", str(metrics_path)),
            "checkpoint": result.get("checkpoint", {}),
            "profile": result.get("profile", {}),
            **_profile_snapshot(result.get("profile", {})),
            "metrics": metrics,
            **_metric_snapshot(metrics, mode=str(row.get("protocol", "offline"))),
        }

    def _checkpoint_override(self, method_cfg: dict[str, Any], *, allow_missing_checkpoints: bool) -> str | None:
        checkpoint = str(method_cfg.get("checkpoint_path", "")).strip()
        if not checkpoint or "TODO" in checkpoint.upper():
            return None
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            return f"evaluation.checkpoint_path={checkpoint_path}"
        if allow_missing_checkpoints:
            return None
        return None

    def _selected_rows(
        self,
        *,
        row_ids: list[str] | None,
        methods: list[str] | None,
        datasets: list[str] | None,
        degradations: list[str] | None,
        scales: list[str] | None,
        protocols: list[str] | None,
        max_runs: int | None,
        include_external: bool,
    ) -> list[dict[str, Any]]:
        filters = {
            "id": set(row_ids or []),
            "method": set(methods or []),
            "dataset": set(datasets or []),
            "degradation": set(degradations or []),
            "scale": set(scales or []),
            "protocol": set(protocols or []),
        }
        rows = []
        for row in self.plan.rows():
            if not include_external and row.get("kind") != "internal":
                continue
            if any(values and str(row.get(key)) not in values for key, values in filters.items()):
                continue
            rows.append(row)
        if max_runs is not None:
            rows = rows[: max(int(max_runs), 0)]
        return rows

    def _write_run_artifacts(
        self,
        root: Path,
        *,
        rows: list[dict[str, Any]],
        records: list[dict[str, Any]],
    ) -> BenchmarkRunExport:
        plan_payload = {
            "name": self._name(),
            "metrics": self.benchmark.get("metrics", []),
            "run_count": len(rows),
            "rows": rows,
        }
        summary_payload = {
            "name": self._name(),
            "records": records,
        }
        plan_path = root / "selected_plan.json"
        summary_path = root / "summary.json"
        csv_path = root / "summary.csv"
        md_path = root / "summary.md"
        plan_path.write_text(json.dumps(plan_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        csv_path.write_text(_run_records_to_csv(records), encoding="utf-8")
        md_path.write_text(_run_records_to_markdown(records), encoding="utf-8")
        files = {
            "selected_plan": str(plan_path),
            "summary": str(summary_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
        }
        export = BenchmarkRunExport(
            output_dir=str(root),
            files=files,
            run_count=len(records),
            ok_count=sum(1 for record in records if record.get("status") == "ok"),
            skipped_count=sum(1 for record in records if str(record.get("status", "")).startswith("skipped")),
            failed_count=sum(1 for record in records if record.get("status") == "failed"),
        )
        manifest = root / "benchmark_run_export.json"
        manifest.write_text(export.to_json(), encoding="utf-8")
        files["manifest"] = str(manifest)
        return BenchmarkRunExport(
            output_dir=export.output_dir,
            files=files,
            run_count=export.run_count,
            ok_count=export.ok_count,
            skipped_count=export.skipped_count,
            failed_count=export.failed_count,
        )

    def _skipped_record(
        self,
        row: dict[str, Any],
        *,
        status: str,
        metrics_path: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        record = {
            **self._record_identity(row),
            "status": status,
            "metrics_path": metrics_path or row.get("metrics_path"),
        }
        if error:
            record["error"] = error
        return record

    def _record_identity(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row.get("id"),
            "method": row.get("method"),
            "display_name": row.get("display_name"),
            "kind": row.get("kind"),
            "dataset": row.get("dataset"),
            "degradation": row.get("degradation"),
            "scale": row.get("scale"),
            "protocol": row.get("protocol"),
        }

    def _metrics_path(self, root: Path, row: dict[str, Any]) -> Path:
        return root / "metrics" / str(row.get("method", "method")) / Path(str(row.get("metrics_path", "metrics.json"))).name

    def _output_dir(self, output_dir: str | Path | None = None) -> Path:
        if output_dir is not None:
            return Path(output_dir)
        configured = self.benchmark.get("run_output_dir")
        if configured:
            return Path(str(configured))
        return Path("outputs") / "benchmark_runs" / _slug(self._name())

    def _name(self) -> str:
        return str(self.benchmark.get("name", "benchmark"))


def _required_mapping(parent: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    raw = parent.get(key)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"benchmark.{key} must be a non-empty mapping")
    mapping = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"benchmark.{key}.{name} must be a mapping")
        mapping[str(name)] = value
    return mapping


def _override_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if raw is None or raw == "":
        return []
    raise ValueError(f"Overrides must be a string or list: {raw}")


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=PLAN_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _format_cell(row.get(column, "")) for column in PLAN_COLUMNS})
    return buffer.getvalue()


def _rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(PLAN_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in PLAN_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(column, "")) for column in PLAN_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def _run_records_to_csv(records: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RUN_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow({column: _format_cell(record.get(column, "")) for column in RUN_COLUMNS})
    return buffer.getvalue()


def _run_records_to_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(RUN_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in RUN_COLUMNS) + " |",
    ]
    for record in records:
        lines.append("| " + " | ".join(_markdown_cell(record.get(column, "")) for column in RUN_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def _metric_snapshot(metrics: Any, *, mode: str) -> dict[str, Any]:
    if not isinstance(metrics, dict) or not metrics:
        return {}
    mode_metrics = metrics.get(mode)
    if not isinstance(mode_metrics, dict):
        mode_metrics = next((value for value in metrics.values() if isinstance(value, dict)), {})
    return {
        key: float(value)
        for key, value in mode_metrics.items()
        if key in RUN_COLUMNS and _is_number(value)
    }


def _profile_snapshot(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    snapshot = {}
    for key in ["parameters", "macs", "gmacs"]:
        value = profile.get(key)
        if _is_number(value):
            snapshot[key] = float(value) if key == "gmacs" else int(value)
    return snapshot


def _has_checkpoint_path(method_cfg: dict[str, Any]) -> bool:
    checkpoint = str(method_cfg.get("checkpoint_path", "")).strip()
    return bool(checkpoint) and "TODO" not in checkpoint.upper()


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _format_cell(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def _markdown_cell(value: Any) -> str:
    return _format_cell(value).replace("|", "\\|")


def _slug(value: str) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join(part for part in "".join(chars).split("_") if part) or "run"
