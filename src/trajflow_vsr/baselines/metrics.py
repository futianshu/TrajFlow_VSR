"""Collect baseline metric files into reproducible comparison tables."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajflow_vsr.baselines.registry import baseline_records


DEFAULT_METRICS = [
    "psnr",
    "ssim",
    "temporal_delta_error",
    "tof",
    "lpips",
    "dists",
    "vmaf",
    "uncertainty_error_correlation",
    "reliability_ece",
    "selective_reconstruction_auc",
    "fps",
    "latency_seconds",
    "megapixels_per_second",
    "vram_gb",
]


BASE_COLUMNS = ["id", "display_name", "category", "status", "metrics_status", "metrics_path"]


@dataclass(frozen=True)
class BaselineMetricsExport:
    """Files written by a baseline metrics export."""

    output_dir: str
    files: dict[str, str]
    baseline_count: int
    available_count: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


def collect_baseline_metrics(
    config: dict[str, Any],
    *,
    metrics: list[str] | None = None,
    root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Read metric files referenced by a baseline registry."""

    selected_metrics = metrics or DEFAULT_METRICS
    root_path = Path(root) if root is not None else Path(".")
    rows = []
    for record in baseline_records(config):
        metrics_path = Path(str(record.get("metrics_path", "")))
        resolved = metrics_path if metrics_path.is_absolute() else root_path / metrics_path
        row: dict[str, Any] = {
            "id": record["id"],
            "display_name": record.get("display_name", record["id"]),
            "category": record.get("category", ""),
            "status": record.get("status", ""),
            "metrics_path": str(metrics_path),
        }
        if not metrics_path or not resolved.exists():
            row["metrics_status"] = "missing_metrics"
            for metric in selected_metrics:
                row[metric] = ""
            rows.append(row)
            continue

        payload = json.loads(resolved.read_text(encoding="utf-8"))
        flat_metrics = _extract_metrics(payload)
        row["metrics_status"] = "ok"
        for metric in selected_metrics:
            row[metric] = flat_metrics.get(metric, "")
        rows.append(row)
    return rows


def export_baseline_metrics(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    name: str = "baseline_metrics",
    metrics: list[str] | None = None,
    root: str | Path | None = None,
) -> BaselineMetricsExport:
    """Export baseline metric rows to JSON, CSV, and Markdown."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected_metrics = metrics or DEFAULT_METRICS
    rows = collect_baseline_metrics(config, metrics=selected_metrics, root=root)
    payload = {
        "baseline_count": len(rows),
        "available_count": sum(1 for row in rows if row.get("metrics_status") == "ok"),
        "metrics": selected_metrics,
        "rows": rows,
        "best_by_metric": _best_by_metric(rows, selected_metrics),
    }

    json_path = out / f"{name}.json"
    csv_path = out / f"{name}.csv"
    md_path = out / f"{name}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path.write_text(_rows_to_csv(rows, selected_metrics), encoding="utf-8")
    md_path.write_text(_rows_to_markdown(rows, selected_metrics), encoding="utf-8")

    files = {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
    manifest = out / f"{name}_export.json"
    export = BaselineMetricsExport(
        output_dir=str(out),
        files=files,
        baseline_count=payload["baseline_count"],
        available_count=payload["available_count"],
    )
    manifest.write_text(export.to_json(), encoding="utf-8")
    files["manifest"] = str(manifest)
    return BaselineMetricsExport(
        output_dir=str(out),
        files=files,
        baseline_count=payload["baseline_count"],
        available_count=payload["available_count"],
    )


def _extract_metrics(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Metric file must contain a JSON object")
    if isinstance(payload.get("metrics"), dict):
        return _extract_metrics(payload["metrics"])
    if isinstance(payload.get("offline"), dict):
        return _numeric_items(payload["offline"])
    if isinstance(payload.get("streaming"), dict):
        return _numeric_items(payload["streaming"])
    return _numeric_items(payload)


def _numeric_items(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in payload.items() if _is_number(value)}


def _best_by_metric(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for metric in metrics:
        candidates = [row for row in rows if _is_number(row.get(metric))]
        if not candidates:
            continue
        reverse = _metric_direction(metric) == "max"
        winner = sorted(candidates, key=lambda row: float(row[metric]), reverse=reverse)[0]
        best[metric] = {
            "id": winner.get("id"),
            "display_name": winner.get("display_name"),
            "value": winner.get(metric),
            "direction": _metric_direction(metric),
        }
    return best


def _rows_to_csv(rows: list[dict[str, Any]], metrics: list[str]) -> str:
    buffer = io.StringIO()
    columns = [*BASE_COLUMNS, *metrics]
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _format_cell(row.get(column, "")) for column in columns})
    return buffer.getvalue()


def _rows_to_markdown(rows: list[dict[str, Any]], metrics: list[str]) -> str:
    columns = [*BASE_COLUMNS, *metrics]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _metric_direction(metric: str) -> str:
    if metric in {
        "temporal_delta_error",
        "tof",
        "lpips",
        "dists",
        "latency_seconds",
        "vram_gb",
        "reliability_ece",
        "selective_reconstruction_auc",
    }:
        return "min"
    return "max"


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_cell(value: Any) -> str:
    return _format_cell(value).replace("|", "\\|")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
