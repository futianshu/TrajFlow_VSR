"""Baseline registry tracking for reproducible paper experiments."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajflow_vsr.utils.config import get_by_dotted_path, load_config


REQUIRED_FIELDS = [
    "display_name",
    "category",
    "repo_url",
    "commit",
    "weights.source",
    "weights.local_path",
    "commands.setup",
    "commands.infer",
    "commands.evaluate",
    "metrics_path",
]


TABLE_COLUMNS = [
    "id",
    "display_name",
    "category",
    "status",
    "repo_url",
    "commit",
    "weights.source",
    "weights.local_path",
    "metrics_path",
    "complete",
    "missing_fields",
]


PLACEHOLDER_TOKENS = ("TODO", "TBD", "FILL_ME")


@dataclass(frozen=True)
class BaselineRegistryExport:
    """Files written by a baseline registry export."""

    output_dir: str
    files: dict[str, str]
    baseline_count: int
    complete_count: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


def load_baseline_registry(path: str | Path) -> dict[str, Any]:
    """Load a baseline registry config."""

    config = load_config(path)
    if not isinstance(config.get("baselines"), dict) or not config["baselines"]:
        raise ValueError("Baseline registry must define a non-empty 'baselines' mapping")
    return config


def baseline_records(config: dict[str, Any], required_fields: list[str] | None = None) -> list[dict[str, Any]]:
    """Normalize registry entries and attach completeness metadata."""

    required = required_fields or REQUIRED_FIELDS
    records = []
    for baseline_id, raw in config.get("baselines", {}).items():
        if not isinstance(raw, dict):
            raise ValueError(f"Baseline entry must be a mapping: {baseline_id}")
        record = {"id": str(baseline_id), **raw}
        missing = [field for field in required if _is_missing(get_by_dotted_path(record, field))]
        record["missing_fields"] = missing
        record["complete"] = not missing
        records.append(record)
    return records


def export_baseline_registry(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    name: str = "baseline_registry",
    required_fields: list[str] | None = None,
) -> BaselineRegistryExport:
    """Export baseline records to JSON, CSV, and Markdown checklists."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = baseline_records(config, required_fields=required_fields)
    payload = {
        "registry": config.get("registry", {}),
        "baseline_count": len(records),
        "complete_count": sum(1 for record in records if record["complete"]),
        "records": records,
    }

    json_path = out / f"{name}.json"
    csv_path = out / f"{name}.csv"
    table_path = out / f"{name}.md"
    checklist_path = out / f"{name}_checklist.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path.write_text(_records_to_csv(records), encoding="utf-8")
    table_path.write_text(_records_to_markdown(records), encoding="utf-8")
    checklist_path.write_text(_records_to_checklist(config, records), encoding="utf-8")

    files = {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(table_path),
        "checklist": str(checklist_path),
    }
    manifest = out / f"{name}_export.json"
    export = BaselineRegistryExport(
        output_dir=str(out),
        files=files,
        baseline_count=len(records),
        complete_count=payload["complete_count"],
    )
    manifest.write_text(export.to_json(), encoding="utf-8")
    files["manifest"] = str(manifest)
    return BaselineRegistryExport(
        output_dir=str(out),
        files=files,
        baseline_count=len(records),
        complete_count=payload["complete_count"],
    )


def _records_to_csv(records: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow(_flat_record(record))
    return buffer.getvalue()


def _records_to_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |",
    ]
    for record in records:
        flat = _flat_record(record)
        lines.append("| " + " | ".join(_markdown_cell(flat.get(column, "")) for column in TABLE_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def _records_to_checklist(config: dict[str, Any], records: list[dict[str, Any]]) -> str:
    registry = config.get("registry", {})
    lines = [
        f"# {registry.get('name', 'Baseline Registry')}",
        "",
        f"Purpose: {registry.get('purpose', 'track reproducible baseline runs')}",
        "",
        f"Baselines: {len(records)}",
        f"Complete: {sum(1 for record in records if record['complete'])}",
        "",
    ]
    for record in records:
        status = "complete" if record["complete"] else "pending"
        lines.extend(
            [
                f"## {record['id']} - {record.get('display_name', record['id'])}",
                "",
                f"- status: {record.get('status', status)}",
                f"- category: {record.get('category', '')}",
                f"- repo: {record.get('repo_url', '')}",
                f"- commit: {record.get('commit', '')}",
                f"- weights: {get_by_dotted_path(record, 'weights.source', '')}",
                f"- metrics: {record.get('metrics_path', '')}",
                f"- missing: {', '.join(record['missing_fields']) or 'none'}",
                "",
                "Commands:",
                "",
                "```bash",
                str(get_by_dotted_path(record, "commands.setup", "")),
                str(get_by_dotted_path(record, "commands.infer", "")),
                str(get_by_dotted_path(record, "commands.evaluate", "")),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _flat_record(record: dict[str, Any]) -> dict[str, str]:
    flat = {}
    for column in TABLE_COLUMNS:
        value = record.get(column)
        if "." in column:
            value = get_by_dotted_path(record, column)
        if column == "missing_fields":
            value = ", ".join(str(item) for item in record.get("missing_fields", []))
        flat[column] = _format_cell(value)
    return flat


def _format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _markdown_cell(value: Any) -> str:
    return _format_cell(value).replace("|", "\\|")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    upper = text.upper()
    return any(token in upper for token in PLACEHOLDER_TOKENS)
