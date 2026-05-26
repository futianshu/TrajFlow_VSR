"""Data manifest inventory and protocol audit helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajflow_vsr.data.manifest import load_frame_manifest
from trajflow_vsr.utils.config import load_config


@dataclass(frozen=True)
class DataInventoryExport:
    """Files written by a data inventory export."""

    output_dir: str
    files: dict[str, str]
    manifest_count: int
    issue_count: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


def build_data_inventory(root: str | Path = ".", manifest_paths: list[str] | None = None) -> dict[str, Any]:
    """Collect lightweight information about manifests without reading image tensors."""

    project_root = Path(root)
    paths = set(manifest_paths or [])
    paths.update(_configured_manifest_paths(project_root))
    paths.update(str(path.relative_to(project_root)) for path in sorted((project_root / "data/splits").glob("*.json")))

    manifests = []
    issues = []
    for manifest_path in sorted(paths):
        entry = _manifest_entry(project_root, manifest_path)
        manifests.append(entry)
        for warning in entry.get("warnings", []):
            issues.append({"path": entry["path"], "severity": "warning", "message": warning})
        if not entry.get("exists", False) and not _is_placeholder(manifest_path):
            issues.append({"path": entry["path"], "severity": "missing", "message": "manifest file does not exist"})

    return {
        "name": "trajflow_vsr_data_inventory",
        "root": str(project_root.resolve()),
        "manifest_count": len(manifests),
        "issue_count": len(issues),
        "manifests": manifests,
        "issues": issues,
    }


def export_data_inventory(
    inventory: dict[str, Any],
    output_dir: str | Path,
    *,
    name: str = "data_inventory",
) -> DataInventoryExport:
    """Write data inventory artifacts as JSON and Markdown."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{name}.json"
    md_path = out / f"{name}.md"
    json_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_inventory_markdown(inventory), encoding="utf-8")
    files = {"json": str(json_path), "markdown": str(md_path)}
    export = DataInventoryExport(
        output_dir=str(out),
        files=files,
        manifest_count=int(inventory.get("manifest_count", 0)),
        issue_count=int(inventory.get("issue_count", 0)),
    )
    manifest_path = out / f"{name}_export.json"
    manifest_path.write_text(export.to_json(), encoding="utf-8")
    files["manifest"] = str(manifest_path)
    return DataInventoryExport(
        output_dir=export.output_dir,
        files=files,
        manifest_count=export.manifest_count,
        issue_count=export.issue_count,
    )


def _manifest_entry(root: Path, manifest_path: str) -> dict[str, Any]:
    full_path = _resolve_path(root, manifest_path)
    entry: dict[str, Any] = {
        "path": manifest_path,
        "exists": full_path.exists(),
        "placeholder": _is_placeholder(manifest_path),
        "warnings": [],
    }
    if not full_path.exists():
        return entry
    try:
        manifest = load_frame_manifest(full_path)
    except Exception as exc:
        entry["warnings"].append(f"failed to load manifest: {exc}")
        return entry

    sequences = manifest.get("sequences", [])
    clips = manifest.get("clips", [])
    paired_sequences = sum(1 for sequence in sequences if sequence.get("hr_frames"))
    frame_count = sum(len(sequence.get("frames", [])) for sequence in sequences)
    role = _infer_role(manifest)
    paired = bool(manifest.get("paired", False))
    entry.update(
        {
            "dataset": manifest.get("dataset"),
            "split": manifest.get("split"),
            "role": role,
            "layout": manifest.get("layout"),
            "root": manifest.get("root"),
            "hr_root": manifest.get("hr_root"),
            "paired": paired,
            "paired_sequences": paired_sequences,
            "sequences": len(sequences),
            "clips": len(clips),
            "frames": frame_count,
            "degradation": manifest.get("degradation"),
        }
    )
    _attach_protocol_warnings(entry)
    return entry


def _attach_protocol_warnings(entry: dict[str, Any]) -> None:
    dataset = str(entry.get("dataset", "")).lower()
    role = str(entry.get("role", "")).lower()
    paired = bool(entry.get("paired", False))
    paired_sequences = int(entry.get("paired_sequences", 0))
    sequences = int(entry.get("sequences", 0))
    if role == "train" and not paired and dataset != "videolq":
        entry["warnings"].append("training manifest is unpaired; formal Stage B-E training should require HR targets")
    if paired and paired_sequences != sequences:
        entry["warnings"].append("paired manifest has sequences without HR targets")
    if dataset == "videolq" and paired:
        entry["warnings"].append("VideoLQ should remain no-reference unless GT is explicitly added")


def _configured_manifest_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    config_root = root / "configs"
    if not config_root.exists():
        return paths
    for config_path in sorted(config_root.glob("**/*.yaml")):
        try:
            config = load_config(config_path)
        except Exception:
            continue
        paths.update(_find_manifest_paths(config))
    return paths


def _find_manifest_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in {"manifest_path", "manifest_output", "bdx4_manifest_path"} and isinstance(item, str):
                paths.add(item)
            else:
                paths.update(_find_manifest_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.update(_find_manifest_paths(item))
    elif isinstance(value, str) and "manifest_path=" in value:
        paths.add(value.split("manifest_path=", 1)[1].split(",", 1)[0])
    return paths


def _infer_role(manifest: dict[str, Any]) -> str:
    split = str(manifest.get("split", "")).lower()
    if "train" in split:
        return "train"
    if "val" in split or "valid" in split:
        return "val"
    if "test" in split or "real" in split:
        return "test"
    return split or "unknown"


def _resolve_path(root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _is_placeholder(path: str) -> bool:
    upper = str(path).upper()
    return any(token in upper for token in ["YOUR_", "TODO", "TBD", "FILL_ME"])


def _inventory_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# Data Inventory",
        "",
        f"- manifests: {inventory.get('manifest_count', 0)}",
        f"- issues: {inventory.get('issue_count', 0)}",
        "",
        "| path | dataset | split | role | paired | sequences | clips | frames | warnings |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in inventory.get("manifests", []):
        warnings = "; ".join(str(warning) for warning in item.get("warnings", []))
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(item.get("path")),
                    _cell(item.get("dataset")),
                    _cell(item.get("split")),
                    _cell(item.get("role")),
                    _cell(item.get("paired")),
                    _cell(item.get("sequences")),
                    _cell(item.get("clips")),
                    _cell(item.get("frames")),
                    _cell(warnings),
                ]
            )
            + " |"
        )
    if inventory.get("issues"):
        lines.extend(["", "## Issues", ""])
        for issue in inventory.get("issues", []):
            lines.append(f"- {issue.get('severity')}: {issue.get('path')} - {issue.get('message')}")
    return "\n".join(lines) + "\n"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("|", "\\|")
