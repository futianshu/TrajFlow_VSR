"""CCF-A proposal readiness audit utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajflow_vsr.baselines.registry import baseline_records, load_baseline_registry
from trajflow_vsr.metrics import metric_backend_status
from trajflow_vsr.utils.config import load_config


STATUS_OK = "ok"
STATUS_PENDING = "pending"
STATUS_EXTERNAL = "external_required"


@dataclass(frozen=True)
class ReadinessExport:
    """Files written by a readiness audit export."""

    output_dir: str
    files: dict[str, str]
    ok_count: int
    pending_count: int
    external_required_count: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


def audit_project_readiness(root: str | Path = ".") -> dict[str, Any]:
    """Audit proposal-facing gaps without fabricating external results."""

    project_root = Path(root)
    checks: list[dict[str, Any]] = []
    _check_core_code(project_root, checks)
    _check_configs(project_root, checks)
    _check_manifests(project_root, checks)
    _check_checkpoints(project_root, checks)
    _check_baselines(project_root, checks)
    _check_metric_backends(checks)
    _check_paper_docs(project_root, checks)

    counts = _status_counts(checks)
    return {
        "name": "ccfa_proposal_readiness",
        "root": str(project_root.resolve()),
        "ready_for_paper_claims": counts.get(STATUS_PENDING, 0) == 0 and counts.get(STATUS_EXTERNAL, 0) == 0,
        "counts": counts,
        "checks": checks,
    }


def export_readiness_audit(
    report: dict[str, Any],
    output_dir: str | Path,
    *,
    name: str = "ccfa_readiness",
) -> ReadinessExport:
    """Write readiness audit artifacts as JSON and Markdown."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{name}.json"
    md_path = out / f"{name}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_readiness_markdown(report), encoding="utf-8")
    files = {"json": str(json_path), "markdown": str(md_path)}
    counts = report.get("counts", {})
    export = ReadinessExport(
        output_dir=str(out),
        files=files,
        ok_count=int(counts.get(STATUS_OK, 0)),
        pending_count=int(counts.get(STATUS_PENDING, 0)),
        external_required_count=int(counts.get(STATUS_EXTERNAL, 0)),
    )
    manifest = out / f"{name}_export.json"
    manifest.write_text(export.to_json(), encoding="utf-8")
    files["manifest"] = str(manifest)
    return ReadinessExport(
        output_dir=export.output_dir,
        files=files,
        ok_count=export.ok_count,
        pending_count=export.pending_count,
        external_required_count=export.external_required_count,
    )


def _check_core_code(root: Path, checks: list[dict[str, Any]]) -> None:
    required = [
        "src/trajflow_vsr/models/tokenizer/evidence_tokenizer.py",
        "src/trajflow_vsr/models/uncertainty/degradation_encoder.py",
        "src/trajflow_vsr/models/transport/ot_sb_bridge.py",
        "src/trajflow_vsr/models/memory/trajectory_koopman_ssm.py",
        "src/trajflow_vsr/models/flow/rectified_flow.py",
        "src/trajflow_vsr/models/decoder/wavelet_operator_decoder.py",
        "src/trajflow_vsr/models/consistency/data_projection.py",
        "src/trajflow_vsr/evaluation/runner.py",
    ]
    for relative in required:
        _add_path_check(checks, "core_code", relative, root / relative)


def _check_configs(root: Path, checks: list[dict[str, Any]]) -> None:
    required = [
        "configs/train/stage_a_tokenizer.yaml",
        "configs/train/stage_b_frame_manifest_full.yaml",
        "configs/train/stage_c_rectified_flow_full.yaml",
        "configs/train/stage_d_distill_full.yaml",
        "configs/train/stage_e_streaming_full.yaml",
        "configs/eval/paper_official.yaml",
        "configs/benchmark/ccfa_core_benchmark.yaml",
        "configs/baselines/core_vsr_baselines.yaml",
    ]
    for relative in required:
        _add_path_check(checks, "configs", relative, root / relative)


def _check_manifests(root: Path, checks: list[dict[str, Any]]) -> None:
    paths = set()
    for config_path in sorted((root / "configs").glob("**/*.yaml")):
        try:
            config = load_config(config_path)
        except Exception as exc:
            _add_check(checks, "config_parse", str(config_path), STATUS_PENDING, detail=str(exc))
            continue
        for manifest_path in _find_manifest_paths(config):
            paths.add(manifest_path)
    for manifest_path in sorted(paths):
        status = STATUS_EXTERNAL if _is_placeholder_path(manifest_path) else STATUS_PENDING
        full_path = root / manifest_path
        if full_path.exists():
            status = STATUS_OK
        _add_check(
            checks,
            "data_manifests",
            manifest_path,
            status,
            path=manifest_path,
            detail="create with scripts/prepare_data.py or scripts/degrade_data.py",
        )


def _check_checkpoints(root: Path, checks: list[dict[str, Any]]) -> None:
    benchmark_path = root / "configs/benchmark/ccfa_core_benchmark.yaml"
    if not benchmark_path.exists():
        return
    benchmark = load_config(benchmark_path).get("benchmark", {})
    methods = benchmark.get("methods", {}) if isinstance(benchmark, dict) else {}
    for method_name, method in methods.items():
        if not isinstance(method, dict):
            continue
        checkpoint = str(method.get("checkpoint_path", "")).strip()
        if not checkpoint:
            continue
        status = STATUS_EXTERNAL if _is_placeholder_path(checkpoint) else STATUS_PENDING
        if (root / checkpoint).exists():
            status = STATUS_OK
        _add_check(
            checks,
            "checkpoints",
            str(method_name),
            status,
            path=checkpoint,
            detail="train the configured method before claiming benchmark numbers",
        )


def _check_baselines(root: Path, checks: list[dict[str, Any]]) -> None:
    registry_path = root / "configs/baselines/core_vsr_baselines.yaml"
    if not registry_path.exists():
        _add_check(checks, "baselines", "core_vsr_baselines", STATUS_PENDING, path=str(registry_path))
        return
    records = baseline_records(load_baseline_registry(registry_path))
    for record in records:
        status = STATUS_OK if record.get("complete") else STATUS_EXTERNAL
        _add_check(
            checks,
            "baselines",
            str(record.get("id")),
            status,
            path=str(registry_path),
            detail=", ".join(record.get("missing_fields", [])) or "complete",
        )


def _check_metric_backends(checks: list[dict[str, Any]]) -> None:
    required = ["lpips", "dists", "niqe", "vmaf", "fvd", "musiq", "clipiqa"]
    status = metric_backend_status(required, requested_backend="official")
    for metric, info in status.items():
        check_status = STATUS_OK if info.get("official_available") else STATUS_EXTERNAL
        detail = f"official_backend={info.get('official_backend')}; value_kind={info.get('value_kind')}"
        _add_check(checks, "official_metrics", metric, check_status, detail=detail)


def _check_paper_docs(root: Path, checks: list[dict[str, Any]]) -> None:
    required = [
        "docs/paper/method_outline.md",
        "docs/paper/related_work_plan.md",
        "docs/paper/submission_checklist.md",
        "docs/paper/experiment_tables.md",
        "docs/experiments/formal_runbook.md",
    ]
    for relative in required:
        _add_path_check(checks, "paper_docs", relative, root / relative)


def _find_manifest_paths(value: Any) -> list[str]:
    paths = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in {"manifest_path", "manifest_output"} and isinstance(item, str):
                paths.append(item)
            else:
                paths.extend(_find_manifest_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_find_manifest_paths(item))
    elif isinstance(value, str) and "manifest_path=" in value:
        paths.append(value.split("manifest_path=", 1)[1].split(",", 1)[0])
    return paths


def _add_path_check(checks: list[dict[str, Any]], category: str, item: str, path: Path) -> None:
    status = STATUS_OK if path.exists() else STATUS_PENDING
    _add_check(checks, category, item, status, path=str(path))


def _add_check(
    checks: list[dict[str, Any]],
    category: str,
    item: str,
    status: str,
    *,
    path: str | None = None,
    detail: str = "",
) -> None:
    checks.append(
        {
            "category": category,
            "item": item,
            "status": status,
            "path": path,
            "detail": detail,
        }
    )


def _status_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts = {STATUS_OK: 0, STATUS_PENDING: 0, STATUS_EXTERNAL: 0}
    for check in checks:
        status = str(check.get("status", STATUS_PENDING))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _is_placeholder_path(path: str) -> bool:
    upper = str(path).upper()
    return any(token in upper for token in ["YOUR_", "TODO", "TBD", "FILL_ME"])


def _readiness_markdown(report: dict[str, Any]) -> str:
    checks = report.get("checks", [])
    counts = report.get("counts", {})
    lines = [
        "# CCF-A Readiness Audit",
        "",
        f"Ready for paper claims: {str(report.get('ready_for_paper_claims', False)).lower()}",
        "",
        f"- ok: {counts.get(STATUS_OK, 0)}",
        f"- pending: {counts.get(STATUS_PENDING, 0)}",
        f"- external_required: {counts.get(STATUS_EXTERNAL, 0)}",
        "",
        "| category | item | status | path | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for check in checks:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(check.get(key, ""))
                for key in ["category", "item", "status", "path", "detail"]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|")
