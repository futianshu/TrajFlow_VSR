"""Paper-facing table export utilities for experiment comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RANKING_COLUMNS = [
    "rank",
    "name",
    "selection_score",
    "psnr",
    "ssim",
    "temporal_delta_error",
    "fps",
    "latency_seconds",
    "profile.parameters",
]


@dataclass(frozen=True)
class PaperTableExport:
    """Paths written by a paper table export."""

    output_dir: str
    files: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


def load_comparison(path: str | Path) -> dict[str, Any]:
    """Load a comparison object from either comparison.json or summary.json."""

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Comparison input must be a JSON object: {source}")
    comparison = data.get("comparison", data)
    if not isinstance(comparison, dict):
        raise ValueError(f"Comparison input has invalid 'comparison' payload: {source}")
    if not isinstance(comparison.get("rows"), list):
        raise ValueError(f"Comparison input must contain rows: {source}")
    return comparison


def export_paper_tables(
    comparison: dict[str, Any],
    output_dir: str | Path,
    *,
    name: str = "ablation",
    metrics: list[str] | None = None,
    matrix_separator: str = "_x_",
) -> PaperTableExport:
    """Write ranking and optional matrix tables as Markdown and LaTeX."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected_metrics = metrics or _default_metrics(comparison)
    files: dict[str, str] = {}

    ranking_rows = _ranking_rows(comparison)
    ranking_md = out / f"{name}_ranking.md"
    ranking_tex = out / f"{name}_ranking.tex"
    ranking_columns = _available_columns(ranking_rows, DEFAULT_RANKING_COLUMNS)
    ranking_md.write_text(_rows_to_markdown(ranking_rows, ranking_columns), encoding="utf-8")
    ranking_tex.write_text(_rows_to_latex(ranking_rows, ranking_columns, caption=f"{name} ranking"), encoding="utf-8")
    files["ranking_markdown"] = str(ranking_md)
    files["ranking_latex"] = str(ranking_tex)

    for metric in selected_metrics:
        matrix_rows, matrix_columns = _metric_matrix(comparison, metric, matrix_separator=matrix_separator)
        if not matrix_rows:
            continue
        metric_slug = _slug(metric)
        matrix_md = out / f"{name}_{metric_slug}_matrix.md"
        matrix_tex = out / f"{name}_{metric_slug}_matrix.tex"
        matrix_md.write_text(_rows_to_markdown(matrix_rows, matrix_columns), encoding="utf-8")
        matrix_tex.write_text(
            _rows_to_latex(matrix_rows, matrix_columns, caption=f"{name} {metric} matrix"),
            encoding="utf-8",
        )
        files[f"{metric_slug}_matrix_markdown"] = str(matrix_md)
        files[f"{metric_slug}_matrix_latex"] = str(matrix_tex)

    manifest = out / f"{name}_paper_tables.json"
    export = PaperTableExport(output_dir=str(out), files=files)
    manifest.write_text(export.to_json(), encoding="utf-8")
    files["manifest"] = str(manifest)
    return PaperTableExport(output_dir=str(out), files=files)


def _ranking_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows = comparison.get("rows", [])
    ranking = comparison.get("ranking", [])
    rank_by_name = {
        item.get("name"): {
            "rank": item.get("rank"),
            "selection_score": item.get("selection_score"),
        }
        for item in ranking
        if isinstance(item, dict)
    }
    merged = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rank = rank_by_name.get(row.get("name"), {})
        merged.append(
            {
                "rank": rank.get("rank", ""),
                **row,
                "selection_score": rank.get("selection_score", ""),
            }
        )
    return sorted(merged, key=lambda item: _rank_key(item.get("rank")))


def _metric_matrix(
    comparison: dict[str, Any],
    metric: str,
    *,
    matrix_separator: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    row_map: dict[str, dict[str, Any]] = {}
    columns: list[str] = ["variant"]
    for row in comparison.get("rows", []):
        if not isinstance(row, dict) or metric not in row:
            continue
        name = str(row.get("name", ""))
        if matrix_separator not in name:
            continue
        row_name, column_name = name.split(matrix_separator, 1)
        row_entry = row_map.setdefault(row_name, {"variant": row_name})
        row_entry[column_name] = row[metric]
        if column_name not in columns:
            columns.append(column_name)
    matrix_rows = [row_map[key] for key in sorted(row_map, key=_variant_sort_key)]
    return matrix_rows, columns


def _default_metrics(comparison: dict[str, Any]) -> list[str]:
    metrics = comparison.get("metrics")
    if isinstance(metrics, list) and metrics:
        return [str(metric) for metric in metrics]
    return ["psnr", "ssim", "temporal_delta_error", "fps"]


def _available_columns(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    return [column for column in preferred if any(column in row for row in rows)]


def _rows_to_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _rows_to_latex(rows: list[dict[str, Any]], columns: list[str], *, caption: str) -> str:
    alignment = "l" + "r" * max(len(columns) - 1, 0)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(_format_cell(row.get(column, ""))) for column in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _markdown_cell(value: Any) -> str:
    return _format_cell(value).replace("|", "\\|")


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _rank_key(value: Any) -> tuple[int, str]:
    if isinstance(value, int | float):
        return (int(value), "")
    try:
        return (int(str(value)), "")
    except ValueError:
        return (10**9, str(value))


def _variant_sort_key(value: str) -> tuple[str, int, str]:
    prefix, number = _split_trailing_number(value)
    return (prefix, number, value)


def _split_trailing_number(value: str) -> tuple[str, int]:
    digits = []
    for char in reversed(value):
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return (value, -1)
    number = int("".join(reversed(digits)))
    return (value[: -len(digits)], number)


def _slug(value: str) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join(part for part in "".join(chars).split("_") if part) or "metric"
