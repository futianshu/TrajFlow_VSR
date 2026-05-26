"""Configuration loading helpers.

The project uses YAML-like files for experiment readability, but keeps this
loader dependency-free so the initial scaffold works before PyYAML is added.
The parser intentionally supports a small subset: nested dictionaries and
inline scalar/list values.
"""

from __future__ import annotations

import ast
import copy
import json
import tomllib
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a config file or override cannot be parsed."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON, TOML, or simple YAML config file."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(text)
    elif suffix == ".toml":
        data = tomllib.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = parse_simple_yaml(text)
    else:
        raise ConfigError(f"Unsupported config extension: {suffix}")

    if not isinstance(data, dict):
        raise ConfigError(f"Top-level config must be a mapping: {config_path}")
    return data


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a minimal YAML subset used by this repository's configs."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ConfigError(f"Indent must use multiples of two spaces at line {lineno}")

        stripped = line.strip()
        if ":" not in stripped:
            raise ConfigError(f"Expected 'key: value' at line {lineno}: {raw_line}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ConfigError(f"Missing key at line {lineno}")

        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            raise ConfigError(f"Invalid indentation at line {lineno}")

        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)

    return root


def parse_scalar(value: str) -> Any:
    """Parse a scalar value from the simple YAML subset."""

    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none", "~"}:
        return None

    if value.startswith("[") or value.startswith("{") or value.startswith(("'", '"')):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ConfigError(f"Invalid literal value: {value}") from exc

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Return a copy of config with dotted CLI overrides applied."""

    merged = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ConfigError(f"Override must look like key.path=value: {override}")
        path, raw_value = override.split("=", 1)
        set_by_dotted_path(merged, path.strip(), parse_scalar(raw_value.strip()))
    return merged


def set_by_dotted_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a nested mapping value using a dotted path."""

    if not dotted_path:
        raise ConfigError("Override key path cannot be empty")

    node: dict[str, Any] = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot override through non-mapping key: {part}")
        node = child
    node[parts[-1]] = value


def get_by_dotted_path(config: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    """Read a nested mapping value using a dotted path."""

    node: Any = config
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _strip_comment(line: str) -> str:
    """Remove comments while preserving hashes inside quoted strings."""

    in_single = False
    in_double = False
    escaped = False
    for idx, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line
