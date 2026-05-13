"""YAML-backed configuration loading."""

from pathlib import Path
from typing import Any

import yaml

from dockar.config.models import DocKarConfig


def load_config(path: str | Path) -> DocKarConfig:
    """Load a DocKar configuration from a YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    if not isinstance(raw_config, dict):
        message = f"Config file must contain a YAML mapping: {config_path}"
        raise ValueError(message)

    return DocKarConfig.model_validate(_resolve_paths(raw_config, config_path.parent))


def _resolve_paths(raw_config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    execution = raw_config.get("execution")
    if not isinstance(execution, dict):
        return raw_config

    resolved = dict(raw_config)
    resolved_execution = dict(execution)
    for key in ("docs_path", "schema_path", "labels_path", "runs_path"):
        value = resolved_execution.get(key)
        if isinstance(value, str):
            path = Path(value)
            resolved_execution[key] = path if path.is_absolute() else base_dir / path

    resolved["execution"] = resolved_execution
    return resolved
