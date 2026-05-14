"""YAML-backed configuration loading with environment overrides."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from dockar.config.models import ConfigMapping, DocKarConfig


class ConfigError(ValueError):
    """Raised when configuration cannot be loaded or validated."""


class ConfigLoader:
    """Load DocKar configuration from YAML and environment variables.

    Environment overrides use ``DOCKAR`` by default and ``__`` as the nested delimiter.
    For example: ``DOCKAR__MODEL__DEFAULT_MODEL=gpt-4o``.
    """

    def __init__(self, env_prefix: str = "DOCKAR", env_delimiter: str = "__") -> None:
        self.env_prefix = env_prefix
        self.env_delimiter = env_delimiter

    def load(self, path: str | Path) -> DocKarConfig:
        """Load and validate a YAML configuration file."""

        config_path = Path(path)
        raw_config = self._read_yaml(config_path)
        self._validate_required_sections(raw_config, config_path)

        merged_config = self._deep_merge(
            raw_config,
            self._environment_overrides(os.environ),
        )
        resolved_config = self._resolve_paths(merged_config, config_path.parent)
        return self._validate(resolved_config, config_path)

    def _read_yaml(self, path: Path) -> ConfigMapping:
        if not path.exists():
            message = f"Config file does not exist: {path}"
            raise ConfigError(message)
        if not path.is_file():
            message = f"Config path must be a file: {path}"
            raise ConfigError(message)

        try:
            with path.open("r", encoding="utf-8") as config_file:
                raw_config = yaml.safe_load(config_file) or {}
        except yaml.YAMLError as exc:
            message = f"Could not parse YAML config {path}: {exc}"
            raise ConfigError(message) from exc
        except OSError as exc:
            message = f"Could not read config file {path}: {exc}"
            raise ConfigError(message) from exc

        if not isinstance(raw_config, dict):
            message = f"Config file must contain a YAML mapping at the top level: {path}"
            raise ConfigError(message)
        return raw_config

    def _validate_required_sections(self, raw_config: ConfigMapping, path: Path) -> None:
        missing = sorted(DocKarConfig.required_sections() - raw_config.keys())
        if missing:
            missing_sections = ", ".join(missing)
            message = f"Config file {path} is missing required section(s): {missing_sections}"
            raise ConfigError(message)

    def _environment_overrides(self, environ: Mapping[str, str]) -> ConfigMapping:
        overrides: ConfigMapping = {}
        prefix = f"{self.env_prefix}{self.env_delimiter}"

        for key, value in environ.items():
            if not key.startswith(prefix):
                continue

            path = [part.lower() for part in key[len(prefix) :].split(self.env_delimiter)]
            if not path or any(not part for part in path):
                continue

            cursor: ConfigMapping = overrides
            for part in path[:-1]:
                nested = cursor.setdefault(part, {})
                if not isinstance(nested, dict):
                    message = f"Environment override {key} conflicts with another override"
                    raise ConfigError(message)
                cursor = nested
            cursor[path[-1]] = self._parse_env_value(value)

        return overrides

    def _parse_env_value(self, value: str) -> Any:
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError:
            return value

    def _deep_merge(self, base: ConfigMapping, overrides: ConfigMapping) -> ConfigMapping:
        merged = dict(base)
        for key, value in overrides.items():
            existing = merged.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(existing, value)
            else:
                merged[key] = value
        return merged

    def _resolve_paths(self, raw_config: ConfigMapping, base_dir: Path) -> ConfigMapping:
        resolved = dict(raw_config)
        logging_config = resolved.get("logging")
        if isinstance(logging_config, dict):
            resolved_logging = dict(logging_config)
            output_dir = resolved_logging.get("output_dir")
            if isinstance(output_dir, str):
                path = Path(output_dir)
                resolved_logging["output_dir"] = path if path.is_absolute() else base_dir / path
            resolved["logging"] = resolved_logging
        return resolved

    def _validate(self, raw_config: ConfigMapping, path: Path) -> DocKarConfig:
        try:
            return DocKarConfig.model_validate(raw_config)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            message = f"Invalid DocKar config {path}: {details}"
            raise ConfigError(message) from exc


def load_config(path: str | Path) -> DocKarConfig:
    """Load a DocKar configuration from a YAML file."""

    return ConfigLoader().load(path)
