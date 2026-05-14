from pathlib import Path

import pytest

from dockar.config import ConfigError, ConfigLoader, load_config


def write_config(path: Path, extra: str = "") -> None:
    path.write_text(
        f"""
project_name: Example
model:
  default_model: gpt-4o-mini
  fallback_model: gpt-4o
  temperature: 0.2
  max_tokens: 2048
loop:
  max_iterations: 10
  budget_usd: 5.0
  early_stopping_rounds: 2
  candidates_per_iteration: 4
  top_k: 2
evaluation:
  weights:
    accuracy: 0.7
    cost: 0.2
    latency: 0.1
  field_scoring:
    enabled: true
    exact_match_fields: ["invoice_id"]
    semantic_fields: ["description"]
    numeric_tolerance: 0.01
    per_field_weights:
      invoice_id: 2.0
ingestion:
  ocr_enabled: true
  chunk_size: 3000
  max_doc_pages: 50
logging:
  log_level: debug
  output_dir: logs
{extra}
""",
        encoding="utf-8",
    )


def test_load_config_resolves_logging_output_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "dockar.yaml"
    write_config(config_path)

    config = load_config(config_path)

    assert config.project_name == "Example"
    assert config.model.default_model == "gpt-4o-mini"
    assert config.loop.budget_usd == 5.0
    assert config.logging.log_level == "DEBUG"
    assert config.logging.output_dir == tmp_path / "logs"


def test_environment_overrides_nested_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "dockar.yaml"
    write_config(config_path)

    monkeypatch.setenv("DOCKAR__MODEL__DEFAULT_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("DOCKAR__LOOP__TOP_K", "3")
    monkeypatch.setenv("DOCKAR__INGESTION__OCR_ENABLED", "false")

    config = ConfigLoader().load(config_path)

    assert config.model.default_model == "gpt-4.1-mini"
    assert config.loop.top_k == 3
    assert config.ingestion.ocr_enabled is False


def test_missing_required_sections_get_helpful_error(tmp_path: Path) -> None:
    config_path = tmp_path / "dockar.yaml"
    config_path.write_text("model:\n  default_model: gpt-4o-mini\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="missing required section"):
        load_config(config_path)


def test_invalid_config_gets_field_path_in_error(tmp_path: Path) -> None:
    config_path = tmp_path / "dockar.yaml"
    write_config(
        config_path,
        extra="""
""",
    )
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace("top_k: 2", "top_k: 9"), encoding="utf-8")

    with pytest.raises(ConfigError, match="loop"):
        load_config(config_path)
