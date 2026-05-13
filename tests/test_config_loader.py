from pathlib import Path

from dockar.config import load_config


def test_load_config_resolves_execution_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "dockar.yaml"
    config_path.write_text(
        """
project_name: Example
execution:
  docs_path: docs
  schema_path: schema.json
  labels_path: labels.json
  runs_path: output/runs
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project_name == "Example"
    assert config.execution.docs_path == tmp_path / "docs"
    assert config.execution.schema_path == tmp_path / "schema.json"
    assert config.execution.labels_path == tmp_path / "labels.json"
    assert config.execution.runs_path == tmp_path / "output/runs"


def test_load_config_allows_empty_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "dockar.yaml"
    config_path.write_text("", encoding="utf-8")

    config = load_config(config_path)

    assert config.project_name == "DocKar"
