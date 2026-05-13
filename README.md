# DocKar: Self-Improving Document Extraction System

*(DOCument extraction with KARpathy-style Training Loop)*

DocKar is a production-oriented Python 3.12 toolkit for document extraction experiments.
It is designed as a CLI and SDK, with no UI layer, and follows clean architecture so each
pipeline capability can be implemented, tested, and swapped independently.

## Goals

- Learn effective extraction prompts from a small labeled dataset.
- Support PDF ingestion, OCR, chunking, prompt generation, execution, evaluation, and
  post-processing through explicit interfaces.
- Optimize for accuracy, cost, and latency in a closed-loop workflow.
- Keep runtime side effects out of imports so the SDK is safe to embed.

## Project Layout

```text
dockar/
  config/          YAML loading and Pydantic configuration models
  ingestion/       document loading interfaces
  ocr/             OCR interfaces
  chunking/        text chunking interfaces
  prompt_engine/   prompt candidate interfaces
  execution/       extraction execution interfaces
  evaluation/      scoring interfaces
  loop/            optimization loop interfaces
  postprocessing/  cleanup and validation interfaces
  logging/         structured logging setup
  models/          shared domain models
  utils/           common helpers
cli/               Typer CLI entrypoints
tests/             automated tests
```

## Installation

```bash
poetry install
```

## Configuration

DocKar configuration is YAML-backed and validated with Pydantic.

```yaml
project_name: DocKar
model:
  provider: openai
  name: gpt-4o-mini
  timeout_seconds: 60
  max_retries: 2
budget:
  max_cost_usd: 10
  max_iterations: 20
  early_stopping_rounds: 3
execution:
  docs_path: ./docs
  schema_path: ./schema.json
  labels_path: ./labels.json
  runs_path: ./runs
  chunk_size: 4000
logging:
  level: INFO
  json: true
```

Relative paths under `execution` are resolved relative to the YAML file.

## CLI

Validate a configuration file:

```bash
poetry run dockar config-check ./dockar.yaml
```

Prepare a run and print the effective configuration:

```bash
poetry run dockar run \
  --docs ./docs \
  --schema ./schema.json \
  --labels ./labels.json \
  --budget 10 \
  --iterations 20
```

## SDK

```python
from dockar import load_config

config = load_config("dockar.yaml")
```

## Development

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy dockar cli
```
