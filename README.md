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

DocKar configuration is YAML-backed, validated with Pydantic, and supports
environment overrides. See [config/default.yaml](/home/ubuntu/codebase/my_github/DocKar/config/default.yaml)
for a complete example.

```yaml
project_name: DocKar

model:
  default_model: gpt-4o-mini
  fallback_model: gpt-4o
  temperature: 0.0
  max_tokens: 4096

loop:
  max_iterations: 20
  budget_usd: 10.0
  early_stopping_rounds: 3
  candidates_per_iteration: 4
  top_k: 2

evaluation:
  weights:
    accuracy: 0.8
    cost: 0.1
    latency: 0.1
  field_scoring:
    enabled: true
    exact_match_fields: []
    semantic_fields: []
    numeric_tolerance: 0.0
    per_field_weights: {}

ingestion:
  ocr_enabled: true
  chunk_size: 4000
  max_doc_pages: 100

logging:
  log_level: INFO
  output_dir: ./runs/logs
```

Relative `logging.output_dir` paths are resolved relative to the YAML file.

Environment overrides use the `DOCKAR__` prefix and `__` nested delimiter:

```bash
DOCKAR__MODEL__DEFAULT_MODEL=gpt-4o poetry run dockar config-check config/default.yaml
DOCKAR__LOOP__BUDGET_USD=25 poetry run dockar config-check config/default.yaml
DOCKAR__INGESTION__OCR_ENABLED=false poetry run dockar config-check config/default.yaml
```

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
from dockar.config import ConfigLoader

config = ConfigLoader().load("config/default.yaml")
```

Load a PDF document:

```python
from pathlib import Path

from dockar.ingestion import DocumentLoader

document = DocumentLoader().load(Path("examples/invoice.pdf"))
print(document.raw_text)
print(document.pages[0].extraction_method)
```

PDF ingestion uses embedded text first and falls back to Tesseract OCR when page text
density is low. The Python OCR package is installed through Poetry; the `tesseract`
system binary must also be available on the machine for OCR fallback to run.

Chunk a loaded document:

```python
from dockar.chunking import Chunker

chunks = Chunker(chunk_size=4000).chunk(document)
print(chunks[0].chunk_id)
print(chunks[0].page_range)
```

## Development

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy dockar cli
```
