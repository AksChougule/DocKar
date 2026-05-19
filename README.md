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
  provider: openai
  default_model: gpt-4o-mini
  fallback_model: gpt-4o
  temperature: 0.0
  max_tokens: 4096
  timeout_seconds: 60
  max_retries: 2
  openai_api_key:
  openai_base_url: https://api.openai.com/v1
  ollama_base_url: http://localhost:11434

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

Generate with the LLM router:

```python
from dockar.config import ConfigLoader
from dockar.execution import RouterClient

config = ConfigLoader().load("config/default.yaml")
router = RouterClient.from_model_config(config.model)
response = router.generate("Extract invoice fields as JSON.")
print(response.text)
print(response.cost_usd, response.latency_ms)
```

For OpenAI, set `OPENAI_API_KEY` or `model.openai_api_key`. For Ollama, set
`model.provider: ollama` and point `model.ollama_base_url` at the local server.

Generate extraction prompt candidates:

```python
from dockar.prompt_engine import PromptGenerator

schema = {"type": "object", "properties": {"invoice_id": {"type": "string"}}}
examples = [{"invoice_id": "INV-001"}]

candidates = PromptGenerator(deterministic=True).generate(
    task_description="Extract invoice fields.",
    schema=schema,
    examples=examples,
    candidate_count=4,
)
print(candidates[0].text)
```

Execute extraction with an LLM client:

```python
from dockar.execution import Executor, RouterClient

router = RouterClient.from_model_config(config.model)
result = Executor(router).execute_document(document, candidates[0])

print(result.output.data if result.output else result.error)
print(result.raw_outputs)
print(result.metadata)
```

Post-process an extraction:

```python
from dockar.postprocessing import PostProcessor

processed = PostProcessor().process(result.output, schema)
print(processed.data)
print(processed.metadata["postprocessing"]["fixes"])
```

Evaluate extraction quality:

```python
from dockar.evaluation import Evaluator

expected = {"invoice_id": "INV-001"}
report = Evaluator(config=config.evaluation).evaluate(processed, expected)

print(report.accuracy)
print(report.cost, report.latency_ms)
print(report.documents[0].field_scores)
```

## Development

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy dockar cli
```
