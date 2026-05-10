# DocKar: Self-Improving Document Extraction System

*(DOCument extraction with KARpathy-style Training Loop)*

---

## 1. Vision

Build a **general-purpose document extraction system** that:

* Learns from **5–10 labeled examples**
* Automatically **generates and improves prompts**
* Optimizes for **user-defined tradeoffs (accuracy, cost, latency)**
* Iteratively improves using a **closed-loop evaluation system**
* Outputs a **best prompt + metrics report**

This is essentially:

> “Prompt training without fine-tuning”

---

## 2. Phase 1 Scope (MVP)

### Supported Inputs

* PDFs:

  * Text-based
  * Scanned (OCR required)

### Supported Extraction Types

* Key-value pairs
* Nested JSON
* Tables (via parser + LLM fallback)

### Schema Definition

User can provide:

* JSON schema
* Example output
* Natural language → system generates schema

---

## 3. System Architecture

### Core Modules

```
/core
  /ingestion
  /ocr
  /chunking
  /prompt_engine
  /execution
  /evaluation
  /loop
  /postprocessing
  /logging
```

---

## 4. End-to-End Workflow

### Step 1: Input

User provides:

* Documents (PDFs)
* Target schema
* 5–10 labeled examples (golden dataset)

---

### Step 2: Preprocessing

#### OCR Strategy (Hybrid)

* Use Tesseract OCR for scanned PDFs
* Store:

  * Raw text
  * Optional page images (for fallback)

#### Table Extraction

* Try parser first:

  * Camelot / Tabula
* Fallback → LLM

#### Chunking

* Basic chunking (size-based)
* Merge outputs later

---

### Step 3: Initial Prompt Generation

Input:

* Task description
* Schema
* 3 example outputs (fixed subset)

Output:

* Base prompt

---

### Step 4: Extraction Execution

For each document:

* Run extraction
* Handle failures:

  * Retry
  * Prompt variation
  * Escalate model

---

### Step 5: Post-processing

Pipeline:

1. Rule-based cleanup
2. LLM repair (if needed)
3. Schema validation (semi-strict)

---

### Step 6: Evaluation Engine

#### Multi-metric scoring

| Type        | Method                 |
| ----------- | ---------------------- |
| Exact match | For categorical fields |
| Field-level | Partial credit         |
| Semantic    | For long text          |

---

### Step 7: Training Loop (Core Innovation)

#### Strategy: Beam + Refinement

Each iteration:

1. Generate multiple candidate prompts
2. Run on full dataset
3. Score each prompt
4. Keep best-performing prompts
5. Generate next iteration from best

---

### Loop Stops When:

* Budget reached
* Max iterations reached
* No improvement (early stopping)

---

## 5. Prompt Optimization Strategy

### Combined Approach (Your Choice)

* Iterative refinement (A)
* Multi-candidate generation (B)

#### Prompt Mutation Examples

* Change instructions
* Add/remove examples
* Rephrase schema guidance
* Add constraints

---

## 6. Observability (Full Trace)

Per run:

* Prompt
* Model used
* Cost
* Latency
* Output
* Errors

Per document:

* Field-level scores
* Failures

Stored as:

```
/runs/{run_id}/
  prompts.json
  results.json
  metrics.json
  logs.json
```

---

## 7. Cost + Control System

### Constraints

* Budget cap
* Max iterations
* Cost-aware scoring

### Model Strategy

* Default: cheap model
* Escalate only on failure

---

## 8. Output

Final artifact:

```
{
  "best_prompt": "...",
  "metrics": {
    "accuracy": ...,
    "cost": ...,
    "latency": ...
  }
}
```

---

## 9. Interfaces

### CLI

```
extractor run \
  --docs ./docs \
  --schema schema.json \
  --labels labels.json \
  --budget 10 \
  --iterations 20
```

---

### Python SDK

```python
from extractor import Runner

runner = Runner(config)

result = runner.run(
    docs=docs,
    schema=schema,
    labels=labels
)
```

---

## 10. Domain Packs (Phase 1)

Predefined configs:

* Invoice
* Clinical document (important for your domain)
* Generic structured form

---

## 11. Explicit Non-Goals (Phase 1)

* UI
* Large documents (>10 pages)
* Parallelization
* Advanced layout parsing
* Fine-tuning models

---