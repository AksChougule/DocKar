"""Domain models used across the extraction pipeline."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Input document metadata and text payload."""

    id: str
    source_path: Path
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedDocument(BaseModel):
    """Structured output produced for a document."""

    document_id: str
    data: dict[str, Any]
    raw_output: str | None = None


class ExtractionResult(BaseModel):
    """Execution result with observability metadata."""

    document_id: str
    output: ExtractedDocument | None = None
    error: str | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
