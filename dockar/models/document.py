"""Domain models used across the extraction pipeline."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """Page-level content extracted from a source document."""

    page_number: int = Field(ge=1)
    raw_text: str = ""
    extraction_method: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Input document with full and page-level text payloads."""

    id: str
    source_path: Path
    raw_text: str = ""
    pages: list[DocumentPage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """Backward-compatible alias for raw document text."""

        return self.raw_text


class Chunk(BaseModel):
    """Text chunk with provenance back to the source document."""

    chunk_id: str
    text: str
    document_id: str
    page_range: tuple[int, int] | None = None


class ExtractedDocument(BaseModel):
    """Structured output produced for a document."""

    document_id: str
    data: dict[str, Any]
    raw_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Execution result with observability metadata."""

    document_id: str
    chunk_id: str | None = None
    output: ExtractedDocument | None = None
    error: str | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    raw_outputs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
