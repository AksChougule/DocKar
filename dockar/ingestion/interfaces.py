"""Interfaces for document ingestion."""

from pathlib import Path
from typing import Protocol

from dockar.models import Document, DocumentPage


class TextExtractor(Protocol):
    """Extracts embedded text from a source file."""

    def extract(self, path: Path) -> list[DocumentPage]:
        """Return page-level text extracted from a file."""
        ...


class DocumentIngestor(Protocol):
    """Loads source files into domain documents."""

    def load(self, path: Path) -> Document:
        """Load one file into a document."""
        ...

    def ingest(self, path: Path) -> list[Document]:
        """Ingest one file or directory path into documents."""
        ...
