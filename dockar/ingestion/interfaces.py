"""Interfaces for document ingestion."""

from pathlib import Path
from typing import Protocol

from dockar.models import Document


class DocumentIngestor(Protocol):
    """Loads source files into domain documents."""

    def ingest(self, path: Path) -> list[Document]:
        """Ingest one path into documents."""
        ...
