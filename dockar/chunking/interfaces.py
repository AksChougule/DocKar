"""Interfaces for chunking document text."""

from typing import Protocol

from dockar.models import Document


class Chunker(Protocol):
    """Splits documents into model-sized chunks."""

    def chunk(self, document: Document) -> list[str]:
        """Return ordered chunks for one document."""
        ...
