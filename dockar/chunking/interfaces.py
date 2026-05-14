"""Interfaces for chunking document text."""

from typing import Protocol

from dockar.models import Chunk, Document


class ChunkingStrategy(Protocol):
    """Strategy interface for splitting documents into chunks."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Return ordered chunks for one document."""
        ...
