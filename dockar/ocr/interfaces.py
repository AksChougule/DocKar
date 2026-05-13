"""Interfaces for OCR engines."""

from typing import Protocol

from dockar.models import Document


class OcrEngine(Protocol):
    """Extracts text from scanned documents."""

    def extract_text(self, document: Document) -> Document:
        """Return a document with text populated."""
        ...
