"""Interfaces for OCR engines."""

from typing import Protocol

from dockar.models import DocumentPage


class OcrEngine(Protocol):
    """Extracts text from scanned documents."""

    def extract_pages(self, pages: list[DocumentPage]) -> list[DocumentPage]:
        """Return OCR-populated pages."""
        ...
