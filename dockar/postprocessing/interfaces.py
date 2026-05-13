"""Interfaces for cleanup and schema repair."""

from typing import Any, Protocol

from dockar.models import ExtractedDocument


class Postprocessor(Protocol):
    """Cleans and validates extraction outputs."""

    def process(self, extracted: ExtractedDocument, schema: dict[str, Any]) -> ExtractedDocument:
        """Return a cleaned extraction."""
        ...
