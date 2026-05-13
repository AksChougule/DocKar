"""Interfaces for running extraction prompts."""

from typing import Protocol

from dockar.models import Document, ExtractionResult
from dockar.prompt_engine import PromptCandidate


class ExtractionExecutor(Protocol):
    """Runs a prompt against a document."""

    async def execute(self, document: Document, prompt: PromptCandidate) -> ExtractionResult:
        """Execute extraction asynchronously."""
        ...
