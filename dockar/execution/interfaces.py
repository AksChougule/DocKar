"""Interfaces for running extraction prompts."""

from typing import Protocol

from dockar.models import Document, ExtractionResult, LLMRequestConfig, LLMResponse
from dockar.prompt_engine import PromptCandidate


class LLMClient(Protocol):
    """Provider-neutral LLM generation client."""

    def generate(self, prompt: str, config: LLMRequestConfig) -> LLMResponse:
        """Generate a completion for a prompt."""
        ...


class ExtractionExecutor(Protocol):
    """Runs a prompt against a document."""

    async def execute(self, document: Document, prompt: PromptCandidate) -> ExtractionResult:
        """Execute extraction asynchronously."""
        ...
