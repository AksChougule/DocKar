"""Interfaces for prompt generation and mutation."""

from typing import Any, Protocol

from pydantic import BaseModel, Field


class PromptCandidate(BaseModel):
    """A prompt under evaluation."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptEngine(Protocol):
    """Creates and refines prompt candidates."""

    def generate(self, task_description: str, schema: dict[str, Any]) -> list[PromptCandidate]:
        """Generate initial prompts."""
        ...

    def refine(self, candidates: list[PromptCandidate]) -> list[PromptCandidate]:
        """Generate improved prompts from previous candidates."""
        ...
