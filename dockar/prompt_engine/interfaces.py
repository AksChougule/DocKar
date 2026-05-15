"""Interfaces and models for prompt generation and mutation."""

from typing import Any, Protocol

from pydantic import BaseModel, Field


class PromptCandidate(BaseModel):
    """A prompt under evaluation."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptInputs(BaseModel):
    """Inputs used to build extraction prompts."""

    task_description: str = Field(min_length=1)
    output_schema: dict[str, Any] = Field(alias="schema")
    examples: list[dict[str, Any]] = Field(default_factory=list, max_length=3)


class PromptEngine(Protocol):
    """Creates and refines prompt candidates."""

    def generate(
        self,
        task_description: str,
        schema: dict[str, Any],
        examples: list[dict[str, Any]] | None = None,
        candidate_count: int = 1,
    ) -> list[PromptCandidate]:
        """Generate initial prompts."""
        ...

    def refine(self, candidates: list[PromptCandidate]) -> list[PromptCandidate]:
        """Generate improved prompts from previous candidates."""
        ...
