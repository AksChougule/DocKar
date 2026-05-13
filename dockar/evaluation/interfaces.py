"""Interfaces for scoring extraction quality."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from dockar.models import ExtractedDocument


class EvaluationReport(BaseModel):
    """Aggregate and field-level scoring results."""

    score: float = Field(ge=0, le=1)
    field_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evaluator(Protocol):
    """Scores an extraction against a gold label."""

    def evaluate(self, prediction: ExtractedDocument, expected: dict[str, Any]) -> EvaluationReport:
        """Evaluate one prediction."""
        ...
