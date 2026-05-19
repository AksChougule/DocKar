"""Interfaces and report models for scoring extraction quality."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from dockar.models import ExtractedDocument, ExtractionResult


class FieldScore(BaseModel):
    """Score details for one extracted field."""

    field: str
    score: float = Field(ge=0, le=1)
    rule: str
    expected: Any = None
    predicted: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentEvaluation(BaseModel):
    """Evaluation breakdown for one document."""

    document_id: str
    accuracy: float = Field(ge=0, le=1)
    cost: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    field_scores: dict[str, float] = Field(default_factory=dict)
    fields: list[FieldScore] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Aggregate and field-level scoring results."""

    score: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    cost: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    field_scores: dict[str, float] = Field(default_factory=dict)
    documents: list[DocumentEvaluation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationEngine(Protocol):
    """Scores an extraction against a gold label."""

    def evaluate(self, prediction: ExtractedDocument, expected: dict[str, Any]) -> EvaluationReport:
        """Evaluate one prediction."""
        ...

    def evaluate_results(
        self,
        predictions: list[ExtractionResult],
        expected_by_document: dict[str, dict[str, Any]],
    ) -> EvaluationReport:
        """Evaluate multiple extraction results."""
        ...
