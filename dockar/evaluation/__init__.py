"""Evaluation boundary."""

from dockar.evaluation.evaluator import Evaluator
from dockar.evaluation.interfaces import (
    DocumentEvaluation,
    EvaluationEngine,
    EvaluationReport,
    FieldScore,
)

__all__ = [
    "DocumentEvaluation",
    "EvaluationEngine",
    "EvaluationReport",
    "Evaluator",
    "FieldScore",
]
