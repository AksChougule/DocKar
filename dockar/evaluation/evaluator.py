"""Evaluation engine for extraction results."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from dockar.config import EvaluationConfig
from dockar.evaluation.interfaces import DocumentEvaluation, EvaluationReport, FieldScore
from dockar.execution.interfaces import LLMClient
from dockar.execution.llm import LLMError
from dockar.models import ExtractedDocument, ExtractionResult, LLMRequestConfig


class Evaluator:
    """Evaluate extraction accuracy, cost, and latency with per-field breakdowns."""

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        semantic_client: LLMClient | None = None,
        semantic_config: LLMRequestConfig | None = None,
        max_cost_usd: float = 1.0,
        max_latency_ms: float = 60_000.0,
    ) -> None:
        self.config = config or EvaluationConfig()
        self.semantic_client = semantic_client
        self.semantic_config = semantic_config
        self.max_cost_usd = max_cost_usd
        self.max_latency_ms = max_latency_ms

    def evaluate(
        self,
        prediction: ExtractedDocument,
        expected: dict[str, Any],
    ) -> EvaluationReport:
        """Evaluate one extracted document."""

        result = ExtractionResult(
            document_id=prediction.document_id,
            output=prediction,
            cost_usd=float(prediction.metadata.get("cost_usd", 0) or 0),
            latency_ms=float(prediction.metadata.get("latency_ms", 0) or 0),
        )
        return self.evaluate_results([result], {prediction.document_id: expected})

    def evaluate_results(
        self,
        predictions: list[ExtractionResult],
        expected_by_document: dict[str, dict[str, Any]],
    ) -> EvaluationReport:
        """Evaluate multiple extraction results."""

        document_reports = [
            self._evaluate_document(result, expected_by_document.get(result.document_id, {}))
            for result in predictions
        ]
        accuracy = self._average([report.accuracy for report in document_reports])
        cost = sum(report.cost for report in document_reports)
        latency_ms = self._average([report.latency_ms for report in document_reports])
        field_scores = self._aggregate_field_scores(document_reports)
        score = self._weighted_score(accuracy, cost, latency_ms)

        return EvaluationReport(
            score=score,
            accuracy=accuracy,
            cost=cost,
            latency_ms=latency_ms,
            field_scores=field_scores,
            documents=document_reports,
            metadata={
                "document_count": len(document_reports),
                "weights": self.config.weights.model_dump(),
                "cost_score": self._cost_score(cost),
                "latency_score": self._latency_score(latency_ms),
            },
        )

    def _evaluate_document(
        self,
        result: ExtractionResult,
        expected: dict[str, Any],
    ) -> DocumentEvaluation:
        errors = list(result.errors)
        if result.error:
            errors.append(result.error)
        predicted = result.output.data if result.output is not None else {}
        fields = sorted(set(expected) | set(predicted))
        field_details = [
            self._score_field(field, predicted.get(field), expected.get(field))
            for field in fields
        ]
        field_scores = {detail.field: detail.score for detail in field_details}
        accuracy = self._weighted_field_accuracy(field_details)

        return DocumentEvaluation(
            document_id=result.document_id,
            accuracy=accuracy,
            cost=float(result.cost_usd or 0),
            latency_ms=float(result.latency_ms or 0),
            field_scores=field_scores,
            fields=field_details,
            errors=errors,
            metadata={
                "chunk_id": result.chunk_id,
                "source_metadata": result.metadata,
            },
        )

    def _score_field(self, field: str, predicted: Any, expected: Any) -> FieldScore:
        if field in self.config.field_scoring.semantic_fields:
            score, metadata = self._semantic_score(predicted, expected)
            return FieldScore(
                field=field,
                score=score,
                rule="semantic",
                expected=expected,
                predicted=predicted,
                metadata=metadata,
            )

        if field in self.config.field_scoring.exact_match_fields:
            score = (
                1.0
                if self._normalize_exact(predicted) == self._normalize_exact(expected)
                else 0.0
            )
            return FieldScore(
                field=field,
                score=score,
                rule="exact",
                expected=expected,
                predicted=predicted,
            )

        score = self._field_level_score(predicted, expected)
        return FieldScore(
            field=field,
            score=score,
            rule="field_level",
            expected=expected,
            predicted=predicted,
        )

    def _field_level_score(self, predicted: Any, expected: Any) -> float:
        if predicted == expected:
            return 1.0
        if predicted is None or expected is None:
            return 0.0
        if self._both_numbers(predicted, expected):
            tolerance = self.config.field_scoring.numeric_tolerance
            predicted_number = float(predicted)
            expected_number = float(expected)
            if abs(predicted_number - expected_number) <= tolerance:
                return 1.0
            denominator = max(abs(expected_number), 1.0)
            return max(0.0, 1.0 - abs(predicted_number - expected_number) / denominator)
        if isinstance(predicted, str) and isinstance(expected, str):
            return SequenceMatcher(
                None,
                self._normalize_string(predicted),
                self._normalize_string(expected),
            ).ratio()
        if isinstance(predicted, list) and isinstance(expected, list):
            return self._list_score(predicted, expected)
        if isinstance(predicted, dict) and isinstance(expected, dict):
            return self._dict_score(predicted, expected)
        return 0.0

    def _semantic_score(self, predicted: Any, expected: Any) -> tuple[float, dict[str, Any]]:
        fallback = self._field_level_score(predicted, expected)
        if self.semantic_client is None or self.semantic_config is None:
            return fallback, {"semantic_provider": "fallback_similarity"}

        prompt = (
            "Score whether the predicted value semantically matches the expected value. "
            "Return JSON only with a numeric score from 0 to 1 under key score.\n\n"
            f"Expected:\n{json.dumps(expected, ensure_ascii=False)}\n\n"
            f"Predicted:\n{json.dumps(predicted, ensure_ascii=False)}"
        )
        try:
            response = self.semantic_client.generate(prompt, self.semantic_config)
            score = self._parse_semantic_score(response.text)
            return score, {
                "semantic_provider": response.provider,
                "semantic_model": response.model,
                "semantic_cost_usd": response.cost_usd,
                "semantic_latency_ms": response.latency_ms,
            }
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            return fallback, {
                "semantic_provider": "fallback_similarity",
                "semantic_error": str(exc),
            }

    def _parse_semantic_score(self, text: str) -> float:
        cleaned = self._strip_markdown_fence(text.strip())
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict) or "score" not in parsed:
            message = "semantic scorer response must contain score"
            raise ValueError(message)
        return min(1.0, max(0.0, float(parsed["score"])))

    def _weighted_field_accuracy(self, fields: list[FieldScore]) -> float:
        if not fields:
            return 0.0
        weights = self.config.field_scoring.per_field_weights
        total_weight = sum(weights.get(field.field, 1.0) for field in fields)
        if total_weight <= 0:
            return 0.0
        return sum(field.score * weights.get(field.field, 1.0) for field in fields) / total_weight

    def _weighted_score(self, accuracy: float, cost: float, latency_ms: float) -> float:
        weights = self.config.weights
        total_weight = weights.accuracy + weights.cost + weights.latency
        return (
            accuracy * weights.accuracy
            + self._cost_score(cost) * weights.cost
            + self._latency_score(latency_ms) * weights.latency
        ) / total_weight

    def _cost_score(self, cost: float) -> float:
        if self.max_cost_usd <= 0:
            return 1.0
        return max(0.0, 1.0 - cost / self.max_cost_usd)

    def _latency_score(self, latency_ms: float) -> float:
        if self.max_latency_ms <= 0:
            return 1.0
        return max(0.0, 1.0 - latency_ms / self.max_latency_ms)

    def _aggregate_field_scores(
        self,
        documents: list[DocumentEvaluation],
    ) -> dict[str, float]:
        scores: dict[str, list[float]] = {}
        for document in documents:
            for field, score in document.field_scores.items():
                scores.setdefault(field, []).append(score)
        return {field: self._average(values) for field, values in scores.items()}

    def _average(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _normalize_exact(self, value: Any) -> Any:
        return self._normalize_string(value) if isinstance(value, str) else value

    def _normalize_string(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().lower()

    def _both_numbers(self, left: Any, right: Any) -> bool:
        return isinstance(left, int | float) and isinstance(right, int | float)

    def _list_score(self, predicted: list[Any], expected: list[Any]) -> float:
        if not predicted and not expected:
            return 1.0
        if not predicted or not expected:
            return 0.0
        matched = sum(1 for item in predicted if item in expected)
        return matched / max(len(predicted), len(expected))

    def _dict_score(self, predicted: dict[str, Any], expected: dict[str, Any]) -> float:
        fields = sorted(set(predicted) | set(expected))
        if not fields:
            return 1.0
        return self._average([
            self._field_level_score(predicted.get(field), expected.get(field))
            for field in fields
        ])

    def _strip_markdown_fence(self, text: str) -> str:
        fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if fence_match is None:
            return text
        return fence_match.group(1).strip()
