from dockar.config import EvaluationConfig, EvaluationWeightsConfig, FieldScoringConfig
from dockar.evaluation import Evaluator
from dockar.models import (
    ExtractedDocument,
    ExtractionResult,
    LLMRequestConfig,
    LLMResponse,
    LLMUsage,
)


def test_evaluator_scores_exact_and_field_level_matches() -> None:
    config = EvaluationConfig(
        weights=EvaluationWeightsConfig(accuracy=1.0, cost=0.0, latency=0.0),
        field_scoring=FieldScoringConfig(
            exact_match_fields=["invoice_id"],
            numeric_tolerance=0.5,
        ),
    )
    prediction = ExtractionResult(
        document_id="doc-1",
        output=ExtractedDocument(
            document_id="doc-1",
            data={
                "invoice_id": "INV-001",
                "total": 100.25,
                "vendor": "Acme Inc",
            },
        ),
        cost_usd=0.02,
        latency_ms=250,
    )

    report = Evaluator(config=config).evaluate_results(
        [prediction],
        {
            "doc-1": {
                "invoice_id": "INV-001",
                "total": 100.0,
                "vendor": "Acme Incorporated",
            }
        },
    )

    assert report.accuracy > 0.8
    assert report.score == report.accuracy
    assert report.cost == 0.02
    assert report.latency_ms == 250
    assert report.field_scores["invoice_id"] == 1.0
    assert report.field_scores["total"] == 1.0
    assert report.documents[0].fields[0].field == "invoice_id"


def test_evaluator_applies_configurable_weights_to_aggregate_score() -> None:
    config = EvaluationConfig(
        weights=EvaluationWeightsConfig(accuracy=0.5, cost=0.25, latency=0.25),
        field_scoring=FieldScoringConfig(),
    )
    prediction = ExtractionResult(
        document_id="doc-1",
        output=ExtractedDocument(document_id="doc-1", data={"field": "wrong"}),
        cost_usd=0.5,
        latency_ms=500,
    )

    report = Evaluator(
        config=config,
        max_cost_usd=1.0,
        max_latency_ms=1000,
    ).evaluate_results([prediction], {"doc-1": {"field": "right"}})

    assert report.accuracy < 1.0
    assert report.metadata["cost_score"] == 0.5
    assert report.metadata["latency_score"] == 0.5
    assert 0.0 <= report.score <= 1.0


class SemanticClient:
    def generate(self, prompt: str, config: LLMRequestConfig) -> LLMResponse:
        return LLMResponse(
            text='{"score": 0.9}',
            model=config.model,
            provider="test",
            usage=LLMUsage(total_tokens=10),
            latency_ms=10,
            cost_usd=0.001,
        )


def test_evaluator_supports_llm_semantic_scoring() -> None:
    config = EvaluationConfig(
        weights=EvaluationWeightsConfig(accuracy=1.0, cost=0.0, latency=0.0),
        field_scoring=FieldScoringConfig(semantic_fields=["description"]),
    )
    prediction = ExtractionResult(
        document_id="doc-1",
        output=ExtractedDocument(
            document_id="doc-1",
            data={"description": "Payment is due at the end of the month."},
        ),
    )

    report = Evaluator(
        config=config,
        semantic_client=SemanticClient(),
        semantic_config=LLMRequestConfig(model="semantic-model"),
    ).evaluate_results(
        [prediction],
        {"doc-1": {"description": "Invoice should be paid by month end."}},
    )

    field = report.documents[0].fields[0]
    assert field.rule == "semantic"
    assert field.score == 0.9
    assert field.metadata["semantic_provider"] == "test"


def test_evaluator_includes_error_results_in_breakdown() -> None:
    result = ExtractionResult(
        document_id="doc-err",
        error="model failed",
        errors=["timeout"],
        cost_usd=0.1,
        latency_ms=1000,
    )

    report = Evaluator().evaluate_results([result], {"doc-err": {"invoice_id": "INV-1"}})

    assert report.accuracy == 0.0
    assert report.documents[0].errors == ["timeout", "model failed"]
    assert report.documents[0].field_scores["invoice_id"] == 0.0


def test_evaluator_applies_per_field_weights() -> None:
    config = EvaluationConfig(
        weights=EvaluationWeightsConfig(accuracy=1.0, cost=0.0, latency=0.0),
        field_scoring=FieldScoringConfig(
            exact_match_fields=["important"],
            per_field_weights={"important": 3.0},
        ),
    )
    prediction = ExtractionResult(
        document_id="doc-1",
        output=ExtractedDocument(
            document_id="doc-1",
            data={"important": "wrong", "minor": "ok"},
        ),
    )

    report = Evaluator(config=config).evaluate_results(
        [prediction],
        {"doc-1": {"important": "right", "minor": "ok"}},
    )

    assert report.accuracy == 0.25
