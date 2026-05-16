from dockar.models import ExtractedDocument, LLMRequestConfig, LLMResponse, LLMUsage
from dockar.postprocessing import PostProcessor


def invoice_schema(additional_properties: bool = False) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": additional_properties,
        "properties": {
            "invoice_id": {"type": "string"},
            "total": {"type": "number"},
            "paid": {"type": "boolean"},
            "items": {"type": "array"},
        },
        "required": ["invoice_id", "total"],
    }


def test_postprocessor_cleans_types_strings_and_noise() -> None:
    extracted = ExtractedDocument(
        document_id="doc-1",
        data={
            " invoice_id ": "  INV-001\n",
            "total": "$1,234.50",
            "paid": "yes",
            "items": "widget",
            "explanation": "I found it",
        },
    )

    processed = PostProcessor().process(extracted, invoice_schema())

    assert processed.data == {
        "invoice_id": "INV-001",
        "total": 1234.5,
        "paid": True,
        "items": ["widget"],
    }
    fixes = processed.metadata["postprocessing"]["fixes"]
    assert any(fix["message"] == "Removed noise field" for fix in fixes)
    assert any(fix["message"] == "Coerced to number" for fix in fixes)
    assert processed.metadata["postprocessing"]["valid"] is True


def test_postprocessor_recovers_partially_broken_raw_json() -> None:
    extracted = ExtractedDocument(
        document_id="doc-2",
        data={},
        raw_output='Here is the result: {"invoice_id": "INV-002", "total": "42",}',
    )

    processed = PostProcessor().process(extracted, invoice_schema())

    assert processed.data["invoice_id"] == "INV-002"
    assert processed.data["total"] == 42.0
    assert processed.metadata["postprocessing"]["valid"] is True


def test_postprocessor_adds_missing_required_fields_and_logs_validation() -> None:
    extracted = ExtractedDocument(document_id="doc-3", data={"invoice_id": "INV-003"})

    processed = PostProcessor().process(extracted, invoice_schema())

    assert processed.data["total"] is None
    assert processed.metadata["postprocessing"]["valid"] is False
    assert "total: required field is missing" in processed.metadata["postprocessing"][
        "validation_errors"
    ]


def test_postprocessor_removes_extra_fields_when_schema_is_strict() -> None:
    extracted = ExtractedDocument(
        document_id="doc-4",
        data={"invoice_id": "INV-004", "total": 10, "extra": "drop me"},
    )

    processed = PostProcessor().process(extracted, invoice_schema(additional_properties=False))

    assert "extra" not in processed.data
    assert processed.metadata["postprocessing"]["valid"] is True


class RepairClient:
    def generate(self, prompt: str, config: LLMRequestConfig) -> LLMResponse:
        return LLMResponse(
            text='{"invoice_id": "INV-005", "total": 99.5}',
            model=config.model,
            provider="test",
            usage=LLMUsage(total_tokens=5),
            latency_ms=1.0,
        )


def test_postprocessor_can_use_llm_repair_for_invalid_output() -> None:
    extracted = ExtractedDocument(
        document_id="doc-5",
        data={},
        raw_output="invoice id INV-005 total 99.5",
    )
    processor = PostProcessor(
        repair_client=RepairClient(),
        repair_config=LLMRequestConfig(model="repair-model"),
    )

    processed = processor.process(extracted, invoice_schema())

    assert processed.data["invoice_id"] == "INV-005"
    assert processed.data["total"] == 99.5
    assert processed.metadata["postprocessing"]["valid"] is True
    assert any(
        fix["step"] == "llm_repair"
        for fix in processed.metadata["postprocessing"]["fixes"]
    )
