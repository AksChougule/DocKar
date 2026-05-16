from pathlib import Path

from dockar.execution import Executor
from dockar.models import Chunk, Document, LLMRequestConfig, LLMResponse, LLMUsage
from dockar.prompt_engine import PromptCandidate


class FakeLLMClient:
    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, prompt: str, config: LLMRequestConfig) -> LLMResponse:
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def llm_response(
    text: str,
    latency_ms: float = 5.0,
    cost_usd: float = 0.01,
    fallback_used: bool = False,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="test-model",
        provider="test",
        usage=LLMUsage(total_tokens=10),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        fallback_used=fallback_used,
    )


def prompt() -> PromptCandidate:
    return PromptCandidate(id="prompt-1", text="Extract JSON.\n\n{{document_text}}")


def test_executor_runs_prompt_for_document_and_returns_structured_output() -> None:
    client = FakeLLMClient([llm_response('{"invoice_id": "INV-001"}')])
    executor = Executor(client, LLMRequestConfig(model="test-model"))
    document = Document(
        id="doc-1",
        source_path=Path("invoice.pdf"),
        raw_text="Invoice INV-001",
    )

    result = executor.execute_document(document, prompt())

    assert result.output is not None
    assert result.output.data == {"invoice_id": "INV-001"}
    assert result.raw_outputs == ['{"invoice_id": "INV-001"}']
    assert result.error is None
    assert result.cost_usd == 0.01
    assert result.metadata["attempts"] == 1
    assert "Invoice INV-001" in client.prompts[0]


def test_executor_retries_after_invalid_json() -> None:
    client = FakeLLMClient(
        [
            llm_response("not json"),
            llm_response('{"invoice_id": "INV-002"}'),
        ]
    )
    executor = Executor(client, LLMRequestConfig(model="test-model"), max_retries=1)
    document = Document(id="doc-2", source_path=Path("invoice.pdf"), raw_text="Invoice INV-002")

    result = executor.execute_document(document, prompt())

    assert result.output is not None
    assert result.output.data["invoice_id"] == "INV-002"
    assert result.raw_outputs == ["not json", '{"invoice_id": "INV-002"}']
    assert len(result.errors) == 1
    assert result.metadata["attempts"] == 2


def test_executor_uses_prompt_variation_after_retry_failure() -> None:
    client = FakeLLMClient(
        [
            llm_response("not json"),
            llm_response("still not json"),
            llm_response('{"invoice_id": "INV-003"}'),
        ]
    )
    executor = Executor(client, LLMRequestConfig(model="test-model"), max_retries=1)
    document = Document(id="doc-3", source_path=Path("invoice.pdf"), raw_text="Invoice INV-003")

    result = executor.execute_document(document, prompt())

    assert result.output is not None
    assert result.output.data["invoice_id"] == "INV-003"
    assert "## Retry Constraint" in client.prompts[-1]
    assert result.metadata["prompt_variation_used"] is True
    assert result.metadata["attempts"] == 3


def test_executor_preserves_chunk_metadata() -> None:
    client = FakeLLMClient([llm_response('{"total": 12.5}', fallback_used=True)])
    executor = Executor(client, LLMRequestConfig(model="test-model"))
    chunk = Chunk(
        chunk_id="doc-4::chunk-0001",
        document_id="doc-4",
        text="Total 12.50",
        page_range=(2, 3),
    )

    result = executor.execute_chunk(chunk, prompt())

    assert result.document_id == "doc-4"
    assert result.chunk_id == "doc-4::chunk-0001"
    assert result.output is not None
    assert result.output.data == {"total": 12.5}
    assert result.metadata["source_type"] == "chunk"
    assert result.metadata["page_range"] == (2, 3)
    assert result.metadata["fallback_used"] is True


def test_executor_returns_error_result_when_all_attempts_fail() -> None:
    client = FakeLLMClient([llm_response("nope"), llm_response("still nope")])
    executor = Executor(client, LLMRequestConfig(model="test-model"), max_retries=0)
    document = Document(id="doc-5", source_path=Path("invoice.pdf"), raw_text="Invoice")

    result = executor.execute_document(document, prompt())

    assert result.output is None
    assert result.error is not None
    assert result.raw_outputs == ["nope", "still nope"]
    assert len(result.errors) == 2
