import httpx
import pytest

from dockar.execution import LLMError, OllamaClient, OpenAIClient, RouterClient
from dockar.models import LLMRequestConfig, LLMResponse, LLMUsage


def test_openai_client_generates_response_and_estimates_cost() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "extracted text"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    client = OpenAIClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.generate(
        "extract this",
        LLMRequestConfig(model="gpt-4o-mini", max_tokens=50),
    )

    assert response.text == "extracted text"
    assert response.provider == "openai"
    assert response.usage.total_tokens == 15
    assert response.cost_usd > 0


def test_openai_client_retries_server_errors() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "try again"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAIClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
    )

    response = client.generate("hello", LLMRequestConfig(model="gpt-4o-mini"))

    assert calls == 2
    assert response.attempts == 2
    assert response.text == "ok"


def test_openai_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIClient(api_key=None)

    with pytest.raises(LLMError, match="API key"):
        client.generate("hello", LLMRequestConfig(model="gpt-4o-mini"))


def test_ollama_client_generates_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        return httpx.Response(
            200,
            json={
                "response": "local answer",
                "prompt_eval_count": 4,
                "eval_count": 3,
            },
        )

    client = OllamaClient(http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    response = client.generate("hello", LLMRequestConfig(model="llama3.1"))

    assert response.text == "local answer"
    assert response.provider == "ollama"
    assert response.cost_usd == 0
    assert response.usage.total_tokens == 7


class FailingClient:
    def generate(self, prompt: str, config: LLMRequestConfig) -> LLMResponse:
        raise LLMError("default failed")


class SuccessfulClient:
    def generate(self, prompt: str, config: LLMRequestConfig) -> LLMResponse:
        return LLMResponse(
            text=f"{config.model}: {prompt}",
            model=config.model,
            provider="test",
            usage=LLMUsage(total_tokens=1),
            latency_ms=1.0,
        )


def test_router_falls_back_to_stronger_model() -> None:
    router = RouterClient(
        default_client=FailingClient(),
        fallback_client=SuccessfulClient(),
        default_model="cheap-model",
        fallback_model="strong-model",
    )

    response = router.generate("extract")

    assert response.text == "strong-model: extract"
    assert response.model == "strong-model"
    assert response.fallback_used is True
    assert response.attempts == 2
