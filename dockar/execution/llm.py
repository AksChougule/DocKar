"""Provider-neutral LLM clients."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from dockar.config import ModelConfig
from dockar.models import LLMRequestConfig, LLMResponse, LLMUsage


class LLMError(RuntimeError):
    """Raised when an LLM provider fails to generate a response."""


RetryableErrors = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


class _GeneratesText(Protocol):
    def generate(self, prompt: str, config: LLMRequestConfig) -> LLMResponse:
        """Generate text for router use."""
        ...


class OpenAIClient:
    """OpenAI Chat Completions client using httpx."""

    provider = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        http_client: httpx.Client | None = None,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client or httpx.Client()
        self.max_retries = max_retries

    def generate(self, prompt: str, config: LLMRequestConfig) -> LLMResponse:
        """Generate text with OpenAI."""

        if not self.api_key:
            message = "OpenAI API key is required. Set model.openai_api_key or OPENAI_API_KEY."
            raise LLMError(message)

        started = time.perf_counter()
        payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        data, attempts = self._request_with_retries(
            lambda: self.http_client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=config.timeout_seconds,
            )
        )
        latency_ms = (time.perf_counter() - started) * 1000
        text = self._extract_text(data)
        usage = self._usage(data, prompt, text)
        return LLMResponse(
            text=text,
            model=config.model,
            provider=self.provider,
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=self._estimate_cost(config.model, usage),
            attempts=attempts,
        )

    def _request_with_retries(
        self,
        request: Callable[[], httpx.Response],
    ) -> tuple[dict[str, Any], int]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                response = request()
                if response.status_code >= 500:
                    response.raise_for_status()
                if response.status_code >= 400:
                    message = (
                        f"OpenAI request failed with status {response.status_code}: "
                        f"{response.text}"
                    )
                    raise LLMError(message)
                data = response.json()
                if not isinstance(data, dict):
                    raise LLMError("OpenAI response must be a JSON object")
                return data, attempt
            except RetryableErrors as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                last_error = exc

        message = f"OpenAI request failed after retries: {last_error}"
        raise LLMError(message) from last_error

    def _extract_text(self, data: dict[str, Any]) -> str:
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI response did not include choices[0].message.content") from exc

        if not isinstance(text, str) or not text.strip():
            raise LLMError("OpenAI response text was empty")
        return text

    def _usage(self, data: dict[str, Any], prompt: str, text: str) -> LLMUsage:
        usage = data.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
            return LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated=False,
            )

        prompt_tokens = self._estimate_tokens(prompt)
        completion_tokens = self._estimate_tokens(text)
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated=True,
        )

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _estimate_cost(self, model: str, usage: LLMUsage) -> float:
        input_per_million, output_per_million = self._prices(model)
        return (
            usage.prompt_tokens * input_per_million
            + usage.completion_tokens * output_per_million
        ) / 1_000_000

    def _prices(self, model: str) -> tuple[float, float]:
        prices = {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
            "gpt-4.1-mini": (0.40, 1.60),
            "gpt-4.1": (2.00, 8.00),
        }
        return prices.get(model, (1.00, 3.00))


class OllamaClient:
    """Ollama local model client using httpx."""

    provider = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        http_client: httpx.Client | None = None,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client or httpx.Client()
        self.max_retries = max_retries

    def generate(self, prompt: str, config: LLMRequestConfig) -> LLMResponse:
        """Generate text with an Ollama local model."""

        started = time.perf_counter()
        payload = {
            "model": config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }
        data, attempts = self._request_with_retries(
            lambda: self.http_client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=config.timeout_seconds,
            )
        )
        latency_ms = (time.perf_counter() - started) * 1000
        text = self._extract_text(data)
        usage = self._usage(data, prompt, text)
        return LLMResponse(
            text=text,
            model=config.model,
            provider=self.provider,
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=0.0,
            attempts=attempts,
        )

    def _request_with_retries(
        self,
        request: Callable[[], httpx.Response],
    ) -> tuple[dict[str, Any], int]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                response = request()
                if response.status_code >= 500:
                    response.raise_for_status()
                if response.status_code >= 400:
                    message = (
                        f"Ollama request failed with status {response.status_code}: "
                        f"{response.text}"
                    )
                    raise LLMError(message)
                data = response.json()
                if not isinstance(data, dict):
                    raise LLMError("Ollama response must be a JSON object")
                return data, attempt
            except RetryableErrors as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                last_error = exc

        message = f"Ollama request failed after retries: {last_error}"
        raise LLMError(message) from last_error

    def _extract_text(self, data: dict[str, Any]) -> str:
        text = data.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Ollama response text was empty")
        return text

    def _usage(self, data: dict[str, Any], prompt: str, text: str) -> LLMUsage:
        prompt_tokens = int(data.get("prompt_eval_count") or self._estimate_tokens(prompt))
        completion_tokens = int(data.get("eval_count") or self._estimate_tokens(text))
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated="prompt_eval_count" not in data or "eval_count" not in data,
        )

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class RouterClient:
    """Route generation to a default model and fall back on failure."""

    def __init__(
        self,
        default_client: _GeneratesText,
        fallback_client: _GeneratesText,
        default_model: str,
        fallback_model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.default_client = default_client
        self.fallback_client = fallback_client
        self.default_model = default_model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> RouterClient:
        """Create a router from DocKar model configuration."""

        if config.provider == "ollama":
            default_client: _GeneratesText = OllamaClient(
                base_url=config.ollama_base_url,
                max_retries=config.max_retries,
            )
            fallback_client: _GeneratesText = default_client
        else:
            default_client = OpenAIClient(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                max_retries=config.max_retries,
            )
            fallback_client = default_client

        return cls(
            default_client=default_client,
            fallback_client=fallback_client,
            default_model=config.default_model,
            fallback_model=config.fallback_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
        )

    def generate(self, prompt: str, config: LLMRequestConfig | None = None) -> LLMResponse:
        """Generate with the default model, then fallback model on failure."""

        default_config = config or self._request_config(self.default_model)
        try:
            return self.default_client.generate(prompt, default_config)
        except LLMError as default_error:
            fallback_config = default_config.model_copy(update={"model": self.fallback_model})
            try:
                response = self.fallback_client.generate(prompt, fallback_config)
            except LLMError as fallback_error:
                message = (
                    "LLM router failed for default and fallback models. "
                    f"default={default_error}; fallback={fallback_error}"
                )
                raise LLMError(message) from fallback_error

            return response.model_copy(
                update={
                    "fallback_used": True,
                    "attempts": response.attempts + 1,
                }
            )

    def _request_config(self, model: str) -> LLMRequestConfig:
        return LLMRequestConfig(
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )
