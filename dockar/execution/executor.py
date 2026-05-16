"""Extraction execution engine."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any

from dockar.execution.interfaces import LLMClient
from dockar.execution.llm import LLMError, RouterClient
from dockar.models import (
    Chunk,
    Document,
    ExtractedDocument,
    ExtractionResult,
    LLMRequestConfig,
    LLMResponse,
)
from dockar.prompt_engine import PromptCandidate


class ExtractionExecutionError(RuntimeError):
    """Raised when extraction execution cannot produce a structured result."""


class Executor:
    """Run extraction prompts against documents or chunks."""

    def __init__(
        self,
        llm_client: LLMClient | RouterClient,
        llm_config: LLMRequestConfig | None = None,
        max_retries: int = 1,
    ) -> None:
        if max_retries < 0:
            message = "max_retries must be greater than or equal to 0"
            raise ValueError(message)
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.max_retries = max_retries

    async def execute(self, document: Document, prompt: PromptCandidate) -> ExtractionResult:
        """Execute extraction for a whole document."""

        return self.execute_document(document, prompt)

    def execute_document(self, document: Document, prompt: PromptCandidate) -> ExtractionResult:
        """Execute extraction for one document."""

        return self._execute_text(
            document_id=document.id,
            source_text=document.raw_text,
            prompt=prompt,
            metadata={
                "source_type": "document",
                "page_count": len(document.pages),
            },
        )

    def execute_chunk(self, chunk: Chunk, prompt: PromptCandidate) -> ExtractionResult:
        """Execute extraction for one chunk."""

        return self._execute_text(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            source_text=chunk.text,
            prompt=prompt,
            metadata={
                "source_type": "chunk",
                "page_range": chunk.page_range,
            },
        )

    def execute_documents(
        self,
        documents: Sequence[Document],
        prompt: PromptCandidate,
    ) -> list[ExtractionResult]:
        """Execute extraction for multiple documents."""

        return [self.execute_document(document, prompt) for document in documents]

    def execute_chunks(
        self,
        chunks: Sequence[Chunk],
        prompt: PromptCandidate,
    ) -> list[ExtractionResult]:
        """Execute extraction for multiple chunks."""

        return [self.execute_chunk(chunk, prompt) for chunk in chunks]

    def _execute_text(
        self,
        document_id: str,
        source_text: str,
        prompt: PromptCandidate,
        metadata: dict[str, Any],
        chunk_id: str | None = None,
    ) -> ExtractionResult:
        started = time.perf_counter()
        raw_outputs: list[str] = []
        errors: list[str] = []
        total_cost = 0.0
        llm_latency = 0.0
        fallback_used = False
        model: str | None = None
        provider: str | None = None

        prompts = self._attempt_prompts(prompt.text)
        for attempt_index, attempt_prompt in enumerate(prompts, start=1):
            rendered_prompt = self._render_prompt(attempt_prompt, source_text)
            try:
                response = self._generate(rendered_prompt)
                raw_outputs.append(response.text)
                total_cost += response.cost_usd
                llm_latency += response.latency_ms
                fallback_used = fallback_used or response.fallback_used
                model = response.model
                provider = response.provider
                data = self._parse_json(response.text)
                elapsed_ms = (time.perf_counter() - started) * 1000
                return ExtractionResult(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    output=ExtractedDocument(
                        document_id=document_id,
                        data=data,
                        raw_output=response.text,
                    ),
                    latency_ms=elapsed_ms,
                    cost_usd=total_cost,
                    raw_outputs=raw_outputs,
                    errors=errors,
                    metadata={
                        **metadata,
                        "attempts": attempt_index,
                        "llm_latency_ms": llm_latency,
                        "fallback_used": fallback_used,
                        "model": model,
                        "provider": provider,
                        "prompt_id": prompt.id,
                        "prompt_variation_used": attempt_index > self.max_retries + 1,
                    },
                )
            except (LLMError, ExtractionExecutionError) as exc:
                errors.append(str(exc))

        elapsed_ms = (time.perf_counter() - started) * 1000
        return ExtractionResult(
            document_id=document_id,
            chunk_id=chunk_id,
            error=errors[-1] if errors else "Extraction failed",
            latency_ms=elapsed_ms,
            cost_usd=total_cost,
            raw_outputs=raw_outputs,
            errors=errors,
            metadata={
                **metadata,
                "attempts": len(prompts),
                "llm_latency_ms": llm_latency,
                "fallback_used": fallback_used,
                "model": model,
                "provider": provider,
                "prompt_id": prompt.id,
                "prompt_variation_used": len(prompts) > self.max_retries + 1,
            },
        )

    def _generate(self, prompt: str) -> LLMResponse:
        if isinstance(self.llm_client, RouterClient) and self.llm_config is None:
            return self.llm_client.generate(prompt)
        if self.llm_config is None:
            message = "llm_config is required when the LLM client is not a RouterClient"
            raise LLMError(message)
        return self.llm_client.generate(prompt, self.llm_config)

    def _attempt_prompts(self, prompt: str) -> list[str]:
        prompts = [prompt for _ in range(self.max_retries + 1)]
        prompts.append(self._vary_prompt(prompt))
        return prompts

    def _render_prompt(self, prompt: str, source_text: str) -> str:
        if "{{document_text}}" in prompt:
            return prompt.replace("{{document_text}}", source_text)
        return f"{prompt.rstrip()}\n\n## Document Text\n{source_text}"

    def _vary_prompt(self, prompt: str) -> str:
        variation = (
            "Important: respond with a single valid JSON object only. "
            "If a field is unavailable, use null."
        )
        return f"{prompt.rstrip()}\n\n## Retry Constraint\n{variation}"

    def _parse_json(self, raw_output: str) -> dict[str, Any]:
        cleaned = self._strip_markdown_fence(raw_output.strip())
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = self._parse_embedded_json(cleaned)

        if not isinstance(parsed, dict):
            message = "LLM output must be a JSON object"
            raise ExtractionExecutionError(message)
        return parsed

    def _parse_embedded_json(self, text: str) -> Any:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            message = "LLM output did not contain valid JSON"
            raise ExtractionExecutionError(message)
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            message = f"LLM output did not contain valid JSON: {exc.msg}"
            raise ExtractionExecutionError(message) from exc

    def _strip_markdown_fence(self, text: str) -> str:
        fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if fence_match is None:
            return text
        return fence_match.group(1).strip()
