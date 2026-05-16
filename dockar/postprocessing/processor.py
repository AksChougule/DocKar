"""Post-processing for extraction outputs."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from dockar.execution.interfaces import LLMClient
from dockar.execution.llm import LLMError
from dockar.models import ExtractedDocument, LLMRequestConfig


class PostProcessingError(RuntimeError):
    """Raised when post-processing cannot produce usable data."""


class PostProcessor:
    """Clean, repair, and semi-strictly validate extraction output."""

    def __init__(
        self,
        repair_client: LLMClient | None = None,
        repair_config: LLMRequestConfig | None = None,
    ) -> None:
        self.repair_client = repair_client
        self.repair_config = repair_config

    def process(self, extracted: ExtractedDocument, schema: dict[str, Any]) -> ExtractedDocument:
        """Return a cleaned and schema-aligned extraction."""

        fixes: list[dict[str, Any]] = []
        data = self._initial_data(extracted, fixes)
        data = self._cleanup(data, schema, fixes)
        data = self._align_to_schema(data, schema, fixes)
        validation_errors = self._validate(data, schema)

        if validation_errors and extracted.raw_output:
            repaired = self._repair_with_llm(extracted.raw_output, schema, fixes)
            if repaired is not None:
                data = self._cleanup(repaired, schema, fixes)
                data = self._align_to_schema(data, schema, fixes)
                validation_errors = self._validate(data, schema)

        metadata = {
            **getattr(extracted, "metadata", {}),
            "postprocessing": {
                "fixes": fixes,
                "validation_errors": validation_errors,
                "valid": not validation_errors,
            },
        }
        return ExtractedDocument(
            document_id=extracted.document_id,
            data=data,
            raw_output=extracted.raw_output,
            metadata=metadata,
        )

    def _initial_data(
        self,
        extracted: ExtractedDocument,
        fixes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if extracted.data:
            return deepcopy(extracted.data)
        if not extracted.raw_output:
            fixes.append({"step": "parse", "message": "No data or raw output available"})
            return {}
        parsed = self._parse_json_like(extracted.raw_output, fixes)
        return parsed if isinstance(parsed, dict) else {}

    def _parse_json_like(self, raw_output: str, fixes: list[dict[str, Any]]) -> Any:
        cleaned = self._strip_markdown_fence(raw_output.strip())
        embedded = self._extract_json_object(cleaned)
        candidates = (
            cleaned,
            embedded,
            self._repair_common_json(cleaned),
            self._repair_common_json(embedded) if embedded is not None else None,
        )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                parsed = json.loads(candidate)
                if candidate != cleaned:
                    fixes.append({"step": "json_repair", "message": "Recovered JSON object"})
                return parsed
            except json.JSONDecodeError:
                continue

        fixes.append({"step": "json_repair", "message": "Could not parse raw output as JSON"})
        return {}

    def _cleanup(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
        fixes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        properties = self._properties(schema)

        for key, value in data.items():
            if self._is_noise_key(key):
                fixes.append({"step": "cleanup", "field": key, "message": "Removed noise field"})
                continue

            normalized_key = key.strip()
            if normalized_key != key:
                fixes.append({"step": "cleanup", "field": key, "message": "Normalized field name"})

            field_schema = properties.get(normalized_key, {})
            cleaned[normalized_key] = self._clean_value(normalized_key, value, field_schema, fixes)

        return cleaned

    def _clean_value(
        self,
        field: str,
        value: Any,
        field_schema: dict[str, Any],
        fixes: list[dict[str, Any]],
    ) -> Any:
        if isinstance(value, str):
            normalized = self._normalize_string(value)
            if normalized != value:
                fixes.append({"step": "cleanup", "field": field, "message": "Normalized string"})
            value = normalized

        expected_type = field_schema.get("type")
        if isinstance(expected_type, list):
            expected_type = next(
                (item for item in expected_type if item != "null"),
                expected_type[0],
            )

        try:
            return self._coerce_type(field, value, expected_type, fixes)
        except (TypeError, ValueError):
            fixes.append({"step": "cleanup", "field": field, "message": "Could not coerce type"})
            return value

    def _coerce_type(
        self,
        field: str,
        value: Any,
        expected_type: Any,
        fixes: list[dict[str, Any]],
    ) -> Any:
        if value is None or expected_type is None:
            return value
        if expected_type == "string" and not isinstance(value, str):
            fixes.append({"step": "cleanup", "field": field, "message": "Coerced to string"})
            return str(value)
        if expected_type in {"number", "integer"} and isinstance(value, str):
            numeric = self._parse_number(value)
            if numeric is not None:
                fixes.append({"step": "cleanup", "field": field, "message": "Coerced to number"})
                return int(numeric) if expected_type == "integer" else numeric
        if expected_type == "boolean" and isinstance(value, str):
            boolean = self._parse_bool(value)
            if boolean is not None:
                fixes.append({"step": "cleanup", "field": field, "message": "Coerced to boolean"})
                return boolean
        if expected_type == "array" and not isinstance(value, list):
            fixes.append({"step": "cleanup", "field": field, "message": "Wrapped value in array"})
            return [value]
        return value

    def _align_to_schema(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
        fixes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        properties = self._properties(schema)
        if not properties:
            return data

        aligned: dict[str, Any] = {}
        additional_allowed = schema.get("additionalProperties", True)
        for field in properties:
            if field in data:
                aligned[field] = data[field]
            elif field in schema.get("required", []):
                aligned[field] = None
                fixes.append(
                    {"step": "schema_alignment", "field": field, "message": "Added missing field"}
                )

        if additional_allowed:
            for field, value in data.items():
                if field not in aligned:
                    aligned[field] = value
        else:
            removed = sorted(field for field in data if field not in properties)
            for field in removed:
                fixes.append(
                    {"step": "schema_alignment", "field": field, "message": "Removed extra field"}
                )

        return aligned

    def _validate(self, data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        properties = self._properties(schema)
        for required_field in schema.get("required", []):
            if required_field not in data or data[required_field] is None:
                errors.append(f"{required_field}: required field is missing")

        for field, field_schema in properties.items():
            if field not in data or data[field] is None:
                continue
            expected_type = field_schema.get("type")
            if not self._matches_type(data[field], expected_type):
                errors.append(
                    f"{field}: expected {expected_type}, got {type(data[field]).__name__}"
                )
            enum = field_schema.get("enum")
            if isinstance(enum, list) and data[field] not in enum:
                errors.append(f"{field}: value is not in enum")

        return errors

    def _repair_with_llm(
        self,
        raw_output: str,
        schema: dict[str, Any],
        fixes: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if self.repair_client is None or self.repair_config is None:
            return None

        prompt = (
            "Repair this extraction output so it is one valid JSON object matching the schema. "
            "Return JSON only.\n\n"
            f"Schema:\n{json.dumps(schema, indent=2, sort_keys=True)}\n\n"
            f"Output:\n{raw_output}"
        )
        try:
            response = self.repair_client.generate(prompt, self.repair_config)
        except LLMError as exc:
            fixes.append({"step": "llm_repair", "message": f"Repair failed: {exc}"})
            return None

        parsed = self._parse_json_like(response.text, fixes)
        if isinstance(parsed, dict):
            fixes.append({"step": "llm_repair", "message": "Applied LLM repair"})
            return parsed
        return None

    def _properties(self, schema: dict[str, Any]) -> dict[str, Any]:
        properties = schema.get("properties")
        return properties if isinstance(properties, dict) else {}

    def _normalize_string(self, value: str) -> str:
        normalized = re.sub(r"\s+", " ", value).strip()
        normalized = normalized.strip("\u200b")
        return normalized

    def _is_noise_key(self, key: str) -> bool:
        normalized = key.strip().lower()
        return normalized in {"", "note", "notes", "commentary", "explanation", "_metadata"}

    def _parse_number(self, value: str) -> float | None:
        cleaned = re.sub(r"[^0-9.\-]", "", value)
        if cleaned in {"", "-", ".", "-."}:
            return None
        return float(cleaned)

    def _parse_bool(self, value: str) -> bool | None:
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
        return None

    def _matches_type(self, value: Any, expected_type: Any) -> bool:
        if expected_type is None:
            return True
        if isinstance(expected_type, list):
            return any(self._matches_type(value, item) for item in expected_type)
        type_checks = {
            "string": lambda item: isinstance(item, str),
            "number": lambda item: isinstance(item, int | float) and not isinstance(item, bool),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
            "null": lambda item: item is None,
        }
        check = type_checks.get(expected_type)
        return True if check is None else check(value)

    def _strip_markdown_fence(self, text: str) -> str:
        fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if fence_match is None:
            return text
        return fence_match.group(1).strip()

    def _extract_json_object(self, text: str) -> str | None:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else None

    def _repair_common_json(self, text: str) -> str:
        repaired = self._strip_markdown_fence(text)
        repaired = repaired.replace("“", '"').replace("”", '"').replace("'", '"')
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return repaired
