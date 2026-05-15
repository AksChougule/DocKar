"""Deterministic prompt construction."""

import json
from typing import Any


class PromptBuilder:
    """Build a clean JSON-only extraction prompt."""

    default_constraints = [
        "Return only valid JSON.",
        "Do not include markdown fences, commentary, or explanations.",
        "Use null when a requested value is missing or cannot be inferred.",
        "Preserve field names exactly as defined in the schema.",
        "Do not invent values that are not supported by the document text.",
    ]

    def build(
        self,
        task_description: str,
        schema: dict[str, Any],
        examples: list[dict[str, Any]] | None = None,
        constraints: list[str] | None = None,
        section_order: list[str] | None = None,
        instruction_style: str | None = None,
    ) -> str:
        """Build a prompt string from structured inputs."""

        selected_examples = (examples or [])[:3]
        prompt_constraints = constraints or self.default_constraints
        sections = {
            "role": self._role_section(instruction_style),
            "task": self._task_section(task_description),
            "schema": self._schema_section(schema),
            "constraints": self._constraints_section(prompt_constraints),
            "examples": self._examples_section(selected_examples),
            "document": self._document_section(),
            "output": self._output_section(),
        }
        order = section_order or [
            "role",
            "task",
            "schema",
            "constraints",
            "examples",
            "document",
            "output",
        ]
        return "\n\n".join(sections[name] for name in order if sections.get(name)).strip()

    def _role_section(self, instruction_style: str | None) -> str:
        if instruction_style == "terse":
            return "You extract structured data from documents."
        if instruction_style == "auditor":
            return (
                "You are a careful document extraction auditor. Prefer explicit evidence "
                "from the document over assumptions."
            )
        return (
            "You are a precise document extraction system. Extract the requested fields "
            "from the document text."
        )

    def _task_section(self, task_description: str) -> str:
        return f"## Task\n{task_description.strip()}"

    def _schema_section(self, schema: dict[str, Any]) -> str:
        return f"## Output Schema\n{self._json(schema)}"

    def _constraints_section(self, constraints: list[str]) -> str:
        lines = "\n".join(f"- {constraint}" for constraint in constraints)
        return f"## Constraints\n{lines}"

    def _examples_section(self, examples: list[dict[str, Any]]) -> str:
        if not examples:
            return ""

        formatted_examples = []
        for index, example in enumerate(examples, start=1):
            formatted_examples.append(f"Example {index}:\n{self._json(example)}")
        return "## Examples\n" + "\n\n".join(formatted_examples)

    def _document_section(self) -> str:
        return "## Document Text\n{{document_text}}"

    def _output_section(self) -> str:
        return "## Response\nReturn the JSON object only."

    def _json(self, value: dict[str, Any]) -> str:
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
