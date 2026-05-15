"""Prompt mutation strategies."""

import random
from dataclasses import dataclass
from typing import Any

from dockar.prompt_engine.builder import PromptBuilder


@dataclass(frozen=True)
class PromptMutation:
    """Concrete mutation recipe for one prompt variant."""

    strategy: str
    instruction_style: str | None = None
    section_order: list[str] | None = None
    constraints: list[str] | None = None
    example_indexes: list[int] | None = None


class PromptMutator:
    """Create deterministic or randomized prompt mutation recipes."""

    base_constraints = PromptBuilder.default_constraints

    def __init__(self, deterministic: bool = True, seed: int | None = None) -> None:
        self.deterministic = deterministic
        self.random = random.Random(seed)

    def mutations(self, candidate_count: int, example_count: int) -> list[PromptMutation]:
        """Return mutation recipes for requested candidates."""

        if candidate_count <= 0:
            message = "candidate_count must be greater than 0"
            raise ValueError(message)

        recipes = self._deterministic_recipes(example_count)
        if self.deterministic:
            return [recipes[index % len(recipes)] for index in range(candidate_count)]

        shuffled = recipes[:]
        self.random.shuffle(shuffled)
        return [shuffled[index % len(shuffled)] for index in range(candidate_count)]

    def _deterministic_recipes(self, example_count: int) -> list[PromptMutation]:
        all_examples = list(range(min(example_count, 3)))
        reversed_examples = list(reversed(all_examples))
        limited_examples = all_examples[:2]

        return [
            PromptMutation(strategy="base", example_indexes=all_examples),
            PromptMutation(
                strategy="rephrase_instructions",
                instruction_style="auditor",
                example_indexes=all_examples,
            ),
            PromptMutation(
                strategy="reorder_sections",
                section_order=[
                    "role",
                    "schema",
                    "task",
                    "constraints",
                    "examples",
                    "document",
                    "output",
                ],
                example_indexes=all_examples,
            ),
            PromptMutation(
                strategy="add_constraints",
                constraints=[
                    *self.base_constraints,
                    (
                        "If multiple candidates exist, choose the value with the strongest "
                        "textual support."
                    ),
                    "Normalize dates and numbers only when the schema implies a normalized format.",
                ],
                example_indexes=all_examples,
            ),
            PromptMutation(
                strategy="remove_constraints",
                constraints=[
                    "Return only valid JSON.",
                    "Do not include markdown fences, commentary, or explanations.",
                    "Preserve field names exactly as defined in the schema.",
                ],
                example_indexes=all_examples,
            ),
            PromptMutation(
                strategy="vary_examples",
                instruction_style="terse",
                example_indexes=reversed_examples or limited_examples,
            ),
            PromptMutation(
                strategy="vary_examples_limited",
                example_indexes=limited_examples,
            ),
        ]

    def select_examples(
        self,
        examples: list[dict[str, Any]],
        mutation: PromptMutation,
    ) -> list[dict[str, Any]]:
        """Select examples for a mutation recipe."""

        if mutation.example_indexes is None:
            return examples[:3]
        return [examples[index] for index in mutation.example_indexes if index < len(examples)]
