"""Prompt candidate generation."""

from typing import Any

from dockar.prompt_engine.builder import PromptBuilder
from dockar.prompt_engine.interfaces import PromptCandidate
from dockar.prompt_engine.mutator import PromptMutator


class PromptGenerator:
    """Generate base prompts and mutated prompt candidates."""

    def __init__(
        self,
        builder: PromptBuilder | None = None,
        mutator: PromptMutator | None = None,
        deterministic: bool = True,
    ) -> None:
        self.builder = builder or PromptBuilder()
        self.mutator = mutator or PromptMutator(deterministic=deterministic)

    def build_prompt(
        self,
        task_description: str,
        schema: dict[str, Any],
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build the base prompt string."""

        return self.builder.build(task_description, schema, examples)

    def generate(
        self,
        task_description: str,
        schema: dict[str, Any],
        examples: list[dict[str, Any]] | None = None,
        candidate_count: int = 1,
    ) -> list[PromptCandidate]:
        """Generate one or more prompt candidates."""

        prompt_examples = (examples or [])[:3]
        mutations = self.mutator.mutations(candidate_count, len(prompt_examples))
        candidates: list[PromptCandidate] = []
        for index, mutation in enumerate(mutations, start=1):
            selected_examples = self.mutator.select_examples(prompt_examples, mutation)
            prompt = self.builder.build(
                task_description=task_description,
                schema=schema,
                examples=selected_examples,
                constraints=mutation.constraints,
                section_order=mutation.section_order,
                instruction_style=mutation.instruction_style,
            )
            candidates.append(
                PromptCandidate(
                    id=f"prompt-{index:04d}",
                    text=prompt,
                    metadata={
                        "strategy": mutation.strategy,
                        "example_count": len(selected_examples),
                    },
                )
            )
        return candidates

    def refine(self, candidates: list[PromptCandidate]) -> list[PromptCandidate]:
        """Return candidates unchanged until scoring feedback is available."""

        return candidates
