"""Interfaces for prompt optimization loops."""

from typing import Protocol

from pydantic import BaseModel

from dockar.evaluation import EvaluationReport
from dockar.prompt_engine import PromptCandidate


class OptimizationResult(BaseModel):
    """Best prompt and metrics emitted by a loop run."""

    best_prompt: PromptCandidate
    metrics: EvaluationReport


class OptimizationLoop(Protocol):
    """Coordinates prompt generation, execution, and evaluation."""

    async def run(self) -> OptimizationResult:
        """Run the optimization loop."""
        ...
