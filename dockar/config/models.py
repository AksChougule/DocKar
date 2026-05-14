"""Pydantic configuration models for DocKar."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator


class ModelConfig(BaseModel):
    """LLM provider and model settings."""

    provider: str = Field(default="openai", pattern="^(openai|ollama)$")
    default_model: str = Field(default="gpt-4o-mini", min_length=1)
    fallback_model: str = Field(default="gpt-4o", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: PositiveInt = 4096
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434"


class LoopConfig(BaseModel):
    """Prompt optimization loop settings."""

    max_iterations: PositiveInt = 20
    budget_usd: float = Field(default=10.0, gt=0)
    early_stopping_rounds: PositiveInt = 3
    candidates_per_iteration: PositiveInt = 4
    top_k: PositiveInt = 2

    @model_validator(mode="after")
    def validate_top_k(self) -> "LoopConfig":
        """Keep beam size compatible with candidate generation."""

        if self.top_k > self.candidates_per_iteration:
            message = "loop.top_k must be less than or equal to loop.candidates_per_iteration"
            raise ValueError(message)
        return self


class EvaluationWeightsConfig(BaseModel):
    """Weights used to combine evaluation metrics."""

    accuracy: float = Field(default=0.8, ge=0)
    cost: float = Field(default=0.1, ge=0)
    latency: float = Field(default=0.1, ge=0)

    @model_validator(mode="after")
    def validate_weight_sum(self) -> "EvaluationWeightsConfig":
        """Ensure at least one scoring dimension contributes."""

        if self.accuracy + self.cost + self.latency <= 0:
            message = "evaluation.weights must include at least one positive weight"
            raise ValueError(message)
        return self


class FieldScoringConfig(BaseModel):
    """Field-level evaluation behavior."""

    enabled: bool = True
    exact_match_fields: list[str] = Field(default_factory=list)
    semantic_fields: list[str] = Field(default_factory=list)
    numeric_tolerance: float = Field(default=0.0, ge=0)
    per_field_weights: dict[str, float] = Field(default_factory=dict)

    @field_validator("per_field_weights")
    @classmethod
    def validate_per_field_weights(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject negative per-field weights."""

        negative_fields = [field for field, weight in value.items() if weight < 0]
        if negative_fields:
            fields = ", ".join(sorted(negative_fields))
            message = f"field_scoring.per_field_weights cannot be negative for: {fields}"
            raise ValueError(message)
        return value


class EvaluationConfig(BaseModel):
    """Extraction scoring settings."""

    weights: EvaluationWeightsConfig = Field(default_factory=EvaluationWeightsConfig)
    field_scoring: FieldScoringConfig = Field(default_factory=FieldScoringConfig)


class IngestionConfig(BaseModel):
    """Document ingestion settings."""

    ocr_enabled: bool = True
    chunk_size: PositiveInt = 4000
    max_doc_pages: PositiveInt = 100


class LoggingConfig(BaseModel):
    """Structured logging settings."""

    log_level: str = "INFO"
    output_dir: Path = Path("runs/logs")

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Validate and normalize logging levels."""

        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            message = f"logging.log_level must be one of: {', '.join(sorted(allowed))}"
            raise ValueError(message)
        return normalized


class DocKarConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = "DocKar"
    model: ModelConfig = Field(default_factory=ModelConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def required_sections(cls) -> set[str]:
        """Sections expected in project YAML files."""

        return {"model", "loop", "evaluation", "ingestion", "logging"}


ConfigMapping = dict[str, Any]
