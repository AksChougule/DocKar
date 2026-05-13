"""Pydantic configuration models for DocKar."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class ModelConfig(BaseModel):
    """LLM provider and model settings."""

    provider: str = Field(default="openai", description="LLM provider identifier.")
    name: str = Field(default="gpt-4o-mini", description="Default extraction model.")
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)


class BudgetConfig(BaseModel):
    """Run-level budget constraints."""

    max_cost_usd: float = Field(default=10.0, gt=0)
    max_iterations: PositiveInt = 20
    early_stopping_rounds: PositiveInt = 3


class ExecutionConfig(BaseModel):
    """Runtime paths and execution behavior."""

    docs_path: Path | None = None
    schema_path: Path | None = None
    labels_path: Path | None = None
    runs_path: Path = Path("runs")
    chunk_size: PositiveInt = 4000


class LoggingConfig(BaseModel):
    """Structured logging settings."""

    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    json_logs: bool = Field(default=True, alias="json")


class DocKarConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = "DocKar"
    model: ModelConfig = Field(default_factory=ModelConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
