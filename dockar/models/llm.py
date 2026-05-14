"""LLM request and response models."""

from pydantic import BaseModel, Field, PositiveInt


class LLMRequestConfig(BaseModel):
    """Provider-neutral generation configuration."""

    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: PositiveInt = 4096
    timeout_seconds: float = Field(default=60.0, gt=0)


class LLMUsage(BaseModel):
    """Token usage reported or estimated for a generation."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated: bool = False


class LLMResponse(BaseModel):
    """Provider-neutral generation response."""

    text: str
    model: str
    provider: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latency_ms: float
    cost_usd: float = Field(default=0.0, ge=0)
    attempts: int = Field(default=1, ge=1)
    fallback_used: bool = False
