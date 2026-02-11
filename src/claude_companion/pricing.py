"""Model pricing for cost estimation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Pricing per million tokens for a model."""

    input_per_mtok: float        # $/M input tokens
    output_per_mtok: float       # $/M output tokens
    cache_write_per_mtok: float  # $/M cache creation tokens
    cache_read_per_mtok: float   # $/M cache read tokens


# Pricing data for known Claude models
# Source: https://docs.anthropic.com/en/docs/about-claude/models
MODEL_PRICING: dict[str, ModelPricing] = {
    # Opus 4
    "claude-opus-4": ModelPricing(15.0, 75.0, 18.75, 1.50),
    # Sonnet 4
    "claude-sonnet-4": ModelPricing(3.0, 15.0, 3.75, 0.30),
    # Haiku 3.5
    "claude-haiku-4-5": ModelPricing(0.80, 4.0, 1.0, 0.08),
    # Sonnet 3.5 v2
    "claude-3-5-sonnet": ModelPricing(3.0, 15.0, 3.75, 0.30),
    # Haiku 3
    "claude-3-haiku": ModelPricing(0.25, 1.25, 0.30, 0.03),
}


def get_pricing(model_id: str) -> ModelPricing | None:
    """Look up pricing for a model ID.

    Handles versioned IDs like "claude-haiku-4-5-20251001" by trying
    progressively shorter prefixes.
    """
    if not model_id:
        return None

    # Try exact match first
    if model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]

    # Strip version suffix: "claude-haiku-4-5-20251001" -> "claude-haiku-4-5"
    # Try progressively shorter prefixes
    parts = model_id.split("-")
    for i in range(len(parts), 0, -1):
        prefix = "-".join(parts[:i])
        if prefix in MODEL_PRICING:
            return MODEL_PRICING[prefix]

    return None


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
    pricing: ModelPricing,
) -> float:
    """Estimate cost in USD."""
    return (
        input_tokens * pricing.input_per_mtok
        + output_tokens * pricing.output_per_mtok
        + cache_creation_tokens * pricing.cache_write_per_mtok
        + cache_read_tokens * pricing.cache_read_per_mtok
    ) / 1_000_000
