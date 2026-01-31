"""Configuration management for Claude Companion."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# Default style for TUI
DEFAULT_STYLE = "rich"


@dataclass
class LLMConfig:
    """LLM server configuration for summarization."""

    provider: Literal["local", "openai", "openrouter", "anthropic"]
    model: str
    base_url: str | None = None  # For local providers or custom endpoints
    api_key: str | None = None  # Only for API providers (loaded from env)


@dataclass
class Config:
    """Claude Companion configuration."""

    llm: LLMConfig
    style: str = field(default=DEFAULT_STYLE)
    collapse_tools: bool = field(default=True)  # Collapse tool turns into groups


# Default configuration
DEFAULT_CONFIG = Config(
    llm=LLMConfig(
        provider="local",
        base_url="http://localhost:8008/v1",  # vLLM default
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    ),
    style=DEFAULT_STYLE,
    collapse_tools=True,
)

# Config file path
CONFIG_DIR = Path.home() / ".claude-companion"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> Config:
    """Load configuration from environment, file, or defaults.

    Priority (highest to lowest):
    1. Environment variables (CLAUDE_COMPANION_*)
    2. Config file (~/.claude-companion/config.json)
    3. Hardcoded defaults

    API keys are always loaded from environment variables:
    - OPENAI_API_KEY for OpenAI
    - ANTHROPIC_API_KEY for Anthropic
    """
    # Start with defaults
    provider = DEFAULT_CONFIG.llm.provider
    llm_url = DEFAULT_CONFIG.llm.base_url
    llm_model = DEFAULT_CONFIG.llm.model
    style = DEFAULT_STYLE
    collapse_tools = DEFAULT_CONFIG.collapse_tools

    # Try loading from config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
                llm_data = data.get("llm", {})
                provider = llm_data.get("provider", provider)
                llm_url = llm_data.get("base_url", llm_url)
                llm_model = llm_data.get("model", llm_model)
                style = data.get("style", style)
                collapse_tools = data.get("collapse_tools", collapse_tools)
        except Exception:
            # Silently fall back to defaults if config file is malformed
            pass

    # Environment variables override everything
    provider = os.getenv("CLAUDE_COMPANION_LLM_PROVIDER", provider)
    llm_url = os.getenv("CLAUDE_COMPANION_LLM_URL", llm_url)
    llm_model = os.getenv("CLAUDE_COMPANION_LLM_MODEL", llm_model)
    style = os.getenv("CLAUDE_COMPANION_STYLE", style)
    collapse_tools_env = os.getenv("CLAUDE_COMPANION_COLLAPSE_TOOLS")
    if collapse_tools_env is not None:
        collapse_tools = collapse_tools_env.lower() in ("true", "1", "yes")

    # Load API key from environment based on provider
    api_key = None
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")

    return Config(
        llm=LLMConfig(
            provider=provider,
            base_url=llm_url,
            model=llm_model,
            api_key=api_key,
        ),
        style=style,
        collapse_tools=collapse_tools,
    )


def save_config(config: Config) -> None:
    """Save configuration to file.

    Note: API keys are never saved to the config file, only loaded from environment.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "llm": {
            "provider": config.llm.provider,
            "model": config.llm.model,
        },
        "style": config.style,
        "collapse_tools": config.collapse_tools,
    }

    # Save base_url for local providers and openrouter
    if config.llm.provider in ("local", "openrouter") and config.llm.base_url:
        data["llm"]["base_url"] = config.llm.base_url

    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_example_configs() -> dict[str, dict[str, Any]]:
    """Get example configurations for common LLM providers."""
    return {
        "vllm": {
            "provider": "local",
            "base_url": "http://localhost:8008/v1",
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "description": "vLLM server (default)",
        },
        "lm-studio": {
            "provider": "local",
            "base_url": "http://localhost:1234/v1",
            "model": "openai/gpt-oss-20b",
            "description": "LM Studio",
        },
        "ollama": {
            "provider": "local",
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5:7b",
            "description": "Ollama",
        },
        "openai": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "description": "OpenAI API (requires OPENAI_API_KEY env var)",
        },
        "openrouter": {
            "provider": "openrouter",
            "model": "openai/gpt-4o-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "description": "OpenRouter API (requires OPENROUTER_API_KEY env var)",
        },
        "anthropic": {
            "provider": "anthropic",
            "model": "claude-3-5-haiku-20241022",
            "description": "Anthropic API (requires ANTHROPIC_API_KEY env var)",
        },
    }
