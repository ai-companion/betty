"""Configuration management for Claude Companion."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    """LLM server configuration for summarization."""

    base_url: str
    model: str


@dataclass
class Config:
    """Claude Companion configuration."""

    llm: LLMConfig


# Default configuration
DEFAULT_CONFIG = Config(
    llm=LLMConfig(
        base_url="http://localhost:8008/v1",  # vLLM default
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
)

# Config file path
CONFIG_DIR = Path.home() / ".claude-companion"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> Config:
    """Load configuration from environment, file, or defaults.

    Priority (highest to lowest):
    1. Environment variables (CLAUDE_COMPANION_LLM_URL, CLAUDE_COMPANION_LLM_MODEL)
    2. Config file (~/.claude-companion/config.json)
    3. Hardcoded defaults
    """
    # Start with defaults
    llm_url = DEFAULT_CONFIG.llm.base_url
    llm_model = DEFAULT_CONFIG.llm.model

    # Try loading from config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
                llm_data = data.get("llm", {})
                llm_url = llm_data.get("base_url", llm_url)
                llm_model = llm_data.get("model", llm_model)
        except Exception:
            # Silently fall back to defaults if config file is malformed
            pass

    # Environment variables override everything
    llm_url = os.getenv("CLAUDE_COMPANION_LLM_URL", llm_url)
    llm_model = os.getenv("CLAUDE_COMPANION_LLM_MODEL", llm_model)

    return Config(llm=LLMConfig(base_url=llm_url, model=llm_model))


def save_config(config: Config) -> None:
    """Save configuration to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "llm": {
            "base_url": config.llm.base_url,
            "model": config.llm.model,
        }
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_example_configs() -> dict[str, dict[str, str]]:
    """Get example configurations for common LLM providers."""
    return {
        "vllm": {
            "base_url": "http://localhost:8008/v1",
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "description": "vLLM server (default)",
        },
        "lm-studio": {
            "base_url": "http://localhost:1234/v1",
            "model": "openai/gpt-oss-20b",
            "description": "LM Studio",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5:7b",
            "description": "Ollama",
        },
    }
