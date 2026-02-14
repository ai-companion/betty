"""Configuration management for Betty."""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Use tomllib on Python 3.11+, fall back to tomli for 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

logger = logging.getLogger(__name__)

# Default style for TUI
DEFAULT_STYLE = "rich"


@dataclass
class LLMConfig:
    """LLM server configuration for summarization.

    The model string encodes the provider using litellm conventions:
    - "openai/gpt-4o-mini" → OpenAI API
    - "anthropic/claude-3-5-haiku-20241022" → Anthropic API
    - "openrouter/openai/gpt-4o-mini" → OpenRouter
    - "ollama/qwen2.5:7b" → Ollama (native litellm support)
    - "claude-code/haiku" → Claude Code CLI subprocess (special case)
    """

    model: str
    api_base: str | None = None  # For local providers or custom endpoints
    api_key: str | None = None  # Explicit API key (litellm also reads env vars)

    @property
    def is_claude_code(self) -> bool:
        """Check if this config uses the claude-code subprocess provider."""
        return self.model.startswith("claude-code/")


@dataclass
class AnalyzerConfig:
    """Configuration for the turn/range analyzer."""

    context_budget: int = 20000       # Max chars for context window
    small_range_max: int = 10         # Turns threshold for "small" range (full content)
    large_range_min: int = 31         # Turns threshold for "large" range (span summaries)
    per_turn_budget: int = 2000       # Max chars per turn in small ranges


@dataclass
class SummaryConfig:
    """Configuration for summarization style and prompts."""
    style: str = "default"
    custom_prompt: str | None = None  # Only used when style="custom"


@dataclass
class AgentConfig:
    """Configuration for Betty Agent (continuous observer)."""

    enabled: bool = False           # Off by default (opt-in)
    update_interval: int = 5        # Min turns between LLM updates
    max_observations: int = 50      # Max observations per session


@dataclass
class Config:
    """Betty configuration."""

    llm: LLMConfig
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    style: str = field(default=DEFAULT_STYLE)
    collapse_tools: bool = field(default=True)  # Collapse tool turns into groups
    debug_logging: bool = field(default=False)  # Enable debug logging to file (opt-in)
    manager_open_mode: str = field(default="auto")  # "swap", "expand", or "auto"


# Default configuration
DEFAULT_CONFIG = Config(
    llm=LLMConfig(
        model="claude-code/haiku",
    ),
    analyzer=AnalyzerConfig(),
    summary=SummaryConfig(),
    agent=AgentConfig(),
    style=DEFAULT_STYLE,
    collapse_tools=True,
    debug_logging=False,
    manager_open_mode="auto",
)

# Config file path
CONFIG_DIR = Path.home() / ".betty"
CONFIG_FILE = CONFIG_DIR / "config.toml"
# Legacy JSON config for migration
LEGACY_CONFIG_FILE = CONFIG_DIR / "config.json"


def _migrate_directories():
    """Migrate old ~/.claude-companion/ to ~/.betty/"""
    old_config = Path.home() / ".claude-companion"
    new_config = Path.home() / ".betty"
    if old_config.exists() and not new_config.exists():
        old_config.rename(new_config)

    old_cache = Path.home() / ".cache" / "claude-companion"
    new_cache = Path.home() / ".cache" / "betty"
    if old_cache.exists() and not new_cache.exists():
        old_cache.rename(new_cache)


def _migrate_json_config() -> None:
    """Migrate legacy JSON config to TOML format."""
    import json
    if LEGACY_CONFIG_FILE.exists() and not CONFIG_FILE.exists():
        try:
            with open(LEGACY_CONFIG_FILE) as f:
                data = json.load(f)
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "wb") as f:
                tomli_w.dump(data, f)
            # Remove old JSON after successful migration
            LEGACY_CONFIG_FILE.unlink()
        except Exception:
            pass  # Silently ignore migration errors


def _migrate_provider_config(data: dict) -> bool:
    """Migrate old provider+model config to litellm model string format.

    Converts e.g. {"provider": "openai", "model": "gpt-4o-mini"}
    to {"model": "openai/gpt-4o-mini"} and re-saves.

    Returns True if migration occurred.
    """
    llm_data = data.get("llm", {})
    provider = llm_data.get("provider")
    if not provider:
        return False

    model = llm_data.get("model", "")

    # Map old provider+model to litellm model string
    if provider == "claude-code":
        new_model = f"claude-code/{model}"
    elif provider == "local":
        # Local providers: keep raw model name, api_base handles routing
        new_model = model
    elif provider == "openrouter":
        # Always prefix with openrouter/ — litellm routes natively.
        # Old configs had model like "openai/gpt-4o-mini" which needs
        # the full "openrouter/openai/gpt-4o-mini" for litellm.
        new_model = f"openrouter/{model}" if not model.startswith("openrouter/") else model
        # Drop base_url: litellm handles OpenRouter endpoint natively
        llm_data.pop("base_url", None)
    elif provider in ("openai", "anthropic"):
        new_model = f"{provider}/{model}" if "/" not in model else model
        # Drop base_url: litellm handles these endpoints natively
        llm_data.pop("base_url", None)
    else:
        new_model = model

    # Update data in-place
    llm_data["model"] = new_model
    llm_data.pop("provider", None)

    # Rename base_url -> api_base (only remains for local providers)
    if "base_url" in llm_data:
        llm_data["api_base"] = llm_data.pop("base_url")

    data["llm"] = llm_data

    # Re-save config
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "wb") as f:
            tomli_w.dump(data, f)
        logger.info(f"Migrated config: provider={provider} -> model={new_model}")
    except Exception:
        pass

    return True


def _load_from_json() -> dict | None:
    """Try to load config from legacy JSON file (fallback)."""
    import json
    if LEGACY_CONFIG_FILE.exists():
        try:
            with open(LEGACY_CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def load_config() -> Config:
    """Load configuration from environment, file, or defaults.

    Priority (highest to lowest):
    1. Environment variables (BETTY_*)
    2. Config file (~/.betty/config.toml)
    3. Hardcoded defaults

    API keys for litellm providers are read from standard env vars
    (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) by litellm automatically.
    """
    # Migrate old directories if needed
    _migrate_directories()

    # Migrate legacy JSON config if needed
    _migrate_json_config()

    # Start with defaults
    llm_model = DEFAULT_CONFIG.llm.model
    api_base = DEFAULT_CONFIG.llm.api_base
    style = DEFAULT_STYLE
    collapse_tools = DEFAULT_CONFIG.collapse_tools
    debug_logging = DEFAULT_CONFIG.debug_logging
    manager_open_mode = DEFAULT_CONFIG.manager_open_mode

    # Try loading from config file (TOML first, then JSON for backwards compat)
    data = None
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            pass  # Try JSON fallback

    # Fallback to JSON if TOML not found or failed
    if data is None:
        data = _load_from_json()

    if data is not None:
        # Auto-migrate old provider-based config
        _migrate_provider_config(data)

        llm_data = data.get("llm", {})
        llm_model = llm_data.get("model", llm_model)
        api_base = llm_data.get("api_base", api_base)
        style = data.get("style", style)
        collapse_tools = data.get("collapse_tools", collapse_tools)
        debug_logging = data.get("debug_logging", debug_logging)
        manager_open_mode = data.get("manager_open_mode", manager_open_mode)

    # Load analyzer config from file
    analyzer_config = AnalyzerConfig()
    if data is not None:
        analyzer_data = data.get("analyzer", {})
        if analyzer_data:
            analyzer_config = AnalyzerConfig(
                context_budget=analyzer_data.get("context_budget", 20000),
                small_range_max=analyzer_data.get("small_range_max", 10),
                large_range_min=analyzer_data.get("large_range_min", 31),
                per_turn_budget=analyzer_data.get("per_turn_budget", 2000),
            )

    # Load summary config from file
    summary_config = SummaryConfig()
    if data is not None:
        summary_data = data.get("summary", {})
        if summary_data:
            summary_config = SummaryConfig(
                style=summary_data.get("style", "default"),
                custom_prompt=summary_data.get("custom_prompt"),
            )

    # Load agent config from file
    agent_config = AgentConfig()
    if data is not None:
        agent_data = data.get("agent", {})
        if agent_data:
            agent_config = AgentConfig(
                enabled=agent_data.get("enabled", False),
                update_interval=agent_data.get("update_interval", 5),
                max_observations=agent_data.get("max_observations", 50),
            )

    # Environment variables override everything
    api_base = os.getenv("BETTY_LLM_API_BASE", api_base)
    llm_model = os.getenv("BETTY_LLM_MODEL", llm_model)
    style = os.getenv("BETTY_STYLE", style)
    collapse_tools_env = os.getenv("BETTY_COLLAPSE_TOOLS")
    if collapse_tools_env is not None:
        collapse_tools = collapse_tools_env.lower() in ("true", "1", "yes")
    debug_logging_env = os.getenv("BETTY_DEBUG_LOGGING")
    if debug_logging_env is not None:
        debug_logging = debug_logging_env.lower() in ("true", "1", "yes")
    manager_open_mode_env = os.getenv("BETTY_MANAGER_OPEN_MODE")
    if manager_open_mode_env is not None and manager_open_mode_env in ("swap", "expand", "auto"):
        manager_open_mode = manager_open_mode_env

    agent_enabled_env = os.getenv("BETTY_AGENT_ENABLED")
    if agent_enabled_env is not None:
        agent_config.enabled = agent_enabled_env.lower() in ("true", "1", "yes")

    summary_style_env = os.getenv("BETTY_SUMMARY_STYLE")
    if summary_style_env is not None:
        summary_config = SummaryConfig(
            style=summary_style_env,
            custom_prompt=summary_config.custom_prompt,
        )
    summary_prompt_env = os.getenv("BETTY_SUMMARY_PROMPT")
    if summary_prompt_env is not None:
        summary_config = SummaryConfig(
            style="custom",
            custom_prompt=summary_prompt_env,
        )

    return Config(
        llm=LLMConfig(
            model=llm_model,
            api_base=api_base,
        ),
        analyzer=analyzer_config,
        summary=summary_config,
        agent=agent_config,
        style=style,
        collapse_tools=collapse_tools,
        debug_logging=debug_logging,
        manager_open_mode=manager_open_mode,
    )


def save_config(config: Config) -> None:
    """Save configuration to file.

    Note: API keys are never saved to the config file.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "llm": {
            "model": config.llm.model,
        },
        "style": config.style,
        "collapse_tools": config.collapse_tools,
        "debug_logging": config.debug_logging,
        "manager_open_mode": config.manager_open_mode,
    }

    # Save api_base when set (for local/custom endpoints)
    if config.llm.api_base:
        data["llm"]["api_base"] = config.llm.api_base

    # Save analyzer config only if non-default
    default_analyzer = AnalyzerConfig()
    if config.analyzer != default_analyzer:
        data["analyzer"] = {
            "context_budget": config.analyzer.context_budget,
            "small_range_max": config.analyzer.small_range_max,
            "large_range_min": config.analyzer.large_range_min,
            "per_turn_budget": config.analyzer.per_turn_budget,
        }

    # Save agent config only if non-default
    default_agent = AgentConfig()
    if config.agent != default_agent:
        data["agent"] = {
            "enabled": config.agent.enabled,
            "update_interval": config.agent.update_interval,
            "max_observations": config.agent.max_observations,
        }

    # Save summary config only if non-default
    default_summary = SummaryConfig()
    if config.summary != default_summary:
        summary_data: dict[str, Any] = {"style": config.summary.style}
        if config.summary.custom_prompt:
            summary_data["custom_prompt"] = config.summary.custom_prompt
        data["summary"] = summary_data

    with open(CONFIG_FILE, "wb") as f:
        tomli_w.dump(data, f)


def get_example_configs() -> dict[str, dict[str, Any]]:
    """Get example configurations for common LLM providers."""
    return {
        "vllm": {
            "api_base": "http://localhost:8008/v1",
            "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "description": "vLLM server",
        },
        "lm-studio": {
            "api_base": "http://localhost:1234/v1",
            "model": "gpt-oss-20b",
            "description": "LM Studio",
        },
        "ollama": {
            "model": "ollama/qwen2.5:7b",
            "description": "Ollama (native litellm support, no api_base needed)",
        },
        "openai": {
            "model": "openai/gpt-4o-mini",
            "description": "OpenAI API (requires OPENAI_API_KEY env var)",
        },
        "openrouter": {
            "model": "openrouter/openai/gpt-4o-mini",
            "description": "OpenRouter API (requires OPENROUTER_API_KEY env var)",
        },
        "anthropic": {
            "model": "anthropic/claude-3-5-haiku-20241022",
            "description": "Anthropic API (requires ANTHROPIC_API_KEY env var)",
        },
        "claude-code": {
            "model": "claude-code/haiku",
            "description": "Claude Code CLI (uses `claude -p`, no API key needed)",
        },
    }
