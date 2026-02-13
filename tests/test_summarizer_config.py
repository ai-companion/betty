"""Tests for customizable summarization style and prompts."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from betty.config import Config, LLMConfig, SummaryConfig, AnalyzerConfig, load_config, save_config
from betty.summarizer import (
    SUMMARY_STYLES,
    SYSTEM_PROMPT,
    TOOL_SYSTEM_PROMPT,
    VALID_SUMMARY_STYLES,
    Summarizer,
    _prompt_fingerprint,
    get_summary_prompts,
)


class TestGetSummaryPrompts:
    """Tests for get_summary_prompts helper."""

    def test_default_style(self):
        system, tool = get_summary_prompts("default")
        assert system == SYSTEM_PROMPT
        assert tool == TOOL_SYSTEM_PROMPT

    def test_brief_style(self):
        system, tool = get_summary_prompts("brief")
        assert "under 15 words" in system
        assert "under 15 words" in tool
        assert system != SYSTEM_PROMPT

    def test_detailed_style(self):
        system, tool = get_summary_prompts("detailed")
        assert "2-3 sentences" in system
        assert "2-3 sentences" in tool

    def test_technical_style(self):
        system, tool = get_summary_prompts("technical")
        assert "file paths" in system.lower() or "file path" in system.lower()
        assert "exit codes" in tool.lower() or "exit code" in tool.lower()

    def test_explanatory_style(self):
        system, tool = get_summary_prompts("explanatory")
        assert "WHY" in system
        assert "WHY" in tool

    def test_custom_style_with_prompt(self):
        custom = "Summarize in haiku form"
        system, tool = get_summary_prompts("custom", custom)
        assert system == custom
        assert tool == custom

    def test_custom_style_without_prompt_falls_back(self):
        system, tool = get_summary_prompts("custom", None)
        # Falls back to default when custom prompt is None
        assert system == SYSTEM_PROMPT
        assert tool == TOOL_SYSTEM_PROMPT

    def test_unknown_style_falls_back(self):
        system, tool = get_summary_prompts("nonexistent")
        assert system == SYSTEM_PROMPT
        assert tool == TOOL_SYSTEM_PROMPT

    def test_all_styles_in_dict(self):
        expected = {"default", "brief", "detailed", "technical", "explanatory"}
        assert expected == set(SUMMARY_STYLES.keys())

    def test_valid_styles_includes_custom(self):
        assert "custom" in VALID_SUMMARY_STYLES
        for style in SUMMARY_STYLES:
            assert style in VALID_SUMMARY_STYLES


class TestSummarizerInstancePrompts:
    """Tests for Summarizer storing correct prompts per style."""

    def test_default_prompts(self):
        s = Summarizer(model="test-model", summary_style="default")
        assert s.system_prompt == SYSTEM_PROMPT
        assert s.tool_system_prompt == TOOL_SYSTEM_PROMPT

    def test_brief_prompts(self):
        s = Summarizer(model="test-model", summary_style="brief")
        assert "under 15 words" in s.system_prompt
        assert "under 15 words" in s.tool_system_prompt

    def test_custom_prompts(self):
        custom = "Be a pirate"
        s = Summarizer(model="test-model", summary_style="custom", custom_summary_prompt=custom)
        assert s.system_prompt == custom
        assert s.tool_system_prompt == custom

    def test_no_style_arg_uses_default(self):
        s = Summarizer(model="test-model")
        assert s.system_prompt == SYSTEM_PROMPT
        assert s.tool_system_prompt == TOOL_SYSTEM_PROMPT


class TestCacheFingerprint:
    """Tests that cache fingerprints vary by style."""

    def test_different_styles_produce_different_fingerprints(self):
        model = "test-model"
        default_sys, _ = get_summary_prompts("default")
        brief_sys, _ = get_summary_prompts("brief")
        fp_default = _prompt_fingerprint(model, default_sys)
        fp_brief = _prompt_fingerprint(model, brief_sys)
        assert fp_default != fp_brief

    def test_same_style_produces_same_fingerprint(self):
        model = "test-model"
        sys1, _ = get_summary_prompts("detailed")
        sys2, _ = get_summary_prompts("detailed")
        assert _prompt_fingerprint(model, sys1) == _prompt_fingerprint(model, sys2)

    def test_custom_prompt_produces_unique_fingerprint(self):
        model = "test-model"
        default_sys, _ = get_summary_prompts("default")
        custom_sys, _ = get_summary_prompts("custom", "my custom prompt")
        fp_default = _prompt_fingerprint(model, default_sys)
        fp_custom = _prompt_fingerprint(model, custom_sys)
        assert fp_default != fp_custom


class TestSummaryConfigRoundTrip:
    """Tests for SummaryConfig persistence in TOML."""

    def test_default_config_not_saved(self):
        """Default SummaryConfig should not add a summary section to TOML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.toml"
            config = Config(
                llm=LLMConfig(model="test-model"),
                summary=SummaryConfig(),
            )
            with patch("betty.config.CONFIG_FILE", config_file), \
                 patch("betty.config.CONFIG_DIR", Path(tmpdir)):
                save_config(config)

                content = config_file.read_text()
                assert "summary" not in content

    def test_non_default_style_saved(self):
        """Non-default summary style should be persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.toml"
            config = Config(
                llm=LLMConfig(model="test-model"),
                summary=SummaryConfig(style="brief"),
            )
            with patch("betty.config.CONFIG_FILE", config_file), \
                 patch("betty.config.CONFIG_DIR", Path(tmpdir)):
                save_config(config)
                loaded = load_config()

            assert loaded.summary.style == "brief"

    def test_custom_prompt_saved(self):
        """Custom prompt should be persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.toml"
            config = Config(
                llm=LLMConfig(model="test-model"),
                summary=SummaryConfig(style="custom", custom_prompt="Summarize like a pirate"),
            )
            with patch("betty.config.CONFIG_FILE", config_file), \
                 patch("betty.config.CONFIG_DIR", Path(tmpdir)):
                save_config(config)
                loaded = load_config()

            assert loaded.summary.style == "custom"
            assert loaded.summary.custom_prompt == "Summarize like a pirate"

    def test_env_var_overrides_style(self):
        """BETTY_SUMMARY_STYLE env var should override config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.toml"
            config = Config(
                llm=LLMConfig(model="test-model"),
                summary=SummaryConfig(style="brief"),
            )
            with patch("betty.config.CONFIG_FILE", config_file), \
                 patch("betty.config.CONFIG_DIR", Path(tmpdir)), \
                 patch.dict("os.environ", {"BETTY_SUMMARY_STYLE": "technical"}):
                save_config(config)
                loaded = load_config()

            assert loaded.summary.style == "technical"

    def test_env_var_overrides_prompt(self):
        """BETTY_SUMMARY_PROMPT env var should override and set style to custom."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.toml"
            config = Config(
                llm=LLMConfig(model="test-model"),
            )
            with patch("betty.config.CONFIG_FILE", config_file), \
                 patch("betty.config.CONFIG_DIR", Path(tmpdir)), \
                 patch.dict("os.environ", {"BETTY_SUMMARY_PROMPT": "Be concise"}):
                save_config(config)
                loaded = load_config()

            assert loaded.summary.style == "custom"
            assert loaded.summary.custom_prompt == "Be concise"
