# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Betty is a real-time TUI monitor for Claude Code sessions. It watches project directories for session files and monitors transcript files directly for all conversation activity. No hooks required - just run `betty`.

## Development Commands

```bash
# Install in dev mode
pip install -e .

# Run the companion
betty                # Watch current directory's sessions
betty --global       # Watch all projects
betty --manager      # Start in manager view (all sessions)
betty --worktree     # Watch git worktrees of current repo

# Run with uv (no install needed)
uv run betty

# LLM configuration (for summarization and analysis)
betty config --show                    # Show current config
betty config --llm-preset lm-studio    # Use LM Studio preset
betty config --llm-preset ollama       # Use Ollama preset
betty config --llm-preset anthropic    # Use Anthropic API
betty config --llm-preset openai       # Use OpenAI API
betty config --url URL --model MODEL   # Custom configuration

# UI configuration
betty config --style rich              # Emoji/box-based style
betty config --style claude-code       # Minimal style matching Claude Code CLI
betty config --collapse-tools          # Collapse consecutive tool turns
betty config --manager-open-mode auto  # Manager mode: swap, expand, or auto

# Summary style configuration
betty config --summary-style brief         # Ultra-short, under 15 words
betty config --summary-style detailed      # 2-3 sentences with reasoning
betty config --summary-style technical     # File paths, function names, line numbers
betty config --summary-style explanatory   # Focus on intent and reasoning
betty config --summary-style default       # Reset to default 1-sentence style
betty config --summary-prompt "Custom prompt text"  # Fully custom prompt

# Agent configuration (continuous session observer)
betty config --agent                     # Enable Betty Agent
betty config --no-agent                  # Disable Betty Agent
# Alternatively: BETTY_AGENT_ENABLED=true betty

# Analyzer configuration (for on-demand analysis)
betty config --analyzer-budget N         # Set context budget in chars
betty config --analyzer-small-range N    # Max turns for small range
betty config --analyzer-large-range N    # Min turns for large range

# Test imports
uv run python -c "from betty import tui_textual, store, models; print('OK')"
```

## Architecture

### Data Flow

1. **ProjectWatcher** (`project_watcher.py`) scans `~/.claude/projects/<encoded-cwd>/` for `.jsonl` session files
2. When a session file is discovered, `EventStore` creates a new `Session` and starts a `TranscriptWatcher`
3. `TranscriptWatcher` (`watcher.py`) polls the transcript JSONL file for new entries
4. New turns are parsed and added to the session, triggering TUI updates
5. **PRDetector** (`github.py`) asynchronously detects GitHub PRs for each session's branch

### Key Components

- **`tui_textual.py`** - Main Textual-based TUI: session detail view, manager view, keyboard input, CSS styling
- **`store.py`** - Central `EventStore` class: thread-safe session/turn storage, coordinates watchers/alerts/summarizer/analyzer
- **`project_watcher.py`** - `ProjectWatcher`: scans project directories for session files, notifies on new sessions
- **`watcher.py`** - `TranscriptWatcher`: polls transcript file, parses JSONL entries into `Turn` objects
- **`transcript.py`** - Parses `session.jsonl` files to load conversation history on session start
- **`models.py`** - Data classes: `Turn`, `Session`, `TaskState`, `ToolGroup`
- **`github.py`** - `PRDetector`: async GitHub PR detection via `gh` CLI, caches by (project_dir, branch)
- **`styles.py`** - UI style renderers: `RichStyle` (emoji/box-based) and `ClaudeCodeStyle` (minimal)
- **`alerts.py`** - Pattern matching for dangerous operations (force push, rm -rf, etc.)
- **`summarizer.py`** - Optional LLM summarization of assistant turns via litellm, with configurable style presets
- **`analyzer.py`** - On-demand LLM analysis: structured analysis (summary, critique, sentiment), multi-level analysis (turn/span/session), multi-source goal extraction, cost tracking
- **`cache.py`** - Persistent disk cache for summaries and annotations (`~/.cache/betty/`)
- **`config.py`** - Configuration management (env vars, `~/.betty/config.toml`, defaults)
- **`export.py`** - Export session data to Markdown or JSON formats
- **`pricing.py`** - Model pricing database and cost estimation utilities for token usage tracking
- **`mock_session.py`** - Mock session generator for development/testing
- **`utils.py`** - Utility functions
- **`cli.py`** - Click-based CLI with commands: `config`, `mock`
- **`tui.py`** - Legacy Rich-based TUI (superseded by `tui_textual.py`)

### Session Discovery

Sessions are discovered by watching `~/.claude/projects/<encoded-cwd>/` where `<encoded-cwd>` is the current directory with `/` replaced by `-` (e.g., `/Users/kai/src/foo` becomes `-Users-kai-src-foo`).

By default, only sessions for the current directory are shown. Use `--global` to watch all projects.

### Threading Model

- Main thread: Textual TUI event loop
- Project watcher thread: Directory scanning (daemon)
- Watcher threads: Transcript file polling (daemon, one per session)
- Plan file watcher threads: Monitor PLAN.md / .claude/plan.md per session (daemon)
- Summarizer threads: ThreadPoolExecutor for async LLM calls (2 workers)
- Analyzer threads: ThreadPoolExecutor for async LLM analysis (1 worker)
- PR detection threads: Background threads for `gh` CLI calls (spawned by PRDetector)
- All shared state in `EventStore` protected by `threading.Lock`

### Manager View

The manager view (`M` key) displays all sessions grouped by project, with GitHub PR info on group headers. Sessions show stats (turns, tools, model, branch). Two layout modes:
- **Swap**: Manager replaces the detail view
- **Expand**: Side-by-side manager (left) + detail pane (right), navigable with `h`/`l`
- **Auto**: Chooses based on terminal width

### Optional Features

**LLM Integration**: LLM calls go through litellm, supporting any compatible provider. Presets available:
- **LM Studio** (recommended for Mac): `http://localhost:1234/v1`
- **Ollama**: `http://localhost:11434/v1`
- **vLLM**: `http://localhost:8008/v1` (default)
- **OpenAI**, **Anthropic**, **OpenRouter**: Cloud API providers

Configuration priority:
1. Environment variables: `BETTY_LLM_API_BASE`, `BETTY_LLM_MODEL`
2. Config file: `~/.betty/config.toml`
3. Hardcoded defaults (vLLM)

Use `betty config` to set up your LLM server. Summaries are cached to disk and persist across sessions. The feature gracefully degrades if the server is unavailable.

**Summarization Styles**: Summaries can be customized with preset styles (`default`, `brief`, `detailed`, `technical`, `explanatory`) or a fully custom prompt. Set via `betty config --summary-style <style>` or `betty config --summary-prompt "..."`. Environment variables `BETTY_SUMMARY_STYLE` and `BETTY_SUMMARY_PROMPT` override the config file. Changing styles automatically invalidates the summary cache.

**On-Demand Analysis**: Turns, spans, and sessions can be analyzed on-demand. The analyzer provides structured analysis (summary, critique, sentiment) with multi-source goal extraction (first user message, GitHub issues, plan files, active tasks). Triggered via `A` keybinding in the TUI, with `[`/`]` to zoom between turn, span, and session levels. Toggle analysis panel with `I`. Analysis results are cached to disk.

**GitHub PR Integration**: Automatic PR detection for each session's git branch using the `gh` CLI. PR info (number, title, URL, state) is displayed on manager view headers and session cards. Press `O` to open the PR in a browser.

**Plan File Monitoring**: Auto-detects PLAN.md or .claude/plan.md in the project directory. Real-time updates via dedicated watcher threads. Toggle display with `P`.

**Tool Grouping**: Consecutive tool turns are collapsed into `ToolGroup` objects for compact display. Toggle with `--collapse-tools` config or CLI flag.

**Export**: Sessions can be exported to Markdown or JSON via the `export.py` module (used by TUI keyboard shortcuts).
