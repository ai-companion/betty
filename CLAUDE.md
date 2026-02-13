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

# Run with uv (no install needed)
uv run betty

# LLM configuration (for summarization and analysis)
betty config --show                    # Show current config
betty config --llm-preset lm-studio    # Use LM Studio preset
betty config --llm-preset ollama       # Use Ollama preset
betty config --url URL --model MODEL   # Custom configuration

# Analyzer configuration (for on-demand analysis)
betty config --analyzer-budget N         # Set context budget in chars
betty config --analyzer-small-range N    # Max turns for small range
betty config --analyzer-large-range N    # Min turns for large range

# Test imports
uv run python -c "from betty import tui, store, models; print('OK')"
```

## Architecture

### Data Flow

1. **ProjectWatcher** (`project_watcher.py`) scans `~/.claude/projects/<encoded-cwd>/` for `.jsonl` session files
2. When a session file is discovered, `EventStore` creates a new `Session` and starts a `TranscriptWatcher`
3. `TranscriptWatcher` (`watcher.py`) polls the transcript JSONL file for new entries
4. New turns are parsed and added to the session, triggering TUI updates

### Key Components

- **`store.py`** - Central `EventStore` class: thread-safe session/turn storage, coordinates watchers/alerts/summarizer/analyzer
- **`project_watcher.py`** - `ProjectWatcher`: scans project directories for session files, notifies on new sessions
- **`watcher.py`** - `TranscriptWatcher`: polls transcript file, parses JSONL entries into `Turn` objects
- **`transcript.py`** - Parses `session.jsonl` files to load conversation history on session start
- **`models.py`** - Data classes: `Turn` (conversation turn), `Session` (groups turns), `TaskState`
- **`tui.py`** - Rich-based TUI: renders sessions, turns, handles keyboard input
- **`alerts.py`** - Pattern matching for dangerous operations (force push, rm -rf, etc.)
- **`summarizer.py`** - Optional LLM summarization of assistant turns via OpenAI-compatible server
- **`cache.py`** - Persistent disk cache for summaries (`~/.cache/betty/summaries.json`)
- **`config.py`** - Configuration management for LLM server settings (supports env vars, config file, defaults)
- **`export.py`** - Export session data to Markdown or JSON formats
- **`mock_session.py`** - Mock session generator for development: creates realistic Claude Code session files for testing and cloud-based development without Claude Code
- **`analyzer.py`** - On-demand LLM analysis: structured analysis (summary, critique, sentiment), multi-level analysis (turn/span/session), multi-source goal extraction, cost tracking
- **`pricing.py`** - Model pricing database and cost estimation utilities for token usage tracking
- **`cli.py`** - Click-based CLI with commands: `config`, `mock`

### Session Discovery

Sessions are discovered by watching `~/.claude/projects/<encoded-cwd>/` where `<encoded-cwd>` is the current directory with `/` replaced by `-` (e.g., `/Users/kai/src/foo` becomes `-Users-kai-src-foo`).

By default, only sessions for the current directory are shown. Use `--global` to watch all projects.

### Threading Model

- Main thread: TUI rendering loop
- Project watcher thread: Directory scanning (daemon)
- Watcher threads: Transcript file polling (daemon, one per session)
- Summarizer threads: ThreadPoolExecutor for async LLM calls (2 workers)
- Analyzer threads: ThreadPoolExecutor for async LLM analysis (1 worker)
- All shared state in `EventStore` protected by `threading.Lock`

### Optional Features

**LLM Summarization**: Assistant turns can be summarized using any OpenAI-compatible local LLM server. Supported options:
- **LM Studio** (recommended for Mac): `http://localhost:1234/v1`
- **Ollama**: `http://localhost:11434/v1`
- **vLLM**: `http://localhost:8008/v1` (default)

Configuration priority:
1. Environment variables: `BETTY_LLM_API_BASE`, `BETTY_LLM_MODEL`
2. Config file: `~/.betty/config.toml`
3. Hardcoded defaults (vLLM)

Use `betty config` to set up your LLM server. Summaries are cached to disk and persist across sessions. The feature gracefully degrades if the server is unavailable.

**On-Demand Analysis**: Turns, spans, and sessions can be analyzed on-demand using any OpenAI-compatible local LLM server. The analyzer provides structured analysis (summary, critique, sentiment) with multi-source goal extraction (first user message, GitHub issues, plan files, active tasks). Triggered via `A` keybinding in the TUI, with `[`/`]` to zoom between turn, span, and session levels. Analysis results are cached to disk and persist across sessions. The feature gracefully degrades if the server is unavailable.

**Export**: Sessions can be exported to Markdown or JSON via the `export.py` module (used by TUI keyboard shortcuts).
