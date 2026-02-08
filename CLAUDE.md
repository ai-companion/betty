# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claude Companion is a real-time TUI monitor for Claude Code sessions. It watches project directories for session files and monitors transcript files directly for all conversation activity. No hooks required - just run `claude-companion`.

## Development Commands

```bash
# Install in dev mode
pip install -e .

# Run the companion
claude-companion                # Watch current directory's sessions
claude-companion --global       # Watch all projects

# Run with uv (no install needed)
uv run claude-companion

# LLM configuration (for summarization)
claude-companion config --show                    # Show current config
claude-companion config --llm-preset lm-studio    # Use LM Studio preset
claude-companion config --llm-preset ollama       # Use Ollama preset
claude-companion config --url URL --model MODEL   # Custom configuration

# Test imports
uv run python -c "from claude_companion import tui, store, models; print('OK')"
```

## Architecture

### Data Flow

1. **ProjectWatcher** (`project_watcher.py`) scans `~/.claude/projects/<encoded-cwd>/` for `.jsonl` session files
2. When a session file is discovered, `EventStore` creates a new `Session` and starts a `TranscriptWatcher`
3. `TranscriptWatcher` (`watcher.py`) polls the transcript JSONL file for new entries
4. New turns are parsed and added to the session, triggering TUI updates

### Key Components

- **`store.py`** - Central `EventStore` class: thread-safe session/turn storage, coordinates watchers/alerts/summarizer
- **`project_watcher.py`** - `ProjectWatcher`: scans project directories for session files, notifies on new sessions
- **`watcher.py`** - `TranscriptWatcher`: polls transcript file, parses JSONL entries into `Turn` objects
- **`transcript.py`** - Parses `session.jsonl` files to load conversation history on session start
- **`models.py`** - Data classes: `Turn` (conversation turn), `Session` (groups turns), `TaskState`
- **`tui.py`** - Rich-based TUI: renders sessions, turns, handles keyboard input
- **`alerts.py`** - Pattern matching for dangerous operations (force push, rm -rf, etc.)
- **`summarizer.py`** - Optional LLM summarization of assistant turns via OpenAI-compatible server
- **`cache.py`** - Persistent disk cache for summaries (`~/.cache/claude-companion/summaries.json`)
- **`config.py`** - Configuration management for LLM server settings (supports env vars, config file, defaults)
- **`export.py`** - Export session data to Markdown or JSON formats
- **`mock_session.py`** - Mock session generator for development: creates realistic Claude Code session files for testing and cloud-based development without Claude Code
- **`cli.py`** - Click-based CLI with commands: `config`, `mock`

### Session Discovery

Sessions are discovered by watching `~/.claude/projects/<encoded-cwd>/` where `<encoded-cwd>` is the current directory with `/` replaced by `-` (e.g., `/Users/kai/src/foo` becomes `-Users-kai-src-foo`).

By default, only sessions for the current directory are shown. Use `--global` to watch all projects.

### Threading Model

- Main thread: TUI rendering loop
- Project watcher thread: Directory scanning (daemon)
- Watcher threads: Transcript file polling (daemon, one per session)
- Summarizer threads: ThreadPoolExecutor for async LLM calls (2 workers)
- All shared state in `EventStore` protected by `threading.Lock`

### Optional Features

**LLM Summarization**: Assistant turns can be summarized using any OpenAI-compatible local LLM server. Supported options:
- **LM Studio** (recommended for Mac): `http://localhost:1234/v1`
- **Ollama**: `http://localhost:11434/v1`
- **vLLM**: `http://localhost:8008/v1` (default)

Configuration priority:
1. Environment variables: `CLAUDE_COMPANION_LLM_URL`, `CLAUDE_COMPANION_LLM_MODEL`
2. Config file: `~/.claude-companion/config.toml`
3. Hardcoded defaults (vLLM)

Use `claude-companion config` to set up your LLM server. Summaries are cached to disk and persist across sessions. The feature gracefully degrades if the server is unavailable.

**Export**: Sessions can be exported to Markdown or JSON via the `export.py` module (used by TUI keyboard shortcuts).
