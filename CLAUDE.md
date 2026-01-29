# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claude Companion is a real-time TUI monitor for Claude Code sessions. It uses Claude Code's hook system to detect session starts, then watches the transcript file directly for all conversation activity.

## Development Commands

```bash
# Install in dev mode
pip install -e .

# Run the companion
claude-companion

# Run with uv (no install needed)
uv run claude-companion

# Hook management
claude-companion install        # Install hooks to ~/.claude/settings.json
claude-companion uninstall      # Remove hooks
claude-companion status         # Check hook installation

# LLM configuration (for summarization)
claude-companion config --show                    # Show current config
claude-companion config --preset lm-studio        # Use LM Studio preset
claude-companion config --preset ollama           # Use Ollama preset
claude-companion config --url URL --model MODEL   # Custom configuration

# Test imports
uv run python -c "from claude_companion import tui, store, models; print('OK')"
```

## Architecture

### Data Flow

1. **SessionStart hook** (`hooks.py`) fires when Claude Code starts/resumes a session
2. Hook sends HTTP POST to Flask server (`server.py`) with `transcript_path`
3. `EventStore` (`store.py`) receives event and starts `TranscriptWatcher`
4. `TranscriptWatcher` (`watcher.py`) polls the transcript JSONL file for new entries
5. New turns are parsed and added to the session, triggering TUI updates

### Key Components

- **`store.py`** - Central `EventStore` class: thread-safe session/turn storage, listener pattern for updates, coordinates watcher/alerts/summarizer
- **`watcher.py`** - `TranscriptWatcher`: polls transcript file, parses JSONL entries into `Turn` objects
- **`transcript.py`** - Parses `session.jsonl` files to load conversation history on session start
- **`models.py`** - Data classes: `Event` (raw hook data), `Turn` (conversation turn), `Session` (groups turns)
- **`tui.py`** - Rich-based TUI: renders sessions, turns, handles keyboard input
- **`hooks.py`** - Installs/uninstalls curl hooks in `~/.claude/settings.json`
- **`alerts.py`** - Pattern matching for dangerous operations (force push, rm -rf, etc.)
- **`summarizer.py`** - Optional LLM summarization of assistant turns via OpenAI-compatible server
- **`cache.py`** - Persistent disk cache for summaries (`~/.cache/claude-companion/summaries.json`)
- **`config.py`** - Configuration management for LLM server settings (supports env vars, config file, defaults)
- **`export.py`** - Export session data to Markdown or JSON formats
- **`server.py`** - Flask HTTP server that receives hook events
- **`cli.py`** - Click-based CLI with commands: `install`, `uninstall`, `status`, `config`

### Hook Design

Only `SessionStart` hook is needed. All conversation content (user messages, assistant text, tool calls) comes from watching the transcript file at `~/.claude/projects/<project>/session.jsonl`.

### Threading Model

- Main thread: TUI rendering loop
- Server thread: Flask HTTP server (daemon)
- Watcher thread: Transcript file polling (daemon)
- Summarizer threads: ThreadPoolExecutor for async LLM calls (2 workers)
- All shared state in `EventStore` protected by `threading.Lock`

### Optional Features

**LLM Summarization**: Assistant turns can be summarized using any OpenAI-compatible local LLM server. Supported options:
- **LM Studio** (recommended for Mac): `http://localhost:1234/v1`
- **Ollama**: `http://localhost:11434/v1`
- **vLLM**: `http://localhost:8008/v1` (default)

Configuration priority:
1. Environment variables: `CLAUDE_COMPANION_LLM_URL`, `CLAUDE_COMPANION_LLM_MODEL`
2. Config file: `~/.claude-companion/config.json`
3. Hardcoded defaults (vLLM)

Use `claude-companion config` to set up your LLM server. Summaries are cached to disk and persist across sessions. The feature gracefully degrades if the server is unavailable.

**Export**: Sessions can be exported to Markdown or JSON via the `export.py` module (used by TUI keyboard shortcuts).
