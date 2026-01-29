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

- **`store.py`** - Central `EventStore` class: thread-safe session/turn storage, listener pattern for updates, coordinates watcher and alerts
- **`watcher.py`** - `TranscriptWatcher`: polls transcript file, parses JSONL entries into `Turn` objects
- **`models.py`** - Data classes: `Event` (raw hook data), `Turn` (conversation turn), `Session` (groups turns)
- **`tui.py`** - Rich-based TUI: renders sessions, turns, handles keyboard input
- **`hooks.py`** - Installs/uninstalls curl hooks in `~/.claude/settings.json`
- **`alerts.py`** - Pattern matching for dangerous operations (force push, rm -rf, etc.)

### Hook Design

Only `SessionStart` hook is needed. All conversation content (user messages, assistant text, tool calls) comes from watching the transcript file at `~/.claude/projects/<project>/session.jsonl`.

### Threading Model

- Main thread: TUI rendering loop
- Server thread: Flask HTTP server (daemon)
- Watcher thread: Transcript file polling (daemon)
- All shared state in `EventStore` protected by `threading.Lock`
