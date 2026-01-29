# Claude Companion - Implementation Plan

A Python CLI that supervises Claude Code sessions in real-time via hooks.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code                               │
│  (triggers hooks on SessionStart, PreToolUse, PostToolUse, etc.)│
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP POST (JSON)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Claude Companion                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Flask Server │───▶│ Event Store  │───▶│ Rich TUI Display │  │
│  │ (localhost)  │    │ (in-memory)  │    │ (scrolling log)  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

- **Python 3.10+**
- **click** - CLI framework
- **Flask** - HTTP server for receiving hook events
- **rich** - TUI rendering (Live display, panels, tables)
- **packaging** - `uvx` compatible via pyproject.toml

## Project Structure

```
claude-companion/
├── pyproject.toml              # Package config with entry points
├── README.md                   # Usage documentation
├── PLAN.md                     # This file
├── uv.lock                     # Dependency lock file
└── src/
    └── claude_companion/
        ├── __init__.py         # Package init, version
        ├── cli.py              # Click CLI entry point
        ├── server.py           # Flask HTTP server
        ├── store.py            # In-memory event store
        ├── tui.py              # Rich TUI display
        ├── hooks.py            # Hook installer logic
        └── models.py           # Data models (Event, Session, Turn)
```

## Data Models

### Event (from hooks)
```python
@dataclass
class Event:
    session_id: str
    timestamp: datetime
    event_type: str          # SessionStart, PreToolUse, PostToolUse, Stop, etc.
    tool_name: str | None    # Bash, Write, Read, etc.
    tool_input: dict | None  # Tool parameters
    tool_output: dict | None # Tool result (PostToolUse only)
    model: str | None        # Model name (SessionStart only)
    cwd: str | None          # Working directory
```

### Turn (aggregated view)
```python
@dataclass
class Turn:
    turn_number: int
    role: str                # "user" | "assistant" | "tool"
    content_preview: str     # First ~100 chars
    content_full: str        # Full content
    word_count: int          # Word count as token proxy
    tool_name: str | None    # If tool turn
    timestamp: datetime
```

### Session
```python
@dataclass
class Session:
    session_id: str
    project_path: str
    model: str
    started_at: datetime
    events: list[Event]
    turns: list[Turn]

    @property
    def total_input_words(self) -> int: ...

    @property
    def total_output_words(self) -> int: ...
```

## CLI Commands

### `claude-companion` (default: run)
Start the companion TUI + server.

```bash
# Start on default port
claude-companion

# Start on custom port
claude-companion --port 5433

# Verbose mode (show all events)
claude-companion --verbose
```

### `claude-companion install`
Configure Claude Code hooks in `~/.claude/settings.json`.

```bash
# Install hooks
claude-companion install

# Uninstall hooks
claude-companion uninstall

# Show current hook status
claude-companion status
```

## Hook Configuration

The `install` command will add/update `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "curl -s -X POST http://localhost:5432/event -H 'Content-Type: application/json' -d \"$(cat)\""
      }]
    }],
    "PreToolUse": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "curl -s -X POST http://localhost:5432/event -H 'Content-Type: application/json' -d \"$(cat)\""
      }]
    }],
    "PostToolUse": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "curl -s -X POST http://localhost:5432/event -H 'Content-Type: application/json' -d \"$(cat)\""
      }]
    }],
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "curl -s -X POST http://localhost:5432/event -H 'Content-Type: application/json' -d \"$(cat)\""
      }]
    }],
    "SessionEnd": [{
      "hooks": [{
        "type": "command",
        "command": "curl -s -X POST http://localhost:5432/event -H 'Content-Type: application/json' -d \"$(cat)\""
      }]
    }]
  }
}
```

## HTTP API Endpoints

### `POST /event`
Receive hook events from Claude Code.

**Request:** Raw JSON from hook stdin
**Response:** `{"status": "ok"}`

### `GET /health`
Health check endpoint.

**Response:** `{"status": "healthy", "sessions": 2}`

## TUI Layout

```
╭─ Claude Companion ──────────────────────────────────────────────╮
│ Sessions: [1] project-a (active)  [2] project-b                 │
│ Model: claude-sonnet-4  │  Words: ↓1,234  ↑5,678  │  Turns: 12  │
╰─────────────────────────────────────────────────────────────────╯

╭─ Turn 1 ─ User ─────────────────────────────── 45 words ────────╮
│ Please help me refactor the authentication module to use JWT... │
│ [+] Expand                                                      │
╰─────────────────────────────────────────────────────────────────╯

╭─ Turn 2 ─ Assistant ────────────────────────── 234 words ───────╮
│ I'll help you refactor the authentication module. Let me first  │
│ explore the current implementation...                           │
│ [+] Expand                                                      │
╰─────────────────────────────────────────────────────────────────╯

╭─ Turn 2 ─ Tool: Read ───────────────────────────────────────────╮
│ 📄 src/auth/handler.py                                          │
╰─────────────────────────────────────────────────────────────────╯

╭─ Turn 2 ─ Tool: Edit ───────────────────────────────────────────╮
│ ✏️  src/auth/handler.py (+15, -8)                                │
╰─────────────────────────────────────────────────────────────────╯

───────────────────────────────────────────────────────────────────
 [1-9] Switch session  [q] Quit  [e] Expand all  [c] Collapse all
```

### Visual Styling

| Element | Style |
|---------|-------|
| User message | Blue border, bold "User" label |
| Assistant message | Green border, bold "Assistant" label |
| Tool (Read) | Dim cyan, file icon 📄 |
| Tool (Write/Edit) | Yellow, pencil icon ✏️ |
| Tool (Bash) | Magenta, terminal icon 💻 |
| Tool (error) | Red border, warning icon ⚠️ |
| Word counts | Dim text, arrows ↓↑ for in/out |

## Implementation Phases

### Phase 1: Core Infrastructure (MVP) ✅ COMPLETE
1. **Project setup** ✅
   - pyproject.toml with dependencies and entry points
   - Package structure with src layout
   - uvx compatible

2. **Data models** ✅
   - Event, Turn, Session dataclasses
   - Word counting utility

3. **Flask server** ✅
   - `/event` endpoint to receive hooks
   - `/health` endpoint
   - Thread-safe event store

4. **Basic TUI** ✅
   - Rich Live display
   - Session list header
   - Scrolling turn log
   - Word count stats
   - Tool-specific icons and colors

5. **CLI** ✅
   - `claude-companion` to start server + TUI
   - `claude-companion install` to configure hooks
   - `claude-companion uninstall` to remove hooks
   - `claude-companion status` to check hook status

### Phase 2: Enhanced Display (In Progress)
- Expandable/collapsible content ✅
- Keyboard navigation ✅
  - Arrow keys (↑↓) to navigate turns
  - Enter to expand/collapse selected turn
  - e/c to expand/collapse all
  - g/G to go to beginning/end
  - Scroll indicators when content overflows
- Search/filter turns
- Export session log

### Phase 3: Analysis & Alerts (Future)
- Intent tracking
- Anomaly detection
- System notifications
- Red text warnings for issues

## Key Implementation Details

### Threading Model
- Flask server runs in background thread
- Rich TUI runs in main thread
- Thread-safe queue for events between server and TUI

### Hook Installation
- Read existing `~/.claude/settings.json`
- Merge our hooks (preserve existing hooks)
- Write back atomically
- Validate JSON before writing

### Session Detection
- New session detected on `SessionStart` event
- Session ends on `SessionEnd` event
- Auto-select most recent active session

### Content Extraction
- For user messages: Extract from transcript file using `transcript_path`
- For tool calls: Use `tool_input` from hook data
- For tool results: Use `tool_output` from PostToolUse

### Word Counting
```python
def count_words(text: str) -> int:
    return len(text.split())
```

## Dependencies

```toml
[project]
dependencies = [
    "click>=8.0",
    "flask>=3.0",
    "rich>=13.0",
]
```

## Testing Strategy

1. **Unit tests**: Models, word counting, hook config merging
2. **Integration tests**: Server endpoints, event processing
3. **Manual testing**: Run alongside Claude Code

## Open Questions / Future Considerations

1. **Persistence**: Should we persist session data to disk?
2. **Multiple instances**: What if companion is already running?
3. **Hook failures**: How to handle if companion server is down?
4. **Large content**: Truncation strategy for very long outputs?

---

## Summary

This plan creates a `uvx`-compatible Python CLI that:
1. Installs Claude Code hooks to send events via HTTP
2. Runs a Flask server to receive those events
3. Displays a Rich TUI with session info, turns, and word counts
4. Supports multiple concurrent sessions with number-key switching

The MVP focuses on monitoring and display. Future phases add analysis and alerts.
