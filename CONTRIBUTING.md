# Contributing to Claude Companion

This guide helps you set up a development environment for Claude Companion, including cloud-based development without access to a local Claude Code installation.

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setting Up the Development Environment

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/ai-companion/claude-companion
cd claude-companion

# Install dependencies and run (no explicit install needed)
uv run claude-companion

# Or install in editable mode
uv pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/ai-companion/claude-companion
cd claude-companion

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### Option 3: Using pip with requirements

```bash
# Install dependencies directly
pip install click filelock requests rich openai anthropic textual

# Install package in editable mode
pip install -e .
```

## Development Without Claude Code (Cloud-Based Development)

When developing in cloud environments (GitHub Codespaces, GitPod, remote servers) where you don't have access to Claude Code, use the built-in mock session feature to simulate Claude Code sessions.

### Quick Start with Mock Sessions

```bash
# Start a mock session that simulates a Claude Code conversation
uv run claude-companion mock

# In another terminal, start the companion to watch it
uv run claude-companion
```

### Mock Session Commands

```bash
# Start interactive mock session (adds messages on keypress)
uv run claude-companion mock

# Run a demo that auto-plays a sample conversation
uv run claude-companion mock --demo

# Create a mock session for a specific project path
uv run claude-companion mock --project /path/to/project

# Set custom delay between auto-messages in demo mode
uv run claude-companion mock --demo --delay 2.0
```

### How Mock Sessions Work

The mock session tool:
1. Creates a project directory in `~/.claude/projects/` (same location Claude Code uses)
2. Generates a `.jsonl` transcript file with the correct format
3. Simulates realistic conversation patterns including:
   - User messages
   - Assistant text responses
   - Tool calls (Read, Write, Edit, Bash, Grep, Glob)
   - Multi-turn conversations

This allows you to:
- Test TUI rendering and navigation
- Develop new features without needing Claude Code
- Debug transcript parsing logic
- Test the watcher and session discovery

### Manual Mock Session Creation

For advanced testing, you can create mock session files directly:

```python
import json
from pathlib import Path
from datetime import datetime, timezone

# Create project directory
project_dir = Path.home() / ".claude" / "projects" / "-tmp-my-test-project"
project_dir.mkdir(parents=True, exist_ok=True)

# Create session file
session_file = project_dir / "test-session-001.jsonl"

# Helper for UTC timestamp
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# Write mock entries
entries = [
    {
        "type": "user",
        "timestamp": utc_timestamp(),
        "message": {"content": "Hello, can you help me?"}
    },
    {
        "type": "assistant",
        "timestamp": utc_timestamp(),
        "message": {
            "content": [
                {"type": "text", "text": "Of course! How can I help you today?"}
            ]
        }
    }
]

with open(session_file, "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")
```

## Running Tests

```bash
# Verify imports work correctly
uv run python -c "from claude_companion import tui, store, models; print('OK')"

# Run the TUI briefly to check for import errors
timeout 2 uv run claude-companion || true
```

## Code Structure

See `CLAUDE.md` for detailed architecture documentation.

Key directories:
- `src/claude_companion/` - Main package source code
- `src/claude_companion/cli.py` - CLI entry point and commands
- `src/claude_companion/mock_session.py` - Mock session generator for development

## Working with GitHub Issues and PRs

In cloud environments where the `gh` CLI isn't available, you can use the GitHub API directly with the `GITHUB_TOKEN` environment variable:

```bash
# View an issue
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/ai-companion/claude-companion/issues/73

# List open issues
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/ai-companion/claude-companion/issues?state=open

# Create a pull request
curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/ai-companion/claude-companion/pulls \
  -d '{"title":"My PR","head":"my-branch","base":"main","body":"Description"}'
```

Make sure your `GITHUB_TOKEN` has the appropriate scopes (repo access for private repos, or public_repo for public).

## Submitting Changes

1. Create a feature branch
2. Make your changes
3. Test with mock sessions if you don't have Claude Code access
4. Submit a pull request

## Troubleshooting

### "No sessions found"

If the companion shows no sessions:
- Ensure Claude Code has been run at least once (creates `~/.claude/projects/`)
- Or use `claude-companion mock` to create a mock session
- Check the project path with `--global` to see all projects

### Import errors

If you get import errors:
```bash
# Reinstall dependencies
uv pip install -e . --force-reinstall

# Or with pip
pip install -e . --force-reinstall
```

### Mock session not appearing

- Ensure the mock session is running in another terminal
- The companion polls every 1 second for new sessions
- Check that `~/.claude/projects/` exists and is readable
