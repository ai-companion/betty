# Claude Companion

A CLI supervisor for Claude Code sessions. Monitor your Claude Code sessions in real-time with a rich TUI display.

## Installation

```bash
# Using uvx (recommended)
uvx claude-companion

# Or install with pip
pip install claude-companion
```

## Quick Start

1. **Install the hooks** (one-time setup):
   ```bash
   claude-companion install
   ```

2. **Start the companion** in one terminal:
   ```bash
   claude-companion
   ```

3. **Run Claude Code** in another terminal (side by side):
   ```bash
   claude
   ```

The companion will automatically detect and display your Claude Code session.

## Commands

| Command | Description |
|---------|-------------|
| `claude-companion` | Start the TUI monitor |
| `claude-companion install` | Install Claude Code hooks |
| `claude-companion uninstall` | Remove Claude Code hooks |
| `claude-companion status` | Check hook installation status |

## Options

| Option | Description |
|--------|-------------|
| `--port`, `-p` | HTTP server port (default: 5432) |
| `--version`, `-v` | Show version |

## TUI Keybindings

| Key | Action |
|-----|--------|
| `1-9` | Switch between sessions |
| `q` | Quit |
| `r` | Refresh display |

## How It Works

Claude Companion uses Claude Code's hook system to receive real-time events:

1. **Hooks** are installed in `~/.claude/settings.json`
2. When Claude Code runs, hooks send events via HTTP to the companion
3. The companion displays events in a rich TUI with:
   - Session selector (multiple concurrent sessions supported)
   - Word counts (input/output) as token proxy
   - Color-coded turns (user=blue, assistant=green, tools=various)

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/xukai92/claude-companion
cd claude-companion
pip install -e .

# Run
claude-companion
```

## License

MIT
