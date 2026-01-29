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
| `claude-companion config` | Configure LLM server for summarization |

## Options

| Option | Description |
|--------|-------------|
| `--port`, `-p` | HTTP server port (default: 5432) |
| `--version`, `-v` | Show version |

## LLM Summarization (Optional)

The companion can summarize assistant responses using a local LLM. Supports:

- **LM Studio** (recommended for Mac)
- **Ollama**
- **vLLM**

### Setup with LM Studio

```bash
# 1. Start LM Studio and load a model
# 2. Enable the local server in LM Studio (default: http://localhost:1234)
# 3. Configure the companion
claude-companion config --preset lm-studio

# Or use custom settings
claude-companion config --url http://localhost:1234/v1 --model your-model-name
```

### Setup with Ollama

```bash
# 1. Install Ollama: brew install ollama
# 2. Start Ollama: ollama serve
# 3. Pull a model: ollama pull qwen2.5:7b
# 4. Configure the companion
claude-companion config --preset ollama
```

### Environment Variables

You can also configure via environment variables:

```bash
export CLAUDE_COMPANION_LLM_URL="http://localhost:1234/v1"
export CLAUDE_COMPANION_LLM_MODEL="openai/gpt-oss-20b"
claude-companion
```

## TUI Keybindings

| Key | Action |
|-----|--------|
| `1-9` | Switch between sessions |
| `j/k` | Navigate turns |
| `o` | Expand/collapse turn |
| `s` | Toggle summary mode |
| `f` | Filter turns |
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
