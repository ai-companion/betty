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
| `claude-companion mock` | Generate mock sessions for development |

## Options

| Option | Description |
|--------|-------------|
| `--port`, `-p` | HTTP server port (default: 5432) |
| `--version`, `-v` | Show version |

## LLM Summarization (Optional)

The companion can summarize assistant responses using either **local LLMs** (free) or **API providers** (paid).

### Available Providers

**Local (Free):**
- **LM Studio** - Recommended for Mac
- **Ollama** - Easy setup with Homebrew
- **vLLM** - High-performance server

**API (Paid):**
- **OpenRouter** - Access to multiple models
- **OpenAI** - GPT models
- **Anthropic** - Claude models

### Quick Setup

```bash
# View all available providers
claude-companion config

# Check current configuration
claude-companion config --show
```

### Local Providers

#### LM Studio (Recommended for Mac)

```bash
# 1. Start LM Studio and load a model
# 2. Enable the local server in LM Studio (default: http://localhost:1234)
# 3. Configure the companion
claude-companion config --preset lm-studio

# 4. Start the companion
claude-companion
```

#### Ollama

```bash
# 1. Install and start Ollama
brew install ollama
ollama serve

# 2. Pull a model
ollama pull qwen2.5:7b

# 3. Configure the companion
claude-companion config --preset ollama

# 4. Start the companion
claude-companion
```

### API Providers

#### OpenRouter (Multiple Models)

```bash
# 1. Get API key from https://openrouter.ai/keys
# 2. Configure the companion
claude-companion config --preset openrouter

# 3. Set API key and start
export OPENROUTER_API_KEY="sk-or-v1-..."
claude-companion
```

**Cost:** ~$0.15-0.60 per 1M tokens (very cheap for summarization)

#### OpenAI

```bash
# 1. Get API key from https://platform.openai.com/api-keys
# 2. Configure the companion
claude-companion config --preset openai

# 3. Set API key and start
export OPENAI_API_KEY="sk-..."
claude-companion
```

**Uses:** gpt-4o-mini model by default

#### Anthropic

```bash
# 1. Get API key from https://console.anthropic.com/
# 2. Configure the companion
claude-companion config --preset anthropic

# 3. Set API key and start
export ANTHROPIC_API_KEY="sk-ant-..."
claude-companion
```

**Uses:** Claude 3.5 Haiku by default

### Switching Between Providers

Simply reconfigure and restart:

```bash
# Switch to OpenRouter
claude-companion config --preset openrouter
export OPENROUTER_API_KEY="sk-or-v1-..."
claude-companion

# Switch back to LM Studio
claude-companion config --preset lm-studio
claude-companion  # No API key needed
```

### Permanent API Key Setup

Add to `~/.zshrc` or `~/.bashrc`:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Then simply:
```bash
claude-companion config --preset openrouter
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

### Cloud-Based Development

For development in environments without Claude Code (GitHub Codespaces, GitPod, remote servers), use the mock session feature:

```bash
# Terminal 1: Start mock session that simulates Claude Code
claude-companion mock --demo

# Terminal 2: Watch the mock session
claude-companion --global
```

The mock command creates realistic Claude Code session files, enabling full development and testing without needing Claude Code installed.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup instructions.

## License

MIT
