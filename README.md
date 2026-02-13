<p align="center">
  <img src="docs/assets/logo.png" alt="Betty" width="120">
</p>

# Betty

A real-time TUI monitor for Claude Code sessions.

## Install

```bash
curl -fsSL https://betty4.sh/install.sh | bash
```

Or directly with uv / pip:

```bash
uvx betty              # run without installing
uv tool install betty  # install permanently
pip install betty      # with pip
```

## Use

```bash
# Start betty
betty

# In another terminal, run Claude Code as usual
claude
```

The companion auto-detects your session. No hooks or configuration needed.

## Options

| Flag | Description |
|------|-------------|
| `--global`, `-g` | Watch all projects |
| `--worktree`, `-w` | Watch across git worktrees |
| `--style` | UI style (`rich` or `claude-code`) |
| `--version`, `-v` | Show version |

## Commands

| Command | Description |
|---------|-------------|
| `config` | Configure LLM summarization and UI settings |
| `mock --demo` | Generate mock sessions for development |

## Keybindings

### Navigation

| Key | Action |
|-----|--------|
| `j/k` | Navigate turns |
| `g/G` | Jump to beginning/end |
| `1-9` | Switch sessions |
| `h/l` | Switch panels (manager expand mode) |

### Display

| Key | Action |
|-----|--------|
| `o` / `Space` / `Enter` | Expand/collapse turn or span |
| `e/c` | Expand/collapse all |
| `f` | Cycle filters (All, Spans, Tools, Read, Write, Edit, Bash) |
| `s/S` | Toggle summaries / Summarize all |

### Views

| Key | Action |
|-----|--------|
| `M` | Toggle manager view |
| `T` | Toggle tasks view |
| `P` | Toggle plan view |
| `I` | Toggle insights (analysis) panel |

### Analysis & Annotations

| Key | Action |
|-----|--------|
| `A` | Analyze selected turn/span/session |
| `[`/`]` | Zoom analysis level (turn / span / session) |
| `n` | Annotate selected turn |
| `a` | Toggle/clear alerts |

### Other

| Key | Action |
|-----|--------|
| `O` | Open PR in browser |
| `x` | Export to Markdown |
| `m` | Edit monitor instructions |
| `?` | Ask about trace |
| `D` | Delete session |
| `Esc` | Close panel / clear selection |
| `q` | Quit |

## License

MIT
