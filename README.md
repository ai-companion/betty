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
uvx --from betty-cli betty  # run without installing
uv tool install betty-cli   # install permanently
pip install betty-cli       # with pip
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

### Agent

| Key | Action |
|-----|--------|
| `B` | Toggle agent panel (closed / full / compact) |
| `?` | Ask Betty a question about the session |

### Other

| Key | Action |
|-----|--------|
| `O` | Open PR in browser |
| `x` | Export to Markdown |
| `m` | Edit monitor instructions |
| `D` | Delete session |
| `Esc` | Close panel / clear selection |
| `q` | Quit |

## Betty Agent

Betty Agent is a continuous session observer that tracks what Claude Code is doing and flags problems in real time. It combines heuristic detectors with optional LLM-powered narrative and drift detection.

### Enable

```bash
betty config --agent
```

Or set the environment variable `BETTY_AGENT_ENABLED=true`.

### What it does

- **Goal tracking** — extracts the session goal and current objective, updating as the user gives new instructions
- **Progress assessment** — classifies sessions as `on_track`, `stalled`, or `spinning` using error rates, retry patterns, and tool diversity
- **Error spike detection** — warns when error rate exceeds 40% in recent tool calls
- **Retry loop detection** — flags when the same tool is called 3+ times consecutively
- **Stall detection** — notices gaps of 2+ minutes between turns
- **File change tracking** — logs Read/Write/Edit operations with line counts
- **Milestones** — marks every 10th tool call and 5th user message
- **LLM narrative** (optional) — generates a 2-3 sentence situation report describing current activity
- **Goal drift detection** (optional) — compares recent activity against the session goal and warns if the assistant has gone off track
- **Ask Betty** — press `?` to ask a natural-language question about the session; Betty answers citing turn numbers and file paths

### Configuration

The agent uses your existing LLM configuration (set via `betty config`). LLM features (narrative, drift detection, goal determination, Ask Betty) require a configured LLM provider. Heuristic detectors work without one.

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `false` | Enable the agent (opt-in) |
| `update_interval` | `5` | Minimum turns between LLM updates |
| `max_observations` | `50` | Max observations kept per session |

Observations and reports are cached to disk (`~/.cache/betty/`) and persist across restarts.

## License

MIT
