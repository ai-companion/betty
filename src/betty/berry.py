"""Berry character module — animated blueberry protagonist for the Berry UI style.

Ports the sprite frames from noisy-channels/src/sprites/berry.ts and adds
session-level state inference, speech bubble rendering, activity feed building,
and tool inventory collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Session, Turn

# ============================================================
# Constants
# ============================================================

FRAME_WIDTH = 18
FRAME_HEIGHT = 7

# Tool power-up icons and labels
TOOL_POWERUPS: dict[str, tuple[str, str]] = {
    "Read": ("📖", "Read"),
    "Write": ("✏️", "Write"),
    "Edit": ("✏️", "Edit"),
    "Bash": ("💻", "Bash"),
    "Glob": ("🔍", "Glob"),
    "Grep": ("🔍", "Grep"),
    "Task": ("⚙️", "Task"),
    "TaskCreate": ("⚙️", "Task"),
    "TaskUpdate": ("⚙️", "Task"),
    "TaskGet": ("⚙️", "Task"),
    "TaskList": ("⚙️", "Task"),
    "WebFetch": ("🌐", "Fetch"),
    "WebSearch": ("🌐", "Search"),
    "Agent": ("🤖", "Agent"),
    "NotebookEdit": ("📓", "Notebook"),
}

# Berry color palette (Rich markup styles)
BERRY_COLORS = {
    "primary": "#9333ea",       # Purple
    "secondary": "#ec4899",     # Pink
    "accent": "#a855f7",        # Light purple
    "bg": "#1a0a2e",            # Dark purple bg
    "text": "#e9d5ff",          # Light purple text
    "dim": "#6b21a8",           # Dim purple
    "ground": "#4c1d95",        # Ground color
    "speech_border": "#c084fc", # Speech bubble border
}


def get_tool_powerup(tool_name: str | None) -> tuple[str, str]:
    """Get (icon, label) for a tool name."""
    if not tool_name:
        return ("🔧", "Tool")
    return TOOL_POWERUPS.get(tool_name, ("🔧", tool_name))


# ============================================================
# Sprite Frames — 6 states × 4 frames = 24 frames
# Each frame is a list of 7 strings, each 18 chars wide
# ============================================================

def _pad(line: str, width: int = FRAME_WIDTH) -> str:
    """Pad a line to exact width (accounting for wide Unicode chars)."""
    # Simple length-based padding; works for most terminal emulators
    if len(line) >= width:
        return line
    return line + " " * (width - len(line))


def _frame(lines: list[str]) -> list[str]:
    """Normalize a frame to exactly FRAME_HEIGHT lines, each FRAME_WIDTH wide."""
    padded = [_pad(line) for line in lines]
    while len(padded) < FRAME_HEIGHT:
        padded.append(" " * FRAME_WIDTH)
    return padded[:FRAME_HEIGHT]


# fmt: off
BERRY_SPRITE_FRAMES: dict[str, list[list[str]]] = {
    "idle": [
        _frame([
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◕   ◕ │   ",
            "    │   ‿   │   ",
            "    ╰──┬─┬──╯   ",
            "       │ │      ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "                 ",
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◕   ◕ │   ",
            "    │   ‿   │   ",
            "    ╰──┬─┬──╯   ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ─   ─ │   ",
            "    │   ‿   │   ",
            "    ╰──┬─┬──╯   ",
            "       │ │      ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◕   ◕ │   ",
            "    │   ‿   │   ",
            "    ╰──┬─┬──╯   ",
            "       │ │      ",
            "      ╶┘ └╴     ",
        ]),
    ],
    "thinking": [
        _frame([
            "             ·   ",
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◔   ◔ │   ",
            "   ╭│   ─   │   ",
            "   ╰╰──┬─┬──╯   ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "           · ·   ",
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◔   ◔ │   ",
            "   ╭│   ─   │   ",
            "   ╰╰──┬─┬──╯   ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "         · · ·   ",
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◔   ◔ │   ",
            "   ╭│   ─   │   ",
            "   ╰╰──┬─┬──╯   ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "           ·     ",
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◑   ◑ │   ",
            "   ╭│   ─   │   ",
            "   ╰╰──┬─┬──╯   ",
            "      ╶┘ └╴     ",
        ]),
    ],
    "working": [
        _frame([
            "      ╭───╮  ⚡  ",
            "    ╭─┤ ♠ ├─╮╱  ",
            "    │ ●   ● │   ",
            "    │   ▽   │   ",
            "    ╰──┬─┬──╯   ",
            "       │ │      ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "    ∗ ╭───╮ ∗   ",
            "   ╲╭─┤ ♠ ├─╮╱  ",
            "    │ ◉   ◉ │   ",
            "    │   ▽   │   ",
            "    ╰──┬─┬──╯   ",
            "       │ │      ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "  ⚡  ╭───╮     ",
            "   ╲╭─┤ ♠ ├─╮   ",
            "    │ ●   ● │   ",
            "    │   ▽   │   ",
            "    ╰──┬─┬──╯   ",
            "       │ │      ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "    ∗ ╭───╮  ∗  ",
            "   ╲╭─┤ ♠ ├─╮╱  ",
            "    │ ◉   ◉ │   ",
            "    │   △   │   ",
            "    ╰──┬─┬──╯   ",
            "      ╶┘ └╴     ",
            "                 ",
        ]),
    ],
    "celebrating": [
        _frame([
            "   \\★╭───╮★/   ",
            "    ╭┤ ♠ ├╮    ",
            "    │ ◕ ◕ │    ",
            "    │  ▽  │    ",
            "    ╰─────╯    ",
            "     ╱   ╲     ",
            "                ",
        ]),
        _frame([
            "  ✦\\★╭───╮★/✦  ",
            "    ╭┤ ♠ ├╮    ",
            "    │ ◕ ◕ │    ",
            "    │  ▽  │    ",
            "    ╰─────╯    ",
            "                ",
            "                ",
        ]),
        _frame([
            " ✦  \\╭───╮/  ✦ ",
            "    ╭┤ ♠ ├╮    ",
            "    │ ◕ ◕ │    ",
            "    │  ‿  │    ",
            "    ╰──┬┬─╯    ",
            "       ││      ",
            "      ╶┘└╴     ",
        ]),
        _frame([
            "    ★       ★   ",
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◕   ◕ │   ",
            "    │   ▽   │   ",
            "    ╰──┬─┬──╯   ",
            "      ╶┘ └╴     ",
        ]),
    ],
    "confused": [
        _frame([
            "        ？       ",
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◑   ◐ │   ",
            "    │   ∿   │   ",
            "    ╰──┬─┬──╯   ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "       ？！      ",
            "      ╭───╮     ",
            "     ╭┤ ♠ ├╮    ",
            "     │◑   ◐│    ",
            "     │  ∿  │    ",
            "     ╰─┬─┬─╯    ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "         ～      ",
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◐   ◑ │   ",
            "    │   ∿   │   ",
            "    ╰──┬─┬──╯   ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "       ❓       ",
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◑   ◐ │   ",
            "    │   ○   │   ",
            "    ╰──┬─┬──╯   ",
            "      ╶┘ └╴     ",
        ]),
    ],
    "talking": [
        _frame([
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮◁  ",
            "    │ ◕   ◕ │   ",
            "    │   ○   │   ",
            "    ╰──┬─┬──╯   ",
            "       │ │      ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮ ◁ ",
            "    │ ◕   ◕ │   ",
            "    │   ‿   │   ",
            "    ╰──┬─┬──╯   ",
            "       │ │      ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮◁  ",
            "    │ ◕   ◕ │   ",
            "    │   ◯   │   ",
            "    ╰──┬─┬──╯   ",
            "       │ │      ",
            "      ╶┘ └╴     ",
        ]),
        _frame([
            "      ╭───╮     ",
            "    ╭─┤ ♠ ├─╮   ",
            "    │ ◕   ◕ │   ",
            "    │   ‿   │   ",
            "    ╰──┬─┬──╯   ",
            "       │ │      ",
            "      ╶┘ └╴     ",
        ]),
    ],
}
# fmt: on


def get_sprite_frame(state: str, tick: int, ticks_per_frame: int = 1) -> list[str]:
    """Return the current sprite frame lines for the given state and tick.

    Args:
        state: One of idle/thinking/working/celebrating/confused/talking.
        tick: Monotonically increasing tick counter.
        ticks_per_frame: How many ticks per animation frame (default 1).

    Returns:
        List of FRAME_HEIGHT strings, each FRAME_WIDTH chars wide.
    """
    frames = BERRY_SPRITE_FRAMES.get(state, BERRY_SPRITE_FRAMES["idle"])
    frame_index = (tick // ticks_per_frame) % len(frames)
    return frames[frame_index]


# ============================================================
# State Inference
# ============================================================

def infer_berry_state(turn: "Turn") -> str:
    """Infer Berry state from a single turn (used for per-span rendering)."""
    if turn.role == "user":
        return "thinking"
    elif turn.role == "tool":
        return "working"
    elif turn.role == "assistant":
        return "talking"
    return "idle"


def infer_session_berry_state(session: "Session") -> str:
    """Infer Berry state from the overall session state.

    Returns one of: idle, thinking, working, talking, confused, celebrating.
    """
    if not session.turns:
        return "idle"

    last_turn = session.turns[-1]

    # Check for recent errors in tool output
    for turn in reversed(session.turns[-5:]):
        if turn.role == "tool" and turn.content_preview:
            lower = turn.content_preview.lower()
            if any(kw in lower for kw in ("error", "failed", "traceback", "exception")):
                return "confused"

    if last_turn.role == "user":
        return "thinking"
    elif last_turn.role == "tool":
        return "working"
    elif last_turn.role == "assistant":
        # If session seems done (no activity for a while), celebrate
        if not session.active:
            return "celebrating"
        return "talking"

    return "idle"


# ============================================================
# Speech Bubble
# ============================================================

def _word_wrap(text: str, max_width: int) -> list[str]:
    """Simple word-wrap into lines of at most max_width chars."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        if current_len + len(word) + (1 if current else 0) > max_width:
            if current:
                lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + (1 if len(current) > 1 else 0)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def render_speech_bubble(text: str, max_width: int = 28) -> list[str]:
    """Render a speech bubble with Unicode box drawing.

    Returns lines like:
        ╭──────────────╮
        │ hello world  │
        ╰─▶────────────╯
    """
    if not text:
        return []

    # Truncate very long text
    if len(text) > max_width * 4:
        text = text[: max_width * 4 - 3] + "..."

    wrapped = _word_wrap(text, max_width - 2)  # -2 for padding

    # Calculate actual content width
    content_width = max(len(line) for line in wrapped)
    content_width = max(content_width, 4)  # minimum width

    lines: list[str] = []
    lines.append("╭" + "─" * (content_width + 2) + "╮")
    for line in wrapped:
        padded = line + " " * (content_width - len(line))
        lines.append(f"│ {padded} │")
    lines.append("╰─▶" + "─" * (content_width - 1) + "╯")

    return lines


# ============================================================
# Activity Feed
# ============================================================

@dataclass
class ActivityEntry:
    """A single entry in the activity feed."""
    icon: str
    label: str
    timestamp: datetime


# Role/tool → icon mapping for activity feed
_ACTIVITY_ICONS: dict[str, str] = {
    "user": "👤",
    "assistant": "🤖",
    "Read": "📖",
    "Write": "✏️",
    "Edit": "✏️",
    "Bash": "💻",
    "Glob": "🔍",
    "Grep": "🔍",
    "Task": "⚙️",
    "TaskCreate": "⚙️",
    "TaskUpdate": "⚙️",
    "TaskGet": "⚙️",
    "TaskList": "⚙️",
    "WebFetch": "🌐",
    "WebSearch": "🌐",
    "Agent": "🤖",
}


def build_activity_feed(session: "Session", max_entries: int = 8) -> list[ActivityEntry]:
    """Build a list of recent activity entries from the session's turns.

    Each entry is a one-liner: icon + label + timestamp.
    """
    entries: list[ActivityEntry] = []

    for turn in session.turns[-max_entries:]:
        if turn.role == "user":
            icon = _ACTIVITY_ICONS["user"]
            preview = turn.content_preview[:30]
            if len(turn.content_preview) > 30:
                preview += "..."
            label = preview
        elif turn.role == "assistant":
            icon = _ACTIVITY_ICONS["assistant"]
            if turn.summary:
                label = turn.summary[:35]
            else:
                label = turn.content_preview[:35]
            if len(label) < len(turn.content_preview):
                label += "..."
        else:
            # Tool turn
            icon = _ACTIVITY_ICONS.get(turn.tool_name or "", "🔧")
            label = f"{turn.tool_name or 'Tool'}: {turn.content_preview[:25]}"

        entries.append(ActivityEntry(
            icon=icon,
            label=label,
            timestamp=turn.timestamp,
        ))

    return entries


# ============================================================
# Tool Collection
# ============================================================

def collect_tools_used(session: "Session") -> list[tuple[str, str]]:
    """Collect deduplicated (icon, name) pairs of tools used in the session."""
    seen: set[str] = set()
    tools: list[tuple[str, str]] = []

    for turn in session.turns:
        if turn.role == "tool" and turn.tool_name:
            canonical = turn.tool_name
            # Normalize task tool variants
            if canonical.startswith("Task"):
                canonical = "Task"
            if canonical not in seen:
                seen.add(canonical)
                icon, name = get_tool_powerup(turn.tool_name)
                tools.append((icon, name))

    return tools
