"""Berry character sprites and state inference for Berry visualization style.

Ported from noisy-channels TypeScript (src/sprites/berry.ts).
Berry is a lovable blueberry character that reflects Claude's activity state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import SpanGroup

# One representative frame per state (frame 0 from each animation).
# 7 lines tall, 18 chars wide, using Unicode box-drawing.
BERRY_SPRITES: dict[str, list[str]] = {
    "idle": [
        "      ╭───╮     ",
        "    ╭─┤ ♠ ├─╮   ",
        "    │ ◕   ◕ │   ",
        "    │   ‿   │   ",
        "    ╰──┬─┬──╯   ",
        "       │ │      ",
        "      ╶┘ └╴     ",
    ],
    "thinking": [
        "           · ·   ",
        "      ╭───╮     ",
        "    ╭─┤ ♠ ├─╮   ",
        "    │ ◔   ◔ │   ",
        "   ╭│   ─   │   ",
        "   ╰╰──┬─┬──╯   ",
        "      ╶┘ └╴     ",
    ],
    "working": [
        "    ∗ ╭───╮ ∗   ",
        "   ╲╭─┤ ♠ ├─╮╱  ",
        "    │ ◉   ◉ │   ",
        "    │   ▽   │   ",
        "    ╰──┬─┬──╯   ",
        "       │ │      ",
        "      ╶┘ └╴     ",
    ],
    "celebrating": [
        "  ✦\\★╭───╮★/✦  ",
        "    ╭┤ ♠ ├╮    ",
        "    │ ◕ ◕ │    ",
        "    │  ▽  │    ",
        "    ╰─────╯    ",
        "                ",
        "                ",
    ],
    "confused": [
        "        ？       ",
        "      ╭───╮     ",
        "    ╭─┤ ♠ ├─╮   ",
        "    │ ◑   ◐ │   ",
        "    │   ∿   │   ",
        "    ╰──┬─┬──╯   ",
        "      ╶┘ └╴     ",
    ],
    "talking": [
        "      ╭───╮     ",
        "    ╭─┤ ♠ ├─╮◁  ",
        "    │ ◕   ◕ │   ",
        "    │   ○   │   ",
        "    ╰──┬─┬──╯   ",
        "       │ │      ",
        "      ╶┘ └╴     ",
    ],
}

# Map tool names to power-up icons
TOOL_POWERUPS: dict[str, str] = {
    "Read": "📖",
    "Write": "✏️",
    "Edit": "✏️",
    "Bash": "💻",
    "Glob": "🔍",
    "Grep": "🔍",
    "WebFetch": "🌐",
    "WebSearch": "🌐",
    "Task": "⚙️",
    "default": "🔧",
}

# Berry purple/pink palette
BERRY_COLORS: dict[str, str] = {
    "primary": "#c084fc",    # Purple
    "accent": "#f472b6",     # Pink
    "highlight": "#fbbf24",  # Gold
    "dim": "#a78bfa",        # Lighter purple
    "border": "#9333ea",     # Darker purple for borders
}


def get_tool_powerup(tool_name: str | None) -> str:
    """Get the power-up icon for a tool."""
    if not tool_name:
        return TOOL_POWERUPS["default"]
    return TOOL_POWERUPS.get(tool_name, TOOL_POWERUPS["default"])


def infer_berry_state(group: "SpanGroup", is_last: bool = False) -> str:
    """Examine a span's response turns to pick Berry's animation state.

    Args:
        group: The span group to analyze.
        is_last: Whether this is the last span in the session.

    Returns:
        One of: idle, thinking, working, celebrating, confused, talking.
    """
    if not group.response_turns:
        return "idle"

    tool_count = 0
    assistant_count = 0
    has_error = False

    for turn in group.response_turns:
        if turn.role == "assistant":
            assistant_count += 1
        elif turn.role == "tool":
            tool_count += 1
            # Check for errors in tool output
            preview = turn.content_preview.lower()
            if any(word in preview for word in ("error", "failed", "exception", "traceback", "not found")):
                has_error = True

    # Errors in tool output → confused
    if has_error:
        return "confused"

    # Last span in a completed session → celebrating
    if is_last and assistant_count > 0:
        return "celebrating"

    # More tools than assistant turns → working hard
    if tool_count > assistant_count:
        return "working"

    # Has assistant turns (text-heavy) → talking
    if assistant_count > 0:
        return "talking"

    # Fallback
    return "idle"
