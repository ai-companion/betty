"""Session health metrics for Betty Agent.

Pure computation — no LLM, no UI, no threads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Session, Turn


# Error patterns in tool output
ERROR_PATTERNS = [
    re.compile(r"exit code[:\s]+[1-9]\d*", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*Error:", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"\bFAILED\b"),
    re.compile(r"command not found", re.IGNORECASE),
    re.compile(r"Permission denied", re.IGNORECASE),
    re.compile(r"No such file or directory", re.IGNORECASE),
]


@dataclass
class SessionMetrics:
    """Health metrics computed from a session's turns."""

    token_burn_rate: float = 0.0          # tokens/min (input+output)
    error_rate: float = 0.0               # fraction of tool turns with errors
    retry_count: int = 0                  # consecutive same-tool calls at end
    tool_distribution: dict[str, int] = field(default_factory=dict)
    turn_velocity: float = 0.0            # turns/min overall
    recent_velocity: float = 0.0          # turns/min last 5 min
    files_touched: set[str] = field(default_factory=set)
    seconds_since_last_turn: float = 0.0  # stall indicator
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # Spin detection signals
    repetitive_tool_score: float = 0.0    # 0-1, fraction of last 10 tool calls that repeat
    error_retry_count: int = 0            # tool fails then retried in last 10 turns
    output_shrinking: bool = False        # assistant responses getting shorter
    tool_diversity: float = 0.0           # unique tools / total tools (0-1)


def _is_error_turn(turn: "Turn") -> bool:
    """Check if a tool turn's content indicates an error."""
    text = turn.content_full or turn.content_preview or ""
    return any(p.search(text) for p in ERROR_PATTERNS)


def _extract_file_paths(turn: "Turn") -> set[str]:
    """Extract file paths from a tool turn based on tool_name."""
    paths: set[str] = set()
    if not turn.tool_name:
        return paths

    content = turn.content_full or turn.content_preview or ""

    if turn.tool_name == "Read":
        # content_preview is the file path for Read tools
        path = turn.content_preview.strip()
        if path and path.startswith("/"):
            paths.add(path)
    elif turn.tool_name in ("Write", "Edit"):
        # content_preview starts with file path
        # Format: "path/to/file (N lines)" or "path/to/file (-N, +M)"
        preview = turn.content_preview.strip()
        if preview:
            # Take everything before the first " ("
            path = preview.split(" (")[0].strip()
            if path:
                paths.add(path)

    return paths


def _count_trailing_retries(tool_turns: list["Turn"]) -> int:
    """Count consecutive same-tool calls at the end of the list."""
    if not tool_turns:
        return 0

    last_tool = tool_turns[-1].tool_name
    if not last_tool:
        return 0

    count = 0
    for t in reversed(tool_turns):
        if t.tool_name == last_tool:
            count += 1
        else:
            break

    return count if count >= 2 else 0


def _compute_repetitive_tool_score(tool_turns: list["Turn"], window: int = 10) -> float:
    """Fraction of recent tool calls using the same tool as the most common in the window."""
    recent = tool_turns[-window:]
    if len(recent) < 3:
        return 0.0

    # Count how often the most common tool appears
    tool_counts: dict[str, int] = {}
    for t in recent:
        name = t.tool_name or "unknown"
        tool_counts[name] = tool_counts.get(name, 0) + 1

    max_count = max(tool_counts.values())
    return max_count / len(recent)


def _compute_error_retry_count(tool_turns: list["Turn"], window: int = 10) -> int:
    """Count error→retry sequences in recent tool turns."""
    recent = tool_turns[-window:]
    if len(recent) < 2:
        return 0

    count = 0
    for i in range(1, len(recent)):
        prev = recent[i - 1]
        curr = recent[i]
        if (
            _is_error_turn(prev)
            and prev.tool_name == curr.tool_name
        ):
            count += 1
    return count


def _detect_output_shrinking(turns: list["Turn"], window: int = 5) -> bool:
    """Detect if assistant responses are getting shorter (diminishing output)."""
    assistant_turns = [t for t in turns if t.role == "assistant"]
    recent = assistant_turns[-window:]
    if len(recent) < 3:
        return False

    # Check if word counts are monotonically decreasing (with some tolerance)
    word_counts = [t.word_count for t in recent]
    decreasing_count = sum(
        1 for i in range(1, len(word_counts))
        if word_counts[i] < word_counts[i - 1]
    )
    # If more than 60% of transitions are decreasing, flag it
    return decreasing_count / (len(word_counts) - 1) > 0.6


def compute_session_metrics(session: "Session") -> SessionMetrics:
    """Compute health metrics from a session's turns.

    This is a stateless, pure function — safe to call from any thread.
    """
    from datetime import datetime, timezone

    turns = session.turns
    if not turns:
        return SessionMetrics()

    now = datetime.now()

    # --- Token totals ---
    total_input = session.total_input_tokens
    total_output = session.total_output_tokens

    # --- Timing ---
    first_ts = turns[0].timestamp
    last_ts = turns[-1].timestamp
    duration_seconds = max((last_ts - first_ts).total_seconds(), 1.0)
    duration_minutes = duration_seconds / 60.0
    seconds_since_last = (now - last_ts).total_seconds()

    # --- Token burn rate ---
    token_burn_rate = (total_input + total_output) / duration_minutes if duration_minutes > 0 else 0.0

    # --- Turn velocity ---
    turn_velocity = len(turns) / duration_minutes if duration_minutes > 0 else 0.0

    # --- Recent velocity (last 5 min) ---
    from datetime import timedelta
    five_min_ago = now - timedelta(minutes=5)
    recent_turns = [t for t in turns if t.timestamp >= five_min_ago]
    recent_velocity = len(recent_turns) / 5.0

    # --- Tool analysis ---
    tool_turns = [t for t in turns if t.role == "tool"]
    tool_distribution: dict[str, int] = {}
    error_count = 0
    files_touched: set[str] = set()

    for t in tool_turns:
        name = t.tool_name or "unknown"
        tool_distribution[name] = tool_distribution.get(name, 0) + 1

        if _is_error_turn(t):
            error_count += 1

        files_touched.update(_extract_file_paths(t))

    error_rate = error_count / len(tool_turns) if tool_turns else 0.0

    # --- Retry detection ---
    retry_count = _count_trailing_retries(tool_turns)

    # --- Spin detection ---
    repetitive_tool_score = _compute_repetitive_tool_score(tool_turns)
    error_retry_count = _compute_error_retry_count(tool_turns)
    output_shrinking = _detect_output_shrinking(turns)
    tool_diversity = len(tool_distribution) / len(tool_turns) if tool_turns else 0.0

    return SessionMetrics(
        token_burn_rate=token_burn_rate,
        error_rate=error_rate,
        retry_count=retry_count,
        tool_distribution=tool_distribution,
        turn_velocity=turn_velocity,
        recent_velocity=recent_velocity,
        files_touched=files_touched,
        seconds_since_last_turn=seconds_since_last,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        repetitive_tool_score=repetitive_tool_score,
        error_retry_count=error_retry_count,
        output_shrinking=output_shrinking,
        tool_diversity=tool_diversity,
    )
