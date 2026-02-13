"""Parse Claude Code transcript files to load conversation history."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Turn, count_words, _extract_tool_content, _truncate, parse_task_operation

# Slash command metadata: <command-message>...</command-message> with optional <command-name>
_COMMAND_META_RE = re.compile(
    r"^\s*<command-message>.*</command-message>\s*(?:<command-name>(.*)</command-name>)?\s*$",
    re.DOTALL,
)


def _extract_command_name(content: str) -> str | None:
    """Extract slash command name from metadata entry, or None if not a match."""
    m = _COMMAND_META_RE.match(content)
    if not m:
        return None
    name = m.group(1)
    return name.strip() if name else None


def _detect_command_entry(entry: dict[str, Any]) -> str | None:
    """If entry is a slash command metadata entry, return the command name."""
    if entry.get("type") != "user":
        return None
    content = entry.get("message", {}).get("content", "")
    if isinstance(content, str):
        return _extract_command_name(content)
    return None


def get_transcript_path(session_id: str, cwd: str) -> Path | None:
    """Compute transcript path from session ID and working directory."""
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        return None

    # Encode project path: /Users/kai/src/foo -> -Users-kai-src-foo
    encoded_project = "-" + cwd.lstrip("/").replace("/", "-")
    transcript_file = claude_dir / encoded_project / f"{session_id}.jsonl"

    if transcript_file.exists():
        return transcript_file
    return None


def parse_transcript(transcript_path: Path) -> tuple[list[Turn], int]:
    """Parse a JSONL transcript file and extract turns.

    Returns:
        Tuple of (turns list, file position after last successfully parsed line).
        The file position can be passed to the watcher to avoid missing content.
    """
    turns: list[Turn] = []
    turn_number = 0
    last_good_position = 0
    pending_command: str | None = None

    try:
        with open(transcript_path, "r") as f:
            while True:
                line_start = f.tell()
                line = f.readline()
                if not line:
                    break  # EOF

                line = line.strip()
                if not line:
                    # Empty line - update position and continue
                    last_good_position = f.tell()
                    continue

                try:
                    entry = json.loads(line)

                    # Check for command metadata before parsing
                    cmd_name = _detect_command_entry(entry)
                    if cmd_name is not None:
                        pending_command = cmd_name
                        last_good_position = f.tell()
                        continue

                    new_turns = _parse_entry(entry, turn_number)

                    # Prepend pending command name to next user turn
                    if pending_command and new_turns and new_turns[0].role == "user":
                        t = new_turns[0]
                        merged = f"{pending_command}\n{t.content_full}"
                        t.content_full = merged
                        t.content_preview = _truncate(merged, 100)
                        t.word_count = count_words(merged)
                    pending_command = None

                    for turn in new_turns:
                        turn_number += 1
                        turn.turn_number = turn_number
                        turn.is_historical = True
                        turns.append(turn)
                    # Successfully parsed - update position to after this line
                    last_good_position = f.tell()
                except json.JSONDecodeError:
                    # Failed to parse - don't update position, watcher will retry
                    continue
    except IOError:
        return [], 0

    return turns, last_good_position


def _parse_entry(entry: dict[str, Any], current_turn: int) -> list[Turn]:
    """Parse a single JSONL entry and return turns."""
    turns: list[Turn] = []
    entry_type = entry.get("type", "")
    timestamp = _parse_timestamp(entry.get("timestamp"))

    if entry_type == "user":
        # User message
        message = entry.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str) and content:
            turns.append(Turn(
                turn_number=current_turn,
                role="user",
                content_preview=_truncate(content, 100),
                content_full=content,
                word_count=count_words(content),
                timestamp=timestamp,
            ))
        elif isinstance(content, list):
            # Extract text from list-content (slash command expansions, etc.)
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    t = block.get("text", "")
                    if isinstance(t, str) and t.strip():
                        text_parts.append(t)
            if text_parts:
                combined = "\n".join(text_parts)
                turns.append(Turn(
                    turn_number=current_turn,
                    role="user",
                    content_preview=_truncate(combined, 100),
                    content_full=combined,
                    word_count=count_words(combined),
                    timestamp=timestamp,
                ))

    elif entry_type == "assistant":
        # Assistant message - may contain text and/or tool_use
        message = entry.get("message", {})
        content = message.get("content", [])

        # Extract model and usage from message (shared across all blocks)
        model_id = message.get("model")
        usage = message.get("usage", {})
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        cache_creation = usage.get("cache_creation_input_tokens")
        cache_read = usage.get("cache_read_input_tokens")
        first_turn_created = False

        if isinstance(content, list):
            for block in content:
                block_type = block.get("type", "")

                if block_type == "text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text.strip():
                        turn = Turn(
                            turn_number=current_turn,
                            role="assistant",
                            content_preview=_truncate(text, 100),
                            content_full=text,
                            word_count=count_words(text),
                            timestamp=timestamp,
                        )
                        # Attach token data to first turn only (avoid double-counting)
                        if not first_turn_created:
                            turn.input_tokens = input_tokens
                            turn.output_tokens = output_tokens
                            turn.cache_creation_tokens = cache_creation
                            turn.cache_read_tokens = cache_read
                            turn.model_id = model_id
                            first_turn_created = True
                        turns.append(turn)

                elif block_type == "tool_use":
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})

                    # Parse task operations
                    task_op = parse_task_operation(tool_name, tool_input)

                    content_str = _extract_tool_content(tool_name, tool_input)
                    turn = Turn(
                        turn_number=current_turn,
                        role="tool",
                        content_preview=_truncate(content_str, 100),
                        content_full=content_str,
                        word_count=count_words(content_str),
                        tool_name=tool_name,
                        timestamp=timestamp,
                        task_operation=task_op,
                    )
                    # Attach token data to first turn only (avoid double-counting)
                    if not first_turn_created:
                        turn.input_tokens = input_tokens
                        turn.output_tokens = output_tokens
                        turn.cache_creation_tokens = cache_creation
                        turn.cache_read_tokens = cache_read
                        turn.model_id = model_id
                        first_turn_created = True
                    turns.append(turn)

    return turns


def _parse_timestamp(ts: str | None) -> datetime:
    """Parse ISO 8601 timestamp and convert to local time."""
    if not ts:
        return datetime.now()
    try:
        # Handle ISO format with Z suffix
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        # Convert to local time if timezone-aware
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return datetime.now()
