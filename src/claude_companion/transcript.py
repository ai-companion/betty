"""Parse Claude Code transcript files to load conversation history."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Turn, count_words, _extract_tool_content, _truncate


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
        Tuple of (turns list, file position after reading).
        The file position can be passed to the watcher to avoid missing content.
    """
    turns: list[Turn] = []
    turn_number = 0
    file_position = 0

    try:
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    new_turns = _parse_entry(entry, turn_number)
                    for turn in new_turns:
                        turn_number += 1
                        turn.turn_number = turn_number
                        turn.is_historical = True
                        turns.append(turn)
                except json.JSONDecodeError:
                    continue
            file_position = f.tell()
    except IOError:
        return [], 0

    return turns, file_position


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

    elif entry_type == "assistant":
        # Assistant message - may contain text and/or tool_use
        message = entry.get("message", {})
        content = message.get("content", [])

        if isinstance(content, list):
            for block in content:
                block_type = block.get("type", "")

                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        turns.append(Turn(
                            turn_number=current_turn,
                            role="assistant",
                            content_preview=_truncate(text, 100),
                            content_full=text,
                            word_count=count_words(text),
                            timestamp=timestamp,
                        ))

                elif block_type == "tool_use":
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    content_str = _extract_tool_content(tool_name, tool_input)
                    turns.append(Turn(
                        turn_number=current_turn,
                        role="tool",
                        content_preview=_truncate(content_str, 100),
                        content_full=content_str,
                        word_count=count_words(content_str),
                        tool_name=tool_name,
                        timestamp=timestamp,
                    ))

    return turns


def _parse_timestamp(ts: str | None) -> datetime:
    """Parse ISO 8601 timestamp or return now."""
    if not ts:
        return datetime.now()
    try:
        # Handle ISO format with Z suffix
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return datetime.now()
