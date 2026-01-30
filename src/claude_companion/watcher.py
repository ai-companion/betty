"""Watch transcript files for real-time updates."""

import json
import threading
import time
from pathlib import Path
from typing import Callable

from .models import Turn, count_words, _truncate, _extract_tool_content


class TranscriptWatcher:
    """Watch a transcript file for new entries."""

    def __init__(self, on_turn: Callable[[Turn], None]):
        self._on_turn = on_turn
        self._current_path: Path | None = None
        self._last_position: int = 0
        self._turn_number: int = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def watch(self, transcript_path: str, start_turn: int = 0, start_position: int | None = None) -> None:
        """Start watching a transcript file.

        Args:
            transcript_path: Path to the transcript file.
            start_turn: Turn number to start counting from.
            start_position: File position to start reading from. If None, starts from end of file.
        """
        path = Path(transcript_path)

        # Wait for file to exist (up to 2 seconds)
        for _ in range(20):
            if path.exists():
                break
            time.sleep(0.1)

        with self._lock:
            self._current_path = path
            # Use provided position, or start from end of file
            if start_position is not None:
                self._last_position = start_position
            else:
                self._last_position = path.stat().st_size if path.exists() else 0
            self._turn_number = start_turn

        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _watch_loop(self) -> None:
        """Main watch loop - poll for new content."""
        while self._running:
            try:
                self._check_for_updates()
            except Exception:
                pass  # Don't crash on errors
            time.sleep(0.5)  # Poll every 500ms

    def _check_for_updates(self) -> None:
        """Check for new content in the transcript file."""
        with self._lock:
            path = self._current_path
            pos = self._last_position

        if not path or not path.exists():
            return

        current_size = path.stat().st_size
        if current_size <= pos:
            return  # No new content

        # Read new content
        with open(path, "r") as f:
            f.seek(pos)
            new_content = f.read()
            new_position = f.tell()

        with self._lock:
            self._last_position = new_position

        # Parse new lines
        for line in new_content.strip().split("\n"):
            if not line:
                continue
            try:
                entry = json.loads(line)
                turns = self._parse_entry(entry)
                for turn in turns:
                    self._on_turn(turn)
            except json.JSONDecodeError:
                continue

    def _parse_entry(self, entry: dict) -> list[Turn]:
        """Parse a transcript entry and return turns."""
        turns: list[Turn] = []
        entry_type = entry.get("type", "")

        if entry_type == "user":
            message = entry.get("message", {})
            content = message.get("content", "")
            if isinstance(content, str) and content:
                with self._lock:
                    self._turn_number += 1
                    turn_num = self._turn_number
                turns.append(Turn(
                    turn_number=turn_num,
                    role="user",
                    content_preview=_truncate(content, 100),
                    content_full=content,
                    word_count=count_words(content),
                ))

        elif entry_type == "assistant":
            message = entry.get("message", {})
            content = message.get("content", [])

            if isinstance(content, list):
                for block in content:
                    block_type = block.get("type", "")

                    if block_type == "text":
                        text = block.get("text", "")
                        if text:
                            with self._lock:
                                self._turn_number += 1
                                turn_num = self._turn_number
                            turns.append(Turn(
                                turn_number=turn_num,
                                role="assistant",
                                content_preview=_truncate(text, 100),
                                content_full=text,
                                word_count=count_words(text),
                            ))

                    elif block_type == "tool_use":
                        tool_name = block.get("name", "")
                        tool_input = block.get("input", {})
                        content_str = _extract_tool_content(tool_name, tool_input)
                        with self._lock:
                            self._turn_number += 1
                            turn_num = self._turn_number
                        turns.append(Turn(
                            turn_number=turn_num,
                            role="tool",
                            content_preview=_truncate(content_str, 100),
                            content_full=content_str,
                            word_count=count_words(content_str),
                            tool_name=tool_name,
                        ))

        return turns
