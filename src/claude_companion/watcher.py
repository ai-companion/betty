"""Watch transcript files for real-time updates."""

import json
import threading
import time
from pathlib import Path
from typing import Callable

from .models import Turn, count_words, _truncate, _extract_tool_content, parse_task_operation


class TranscriptWatcher:
    """Watch a transcript file for new entries."""

    def __init__(self, on_turn: Callable[[Turn], None]):
        self._on_turn = on_turn
        self._current_path: Path | None = None
        self._last_position: int = 0
        self._turn_number: int = 0
        self._running = False
        self._cancelled = False  # Set by cancel() to prevent watch() from starting
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

        # Check cancelled flag and set running atomically to prevent race with stop()
        with self._lock:
            if self._cancelled:
                return
            if self._running:
                return
            self._running = True

        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching."""
        with self._lock:
            self._cancelled = True  # Prevent late starts from watch()
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
        """Check for new content in the transcript file.

        Position is only advanced after successfully parsing a line.
        This ensures that if we read a partial line (Claude still writing),
        we'll re-read it on the next poll when it's complete.
        """
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

        # Track position for each successfully parsed line
        current_pos = pos
        lines = new_content.split("\n")

        for i, line in enumerate(lines):
            # Calculate line length in bytes
            line_bytes = len(line.encode('utf-8'))
            # Add 1 for newline, except for the last line if it doesn't end with newline
            has_newline = (i < len(lines) - 1) or new_content.endswith("\n")
            line_length = line_bytes + (1 if has_newline else 0)

            # Skip empty lines but advance position
            if not line.strip():
                current_pos += line_length
                continue

            try:
                entry = json.loads(line)
                turns = self._parse_entry(entry)
                for turn in turns:
                    self._on_turn(turn)
                # Successfully parsed - advance position past this line
                current_pos += line_length
            except json.JSONDecodeError:
                # If this line has a newline, it's complete but malformed - skip it
                # If no newline, it might be incomplete (Claude still writing) - stop here
                if has_newline:
                    # Malformed but complete line - skip and continue
                    current_pos += line_length
                    continue
                else:
                    # Partial line - stop and retry on next poll
                    break

        # Only update position to last successfully parsed line
        with self._lock:
            self._last_position = current_pos

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

                        # Parse task operations
                        task_op = parse_task_operation(tool_name, tool_input)

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
                            task_operation=task_op,
                        ))

        return turns


class PlanFileWatcher:
    """Watches a plan file for changes."""

    def __init__(self, project_dir: str, on_plan_update: Callable[[str, str], None]):
        """Initialize plan file watcher.

        Args:
            project_dir: Absolute path to project directory
            on_plan_update: Callback(content, file_path) called when plan changes
        """
        self.project_dir = project_dir
        self.on_plan_update = on_plan_update
        self.plan_file_path: str | None = None
        self.last_mtime: float | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start watching for plan file."""
        with self._lock:
            if self._running:
                return
            self._running = True

        # Find initial plan file
        from .utils import find_plan_file
        self.plan_file_path = find_plan_file(self.project_dir)

        # If plan exists, read initial content
        if self.plan_file_path:
            self._read_and_notify()

        # Start polling thread
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching."""
        with self._lock:
            self._running = False

    def _watch_loop(self) -> None:
        """Main watch loop (runs in background thread)."""
        while True:
            with self._lock:
                if not self._running:
                    break

            self._check_for_updates()
            time.sleep(2.0)  # Poll every 2 seconds

    def _check_for_updates(self) -> None:
        """Check if plan file has changed."""
        from .utils import find_plan_file

        # Re-check for plan file (handles creation/deletion)
        current_path = find_plan_file(self.project_dir)

        # Plan file was deleted
        if self.plan_file_path and not current_path:
            self.plan_file_path = None
            self.last_mtime = None
            self.on_plan_update("", "")  # Empty content = deleted
            return

        # Plan file was created or moved
        if current_path != self.plan_file_path:
            self.plan_file_path = current_path
            self.last_mtime = None  # Force read

        # Check modification time
        if self.plan_file_path:
            path = Path(self.plan_file_path)
            if path.exists():
                mtime = path.stat().st_mtime
                if self.last_mtime is None or mtime > self.last_mtime:
                    self.last_mtime = mtime
                    self._read_and_notify()

    def _read_and_notify(self) -> None:
        """Read plan file and notify callback."""
        if not self.plan_file_path:
            return

        try:
            path = Path(self.plan_file_path)
            content = path.read_text(encoding="utf-8")
            self.on_plan_update(content, str(path))
        except (OSError, UnicodeDecodeError) as e:
            # File permission error or encoding issue
            error_msg = f"Error reading plan file: {e}"
            self.on_plan_update(error_msg, self.plan_file_path or "")

