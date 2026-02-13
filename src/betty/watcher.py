"""Watch transcript files for real-time updates."""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

from .models import Turn, count_words, _truncate, _extract_tool_content, parse_task_operation
from .transcript import _detect_command_entry, _extract_plan_from_entry

logger = logging.getLogger(__name__)


class TranscriptWatcher:
    """Watch a transcript file for new entries."""

    def __init__(self, on_turn: Callable[[Turn], None], on_plan_update: Callable[[str, str], None] | None = None):
        self._on_turn = on_turn
        self._on_plan_update = on_plan_update
        self._current_path: Path | None = None
        self._last_position: int = 0
        self._turn_number: int = 0
        self._running = False
        self._cancelled = False  # Set by cancel() to prevent watch() from starting
        self._pending_command: str | None = None  # Command name from metadata entry
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
                logger.debug(f"Watcher cancelled before start: {path.name}")
                return
            if self._running:
                logger.debug(f"Watcher already running: {path.name}")
                return
            self._running = True

        logger.debug(f"Starting watcher for {path.name} at position {self._last_position}")
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
            except Exception as e:
                logger.error(f"Error in transcript watcher: {e}", exc_info=True)
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

        logger.debug(f"Detected file change: {path.name} size={current_size} pos={pos}")

        # Read and parse line by line using f.tell() for position tracking
        # This matches how transcript.py works and avoids encoding issues
        last_good_position = pos

        try:
            with open(path, "r") as f:
                f.seek(pos)

                while True:
                    line_start = f.tell()
                    raw_line = f.readline()

                    if not raw_line:
                        break  # EOF

                    # Check for newline BEFORE stripping to distinguish complete vs incomplete lines
                    has_newline = raw_line.endswith("\n")
                    line = raw_line.strip()

                    if not line:
                        # Empty line - update position and continue
                        last_good_position = f.tell()
                        continue

                    try:
                        entry = json.loads(line)

                        # Check for command metadata before parsing
                        cmd_name = _detect_command_entry(entry)
                        if cmd_name is not None:
                            self._pending_command = cmd_name
                            last_good_position = f.tell()
                            continue

                        turns = self._parse_entry(entry)

                        # Prepend pending command name to next user turn
                        if self._pending_command and turns and turns[0].role == "user":
                            t = turns[0]
                            merged = f"{self._pending_command}\n\n{t.content_full}"
                            t.content_full = merged
                            t.content_preview = _truncate(merged, 100)
                            t.word_count = count_words(merged)
                        self._pending_command = None

                        if not turns:
                            logger.debug(f"No turns extracted from entry type: {entry.get('type', 'unknown')}")
                        for turn in turns:
                            self._on_turn(turn)

                        # Check for plan content in this entry
                        if self._on_plan_update:
                            plan_info = _extract_plan_from_entry(entry)
                            if plan_info:
                                self._on_plan_update(plan_info[0], plan_info[1])

                        # Successfully parsed - update position to after this line
                        last_good_position = f.tell()
                    except json.JSONDecodeError as e:
                        # Use newline presence to determine if line is complete
                        if has_newline:
                            # Complete but malformed - skip it
                            logger.warning(f"Malformed JSON line (skipping): {e}")
                            last_good_position = f.tell()
                        else:
                            # Likely incomplete (Claude still writing) - stop here
                            logger.debug(f"Incomplete JSON line (will retry)")
                            break
        except IOError as e:
            logger.error(f"Error reading transcript file: {e}")
            return

        # Update position to last successfully parsed line
        with self._lock:
            self._last_position = last_good_position

    def _parse_entry(self, entry: dict) -> list[Turn]:
        """Parse a transcript entry and return turns."""
        turns: list[Turn] = []
        entry_type = entry.get("type", "")
        logger.debug(f"Parsing entry type: {entry_type}")

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
                    with self._lock:
                        self._turn_number += 1
                        turn_num = self._turn_number
                    turns.append(Turn(
                        turn_number=turn_num,
                        role="user",
                        content_preview=_truncate(combined, 100),
                        content_full=combined,
                        word_count=count_words(combined),
                    ))

        elif entry_type == "assistant":
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
                            with self._lock:
                                self._turn_number += 1
                                turn_num = self._turn_number
                            turn = Turn(
                                turn_number=turn_num,
                                role="assistant",
                                content_preview=_truncate(text, 100),
                                content_full=text,
                                word_count=count_words(text),
                            )
                            # Attach token data to first turn only
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
                        with self._lock:
                            self._turn_number += 1
                            turn_num = self._turn_number
                        turn = Turn(
                            turn_number=turn_num,
                            role="tool",
                            content_preview=_truncate(content_str, 100),
                            content_full=content_str,
                            word_count=count_words(content_str),
                            tool_name=tool_name,
                            task_operation=task_op,
                        )
                        # Attach token data to first turn only
                        if not first_turn_created:
                            turn.input_tokens = input_tokens
                            turn.output_tokens = output_tokens
                            turn.cache_creation_tokens = cache_creation
                            turn.cache_read_tokens = cache_read
                            turn.model_id = model_id
                            first_turn_created = True
                        turns.append(turn)

        return turns

