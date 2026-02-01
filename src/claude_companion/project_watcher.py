"""Watch project directories for session files."""

import threading
import time
from pathlib import Path
from typing import Callable


class ProjectWatcher:
    """Watch project directories for session files.

    Scans project directories for .jsonl session files and notifies
    when new sessions are discovered. Used for auto-discovery of
    Claude Code sessions without requiring hooks.
    """

    def __init__(
        self,
        project_paths: list[Path],
        on_session_discovered: Callable[[str, Path], None],
        max_sessions: int | None = None,
    ):
        """Initialize project watcher.

        Args:
            project_paths: List of project directories to watch
            on_session_discovered: Callback called with (session_id, transcript_path)
                when a new session is discovered
            max_sessions: Maximum number of sessions to load on initial scan.
                If None, all sessions are loaded. New sessions created while
                watching are always detected regardless of this limit.
        """
        self._project_paths = project_paths
        self._on_session_discovered = on_session_discovered
        self._max_sessions = max_sessions
        self._known_sessions: set[str] = set()  # Sessions we've loaded
        self._skipped_sessions: dict[str, tuple[Path, float]] = {}  # session_id -> (path, last_mtime)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Initial scan and start polling thread."""
        # Initial scan - collect all sessions with their modification times
        all_sessions: list[tuple[Path, float]] = []  # (path, mtime)

        for project_path in self._project_paths:
            if not project_path.exists():
                continue
            try:
                for file in project_path.iterdir():
                    if file.suffix == ".jsonl" and file.is_file():
                        try:
                            mtime = file.stat().st_mtime
                            all_sessions.append((file, mtime))
                        except (PermissionError, OSError):
                            pass
            except (PermissionError, OSError):
                pass

        # Sort by modification time (most recent first)
        all_sessions.sort(key=lambda x: x[1], reverse=True)

        # Split into sessions to load vs skip
        if self._max_sessions:
            sessions_to_load = all_sessions[:self._max_sessions]
            sessions_to_skip = all_sessions[self._max_sessions:]
        else:
            sessions_to_load = all_sessions
            sessions_to_skip = []

        # Load the top sessions
        for file, _ in sessions_to_load:
            session_id = file.stem
            self._known_sessions.add(session_id)
            self._on_session_discovered(session_id, file)

        # Track skipped sessions with their mtime (to detect if they become active)
        for file, mtime in sessions_to_skip:
            self._skipped_sessions[file.stem] = (file, mtime)

        # Start background polling thread
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def _scan_for_new_sessions(self) -> None:
        """Scan for new sessions and check if skipped sessions became active."""
        for project_path in self._project_paths:
            if not project_path.exists():
                continue

            try:
                for file in project_path.iterdir():
                    if file.suffix == ".jsonl" and file.is_file():
                        session_id = file.stem

                        # Already loaded - skip
                        if session_id in self._known_sessions:
                            continue

                        # Check if this was a skipped session that got updated
                        if session_id in self._skipped_sessions:
                            old_path, old_mtime = self._skipped_sessions[session_id]
                            try:
                                new_mtime = file.stat().st_mtime
                                if new_mtime > old_mtime:
                                    # Session has new activity - load it now
                                    del self._skipped_sessions[session_id]
                                    self._known_sessions.add(session_id)
                                    self._on_session_discovered(session_id, file)
                            except (PermissionError, OSError):
                                pass
                        else:
                            # Completely new session - load it
                            self._known_sessions.add(session_id)
                            self._on_session_discovered(session_id, file)
            except (PermissionError, OSError):
                pass

    def _watch_loop(self) -> None:
        """Poll project directories for new sessions."""
        while self._running:
            self._scan_for_new_sessions()
            time.sleep(1.0)  # Poll every second

    def stop(self) -> None:
        """Stop the watcher thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
