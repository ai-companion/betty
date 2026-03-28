"""Watch project directories for session files."""

import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent


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
        projects_dir: Path | None = None,
        global_mode: bool = False,
        on_initial_load_done: Callable[[], None] | None = None,
    ):
        """Initialize project watcher.

        Args:
            project_paths: List of project directories to watch
            on_session_discovered: Callback called with (session_id, transcript_path)
                when a new session is discovered
            max_sessions: Maximum number of sessions to load on initial scan.
                If None, all sessions are loaded. New sessions created while
                watching are always detected regardless of this limit.
            projects_dir: Parent directory containing project subdirectories
                (e.g., ~/.claude/projects). Used in global mode to discover new projects.
            global_mode: If True, re-scan projects_dir for new subdirectories on each poll.
            on_initial_load_done: Callback called after the initial scan completes.
        """
        self._project_paths = list(project_paths)
        self._on_session_discovered = on_session_discovered
        self._max_sessions = max_sessions
        self._projects_dir = projects_dir
        self._global_mode = global_mode
        self._on_initial_load_done = on_initial_load_done
        self._known_sessions: set[str] = set()  # Sessions we've loaded
        self._skipped_sessions: dict[str, tuple[Path, float]] = {}  # session_id -> (path, last_mtime)
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer: Observer | None = None
        self._watched_paths: set[str] = set()  # Paths already scheduled in observer
        self._lock = threading.Lock()  # Protects _known_sessions, _skipped_sessions, _watched_paths, _project_paths, _observer

    class _SessionFileHandler(FileSystemEventHandler):
        """Handle filesystem events for session .jsonl files."""

        def __init__(self, watcher: "ProjectWatcher"):
            self._watcher = watcher

        def on_created(self, event):
            if isinstance(event, FileCreatedEvent) and event.src_path.endswith(".jsonl"):
                path = Path(event.src_path)
                if path.is_file():
                    self._watcher._on_new_or_updated_session(path.stem, path)

        def on_modified(self, event):
            if isinstance(event, FileModifiedEvent) and event.src_path.endswith(".jsonl"):
                path = Path(event.src_path)
                if path.is_file():
                    self._watcher._on_new_or_updated_session(path.stem, path)

    def _on_new_or_updated_session(self, session_id: str, file: Path) -> None:
        """Handle a new or updated session file.

        Thread-safe: called from watchdog threads and the global-mode poll thread.
        """
        with self._lock:
            # Already loaded - skip
            if session_id in self._known_sessions:
                return

            # Check if this was a skipped session that got updated
            if session_id in self._skipped_sessions:
                _, old_mtime = self._skipped_sessions[session_id]
                try:
                    st = file.stat()
                    if st.st_mtime > old_mtime and st.st_size > 0:
                        # Session has new activity and content - load it now
                        del self._skipped_sessions[session_id]
                        self._known_sessions.add(session_id)
                    else:
                        # Still empty or not updated
                        if st.st_mtime > old_mtime:
                            self._skipped_sessions[session_id] = (file, st.st_mtime)
                        return
                except (PermissionError, OSError):
                    return
                # Call outside lock (callback may do heavy work)
            else:
                # New session - skip if empty
                try:
                    stat = file.stat()
                    if stat.st_size == 0:
                        self._skipped_sessions[session_id] = (file, stat.st_mtime)
                        return
                except (PermissionError, OSError):
                    pass
                self._known_sessions.add(session_id)

        # Callback outside lock to avoid holding it during heavy I/O
        self._on_session_discovered(session_id, file)

    def _add_watch(self, path: Path) -> None:
        """Add a watchdog watch for a directory if not already watched."""
        path_str = str(path)
        with self._lock:
            if path_str in self._watched_paths or not self._observer:
                return
            observer = self._observer  # Capture under lock
        if not path.exists():
            return
        try:
            observer.schedule(
                self._SessionFileHandler(self), path_str, recursive=False
            )
            with self._lock:
                self._watched_paths.add(path_str)
        except OSError:
            pass

    def start(self) -> None:
        """Initial scan and start watching."""
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

        # Load the top sessions (skip empty files)
        for file, mtime in sessions_to_load:
            session_id = file.stem
            try:
                if file.stat().st_size == 0:
                    self._skipped_sessions[session_id] = (file, mtime)
                    continue
            except (PermissionError, OSError):
                pass
            self._known_sessions.add(session_id)
            self._on_session_discovered(session_id, file)

        # Track skipped sessions with their mtime (to detect if they become active)
        for file, mtime in sessions_to_skip:
            self._skipped_sessions[file.stem] = (file, mtime)

        # Signal that initial load is complete
        if self._on_initial_load_done:
            self._on_initial_load_done()

        # Set up watchdog observer (fall back to polling if it fails)
        self._running = True
        self._stop_event.clear()
        watchdog_ok = False
        try:
            self._observer = Observer()
            for project_path in self._project_paths:
                self._add_watch(project_path)
            self._observer.start()
            watchdog_ok = True
        except Exception:
            self._observer = None

        # In global mode, always run discovery thread.
        # If watchdog failed, run discovery thread even in non-global mode as fallback.
        if self._global_mode or not watchdog_ok:
            self._thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._thread.start()

    def _discover_new_project_dirs(self) -> None:
        """In global mode, discover new project directories and add watches."""
        if not self._projects_dir or not self._projects_dir.exists():
            return
        try:
            with self._lock:
                known_paths = set(self._project_paths)
            new_dirs = []
            for p in self._projects_dir.iterdir():
                if p.is_dir() and p.name.startswith("-") and p not in known_paths:
                    new_dirs.append(p)
            if new_dirs:
                with self._lock:
                    self._project_paths.extend(new_dirs)
                for p in new_dirs:
                    self._add_watch(p)
        except (PermissionError, OSError):
            pass

    def _scan_for_new_sessions(self) -> None:
        """Poll scan for new/updated session files (fallback when watchdog unavailable)."""
        with self._lock:
            paths = list(self._project_paths)
        for project_path in paths:
            if not project_path.exists():
                continue
            try:
                for file in project_path.iterdir():
                    if file.suffix == ".jsonl" and file.is_file():
                        self._on_new_or_updated_session(file.stem, file)
            except (PermissionError, OSError):
                pass

    def _watch_loop(self) -> None:
        """Poll loop for directory discovery and fallback session scanning."""
        while self._running:
            self._discover_new_project_dirs()
            # If watchdog is not running, also poll for new session files
            with self._lock:
                has_observer = self._observer is not None
            if not has_observer:
                self._scan_for_new_sessions()
            # Use event wait so stop() can wake us immediately
            self._stop_event.wait(timeout=5.0)

    def stop(self) -> None:
        """Stop the watcher thread and observer."""
        self._running = False
        self._stop_event.set()  # Wake poll loop immediately
        with self._lock:
            observer = self._observer
            self._observer = None
        if observer:
            observer.stop()
            observer.join(timeout=2.0)
        if self._thread:
            self._thread.join(timeout=6.0)
