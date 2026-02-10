"""Thread-safe session store for Claude Companion."""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from .alerts import Alert, AlertLevel, check_turn_for_alerts, send_system_notification
from .cache import AnnotationCache, SummaryCache
from .config import load_config
from .models import Session, TaskState, Turn
from .project_watcher import ProjectWatcher
from .summarizer import (
    CRITIC_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TOOL_SYSTEM_PROMPT,
    Summarizer,
    _get_turn_context,
    _get_critic_context,
    _prompt_fingerprint,
    make_tool_cache_key,
    make_critic_cache_key,
)
from .transcript import parse_transcript
from .watcher import TranscriptWatcher, PlanFileWatcher


class EventStore:
    """Thread-safe store for Claude Code sessions."""

    def __init__(self, enable_notifications: bool = True) -> None:
        self._sessions: dict[str, Session] = {}
        self._alerts: list[Alert] = []
        self._lock = threading.Lock()
        self._alert_listeners: list[Callable[[Alert], None]] = []
        self._turn_listeners: list[Callable[[Turn], None]] = []
        self._active_session_id: str | None = None
        self._enable_notifications = enable_notifications
        self._transcript_watchers: dict[str, TranscriptWatcher] = {}  # session_id -> watcher
        self._plan_watchers: dict[str, PlanFileWatcher] = {}  # session_id -> watcher
        self._project_watcher: ProjectWatcher | None = None
        self._initial_load_done = False  # Set after initial scan completes

        # Load config and initialize summarizer
        config = load_config()
        self._summarizer = Summarizer(
            model=config.llm.model,
            api_base=config.llm.api_base,
            api_key=config.llm.api_key,
        )
        self._summary_cache = SummaryCache()  # single cache for both assistant and tool summaries
        self._annotation_cache = AnnotationCache()  # user annotations

    def start_watching(
        self,
        project_paths: list[Path],
        max_sessions: int | None = None,
        projects_dir: Path | None = None,
        global_mode: bool = False,
    ) -> None:
        """Start watching project directories for sessions.

        Args:
            project_paths: List of project directories to watch for .jsonl session files
            max_sessions: Maximum number of sessions to load on initial scan.
                If None, all sessions are loaded. New sessions are always detected.
            projects_dir: Parent directory for project subdirectories (for global mode discovery).
            global_mode: If True, dynamically discover new project directories.
        """
        self._project_watcher = ProjectWatcher(
            project_paths,
            on_session_discovered=self._on_session_discovered,
            max_sessions=max_sessions,
            projects_dir=projects_dir,
            global_mode=global_mode,
            on_initial_load_done=self._on_initial_load_done,
        )
        self._project_watcher.start()

    def _on_initial_load_done(self) -> None:
        """Called when ProjectWatcher finishes its initial scan."""
        with self._lock:
            self._initial_load_done = True

    def _on_session_discovered(self, session_id: str, transcript_path: Path) -> None:
        """Handle discovery of a new session file.

        Args:
            session_id: The session ID (stem of the .jsonl file)
            transcript_path: Path to the transcript .jsonl file
        """
        # Extract project_path from transcript path
        project_path = ""
        parts = str(transcript_path).split("/")
        for i, part in enumerate(parts):
            if part == "projects" and i + 1 < len(parts):
                project_path = parts[i + 1]
                break

        with self._lock:
            # Skip if session already exists
            if session_id in self._sessions:
                return

            # Create session
            session = Session(
                session_id=session_id,
                project_path=project_path,
            )
            self._sessions[session_id] = session

            # During initial load, select first session only (most recent due to sort order).
            # After initial load, always auto-activate newly discovered sessions.
            if self._initial_load_done or self._active_session_id is None:
                self._active_session_id = session_id

        # Load transcript history and start watcher (outside lock)
        file_position = self._load_transcript_history(session_id, str(transcript_path))

        # Load cached annotations for this session
        self.load_annotations_for_session(session_id)

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            start_turn = len(session.turns)

        self._create_transcript_watcher(session_id, str(transcript_path), start_turn, file_position)

        # Detect git branch and start plan file watcher
        if project_path:
            from .utils import decode_project_path
            project_dir = decode_project_path(project_path)

            # Try to detect git branch (non-blocking, best-effort)
            if project_dir:
                branch = self._detect_git_branch(project_dir)
                if branch:
                    with self._lock:
                        session = self._sessions.get(session_id)
                        if session:
                            session.branch = branch

            if project_dir:
                plan_watcher = PlanFileWatcher(
                    project_dir,
                    lambda content, path, sid=session_id: self._on_plan_update(sid, content, path)
                )
                plan_watcher.start()
                with self._lock:
                    self._plan_watchers[session_id] = plan_watcher

    @staticmethod
    def _detect_git_branch(project_dir: str) -> str | None:
        """Detect the current git branch for a project directory.

        Returns branch name or None if not a git repo / detection fails.
        """
        import subprocess
        try:
            result = subprocess.run(
                ["git", "-C", project_dir, "branch", "--show-current"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return None

    def _add_alert(self, alert: Alert) -> None:
        """Add an alert and notify listeners."""
        with self._lock:
            self._alerts.append(alert)

        # Send system notification for warnings and dangers
        if self._enable_notifications and alert.level in (AlertLevel.WARNING, AlertLevel.DANGER):
            send_system_notification(alert)

        # Notify alert listeners
        for listener in self._alert_listeners:
            try:
                listener(alert)
            except Exception:
                pass

    def get_alerts(self, level: AlertLevel | None = None) -> list[Alert]:
        """Get alerts, optionally filtered by level."""
        with self._lock:
            if level is None:
                return list(self._alerts)
            return [a for a in self._alerts if a.level == level]

    def get_recent_alerts(self, count: int = 5) -> list[Alert]:
        """Get the most recent alerts."""
        with self._lock:
            return list(self._alerts[-count:])

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()

    def get_sessions(self) -> list[Session]:
        """Get all sessions sorted by start time (most recent first)."""
        with self._lock:
            return sorted(
                self._sessions.values(),
                key=lambda s: s.started_at,
                reverse=True,
            )

    def get_active_session(self) -> Session | None:
        """Get the currently active session."""
        with self._lock:
            if self._active_session_id:
                return self._sessions.get(self._active_session_id)
            return None

    def set_active_session(self, index: int) -> bool:
        """Set active session by index (1-based). Returns True if successful."""
        with self._lock:
            sessions = sorted(
                self._sessions.values(),
                key=lambda s: s.started_at,
                reverse=True,
            )
            if 1 <= index <= len(sessions):
                self._active_session_id = sessions[index - 1].session_id
                return True
            return False

    def set_active_session_by_id(self, session_id: str) -> bool:
        """Set active session by session ID. Returns True if successful."""
        with self._lock:
            if session_id in self._sessions:
                self._active_session_id = session_id
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and clean up its watchers.

        Args:
            session_id: ID of the session to delete

        Returns:
            True if session was deleted, False if not found
        """
        with self._lock:
            if session_id not in self._sessions:
                return False

            # Stop and remove transcript watcher if exists
            transcript_watcher = self._transcript_watchers.pop(session_id, None)
            if transcript_watcher:
                transcript_watcher.stop()

            # Stop and remove plan watcher if exists
            plan_watcher = self._plan_watchers.pop(session_id, None)
            if plan_watcher:
                plan_watcher.stop()

            # Remove session
            del self._sessions[session_id]

            # If this was the active session, switch to another one
            if self._active_session_id == session_id:
                # Get remaining sessions sorted by start time (most recent first)
                remaining = sorted(
                    self._sessions.values(),
                    key=lambda s: s.started_at,
                    reverse=True,
                )
                self._active_session_id = remaining[0].session_id if remaining else None

            return True

    def add_alert_listener(self, listener: Callable[[Alert], None]) -> None:
        """Add a listener to be called on new alerts."""
        self._alert_listeners.append(listener)

    def remove_alert_listener(self, listener: Callable[[Alert], None]) -> None:
        """Remove an alert listener."""
        if listener in self._alert_listeners:
            self._alert_listeners.remove(listener)

    def add_turn_listener(self, listener: Callable[[Turn], None]) -> None:
        """Add a listener to be called on new turns from watcher."""
        self._turn_listeners.append(listener)

    def remove_turn_listener(self, listener: Callable[[Turn], None]) -> None:
        """Remove a turn listener."""
        if listener in self._turn_listeners:
            self._turn_listeners.remove(listener)

    def _process_task_operation(self, session: Session, turn: Turn) -> None:
        """Process task operations from a turn."""
        if not turn.task_operation:
            return

        operation, data = turn.task_operation

        if operation == "create":
            # Generate next available ID (more robust than len-based)
            existing_ids = [int(tid) for tid in session.tasks.keys() if tid.isdigit()]
            task_id = str(max(existing_ids, default=0) + 1)
            session.tasks[task_id] = TaskState(
                task_id=task_id,
                subject=data.get("subject", "Untitled task"),
                description=data.get("description", ""),
                status="pending",
                activeForm=data.get("activeForm"),
            )

        elif operation == "update":
            task_id = data.get("taskId")
            if not task_id:
                return

            # Create task if missing (defensive)
            if task_id not in session.tasks:
                session.tasks[task_id] = TaskState(
                    task_id=task_id,
                    subject=data.get("subject", "Untitled task"),
                    description=data.get("description", ""),
                    status="pending",
                )

            task = session.tasks[task_id]
            task.updated_at = datetime.now()

            # Update fields if provided
            if data.get("status") is not None:
                task.status = data["status"]
            if data.get("subject") is not None:
                task.subject = data["subject"]
            if data.get("description") is not None:
                task.description = data["description"]
            if data.get("activeForm") is not None:
                task.activeForm = data["activeForm"]
            if data.get("owner") is not None:
                task.owner = data["owner"]

            # Handle blocking relationships
            for block_id in data.get("addBlockedBy", []):
                if block_id not in task.blockedBy:
                    task.blockedBy.append(block_id)
            for block_id in data.get("addBlocks", []):
                if block_id not in task.blocks:
                    task.blocks.append(block_id)

    def _on_plan_update(self, session_id: str, content: str, file_path: str) -> None:
        """Handle plan file changes."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logging.debug(f"Plan update ignored for unknown session: {session_id}")
                return

            # Update plan state
            session.plan_content = content if content else None
            session.plan_file_path = file_path if file_path else None
            session.plan_updated_at = datetime.now() if content else None

        # Plan updates will be picked up on next render cycle (TUI polls every 0.2s)

    def _on_watcher_turn(self, turn: Turn, session_id: str) -> None:
        """Handle new turn from transcript watcher.

        Args:
            turn: The turn to add
            session_id: ID of the session this turn belongs to
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                # Renumber turn based on current session turns
                turn.turn_number = len(session.turns) + 1
                session.turns.append(turn)

                # Process task operations
                self._process_task_operation(session, turn)

        # Check for alerts on tool turns
        alerts = check_turn_for_alerts(turn)
        for alert in alerts:
            self._add_alert(alert)

        # Summarize preceding tools when a user or assistant turn arrives
        if turn.role in ("user", "assistant"):
            session = self._sessions.get(session_id)
            if session:
                # Get preceding tool turns
                _, tool_turns = _get_turn_context(session, turn)

                # Summarize tool group (if there are tools)
                if tool_turns:
                    tool_cache_key = make_tool_cache_key(tool_turns, self._summarizer.model, TOOL_SYSTEM_PROMPT)
                    cached_tool = self._summary_cache.get(tool_cache_key)
                    if cached_tool:
                        # Store on first tool turn
                        tool_turns[0].summary = cached_tool
                    else:
                        self._summarizer.summarize_tools_async(
                            tool_turns,
                            self._make_tool_summary_callback(tool_turns, tool_cache_key)
                        )

                # Summarize assistant text (only for assistant turns)
                if turn.role == "assistant":
                    asst_cache_key = self._make_assistant_cache_key(turn.content_full)
                    cached_asst = self._summary_cache.get(asst_cache_key)
                    if cached_asst:
                        turn.summary = cached_asst
                    else:
                        self._summarizer.summarize_async(
                            turn,
                            self._make_summary_callback(turn, asst_cache_key)
                        )

                    # Critique assistant turn (only for non-historical)
                    if not turn.is_historical:
                        critic_context = _get_critic_context(session, turn)
                        critic_cache_key = make_critic_cache_key(turn, critic_context, self._summarizer.model, CRITIC_SYSTEM_PROMPT)
                        cached_critic = self._summary_cache.get(critic_cache_key)
                        if cached_critic:
                            turn.critic = cached_critic
                            # Extract sentiment from cached critic
                            if cached_critic.startswith("✓"):
                                turn.critic_sentiment = "progress"
                            elif cached_critic.startswith("⚠"):
                                turn.critic_sentiment = "concern"
                            elif cached_critic.startswith("✗"):
                                turn.critic_sentiment = "critical"
                        else:
                            self._summarizer.critique_async(
                                turn,
                                critic_context,
                                self._make_critic_callback(turn, critic_cache_key)
                            )

        # Notify turn listeners (outside lock)
        for listener in self._turn_listeners:
            try:
                listener(turn)
            except Exception:
                pass

    def _create_transcript_watcher(
        self,
        session_id: str,
        transcript_path: str,
        start_turn: int,
        start_position: int = 0,
    ) -> None:
        """Create and register a transcript watcher for a session.

        Args:
            session_id: ID of the session to watch
            transcript_path: Path to the transcript file
            start_turn: Turn number to start from
            start_position: File position to start reading from

        Note:
            This method handles thread-safe registration. The watcher is added
            to _transcript_watchers dict before calling watch() to prevent race
            conditions with stop().
        """
        # Create watcher with session_id captured in callback
        watcher = TranscriptWatcher(lambda turn, sid=session_id: self._on_watcher_turn(turn, sid))
        # Register watcher under lock, but start watching outside to avoid blocking.
        # Use cancellation flag to handle race with delete_session().
        with self._lock:
            if session_id not in self._sessions:
                # Session was deleted, don't register watcher
                return
            self._transcript_watchers[session_id] = watcher
        # Start watching outside lock. If delete_session() runs between registration and here,
        # it will call watcher.stop() which sets _cancelled, so watch() will be a no-op.
        watcher.watch(transcript_path, start_turn, start_position=start_position)

    def _make_assistant_cache_key(self, content: str) -> str:
        """Create cache key for assistant text (no tool context).

        Args:
            content: Assistant's text response

        Returns:
            Cache key that uniquely identifies this assistant text
        """
        fingerprint = _prompt_fingerprint(self._summarizer.model, SYSTEM_PROMPT)
        return f"ASST::{fingerprint}::{content}"

    def _make_summary_callback(self, turn: Turn, cache_key: str) -> Callable[[str, bool], None]:
        """Create a callback that updates turn summary and notifies listeners."""
        def callback(summary: str, success: bool) -> None:
            turn.summary = summary
            # Only cache successful summaries
            if success:
                self._summary_cache.set(cache_key, summary)
            # Notify turn listeners to refresh TUI
            for listener in self._turn_listeners:
                try:
                    listener(turn)
                except Exception:
                    pass
        return callback

    def _make_tool_summary_callback(self, tool_turns: list[Turn], cache_key: str) -> Callable[[str, bool], None]:
        """Create a callback that updates tool group summary and notifies listeners."""
        def callback(summary: str, success: bool) -> None:
            # Store summary on first tool turn (ToolGroup reads from there)
            if tool_turns:
                tool_turns[0].summary = summary
            # Only cache successful summaries
            if success:
                self._summary_cache.set(cache_key, summary)
            # Notify turn listeners to refresh TUI
            for listener in self._turn_listeners:
                try:
                    # Notify with first tool turn so TUI knows to refresh
                    if tool_turns:
                        listener(tool_turns[0])
                except Exception:
                    pass
        return callback

    def _make_critic_callback(self, turn: Turn, cache_key: str) -> Callable[[str, str, bool], None]:
        """Create callback that updates turn critic and notifies listeners."""
        def callback(critique: str, sentiment: str, success: bool) -> None:
            turn.critic = critique
            turn.critic_sentiment = sentiment
            if success:
                self._summary_cache.set(cache_key, critique)
                # Fire alert on critical sentiment
                if sentiment == "critical":
                    alert = Alert(
                        level=AlertLevel.DANGER,
                        title="Critic: critical issue",
                        message=critique,
                        turn=turn,
                    )
                    self._add_alert(alert)
            # Notify listeners
            for listener in self._turn_listeners:
                try:
                    listener(turn)
                except Exception:
                    pass
        return callback

    def summarize_tool_group(self, tool_turns: list[Turn]) -> bool:
        """Submit a specific tool group for summarization.

        Args:
            tool_turns: List of consecutive tool turns to summarize

        Returns:
            True if summarization was submitted, False if already has summary or empty.
        """
        if not tool_turns:
            return False

        # Check if already has summary
        if tool_turns[0].summary:
            return False

        tool_cache_key = make_tool_cache_key(tool_turns, self._summarizer.model, TOOL_SYSTEM_PROMPT)
        cached_tool = self._summary_cache.get(tool_cache_key)
        if cached_tool:
            tool_turns[0].summary = cached_tool
            return False

        self._summarizer.summarize_tools_async(
            tool_turns,
            self._make_tool_summary_callback(tool_turns, tool_cache_key)
        )
        return True

    def summarize_historical_turns(self) -> int:
        """Submit all turns without summaries for summarization.

        Submits tool groups (before user or assistant turns, plus trailing tools)
        and assistant text. Returns the count of items submitted.
        """
        count = 0
        with self._lock:
            if not self._active_session_id:
                return 0
            session = self._sessions.get(self._active_session_id)
            if not session:
                return 0

            # Track which tool turns we've already processed
            processed_tool_indices: set[int] = set()

            for i, turn in enumerate(session.turns):
                # Summarize tools before user or assistant turns
                if turn.role in ("user", "assistant"):
                    _, tool_turns = _get_turn_context(session, turn)

                    if tool_turns and (not tool_turns[0].summary):
                        tool_cache_key = make_tool_cache_key(tool_turns, self._summarizer.model, TOOL_SYSTEM_PROMPT)
                        cached_tool = self._summary_cache.get(tool_cache_key)
                        if cached_tool:
                            tool_turns[0].summary = cached_tool
                        else:
                            self._summarizer.summarize_tools_async(
                                tool_turns,
                                self._make_tool_summary_callback(tool_turns, tool_cache_key)
                            )
                            count += 1

                    # Mark these tool turns as processed
                    for tt in tool_turns:
                        for j, st in enumerate(session.turns):
                            if st is tt:
                                processed_tool_indices.add(j)
                                break

                # Summarize assistant text
                if turn.role == "assistant" and not turn.summary:
                    asst_cache_key = self._make_assistant_cache_key(turn.content_full)
                    cached = self._summary_cache.get(asst_cache_key)
                    if cached:
                        turn.summary = cached
                    else:
                        self._summarizer.summarize_async(
                            turn,
                            self._make_summary_callback(turn, asst_cache_key)
                        )
                        count += 1

            # Handle trailing tool groups (tools at the end with no following user/assistant)
            trailing_tools: list[Turn] = []
            for i in range(len(session.turns) - 1, -1, -1):
                turn = session.turns[i]
                if turn.role == "tool" and i not in processed_tool_indices:
                    trailing_tools.insert(0, turn)
                else:
                    break

            if trailing_tools and not trailing_tools[0].summary:
                tool_cache_key = make_tool_cache_key(trailing_tools, self._summarizer.model, TOOL_SYSTEM_PROMPT)
                cached_tool = self._summary_cache.get(tool_cache_key)
                if cached_tool:
                    trailing_tools[0].summary = cached_tool
                else:
                    self._summarizer.summarize_tools_async(
                        trailing_tools,
                        self._make_tool_summary_callback(trailing_tools, tool_cache_key)
                    )
                    count += 1

        return count

    def set_annotation(self, turn_number: int, annotation: str) -> bool:
        """Set annotation on a turn in the active session.

        Args:
            turn_number: The turn number to annotate
            annotation: The annotation text (empty string to clear)

        Returns:
            True if annotation was set, False if turn not found
        """
        with self._lock:
            if not self._active_session_id:
                return False
            session = self._sessions.get(self._active_session_id)
            if not session:
                return False

            # Find turn by number
            for turn in session.turns:
                if turn.turn_number == turn_number:
                    turn.annotation = annotation if annotation else None
                    # Persist to cache
                    self._annotation_cache.set(
                        self._active_session_id, turn_number, annotation
                    )
                    return True
            return False

    def load_annotations_for_session(self, session_id: str) -> None:
        """Load cached annotations for a session's turns."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return
            for turn in session.turns:
                cached = self._annotation_cache.get(session_id, turn.turn_number)
                if cached:
                    turn.annotation = cached

    def stop(self) -> None:
        """Stop all watchers and summarizer."""
        # Stop project watcher first (prevents new sessions from being discovered)
        if self._project_watcher:
            self._project_watcher.stop()
            self._project_watcher = None

        # Stop all transcript watchers (copy to avoid race condition)
        with self._lock:
            transcript_watchers = list(self._transcript_watchers.values())
            self._transcript_watchers.clear()
        for watcher in transcript_watchers:
            watcher.stop()

        # Stop all plan watchers (copy to avoid race condition)
        with self._lock:
            plan_watchers = list(self._plan_watchers.values())
            self._plan_watchers.clear()
        for watcher in plan_watchers:
            watcher.stop()

        self._summarizer.shutdown()

    def _load_transcript_history(self, session_id: str, transcript_path: str | None) -> int:
        """Load historical turns from transcript file.

        Returns:
            File position after reading, to be passed to watcher.
        """
        if not transcript_path:
            return 0

        import time

        path = Path(transcript_path)

        # Wait for file to exist (up to 2 seconds)
        for _ in range(20):
            if path.exists():
                break
            time.sleep(0.1)

        if not path.exists():
            return 0

        # Acquire lock when reading from _sessions to avoid race with delete
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            return 0

        # Parse transcript - use as source of truth for all turns
        transcript_turns, file_position = parse_transcript(path)

        if transcript_turns:
            # Replace session turns with transcript (source of truth)
            # This ensures we have all turns even if some were missed
            with self._lock:
                # Re-check session still exists (could have been deleted)
                if session_id not in self._sessions:
                    return file_position
                session.turns = transcript_turns

                # Process task operations from historical turns
                for turn in transcript_turns:
                    self._process_task_operation(session, turn)

            # Apply cached summaries or submit for summarization
            for turn in transcript_turns:
                # Summarize tools before user or assistant turns
                if turn.role in ("user", "assistant"):
                    _, tool_turns = _get_turn_context(session, turn)

                    if tool_turns:
                        tool_cache_key = make_tool_cache_key(tool_turns, self._summarizer.model, TOOL_SYSTEM_PROMPT)
                        cached_tool = self._summary_cache.get(tool_cache_key)
                        if cached_tool:
                            tool_turns[0].summary = cached_tool
                        else:
                            self._summarizer.summarize_tools_async(
                                tool_turns,
                                self._make_tool_summary_callback(tool_turns, tool_cache_key)
                            )

                # Summarize assistant text
                if turn.role == "assistant":
                    asst_cache_key = self._make_assistant_cache_key(turn.content_full)
                    cached = self._summary_cache.get(asst_cache_key)
                    if cached:
                        turn.summary = cached
                    else:
                        self._summarizer.summarize_async(
                            turn,
                            self._make_summary_callback(turn, asst_cache_key)
                        )

        return file_position
