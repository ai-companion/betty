"""Thread-safe event store for Claude Companion."""

import logging
import threading
from datetime import datetime
from typing import Callable

from .alerts import Alert, AlertLevel, check_event_for_alerts, check_turn_for_alerts, send_system_notification
from .cache import SummaryCache
from .config import load_config
from .history import SessionRecord, save_session
from .models import Event, Session, TaskState, Turn
from .summarizer import Summarizer, _get_turn_context, make_tool_cache_key
from .transcript import parse_transcript
from .watcher import TranscriptWatcher, PlanFileWatcher


class EventStore:
    """Thread-safe store for session events."""

    def __init__(self, enable_notifications: bool = True) -> None:
        self._sessions: dict[str, Session] = {}
        self._alerts: list[Alert] = []
        self._lock = threading.Lock()
        self._listeners: list[Callable[[Event], None]] = []
        self._alert_listeners: list[Callable[[Alert], None]] = []
        self._turn_listeners: list[Callable[[Turn], None]] = []
        self._active_session_id: str | None = None
        self._enable_notifications = enable_notifications
        self._transcript_watchers: dict[str, TranscriptWatcher] = {}  # session_id -> watcher
        self._plan_watchers: dict[str, PlanFileWatcher] = {}  # session_id -> watcher

        # Load config and initialize summarizer
        config = load_config()
        self._summarizer = Summarizer(
            provider=config.llm.provider,
            model=config.llm.model,
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
        )
        self._summary_cache = SummaryCache()  # Single cache for both assistant and tool summaries

    def add_event(self, event: Event) -> None:
        """Add an event to the store."""
        # Variables to start watchers outside lock (avoids deadlock)
        session_id_for_plan = None
        project_path_for_plan = None
        load_transcript_args = None  # (session_id, transcript_path)

        with self._lock:
            session_id = event.session_id

            # Create session if it doesn't exist
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(
                    session_id=session_id,
                    started_at=event.timestamp,
                )
                # Auto-select first session or new session
                if self._active_session_id is None:
                    self._active_session_id = session_id

            # Add event to session
            self._sessions[session_id].add_event(event)

            # Auto-switch to new session on SessionStart
            if event.event_type == "SessionStart":
                self._active_session_id = session_id
                # Prepare data for loading transcript history (done outside lock to avoid deadlock)
                if event.transcript_path:
                    load_transcript_args = (session_id, event.transcript_path)
                    # Save to session history for -c/-r resume features
                    self._save_to_history(session_id, event.transcript_path)

                # Prepare data for plan watcher (start outside lock to avoid blocking)
                session = self._sessions.get(session_id)
                if session and event.transcript_path:
                    session_id_for_plan = session_id
                    project_path_for_plan = session.project_path

        # Load transcript history OUTSIDE lock to avoid deadlock
        # (_load_transcript_history acquires lock internally when needed)
        if load_transcript_args:
            sid, transcript_path = load_transcript_args
            file_position = self._load_transcript_history(sid, transcript_path)
            # Create transcript watcher with position from history load
            # Re-check session exists (could have been deleted during history load)
            with self._lock:
                session = self._sessions.get(sid)
                if session is None:
                    # Session was deleted, bail out
                    pass
                else:
                    start_turn = len(session.turns)
                    # Create watcher outside lock (it acquires lock internally)
            if session is not None:
                self._create_transcript_watcher(sid, transcript_path, start_turn, file_position)

        # Start plan file watcher OUTSIDE lock to avoid blocking SessionStart hook
        if session_id_for_plan and project_path_for_plan:
            from .utils import decode_project_path
            project_dir = decode_project_path(project_path_for_plan)

            if project_dir:
                # Use default argument to capture session_id by value (avoids closure bug)
                plan_watcher = PlanFileWatcher(
                    project_dir,
                    lambda content, path, sid=session_id_for_plan: self._on_plan_update(sid, content, path)
                )
                # Start watcher outside lock (does file I/O)
                plan_watcher.start()
                # Only acquire lock to store the watcher reference AFTER starting
                with self._lock:
                    self._plan_watchers[session_id_for_plan] = plan_watcher

        # Check for alerts
        alerts = check_event_for_alerts(event)
        for alert in alerts:
            self._add_alert(alert)

        # Notify listeners (outside lock to avoid deadlock)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # Don't let listener errors break the store

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

    def add_listener(self, listener: Callable[[Event], None]) -> None:
        """Add a listener to be called on new events."""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[Event], None]) -> None:
        """Remove a listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

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
        from datetime import datetime

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

        # When assistant turn arrives, summarize both the tool group and the assistant
        if turn.role == "assistant":
            session = self._sessions.get(session_id)
            if session:
                user_turn, tool_turns = _get_turn_context(session, turn)

                # Summarize tool group (if there are tools)
                if tool_turns:
                    tool_cache_key = make_tool_cache_key(tool_turns)
                    cached_tool = self._summary_cache.get(tool_cache_key)
                    if cached_tool:
                        # Store on first tool turn
                        tool_turns[0].summary = cached_tool
                    else:
                        self._summarizer.summarize_tools_async(
                            tool_turns,
                            self._make_tool_summary_callback(tool_turns, tool_cache_key)
                        )

                # Summarize assistant (no tool context now)
                asst_cache_key = self._make_assistant_cache_key(turn.content_full)
                cached_asst = self._summary_cache.get(asst_cache_key)
                if cached_asst:
                    turn.summary = cached_asst
                else:
                    self._summarizer.summarize_async(
                        turn,
                        self._make_summary_callback(turn, asst_cache_key)
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
        return f"ASST::{content}"

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

    def summarize_historical_turns(self) -> int:
        """Submit all historical assistant turns without summaries for summarization.

        Also submits tool groups for summarization.
        Returns the count of items submitted.
        """
        count = 0
        with self._lock:
            if not self._active_session_id:
                return 0
            session = self._sessions.get(self._active_session_id)
            if not session:
                return 0

            for turn in session.turns:
                if turn.is_historical and turn.role == "assistant" and not turn.summary:
                    # Get context
                    user_turn, tool_turns = _get_turn_context(session, turn)

                    # Summarize tool group (if there are tools)
                    if tool_turns and (not tool_turns[0].summary):
                        tool_cache_key = make_tool_cache_key(tool_turns)
                        cached_tool = self._summary_cache.get(tool_cache_key)
                        if cached_tool:
                            tool_turns[0].summary = cached_tool
                        else:
                            self._summarizer.summarize_tools_async(
                                tool_turns,
                                self._make_tool_summary_callback(tool_turns, tool_cache_key)
                            )
                            count += 1

                    # Summarize assistant
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
        return count

    def stop(self) -> None:
        """Stop all watchers and summarizer."""
        # Stop all transcript watchers
        for watcher in self._transcript_watchers.values():
            watcher.stop()
        self._transcript_watchers.clear()
        # Stop all plan watchers
        for watcher in self._plan_watchers.values():
            watcher.stop()
        self._plan_watchers.clear()
        self._summarizer.shutdown()

    def _save_to_history(self, session_id: str, transcript_path: str) -> None:
        """Save session to history for -c/-r resume features."""
        from datetime import datetime

        # Extract project path from transcript path
        project_path = ""
        parts = transcript_path.split("/")
        for i, part in enumerate(parts):
            if part == "projects" and i + 1 < len(parts):
                project_path = parts[i + 1]
                break

        now = datetime.now().isoformat()
        record = SessionRecord(
            session_id=session_id,
            project_path=project_path,
            transcript_path=transcript_path,
            started_at=now,
            last_accessed=now,
        )
        save_session(record)

    def load_session_from_transcript(self, transcript_path: str) -> bool:
        """Load a session from a transcript file (for -c/-r resume).

        Returns True if successful, False otherwise.
        """
        from pathlib import Path

        path = Path(transcript_path)
        if not path.exists():
            return False

        # Extract session_id from filename (e.g., "abc123.jsonl" -> "abc123")
        session_id = path.stem

        # Extract project_path from transcript path
        project_path = ""
        parts = transcript_path.split("/")
        for i, part in enumerate(parts):
            if part == "projects" and i + 1 < len(parts):
                project_path = parts[i + 1]
                break

        with self._lock:
            # Create session
            session = Session(
                session_id=session_id,
                project_path=project_path,
            )
            self._sessions[session_id] = session
            self._active_session_id = session_id

        # Load transcript history and get file position
        file_position = self._load_transcript_history(session_id, transcript_path)

        # Create a new watcher for this session and start watching
        with self._lock:
            start_turn = len(self._sessions[session_id].turns)
        self._create_transcript_watcher(session_id, transcript_path, start_turn, file_position)

        # Update history last_accessed
        self._save_to_history(session_id, transcript_path)

        return True

    def _load_transcript_history(self, session_id: str, transcript_path: str | None) -> int:
        """Load historical turns from transcript file.

        Returns:
            File position after reading, to be passed to watcher.
        """
        if not transcript_path:
            return 0

        from pathlib import Path
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
                if turn.role == "assistant":
                    # Get context
                    user_turn, tool_turns = _get_turn_context(session, turn)

                    # Summarize tool group (if there are tools)
                    if tool_turns:
                        tool_cache_key = make_tool_cache_key(tool_turns)
                        cached_tool = self._summary_cache.get(tool_cache_key)
                        if cached_tool:
                            tool_turns[0].summary = cached_tool
                        else:
                            self._summarizer.summarize_tools_async(
                                tool_turns,
                                self._make_tool_summary_callback(tool_turns, tool_cache_key)
                            )

                    # Summarize assistant (no tool context)
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
