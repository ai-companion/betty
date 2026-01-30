"""Thread-safe event store for Claude Companion."""

import threading
from datetime import datetime
from typing import Callable

from .alerts import Alert, AlertLevel, check_event_for_alerts, check_turn_for_alerts, send_system_notification
from .cache import SummaryCache
from .config import load_config
from .history import SessionRecord, save_session
from .models import Event, Session, Turn
from .summarizer import Summarizer
from .transcript import parse_transcript
from .watcher import TranscriptWatcher


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
        self._watcher = TranscriptWatcher(self._on_watcher_turn)

        # Load config and initialize summarizer
        config = load_config()
        self._summarizer = Summarizer(
            provider=config.llm.provider,
            model=config.llm.model,
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
        )
        self._summary_cache = SummaryCache()

    def add_event(self, event: Event) -> None:
        """Add an event to the store."""
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

            # Auto-switch to new session on SessionStart and load history
            if event.event_type == "SessionStart":
                self._active_session_id = session_id
                # Load transcript history using transcript_path from hook
                file_position = self._load_transcript_history(session_id, event.transcript_path)
                # Start watching for new content from where we left off
                if event.transcript_path:
                    start_turn = len(self._sessions[session_id].turns)
                    self._watcher.watch(event.transcript_path, start_turn, start_position=file_position)
                    # Save to session history for -c/-r resume features
                    self._save_to_history(session_id, event.transcript_path)

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

    def _on_watcher_turn(self, turn: Turn) -> None:
        """Handle new turn from transcript watcher."""
        with self._lock:
            if self._active_session_id:
                session = self._sessions.get(self._active_session_id)
                if session:
                    # Renumber turn based on current session turns
                    turn.turn_number = len(session.turns) + 1
                    session.turns.append(turn)

        # Check for alerts on tool turns
        alerts = check_turn_for_alerts(turn)
        for alert in alerts:
            self._add_alert(alert)

        # Submit assistant turns for summarization (check cache first)
        if turn.role == "assistant":
            cached = self._summary_cache.get(turn.content_full)
            if cached:
                turn.summary = cached
            else:
                self._summarizer.summarize_async(turn, self._make_summary_callback(turn))

        # Notify turn listeners (outside lock)
        for listener in self._turn_listeners:
            try:
                listener(turn)
            except Exception:
                pass

    def _make_summary_callback(self, turn: Turn) -> Callable[[str, bool], None]:
        """Create a callback that updates turn summary and notifies listeners."""
        def callback(summary: str, success: bool) -> None:
            turn.summary = summary
            # Only cache successful summaries
            if success:
                self._summary_cache.set(turn.content_full, summary)
            # Notify turn listeners to refresh TUI
            for listener in self._turn_listeners:
                try:
                    listener(turn)
                except Exception:
                    pass
        return callback

    def summarize_historical_turns(self) -> int:
        """Submit all historical assistant turns without summaries for summarization.

        Returns the count of turns submitted.
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
                    # Check cache first
                    cached = self._summary_cache.get(turn.content_full)
                    if cached:
                        turn.summary = cached
                    else:
                        self._summarizer.summarize_async(turn, self._make_summary_callback(turn))
                        count += 1
        return count

    def stop(self) -> None:
        """Stop the watcher and summarizer."""
        self._watcher.stop()
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

        # Start watching for new content from where we left off
        with self._lock:
            start_turn = len(self._sessions[session_id].turns)
        self._watcher.watch(transcript_path, start_turn, start_position=file_position)

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

        session = self._sessions.get(session_id)
        if not session:
            return 0

        # Parse transcript - use as source of truth for all turns
        transcript_turns, file_position = parse_transcript(path)

        if transcript_turns:
            # Apply cached summaries or submit for summarization
            for turn in transcript_turns:
                if turn.role == "assistant":
                    cached = self._summary_cache.get(turn.content_full)
                    if cached:
                        turn.summary = cached
                    else:
                        # Submit for summarization (will be cached when done)
                        self._summarizer.summarize_async(turn, self._make_summary_callback(turn))

            # Replace session turns with transcript (source of truth)
            # This ensures we have all turns even if some were missed
            session.turns = transcript_turns

        return file_position
