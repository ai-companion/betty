"""Thread-safe event store for Claude Companion."""

import threading
from datetime import datetime
from typing import Callable

from .models import Event, Session


class EventStore:
    """Thread-safe store for session events."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._listeners: list[Callable[[Event], None]] = []
        self._active_session_id: str | None = None

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

            # Auto-switch to new session on SessionStart
            if event.event_type == "SessionStart":
                self._active_session_id = session_id

        # Notify listeners (outside lock to avoid deadlock)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # Don't let listener errors break the store

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
