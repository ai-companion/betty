"""Remote API client for Betty server.

Provides ``RemoteStore`` — a polling-based client that exposes the same
interface the TUI expects from ``EventStore``, fetching data from a remote
Betty server over HTTP.
"""

import json
import logging
import threading
import urllib.request
from datetime import datetime
from typing import Any, Callable

from .alerts import Alert, AlertLevel
from .models import Session, TaskState, Turn

logger = logging.getLogger(__name__)

# How often the background thread polls the server (seconds).
_POLL_INTERVAL = 1.5


def _parse_datetime(s: str | None) -> datetime:
    if not s:
        return datetime.now()
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return datetime.now()


def _dict_to_turn(d: dict[str, Any]) -> Turn:
    """Reconstruct a Turn from a server JSON dict."""
    return Turn(
        turn_number=d["turn_number"],
        role=d["role"],
        content_preview=d.get("content_preview", ""),
        content_full=d.get("content_full", ""),
        word_count=d.get("word_count", 0),
        input_tokens=d.get("input_tokens"),
        output_tokens=d.get("output_tokens"),
        cache_creation_tokens=d.get("cache_creation_tokens"),
        cache_read_tokens=d.get("cache_read_tokens"),
        model_id=d.get("model_id"),
        tool_name=d.get("tool_name"),
        timestamp=_parse_datetime(d.get("timestamp")),
        is_historical=d.get("is_historical", False),
        summary=d.get("summary"),
        critic=d.get("critic"),
        critic_sentiment=d.get("critic_sentiment"),
        annotation=d.get("annotation"),
    )


def _dict_to_task(d: dict[str, Any]) -> TaskState:
    return TaskState(
        task_id=d["task_id"],
        subject=d.get("subject", ""),
        description=d.get("description", ""),
        status=d.get("status", "pending"),
        activeForm=d.get("activeForm"),
        owner=d.get("owner"),
        blockedBy=d.get("blockedBy", []),
        blocks=d.get("blocks", []),
        created_at=_parse_datetime(d.get("created_at")),
        updated_at=_parse_datetime(d.get("updated_at")),
    )


def _dict_to_session(d: dict[str, Any], include_turns: bool = False) -> Session:
    """Reconstruct a Session from a server JSON dict."""
    session = Session(
        session_id=d["session_id"],
        project_path=d.get("project_path", ""),
        model=d.get("model", "unknown"),
        started_at=_parse_datetime(d.get("started_at")),
        active=d.get("active", True),
        branch=d.get("branch"),
    )
    # Set pr_info if present
    pr = d.get("pr_info")
    if pr:
        from .github import PRInfo
        session.pr_info = PRInfo(
            number=pr["number"],
            title=pr["title"],
            url=pr["url"],
            state=pr["state"],
        )
    # Set plan
    session.plan_content = d.get("plan_content")
    session.plan_file_path = d.get("plan_file_path")
    # Set tasks
    tasks_raw = d.get("tasks", {})
    session.tasks = {tid: _dict_to_task(td) for tid, td in tasks_raw.items()}
    # Set turns
    if include_turns and "turns" in d:
        session.turns = [_dict_to_turn(t) for t in d["turns"]]
    # Cache display_name from server so we don't need local filesystem
    if "display_name" in d:
        session._display_name_from_path = d["display_name"]
    return session


def _api_get(base_url: str, path: str, timeout: float = 5.0) -> Any:
    """GET a JSON endpoint. Returns parsed JSON or raises."""
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _api_request(base_url: str, path: str, method: str = "GET",
                 data: dict | None = None, timeout: float = 5.0) -> Any:
    """Make a JSON API request. Returns parsed JSON or raises."""
    url = base_url.rstrip("/") + path
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    if body:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def check_server(base_url: str, timeout: float = 2.0) -> bool:
    """Check if a Betty server is reachable."""
    try:
        result = _api_get(base_url, "/api/health", timeout=timeout)
        return result.get("status") == "ok"
    except Exception:
        return False


class RemoteStore:
    """Polling-based client that mirrors the EventStore interface.

    Periodically fetches session data from the Betty server and fires
    turn/alert listeners when changes are detected.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._sessions: dict[str, Session] = {}
        self._active_session_id: str | None = None
        self._alerts: list[Alert] = []
        self._turn_listeners: list[Callable[[Turn], None]] = []
        self._alert_listeners: list[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        self._listener_lock = threading.Lock()
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Track turn counts per session to detect new turns
        self._turn_counts: dict[str, int] = {}

    # --- Polling lifecycle ---

    def start_polling(self) -> None:
        """Start the background polling thread."""
        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        # Do an initial fetch immediately
        self._poll_once()
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_POLL_INTERVAL)
            if not self._stop_event.is_set():
                self._poll_once()

    def _poll_once(self) -> None:
        """Fetch sessions list and active session detail."""
        try:
            data = _api_get(self._base_url, "/api/sessions")
        except Exception:
            return

        server_active_id = data.get("active_session_id")
        session_dicts = data.get("sessions", [])

        with self._lock:
            old_ids = set(self._sessions.keys())
            new_sessions: dict[str, Session] = {}
            for sd in session_dicts:
                sid = sd["session_id"]
                new_sessions[sid] = _dict_to_session(sd)
            # Prune turn counts for removed sessions
            removed_ids = old_ids - set(new_sessions.keys())
            for removed_id in removed_ids:
                self._turn_counts.pop(removed_id, None)
            self._sessions = new_sessions
            self._active_session_id = server_active_id

        # Fetch full detail for active session (with turns)
        if server_active_id:
            try:
                detail = _api_get(self._base_url, f"/api/sessions/{server_active_id}")
                sd = detail.get("session")
                if sd:
                    full_session = _dict_to_session(sd, include_turns=True)
                    with self._lock:
                        self._sessions[server_active_id] = full_session

                    # Detect new turns and fire listeners only for genuinely new ones.
                    # On the very first fetch for a session, initialize the count
                    # without firing (matches EventStore semantics where listeners
                    # only see turns appended after registration).
                    new_count = len(full_session.turns)
                    if server_active_id not in self._turn_counts:
                        # First fetch — set baseline, don't fire
                        self._turn_counts[server_active_id] = new_count
                    else:
                        old_count = self._turn_counts[server_active_id]
                        if new_count > old_count:
                            new_turns = full_session.turns[old_count:]
                            with self._listener_lock:
                                listeners = list(self._turn_listeners)
                            for turn in new_turns:
                                for listener in listeners:
                                    try:
                                        listener(turn)
                                    except Exception:
                                        pass
                        self._turn_counts[server_active_id] = new_count
            except Exception:
                pass

        # Poll alerts
        try:
            alert_data = _api_get(self._base_url, "/api/alerts")
            new_alerts = []
            for ad in alert_data.get("alerts", []):
                new_alerts.append(Alert(
                    level=AlertLevel(ad["level"]),
                    title=ad.get("title", ""),
                    message=ad.get("message", ""),
                ))
            with self._lock:
                old_count = len(self._alerts)
                self._alerts = new_alerts
            # Fire listeners for new alerts
            if len(new_alerts) > old_count:
                with self._listener_lock:
                    alert_listeners = list(self._alert_listeners)
                for alert in new_alerts[old_count:]:
                    for listener in alert_listeners:
                        try:
                            listener(alert)
                        except Exception:
                            pass
        except Exception:
            pass

    # --- EventStore-compatible interface ---

    def get_sessions(self) -> list[Session]:
        with self._lock:
            return sorted(
                self._sessions.values(),
                key=lambda s: s.started_at,
                reverse=True,
            )

    def get_active_session(self) -> Session | None:
        with self._lock:
            if self._active_session_id:
                return self._sessions.get(self._active_session_id)
            return None

    def set_active_session(self, index: int) -> bool:
        try:
            result = _api_request(
                self._base_url, "/api/sessions/active", "PUT",
                data={"index": index},
            )
            return result.get("ok", False)
        except Exception:
            return False

    def set_active_session_by_id(self, session_id: str) -> bool:
        try:
            result = _api_request(
                self._base_url, "/api/sessions/active", "PUT",
                data={"session_id": session_id},
            )
            return result.get("ok", False)
        except Exception:
            return False

    def delete_session(self, session_id: str) -> bool:
        try:
            result = _api_request(
                self._base_url, f"/api/sessions/{session_id}", "DELETE",
            )
            return result.get("ok", False)
        except Exception:
            return False

    def get_alerts(self, level=None) -> list[Alert]:
        with self._lock:
            if level is None:
                return list(self._alerts)
            return [a for a in self._alerts if a.level == level]

    def get_recent_alerts(self, count: int = 5) -> list[Alert]:
        with self._lock:
            return list(self._alerts[-count:])

    def clear_alerts(self) -> None:
        try:
            _api_request(self._base_url, "/api/alerts", "DELETE")
        except Exception:
            pass
        with self._lock:
            self._alerts.clear()

    def add_turn_listener(self, listener: Callable[[Turn], None]) -> None:
        with self._listener_lock:
            self._turn_listeners.append(listener)

    def remove_turn_listener(self, listener: Callable[[Turn], None]) -> None:
        with self._listener_lock:
            if listener in self._turn_listeners:
                self._turn_listeners.remove(listener)

    def add_alert_listener(self, listener: Callable[[Alert], None]) -> None:
        with self._listener_lock:
            self._alert_listeners.append(listener)

    def remove_alert_listener(self, listener: Callable[[Alert], None]) -> None:
        with self._listener_lock:
            if listener in self._alert_listeners:
                self._alert_listeners.remove(listener)

    def update_pr_info(self) -> None:
        # Server handles PR detection; nothing to do client-side.
        pass

    def summarize_tool_group(self, tool_turns: list[Turn]) -> bool:
        # Summarization happens server-side; trigger via summarize_historical_turns.
        return False

    def summarize_historical_turns(self) -> int:
        try:
            result = _api_request(self._base_url, "/api/summarize", "POST")
            return result.get("submitted", 0)
        except Exception:
            return 0

    def set_annotation(self, turn_number: int, annotation: str) -> bool:
        try:
            result = _api_request(
                self._base_url, "/api/annotations", "POST",
                data={"turn_number": turn_number, "annotation": annotation},
            )
            return result.get("ok", False)
        except Exception:
            return False

    def get_agent_report(self, session_id: str | None = None):
        # Agent runs server-side; not exposed via API yet.
        return None

    def ask_agent(self, question: str, callback=None, selected_turns=None) -> str:
        return "no_agent"

    def generate_insight(self, turns: list, label: str, callback=None) -> str:
        return "no_agent"

    @property
    def agent_enabled(self) -> bool:
        return False

    def stop(self) -> None:
        self._stop_event.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=3.0)
