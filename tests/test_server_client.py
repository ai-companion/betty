"""Tests for the Betty server and client (server-client separation)."""

import json
import threading
import time
from datetime import datetime
from http.server import HTTPServer
from unittest.mock import MagicMock, patch

import pytest

from betty.alerts import Alert, AlertLevel
from betty.client import (
    RemoteStore,
    _dict_to_session,
    _dict_to_turn,
    check_server,
)
from betty.models import Session, Turn, TaskState, count_words
from betty.server import (
    DEFAULT_PORT,
    _make_handler,
    _session_to_dict,
    _turn_to_dict,
    _task_to_dict,
)


# --- Helpers ---


def _make_turn(
    turn_number: int = 1,
    role: str = "assistant",
    content: str = "Hello world",
    tool_name: str | None = None,
) -> Turn:
    return Turn(
        turn_number=turn_number,
        role=role,
        content_preview=content[:80],
        content_full=content,
        word_count=count_words(content),
        tool_name=tool_name,
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
    )


def _make_session(session_id: str = "test-session", turns: list[Turn] | None = None) -> Session:
    s = Session(
        session_id=session_id,
        project_path="-home-user-project",
        model="claude-sonnet-4-20250514",
        started_at=datetime(2025, 1, 1, 10, 0, 0),
    )
    if turns:
        s.turns = turns
    return s


# --- Serialization round-trip tests ---


class TestTurnSerialization:
    def test_round_trip(self):
        turn = _make_turn(1, "user", "What is 2+2?")
        turn.summary = "Math question"
        turn.critic = "Good question"
        turn.critic_sentiment = "progress"
        turn.annotation = "test note"

        d = _turn_to_dict(turn)
        restored = _dict_to_turn(d)

        assert restored.turn_number == 1
        assert restored.role == "user"
        assert restored.content_full == "What is 2+2?"
        assert restored.summary == "Math question"
        assert restored.critic == "Good question"
        assert restored.critic_sentiment == "progress"
        assert restored.annotation == "test note"
        assert restored.timestamp == datetime(2025, 1, 1, 12, 0, 0)

    def test_tool_turn(self):
        turn = _make_turn(2, "tool", "Read file.py", tool_name="Read")
        d = _turn_to_dict(turn)
        restored = _dict_to_turn(d)
        assert restored.tool_name == "Read"
        assert restored.role == "tool"

    def test_token_data(self):
        turn = _make_turn()
        turn.input_tokens = 100
        turn.output_tokens = 50
        turn.cache_creation_tokens = 10
        turn.cache_read_tokens = 5
        turn.model_id = "claude-sonnet-4-20250514"

        d = _turn_to_dict(turn)
        restored = _dict_to_turn(d)
        assert restored.input_tokens == 100
        assert restored.output_tokens == 50
        assert restored.cache_creation_tokens == 10
        assert restored.cache_read_tokens == 5
        assert restored.model_id == "claude-sonnet-4-20250514"


class TestTaskSerialization:
    def test_round_trip(self):
        task = TaskState(
            task_id="1",
            subject="Fix bug",
            description="Fix the login bug",
            status="in_progress",
            owner="agent",
            blockedBy=["2"],
            blocks=["3"],
        )
        d = _task_to_dict(task)
        from betty.client import _dict_to_task
        restored = _dict_to_task(d)
        assert restored.task_id == "1"
        assert restored.subject == "Fix bug"
        assert restored.status == "in_progress"
        assert restored.owner == "agent"
        assert restored.blockedBy == ["2"]
        assert restored.blocks == ["3"]


class TestSessionSerialization:
    def test_round_trip_without_turns(self):
        session = _make_session()
        session.branch = "main"
        session.plan_content = "# Plan\n- step 1"

        d = _session_to_dict(session, include_turns=False)
        restored = _dict_to_session(d, include_turns=False)

        assert restored.session_id == "test-session"
        assert restored.project_path == "-home-user-project"
        assert restored.branch == "main"
        assert restored.plan_content == "# Plan\n- step 1"
        assert len(restored.turns) == 0

    def test_round_trip_with_turns(self):
        turns = [
            _make_turn(1, "user", "Hello"),
            _make_turn(2, "assistant", "Hi there"),
        ]
        session = _make_session(turns=turns)

        d = _session_to_dict(session, include_turns=True)
        restored = _dict_to_session(d, include_turns=True)

        assert len(restored.turns) == 2
        assert restored.turns[0].role == "user"
        assert restored.turns[1].role == "assistant"

    def test_pr_info_round_trip(self):
        from betty.github import PRInfo
        session = _make_session()
        session.pr_info = PRInfo(number=42, title="Fix things", url="https://github.com/foo/bar/pull/42", state="OPEN")

        d = _session_to_dict(session)
        restored = _dict_to_session(d)

        assert restored.pr_info is not None
        assert restored.pr_info.number == 42
        assert restored.pr_info.title == "Fix things"
        assert restored.pr_info.state == "OPEN"

    def test_display_name_cached(self):
        session = _make_session()
        d = _session_to_dict(session)
        # Server serializes display_name
        assert "display_name" in d
        restored = _dict_to_session(d)
        # Client caches it to avoid needing local filesystem
        assert restored._display_name_from_path == d["display_name"]

    def test_tasks_round_trip(self):
        session = _make_session()
        session.tasks["1"] = TaskState(
            task_id="1", subject="Do thing", description="desc", status="pending"
        )
        d = _session_to_dict(session)
        restored = _dict_to_session(d)
        assert "1" in restored.tasks
        assert restored.tasks["1"].subject == "Do thing"


# --- Server handler tests (using mock store) ---


def _create_test_server(store):
    """Create a test HTTP server on a random port."""
    handler = _make_handler(store)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


class TestServerEndpoints:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.store = MagicMock()
        self.store.get_sessions.return_value = []
        self.store.get_active_session.return_value = None
        self.store.get_alerts.return_value = []
        self.server, self.base_url = _create_test_server(self.store)
        yield
        self.server.shutdown()

    def _get(self, path):
        import urllib.request
        url = self.base_url + path
        with urllib.request.urlopen(url, timeout=2) as resp:
            return json.loads(resp.read())

    def _request(self, path, method="GET", data=None):
        import urllib.request
        url = self.base_url + path
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, method=method)
        if body:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return json.loads(resp.read())

    def test_health(self):
        result = self._get("/api/health")
        assert result == {"status": "ok"}

    def test_sessions_empty(self):
        result = self._get("/api/sessions")
        assert result["sessions"] == []
        assert result["active_session_id"] is None

    def test_sessions_with_data(self):
        session = _make_session()
        self.store.get_sessions.return_value = [session]
        active = _make_session("active-id")
        self.store.get_active_session.return_value = active

        result = self._get("/api/sessions")
        assert len(result["sessions"]) == 1
        assert result["sessions"][0]["session_id"] == "test-session"
        assert result["active_session_id"] == "active-id"

    def test_session_detail(self):
        turns = [_make_turn(1, "user", "Hello")]
        session = _make_session(turns=turns)
        self.store.get_sessions.return_value = [session]
        self.store.get_active_session.return_value = session

        result = self._get(f"/api/sessions/{session.session_id}")
        assert result["session"]["session_id"] == "test-session"
        assert len(result["session"]["turns"]) == 1
        assert result["is_active"] is True

    def test_session_not_found(self):
        import urllib.error
        self.store.get_sessions.return_value = []
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            self._get("/api/sessions/nonexistent")
        assert exc_info.value.code == 404

    def test_set_active_session_by_id(self):
        self.store.set_active_session_by_id.return_value = True
        result = self._request(
            "/api/sessions/active", "PUT",
            data={"session_id": "test-session"},
        )
        assert result["ok"] is True
        self.store.set_active_session_by_id.assert_called_once_with("test-session")

    def test_set_active_session_by_index(self):
        self.store.set_active_session.return_value = True
        result = self._request(
            "/api/sessions/active", "PUT",
            data={"index": 2},
        )
        assert result["ok"] is True
        self.store.set_active_session.assert_called_once_with(2)

    def test_delete_session(self):
        self.store.delete_session.return_value = True
        result = self._request("/api/sessions/test-session", "DELETE")
        assert result["ok"] is True
        self.store.delete_session.assert_called_once_with("test-session")

    def test_alerts(self):
        self.store.get_alerts.return_value = [
            Alert(level=AlertLevel.WARNING, title="Watch out", message="Careful"),
        ]
        result = self._get("/api/alerts")
        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["level"] == "warning"
        assert result["alerts"][0]["title"] == "Watch out"

    def test_clear_alerts(self):
        result = self._request("/api/alerts", "DELETE")
        assert result["ok"] is True
        self.store.clear_alerts.assert_called_once()

    def test_summarize(self):
        self.store.summarize_historical_turns.return_value = 5
        result = self._request("/api/summarize", "POST")
        assert result["submitted"] == 5

    def test_set_annotation(self):
        self.store.set_annotation.return_value = True
        result = self._request(
            "/api/annotations", "POST",
            data={"turn_number": 3, "annotation": "important"},
        )
        assert result["ok"] is True
        self.store.set_annotation.assert_called_once_with(3, "important")


# --- Client tests ---


class TestCheckServer:
    def test_unreachable(self):
        assert check_server("http://127.0.0.1:19999", timeout=0.5) is False

    def test_reachable(self):
        store = MagicMock()
        store.get_sessions.return_value = []
        store.get_active_session.return_value = None
        store.get_alerts.return_value = []
        server, base_url = _create_test_server(store)
        try:
            assert check_server(base_url) is True
        finally:
            server.shutdown()


class TestRemoteStore:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_store = MagicMock()
        turns = [_make_turn(1, "user", "Hello"), _make_turn(2, "assistant", "Hi")]
        self.session = _make_session(turns=turns)
        self.mock_store.get_sessions.return_value = [self.session]
        self.mock_store.get_active_session.return_value = self.session
        self.mock_store.get_alerts.return_value = []
        self.mock_store.set_active_session_by_id.return_value = True
        self.mock_store.set_active_session.return_value = True
        self.mock_store.delete_session.return_value = True
        self.mock_store.summarize_historical_turns.return_value = 0
        self.mock_store.set_annotation.return_value = True

        self.server, self.base_url = _create_test_server(self.mock_store)
        self.client = RemoteStore(self.base_url)
        yield
        self.client.stop()
        self.server.shutdown()

    def test_get_sessions(self):
        self.client._poll_once()
        sessions = self.client.get_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == "test-session"

    def test_get_active_session_with_turns(self):
        self.client._poll_once()
        active = self.client.get_active_session()
        assert active is not None
        assert active.session_id == "test-session"
        assert len(active.turns) == 2

    def test_turn_listener_fires_on_new_turns(self):
        received = []
        self.client.add_turn_listener(lambda t: received.append(t))

        # First poll sets baseline
        self.client._poll_once()
        assert len(received) == 2  # Initial turns

        # Add a turn on the "server" side
        new_turn = _make_turn(3, "user", "Follow up")
        self.session.turns.append(new_turn)

        # Second poll should detect the new turn
        self.client._poll_once()
        assert len(received) == 3
        assert received[-1].content_full == "Follow up"

    def test_set_active_session(self):
        assert self.client.set_active_session(1) is True
        assert self.client.set_active_session_by_id("test-session") is True

    def test_delete_session(self):
        assert self.client.delete_session("test-session") is True

    def test_clear_alerts(self):
        self.client.clear_alerts()
        self.mock_store.clear_alerts.assert_called()

    def test_summarize(self):
        self.mock_store.summarize_historical_turns.return_value = 3
        count = self.client.summarize_historical_turns()
        assert count == 3

    def test_annotation(self):
        assert self.client.set_annotation(1, "note") is True

    def test_agent_not_available(self):
        assert self.client.agent_enabled is False
        assert self.client.get_agent_report() is None
        assert self.client.ask_agent("question") == "no_agent"

    def test_update_pr_info_noop(self):
        # Should not raise
        self.client.update_pr_info()

    def test_polling_lifecycle(self):
        self.client.start_polling()
        time.sleep(0.3)  # Let at least one poll happen
        sessions = self.client.get_sessions()
        assert len(sessions) == 1
        self.client.stop()


# --- Local fallback test ---


class TestLocalFallback:
    def test_fallback_when_server_unreachable(self):
        """Verify check_server returns False for unreachable server."""
        assert check_server("http://127.0.0.1:19999", timeout=0.3) is False
