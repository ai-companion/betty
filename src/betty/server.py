"""HTTP API server for Betty session data.

Wraps EventStore and exposes session data via a lightweight REST API
using Python's built-in http.server (no extra dependencies).
"""

import json
import logging
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models import Session, Turn, TaskState
from .store import EventStore

logger = logging.getLogger(__name__)

DEFAULT_PORT = 5557

# Maximum request body size (1 MB) to prevent memory exhaustion.
_MAX_BODY_SIZE = 1 * 1024 * 1024


def _turn_to_dict(turn: Turn) -> dict[str, Any]:
    """Serialize a Turn to a JSON-safe dict."""
    return {
        "turn_number": turn.turn_number,
        "role": turn.role,
        "content_preview": turn.content_preview,
        "content_full": turn.content_full,
        "word_count": turn.word_count,
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "cache_creation_tokens": turn.cache_creation_tokens,
        "cache_read_tokens": turn.cache_read_tokens,
        "model_id": turn.model_id,
        "tool_name": turn.tool_name,
        "timestamp": turn.timestamp.isoformat(),
        "is_historical": turn.is_historical,
        "summary": turn.summary,
        "critic": turn.critic,
        "critic_sentiment": turn.critic_sentiment,
        "annotation": turn.annotation,
    }


def _task_to_dict(task: TaskState) -> dict[str, Any]:
    """Serialize a TaskState to a JSON-safe dict."""
    return {
        "task_id": task.task_id,
        "subject": task.subject,
        "description": task.description,
        "status": task.status,
        "activeForm": task.activeForm,
        "owner": task.owner,
        "blockedBy": task.blockedBy,
        "blocks": task.blocks,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
    }


def _session_to_dict(session: Session, include_turns: bool = False) -> dict[str, Any]:
    """Serialize a Session to a JSON-safe dict."""
    pr = None
    if session.pr_info:
        pr = {
            "number": session.pr_info.number,
            "title": session.pr_info.title,
            "url": session.pr_info.url,
            "state": session.pr_info.state,
        }

    d: dict[str, Any] = {
        "session_id": session.session_id,
        "project_path": session.project_path,
        "model": session.model,
        "started_at": session.started_at.isoformat(),
        "display_name": session.display_name,
        "last_activity": session.last_activity.isoformat(),
        "branch": session.branch,
        "pr_info": pr,
        "active": session.active,
        "total_input_tokens": session.total_input_tokens,
        "total_output_tokens": session.total_output_tokens,
        "total_cache_creation_tokens": session.total_cache_creation_tokens,
        "total_cache_read_tokens": session.total_cache_read_tokens,
        "total_tool_calls": session.total_tool_calls,
        "total_input_words": session.total_input_words,
        "total_output_words": session.total_output_words,
        "has_token_data": session.has_token_data,
        "estimated_cost": session.estimated_cost,
        "plan_content": session.plan_content,
        "plan_file_path": session.plan_file_path,
        "turn_count": len(session.turns),
        "tasks": {tid: _task_to_dict(t) for tid, t in session.tasks.items()},
    }
    if include_turns:
        d["turns"] = [_turn_to_dict(t) for t in session.turns]
    return d


def _make_handler(store: EventStore):
    """Create a request handler class bound to the given store."""

    class BettyAPIHandler(BaseHTTPRequestHandler):
        """Handle Betty API requests."""

        def log_message(self, format, *args):
            logger.debug(format, *args)

        def _send_json(self, data: Any, status: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: int, message: str) -> None:
            self._send_json({"error": message}, status)

        def _read_body(self) -> bytes:
            raw = self.headers.get("Content-Length", "0")
            try:
                length = int(raw)
            except (ValueError, TypeError):
                return b""
            if length <= 0:
                return b""
            if length > _MAX_BODY_SIZE:
                return b""
            return self.rfile.read(length)

        def do_GET(self) -> None:
            path = urlparse(self.path).path.rstrip("/")

            if path == "/api/health":
                self._send_json({"status": "ok"})

            elif path == "/api/sessions":
                sessions = store.get_sessions()
                active = store.get_active_session()
                active_id = active.session_id if active else None
                self._send_json({
                    "sessions": [_session_to_dict(s) for s in sessions],
                    "active_session_id": active_id,
                })

            elif path.startswith("/api/sessions/"):
                session_id = path[len("/api/sessions/"):]
                # Find session by ID
                found = None
                for s in store.get_sessions():
                    if s.session_id == session_id:
                        found = s
                        break
                if found:
                    active = store.get_active_session()
                    self._send_json({
                        "session": _session_to_dict(found, include_turns=True),
                        "is_active": found.session_id == (active.session_id if active else None),
                    })
                else:
                    self._send_error(404, "Session not found")

            elif path == "/api/alerts":
                alerts = store.get_alerts()
                self._send_json({
                    "alerts": [
                        {
                            "level": a.level.value,
                            "title": a.title,
                            "message": a.message,
                            "timestamp": a.timestamp.isoformat() if hasattr(a, "timestamp") else None,
                        }
                        for a in alerts
                    ],
                })

            else:
                self._send_error(404, "Not found")

        def do_PUT(self) -> None:
            path = urlparse(self.path).path.rstrip("/")

            if path == "/api/sessions/active":
                body = self._read_body()
                try:
                    data = json.loads(body)
                except (json.JSONDecodeError, ValueError):
                    self._send_error(400, "Invalid JSON")
                    return

                if "session_id" in data:
                    ok = store.set_active_session_by_id(data["session_id"])
                elif "index" in data:
                    ok = store.set_active_session(data["index"])
                else:
                    self._send_error(400, "Provide session_id or index")
                    return
                self._send_json({"ok": ok})

            else:
                self._send_error(404, "Not found")

        def do_POST(self) -> None:
            path = urlparse(self.path).path.rstrip("/")

            if path == "/api/summarize":
                count = store.summarize_historical_turns()
                self._send_json({"submitted": count})

            elif path == "/api/annotations":
                body = self._read_body()
                try:
                    data = json.loads(body)
                except (json.JSONDecodeError, ValueError):
                    self._send_error(400, "Invalid JSON")
                    return
                turn_number = data.get("turn_number")
                annotation = data.get("annotation", "")
                if turn_number is None:
                    self._send_error(400, "Provide turn_number")
                    return
                ok = store.set_annotation(turn_number, annotation)
                self._send_json({"ok": ok})

            else:
                self._send_error(404, "Not found")

        def do_DELETE(self) -> None:
            path = urlparse(self.path).path.rstrip("/")

            if path == "/api/alerts":
                store.clear_alerts()
                self._send_json({"ok": True})

            elif path.startswith("/api/sessions/"):
                session_id = path[len("/api/sessions/"):]
                ok = store.delete_session(session_id)
                if ok:
                    self._send_json({"ok": True})
                else:
                    self._send_error(404, "Session not found")

            else:
                self._send_error(404, "Not found")

    return BettyAPIHandler


def run_server(
    project_paths: list[Path],
    port: int = DEFAULT_PORT,
    host: str = "127.0.0.1",
    global_mode: bool = False,
    projects_dir: Path | None = None,
    backend: str = "file",
    tmux_socket: str | None = None,
) -> None:
    """Start the Betty API server.

    Creates an EventStore, starts watching for sessions, and serves the API.

    Args:
        project_paths: Project directories to watch.
        port: Port to listen on.
        host: Host to bind to.
        global_mode: Watch all projects.
        projects_dir: Parent projects directory (for global mode).
        backend: Discovery backend: ``"file"`` (default) or ``"tmux"``.
        tmux_socket: Tmux socket name for the tmux backend.
    """
    store = EventStore(enable_notifications=False)
    if backend == "tmux":
        store.start_watching_tmux(socket=tmux_socket)
    else:
        store.start_watching(
            project_paths,
            max_sessions=None,
            projects_dir=projects_dir,
            global_mode=global_mode,
        )

    handler = _make_handler(store)
    server = ThreadingHTTPServer((host, port), handler)
    logger.info("Betty server listening on %s:%d", host, port)
    print(f"Betty server listening on http://{host}:{port}")
    print(f"  Health:   http://{host}:{port}/api/health")
    print(f"  Sessions: http://{host}:{port}/api/sessions")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        store.stop()
        print("\nServer stopped.")
