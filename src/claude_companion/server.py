"""Flask HTTP server for receiving Claude Code hook events."""

import logging
import threading
from typing import TYPE_CHECKING

from flask import Flask, jsonify, request

from .models import Event

if TYPE_CHECKING:
    from .store import EventStore


def create_app(store: "EventStore") -> Flask:
    """Create Flask app with the given event store."""
    app = Flask(__name__)

    # Suppress Flask's default logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    @app.route("/event", methods=["POST"])
    def receive_event():
        """Receive hook event from Claude Code."""
        try:
            data = request.get_json(force=True)
            event = Event.from_hook_data(data)
            store.add_event(event)
            return jsonify({"status": "ok"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        sessions = store.get_sessions()
        return jsonify({
            "status": "healthy",
            "sessions": len(sessions),
            "active_sessions": sum(1 for s in sessions if s.active),
        })

    return app


class ServerThread(threading.Thread):
    """Background thread running the Flask server."""

    def __init__(self, store: "EventStore", host: str = "127.0.0.1", port: int = 5432):
        super().__init__(daemon=True)
        self.store = store
        self.host = host
        self.port = port
        self.app = create_app(store)
        self._server = None

    def run(self) -> None:
        """Run the Flask server."""
        from werkzeug.serving import make_server

        self._server = make_server(self.host, self.port, self.app, threaded=True)
        self._server.serve_forever()

    def shutdown(self) -> None:
        """Shutdown the server."""
        if self._server:
            self._server.shutdown()
