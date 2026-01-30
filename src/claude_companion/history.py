"""Session history management for Claude Companion.

Tracks sessions that have been watched, enabling -c/--continue and -r/--resume features.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# History file location
HISTORY_DIR = Path.home() / ".claude-companion"
HISTORY_FILE = HISTORY_DIR / "history.json"
MAX_HISTORY_SIZE = 100  # Keep last N sessions


@dataclass
class SessionRecord:
    """A record of a watched session."""

    session_id: str
    project_path: str  # e.g., "-Users-kai-src-foo"
    transcript_path: str  # Full path to .jsonl file
    started_at: str  # ISO format timestamp
    last_accessed: str  # ISO format timestamp

    @property
    def project_name(self) -> str:
        """Extract human-readable project name from path."""
        # "-Users-kai-src-foo" -> "foo"
        parts = self.project_path.replace("-", "/").split("/")
        return parts[-1] if parts else self.project_path[:20]

    @property
    def started_at_dt(self) -> datetime:
        """Parse started_at as datetime."""
        try:
            return datetime.fromisoformat(self.started_at)
        except (ValueError, TypeError):
            return datetime.now()

    @property
    def last_accessed_dt(self) -> datetime:
        """Parse last_accessed as datetime."""
        try:
            return datetime.fromisoformat(self.last_accessed)
        except (ValueError, TypeError):
            return datetime.now()


def _ensure_history_dir() -> None:
    """Ensure history directory exists."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _load_history() -> list[dict]:
    """Load history from file."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_history(records: list[dict]) -> None:
    """Save history to file."""
    _ensure_history_dir()
    with open(HISTORY_FILE, "w") as f:
        json.dump(records, f, indent=2)


def save_session(record: SessionRecord) -> None:
    """Save or update a session in history."""
    history = _load_history()

    # Check if session already exists (by session_id)
    existing_idx = None
    for i, r in enumerate(history):
        if r.get("session_id") == record.session_id:
            existing_idx = i
            break

    record_dict = asdict(record)

    if existing_idx is not None:
        # Update existing record, but preserve original started_at
        record_dict["started_at"] = history[existing_idx].get("started_at", record.started_at)
        history[existing_idx] = record_dict
    else:
        # Add new record
        history.append(record_dict)

    # Sort by last_accessed (most recent first) and trim
    history.sort(key=lambda r: r.get("last_accessed", ""), reverse=True)
    history = history[:MAX_HISTORY_SIZE]

    _save_history(history)


def cwd_to_project_path(cwd: str | Path | None = None) -> str:
    """Convert a directory path to Claude's project_path format.

    e.g., /Users/kai/src/foo -> -Users-kai-src-foo
    """
    if cwd is None:
        cwd = Path.cwd()
    return str(cwd).replace("/", "-")


def get_history(limit: int = 20, project_path: str | None = None) -> list[SessionRecord]:
    """Get recent session history, sorted by last_accessed (most recent first).

    Args:
        limit: Maximum number of records to return.
        project_path: If provided, only return sessions matching this project path.
    """
    history = _load_history()
    records = []
    for r in history:
        if len(records) >= limit:
            break
        # Filter by project_path if specified
        if project_path and r.get("project_path") != project_path:
            continue
        try:
            records.append(SessionRecord(
                session_id=r["session_id"],
                project_path=r["project_path"],
                transcript_path=r["transcript_path"],
                started_at=r["started_at"],
                last_accessed=r["last_accessed"],
            ))
        except KeyError:
            continue  # Skip malformed records
    return records


def get_most_recent(project_path: str | None = None) -> SessionRecord | None:
    """Get the most recently accessed session.

    Args:
        project_path: If provided, only consider sessions matching this project path.
    """
    history = get_history(limit=1, project_path=project_path)
    return history[0] if history else None
