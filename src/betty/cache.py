"""Persistent cache for summaries and annotations for Betty."""

import hashlib
import json
import logging
from pathlib import Path
from threading import Lock

from filelock import FileLock

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "betty"
SUMMARIES_FILENAME = "summaries.json"
ANNOTATIONS_FILENAME = "annotations.json"
AGENT_REPORTS_FILENAME = "agent_reports.json"
AGENT_REPORT_MAX_AGE_DAYS = 7


def content_hash(content: str) -> str:
    """Generate a short hash for content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class SummaryCache:
    """Thread-safe persistent cache for turn summaries with file locking."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self._cache_dir = cache_dir
        self._cache_file = cache_dir / SUMMARIES_FILENAME
        self._lock_file = cache_dir / f"{SUMMARIES_FILENAME}.lock"
        self._cache: dict[str, str] = {}
        self._thread_lock = Lock()
        self._load()

    def _read_from_disk(self) -> dict[str, str]:
        """Read cache from disk without locking."""
        if not self._cache_file.exists():
            return {}
        try:
            with open(self._cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read summary cache: {e}")
            return {}

    def _write_to_disk(self, data: dict[str, str]) -> None:
        """Write cache to disk without locking."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(data, f)
        except IOError as e:
            logger.warning(f"Failed to write summary cache: {e}")

    def _load(self) -> None:
        """Load cache from disk with file lock."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_file):
            self._cache = self._read_from_disk()
            if self._cache:
                logger.debug(f"Loaded {len(self._cache)} cached summaries")

    def _save(self) -> None:
        """Save cache to disk with file locking and merge."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_file):
            # Re-read to get any changes from other instances
            disk_cache = self._read_from_disk()
            # Merge: summaries are content-addressed, so simple merge is safe
            merged = {**disk_cache, **self._cache}
            self._write_to_disk(merged)
            self._cache = merged

    def get(self, content: str) -> str | None:
        """Get cached summary for content."""
        key = content_hash(content)
        with self._thread_lock:
            return self._cache.get(key)

    def set(self, content: str, summary: str) -> None:
        """Store summary for content."""
        key = content_hash(content)
        with self._thread_lock:
            self._cache[key] = summary
            self._save()

    def __len__(self) -> int:
        """Return number of cached summaries."""
        with self._thread_lock:
            return len(self._cache)


def annotation_key(session_id: str, turn_number: int) -> str:
    """Generate key for annotation lookup."""
    return f"{session_id}:{turn_number}"


class AnnotationCache:
    """Thread-safe persistent cache for user annotations with file locking."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self._cache_dir = cache_dir
        self._cache_file = cache_dir / ANNOTATIONS_FILENAME
        self._lock_file = cache_dir / f"{ANNOTATIONS_FILENAME}.lock"
        self._cache: dict[str, str] = {}
        self._deleted_keys: set[str] = set()  # Track deletions for merge
        self._thread_lock = Lock()
        self._load()

    def _read_from_disk(self) -> dict[str, str]:
        """Read cache from disk without locking."""
        if not self._cache_file.exists():
            return {}
        try:
            with open(self._cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read annotation cache: {e}")
            return {}

    def _write_to_disk(self, data: dict[str, str]) -> None:
        """Write cache to disk without locking."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(data, f)
        except IOError as e:
            logger.warning(f"Failed to write annotation cache: {e}")

    def _load(self) -> None:
        """Load cache from disk with file lock."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_file):
            self._cache = self._read_from_disk()
            if self._cache:
                logger.debug(f"Loaded {len(self._cache)} cached annotations")

    def _save(self) -> None:
        """Save cache to disk with file locking and merge."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_file):
            # Re-read to get any changes from other instances
            disk_cache = self._read_from_disk()
            # Merge: start with disk cache, apply our changes
            merged = dict(disk_cache)
            # Apply our current values (overwrites disk values)
            merged.update(self._cache)
            # Remove keys we explicitly deleted
            for key in self._deleted_keys:
                merged.pop(key, None)
            self._write_to_disk(merged)
            # Update in-memory state
            self._cache = merged
            self._deleted_keys.clear()

    def get(self, session_id: str, turn_number: int) -> str | None:
        """Get cached annotation for a turn."""
        key = annotation_key(session_id, turn_number)
        with self._thread_lock:
            return self._cache.get(key)

    def set(self, session_id: str, turn_number: int, annotation: str) -> None:
        """Store annotation for a turn."""
        key = annotation_key(session_id, turn_number)
        with self._thread_lock:
            if annotation:
                self._cache[key] = annotation
                self._deleted_keys.discard(key)
            elif key in self._cache:
                del self._cache[key]
                self._deleted_keys.add(key)
            self._save()

    def delete(self, session_id: str, turn_number: int) -> None:
        """Delete annotation for a turn."""
        key = annotation_key(session_id, turn_number)
        with self._thread_lock:
            if key in self._cache:
                del self._cache[key]
            self._deleted_keys.add(key)
            self._save()

    def __len__(self) -> int:
        """Return number of cached annotations."""
        with self._thread_lock:
            return len(self._cache)


class AgentCache:
    """Thread-safe persistent cache for agent reports with file locking.

    Reports are keyed by session_id. Each report is stored as a dict with:
    - goal, narrative, progress_assessment
    - observations: list of serialized AgentObservation dicts
    - updated_at: ISO timestamp

    Old reports (>7 days) are pruned on load.
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, max_observations: int = 50):
        self._cache_dir = cache_dir
        self._cache_file = cache_dir / AGENT_REPORTS_FILENAME
        self._lock_file = cache_dir / f"{AGENT_REPORTS_FILENAME}.lock"
        self._cache: dict[str, dict] = {}
        self._thread_lock = Lock()
        self._max_observations = max_observations
        self._dirty_count = 0  # Track changes since last save
        self._load()

    def _read_from_disk(self) -> dict[str, dict]:
        """Read cache from disk without locking."""
        if not self._cache_file.exists():
            return {}
        try:
            with open(self._cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read agent cache: {e}")
            return {}

    def _write_to_disk(self, data: dict[str, dict]) -> None:
        """Write cache to disk without locking."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(data, f)
        except IOError as e:
            logger.warning(f"Failed to write agent cache: {e}")

    def _load(self) -> None:
        """Load cache from disk with file lock and prune old entries."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_file):
            self._cache = self._read_from_disk()
            pruned = self._prune_old(self._cache)
            if len(pruned) < len(self._cache):
                self._cache = pruned
                self._write_to_disk(self._cache)
            if self._cache:
                logger.debug(f"Loaded {len(self._cache)} cached agent reports")

    def _save(self) -> None:
        """Save cache to disk with file locking and merge."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_file):
            disk_cache = self._read_from_disk()
            merged = {**disk_cache, **self._cache}
            self._write_to_disk(merged)
            self._cache = merged

    def _prune_old(self, data: dict[str, dict]) -> dict[str, dict]:
        """Remove reports older than AGENT_REPORT_MAX_AGE_DAYS."""
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=AGENT_REPORT_MAX_AGE_DAYS)
        pruned = {}
        for sid, report_data in data.items():
            updated = report_data.get("updated_at", "")
            try:
                ts = datetime.fromisoformat(updated)
                if ts >= cutoff:
                    pruned[sid] = report_data
            except (ValueError, TypeError):
                # Keep entries with invalid timestamps
                pruned[sid] = report_data
        return pruned

    def get(self, session_id: str) -> dict | None:
        """Get cached report data for a session."""
        with self._thread_lock:
            return self._cache.get(session_id)

    def set(self, session_id: str, report_data: dict) -> None:
        """Store report data for a session.

        Prunes observations to max_observations before saving.
        """
        with self._thread_lock:
            # Prune observations
            obs = report_data.get("observations", [])
            if len(obs) > self._max_observations:
                report_data["observations"] = obs[-self._max_observations:]
            self._cache[session_id] = report_data
            self._dirty_count += 1
            # Save every 5 updates to avoid excessive I/O
            if self._dirty_count >= 5:
                self._save()
                self._dirty_count = 0

    def flush(self) -> None:
        """Force save to disk."""
        with self._thread_lock:
            if self._dirty_count > 0:
                self._save()
                self._dirty_count = 0

    def __len__(self) -> int:
        """Return number of cached reports."""
        with self._thread_lock:
            return len(self._cache)
