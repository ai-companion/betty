"""Persistent cache for summaries and annotations."""

import hashlib
import json
import logging
from pathlib import Path
from threading import Lock

from filelock import FileLock

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "claude-companion"
SUMMARIES_FILENAME = "summaries.json"
ANNOTATIONS_FILENAME = "annotations.json"


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
