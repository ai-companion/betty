"""Persistent cache for summaries and annotations."""

import hashlib
import json
import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "claude-companion"
SUMMARIES_FILENAME = "summaries.json"
ANNOTATIONS_FILENAME = "annotations.json"


def content_hash(content: str) -> str:
    """Generate a short hash for content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class SummaryCache:
    """Thread-safe persistent cache for turn summaries."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self._cache_dir = cache_dir
        self._cache_file = cache_dir / SUMMARIES_FILENAME
        self._cache: dict[str, str] = {}
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if not self._cache_file.exists():
            return
        try:
            with open(self._cache_file, "r") as f:
                self._cache = json.load(f)
            logger.debug(f"Loaded {len(self._cache)} cached summaries")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load summary cache: {e}")
            self._cache = {}

    def _save(self) -> None:
        """Save cache to disk."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f)
        except IOError as e:
            logger.warning(f"Failed to save summary cache: {e}")

    def get(self, content: str) -> str | None:
        """Get cached summary for content."""
        key = content_hash(content)
        with self._lock:
            return self._cache.get(key)

    def set(self, content: str, summary: str) -> None:
        """Store summary for content."""
        key = content_hash(content)
        with self._lock:
            self._cache[key] = summary
            self._save()

    def __len__(self) -> int:
        """Return number of cached summaries."""
        with self._lock:
            return len(self._cache)


def annotation_key(session_id: str, turn_number: int) -> str:
    """Generate key for annotation lookup."""
    return f"{session_id}:{turn_number}"


class AnnotationCache:
    """Thread-safe persistent cache for user annotations."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self._cache_dir = cache_dir
        self._cache_file = cache_dir / ANNOTATIONS_FILENAME
        self._cache: dict[str, str] = {}
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if not self._cache_file.exists():
            return
        try:
            with open(self._cache_file, "r") as f:
                self._cache = json.load(f)
            logger.debug(f"Loaded {len(self._cache)} cached annotations")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load annotation cache: {e}")
            self._cache = {}

    def _save(self) -> None:
        """Save cache to disk."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f)
        except IOError as e:
            logger.warning(f"Failed to save annotation cache: {e}")

    def get(self, session_id: str, turn_number: int) -> str | None:
        """Get cached annotation for a turn."""
        key = annotation_key(session_id, turn_number)
        with self._lock:
            return self._cache.get(key)

    def set(self, session_id: str, turn_number: int, annotation: str) -> None:
        """Store annotation for a turn."""
        key = annotation_key(session_id, turn_number)
        with self._lock:
            if annotation:
                self._cache[key] = annotation
            elif key in self._cache:
                del self._cache[key]
            self._save()

    def delete(self, session_id: str, turn_number: int) -> None:
        """Delete annotation for a turn."""
        key = annotation_key(session_id, turn_number)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._save()

    def __len__(self) -> int:
        """Return number of cached annotations."""
        with self._lock:
            return len(self._cache)
