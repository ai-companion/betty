"""GitHub PR detection for Betty session cards."""

import json
import logging
import re
import subprocess
import threading
from dataclasses import dataclass

# Pattern to extract owner/repo from git remote URLs.
# Supports SSH (git@github.com:owner/repo.git) and HTTPS (https://github.com/owner/repo.git).
_REMOTE_URL_PATTERN = re.compile(
    r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"
)


@dataclass(frozen=True)
class PRInfo:
    """Information about a GitHub pull request."""

    number: int
    title: str
    url: str
    state: str  # "OPEN" | "MERGED" | "CLOSED"


class PRDetector:
    """Thread-safe background PR detector using gh CLI.

    Caches results keyed by (project_dir, branch) to avoid repeated lookups.
    Failed lookups cache None so they are not retried.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], PRInfo | None] = {}
        self._in_flight: set[tuple[str, str]] = set()
        self._remote_cache: dict[str, str | None] = {}  # project_dir -> "owner/repo"

    def detect_async(self, project_dir: str, branch: str) -> None:
        """Start background PR detection for a project/branch pair.

        If a detection is already in-flight or cached, this is a no-op.
        """
        key = (project_dir, branch)
        with self._lock:
            if key in self._cache or key in self._in_flight:
                return
            self._in_flight.add(key)

        threading.Thread(
            target=self._detect,
            args=(project_dir, branch),
            daemon=True,
        ).start()

    def get_pr(self, project_dir: str, branch: str) -> PRInfo | None:
        """Return cached PR info, or None if not yet available."""
        with self._lock:
            return self._cache.get((project_dir, branch))

    def _detect(self, project_dir: str, branch: str) -> None:
        """Background worker: resolve remote, fetch PR, cache result."""
        key = (project_dir, branch)
        try:
            repo = self._get_github_remote(project_dir)
            if not repo:
                return  # Not a GitHub repo
            pr_info = self._fetch_pr(branch, repo)
            with self._lock:
                self._cache[key] = pr_info  # None on no-match is fine
        except Exception:
            logging.debug("PR detection failed for %s@%s", project_dir, branch, exc_info=True)
            with self._lock:
                self._cache[key] = None
        finally:
            with self._lock:
                self._in_flight.discard(key)

    def _get_github_remote(self, project_dir: str) -> str | None:
        """Get 'owner/repo' from git remote origin URL. Cached per project_dir."""
        with self._lock:
            if project_dir in self._remote_cache:
                return self._remote_cache[project_dir]

        try:
            result = subprocess.run(
                ["git", "-C", project_dir, "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                repo = None
            else:
                m = _REMOTE_URL_PATTERN.search(result.stdout.strip())
                repo = f"{m.group('owner')}/{m.group('repo')}" if m else None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            repo = None

        with self._lock:
            self._remote_cache[project_dir] = repo
        return repo

    @staticmethod
    def _fetch_pr(branch: str, repo: str) -> PRInfo | None:
        """Fetch PR for a branch from GitHub via gh CLI."""
        try:
            result = subprocess.run(
                [
                    "gh", "pr", "list",
                    "--repo", repo,
                    "--head", branch,
                    "--json", "number,title,url,state",
                    "--limit", "1",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return None
            data = json.loads(result.stdout)
            if not data:
                return None
            pr = data[0]
            return PRInfo(
                number=pr["number"],
                title=pr["title"],
                url=pr["url"],
                state=pr.get("state", "OPEN"),
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError, KeyError):
            return None
