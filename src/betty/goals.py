"""Goal extraction for Betty.

Extracts session goals from multiple sources (user messages, GitHub issues,
plan files, active tasks) for use by the Agent and other components.
"""

import json
import re
import subprocess
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .models import Session


@dataclass
class GoalSource:
    """A single source of goal context."""

    source_type: str  # "user_request" | "github_issue" | "plan" | "tasks"
    label: str  # Display label for TUI
    content: str  # The actual content
    fresh: bool = True  # Whether from current session


_GH_ISSUE_PATTERN = re.compile(
    r'(?:^|\s)#(\d+)'  # #123
    r'|https?://github\.com/([^/]+/[^/]+)/issues/(\d+)',  # full URL
    re.MULTILINE,
)

GOAL_SYNTHESIS_PROMPT = (
    "You are observing a coding session. Given the following goal sources, "
    "synthesize a clear 1-2 sentence goal statement that captures what "
    "the user is trying to accomplish. Be specific and actionable.\n\n"
    "Return ONLY the synthesized goal text, nothing else."
)


class GoalExtractor:
    """Extract and cache session goal from multiple sources."""

    def __init__(self) -> None:
        self._cache: dict[str, list[GoalSource]] = {}  # session_id -> sources
        self._cache_version: dict[str, tuple] = {}  # session_id -> version key
        self._gh_cache: dict[str, str | None] = {}  # "session:issue" -> text or None
        self._gh_lock = threading.Lock()
        self._synthesized: dict[str, str] = {}  # session_id -> synthesized goal

    def extract(self, session: "Session") -> str:
        """Return formatted goal string for LLM prompt."""
        sources = self._get_sources(session)
        # If synthesized goal available, lead with it
        synthesized = self._synthesized.get(session.session_id)
        if synthesized:
            raw = self._format_sources(sources)
            return f"Synthesized goal: {synthesized}\n\nGoal sources:\n{raw}"
        return self._format_sources(sources)

    def get_sources(self, session: "Session") -> list[GoalSource]:
        """Return raw goal sources (for TUI display)."""
        return self._get_sources(session)

    def get_synthesized_goal(self, session: "Session") -> str | None:
        """Return LLM-synthesized goal if available."""
        return self._synthesized.get(session.session_id)

    def synthesize_goal_async(
        self,
        session: "Session",
        call_llm: Callable[[str, str], str],
        callback: Callable[[str], None] | None = None,
    ) -> None:
        """Kick off async LLM synthesis of goal from sources."""
        if session.session_id in self._synthesized:
            if callback:
                callback(self._synthesized[session.session_id])
            return

        sources = self._get_sources(session)
        if not sources:
            return

        def _do_synthesis() -> None:
            raw_sources = self._format_sources(sources)
            try:
                result = call_llm(raw_sources, GOAL_SYNTHESIS_PROMPT)
                result = result.strip()
                if result:
                    self._synthesized[session.session_id] = result
                    # Invalidate cache so next extract() prepends synthesized goal
                    self._cache.pop(session.session_id, None)
                    self._cache_version.pop(session.session_id, None)
                    if callback:
                        callback(result)
            except Exception:
                pass  # Synthesis is best-effort

        threading.Thread(target=_do_synthesis, daemon=True).start()

    def _get_sources(self, session: "Session") -> list[GoalSource]:
        """Build or return cached goal sources."""
        version = self._compute_version(session)
        if (
            session.session_id in self._cache
            and self._cache_version.get(session.session_id) == version
        ):
            return self._cache[session.session_id]

        sources = self._build_sources(session)
        self._cache[session.session_id] = sources
        self._cache_version[session.session_id] = version
        return sources

    def _compute_version(self, session: "Session") -> tuple:
        """Version key for cache invalidation."""
        user_turn_count = sum(1 for t in session.turns if t.role == "user")
        task_hash = len(session.tasks)
        plan_hash = hash(session.plan_content) if session.plan_content else 0
        gh_count = len([v for v in self._gh_cache.values() if v is not None])
        return (user_turn_count, task_hash, plan_hash, gh_count)

    def _build_sources(self, session: "Session") -> list[GoalSource]:
        """Build goal sources from multiple inputs."""
        sources: list[GoalSource] = []

        # 1. First user message (always)
        first_user = None
        for t in session.turns:
            if t.role == "user":
                first_user = t
                break
        if first_user:
            sources.append(GoalSource(
                source_type="user_request",
                label="User Request",
                content=first_user.content_full[:2000],
            ))
            self._detect_and_fetch_issues(session.session_id, first_user.content_full)
        else:
            sources.append(GoalSource(
                source_type="user_request",
                label="User Request",
                content="[No user message found]",
            ))

        # 2. GitHub issue (if fetched)
        with self._gh_lock:
            for key, text in self._gh_cache.items():
                if key.startswith(f"{session.session_id}:") and text:
                    sources.append(GoalSource(
                        source_type="github_issue",
                        label="GitHub Issue",
                        content=text,
                    ))
                    break  # Only include first matched issue

        # 3. Plan file (with staleness check)
        if session.plan_content:
            is_fresh = True
            if session.plan_updated_at and session.started_at:
                is_fresh = session.plan_updated_at >= session.started_at
            label = "Plan" if is_fresh else "Plan (from previous session)"
            sources.append(GoalSource(
                source_type="plan",
                label=label,
                content=session.plan_content[:500],
                fresh=is_fresh,
            ))

        # 4. Active tasks
        active_tasks = [
            f"[{t.status}] {t.subject}"
            for t in session.tasks.values()
            if not t.is_deleted
        ]
        if active_tasks:
            sources.append(GoalSource(
                source_type="tasks",
                label="Active Tasks",
                content="\n".join(active_tasks[:10]),
            ))

        return sources

    def _detect_and_fetch_issues(self, session_id: str, text: str) -> None:
        """Parse issue references and start background fetch."""
        for match in _GH_ISSUE_PATTERN.finditer(text):
            issue_num = match.group(1) or match.group(3)
            repo = match.group(2)  # None for #123 shorthand
            cache_key = f"{session_id}:{issue_num}"
            with self._gh_lock:
                if cache_key in self._gh_cache:
                    continue  # Already fetching or fetched
                self._gh_cache[cache_key] = None  # Mark as in-flight
            threading.Thread(
                target=self._fetch_gh_issue,
                args=(cache_key, issue_num, repo),
                daemon=True,
            ).start()

    def _fetch_gh_issue(
        self, cache_key: str, issue_num: str, repo: str | None
    ) -> None:
        """Fetch issue via gh CLI in background thread."""
        try:
            cmd = ["gh", "issue", "view", issue_num, "--json", "title,body,labels"]
            if repo:
                cmd.extend(["--repo", repo])
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                title = data.get("title", "")
                body = data.get("body", "")[:500]
                labels = [
                    l.get("name", "")
                    for l in data.get("labels", [])
                    if l.get("name")
                ]
                text = f"#{issue_num}: {title}"
                if labels:
                    text += f" [{', '.join(labels)}]"
                if body:
                    text += f"\n{body}"
                with self._gh_lock:
                    self._gh_cache[cache_key] = text
        except Exception:
            pass  # gh not installed, no repo, network error — silent

    def _format_sources(self, sources: list[GoalSource]) -> str:
        """Format sources into labeled text for the LLM prompt."""
        parts = []
        for source in sources:
            freshness = "" if source.fresh else " (from a previous session)"
            parts.append(f"[{source.label}{freshness}]\n{source.content}")
        return "\n\n".join(parts)
