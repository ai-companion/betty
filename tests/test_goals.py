"""Tests for GoalExtractor and GoalSource."""

import json
import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from betty.goals import (
    GOAL_SYNTHESIS_PROMPT,
    GoalExtractor,
    GoalSource,
    _GH_ISSUE_PATTERN,
)
from betty.models import Session, TaskState, Turn


def _make_turn(
    turn_number: int = 1,
    role: str = "assistant",
    content: str = "Hello world",
    tool_name: str | None = None,
) -> Turn:
    """Create a Turn for testing."""
    from betty.models import count_words

    return Turn(
        turn_number=turn_number,
        role=role,
        content_preview=content[:80],
        content_full=content,
        word_count=count_words(content),
        tool_name=tool_name,
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
    )


def _make_session(turns: list[Turn] | None = None) -> Session:
    """Create a Session for testing."""
    return Session(
        session_id="test-session-123",
        project_path="test-project",
        turns=turns or [],
    )


class TestGoalExtractor:
    def test_basic_extraction(self):
        ge = GoalExtractor()
        turns = [
            _make_turn(1, "user", "Fix the login bug"),
            _make_turn(2, "assistant", "I'll look at it"),
        ]
        session = _make_session(turns)
        result = ge.extract(session)
        assert "Fix the login bug" in result
        assert "[User Request]" in result

    def test_no_user_turns(self):
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "assistant", "Hello")])
        result = ge.extract(session)
        assert "[No user message found]" in result

    def test_truncation(self):
        ge = GoalExtractor()
        long_msg = "x" * 3000
        session = _make_session([_make_turn(1, "user", long_msg)])
        sources = ge.get_sources(session)
        user_source = [s for s in sources if s.source_type == "user_request"][0]
        assert len(user_source.content) == 2000

    def test_caching(self):
        ge = GoalExtractor()
        turns = [_make_turn(1, "user", "Fix bug")]
        session = _make_session(turns)
        result1 = ge.extract(session)
        # Same user turn count — cache still valid
        result2 = ge.extract(session)
        assert result1 == result2  # Still cached

    def test_first_user_message_used(self):
        ge = GoalExtractor()
        turns = [
            _make_turn(1, "assistant", "Welcome"),
            _make_turn(2, "user", "First goal"),
            _make_turn(3, "user", "Second message"),
        ]
        session = _make_session(turns)
        result = ge.extract(session)
        assert "First goal" in result


class TestGoalExtractorMultiSource:
    """Tests for the enriched GoalExtractor."""

    def test_basic_user_message(self):
        """First user message is always included."""
        ge = GoalExtractor()
        session = _make_session([
            _make_turn(1, "user", "Fix the login bug"),
            _make_turn(2, "assistant", "On it"),
        ])
        sources = ge.get_sources(session)
        assert len(sources) >= 1
        user_src = sources[0]
        assert user_src.source_type == "user_request"
        assert user_src.label == "User Request"
        assert user_src.content == "Fix the login bug"

    def test_no_user_message(self):
        """Falls back to '[No user message found]'."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "assistant", "Hello")])
        sources = ge.get_sources(session)
        assert sources[0].content == "[No user message found]"

    def test_plan_included_when_available(self):
        """Plan content is included as a source."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Do stuff")])
        session.plan_content = "## Plan\n- Step 1\n- Step 2"
        session.plan_updated_at = datetime.now()
        sources = ge.get_sources(session)
        plan_sources = [s for s in sources if s.source_type == "plan"]
        assert len(plan_sources) == 1
        assert "Step 1" in plan_sources[0].content
        assert plan_sources[0].fresh is True
        assert plan_sources[0].label == "Plan"

    def test_plan_staleness(self):
        """Plan predating session is marked as stale."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Do stuff")])
        session.started_at = datetime(2025, 6, 15, 12, 0, 0)
        session.plan_content = "Old plan"
        session.plan_updated_at = datetime(2025, 6, 14, 12, 0, 0)  # Before session
        sources = ge.get_sources(session)
        plan_sources = [s for s in sources if s.source_type == "plan"]
        assert len(plan_sources) == 1
        assert plan_sources[0].fresh is False
        assert "previous session" in plan_sources[0].label

    def test_tasks_included(self):
        """Active tasks are included as a source."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Work on tasks")])
        session.tasks["1"] = TaskState(
            task_id="1", subject="Fix bug", description="", status="in_progress"
        )
        session.tasks["2"] = TaskState(
            task_id="2", subject="Add tests", description="", status="pending"
        )
        sources = ge.get_sources(session)
        task_sources = [s for s in sources if s.source_type == "tasks"]
        assert len(task_sources) == 1
        assert "[in_progress] Fix bug" in task_sources[0].content
        assert "[pending] Add tests" in task_sources[0].content

    def test_deleted_tasks_excluded(self):
        """Deleted tasks are not included."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Work")])
        session.tasks["1"] = TaskState(
            task_id="1", subject="Old task", description="", status="deleted"
        )
        sources = ge.get_sources(session)
        task_sources = [s for s in sources if s.source_type == "tasks"]
        assert len(task_sources) == 0

    def test_cache_invalidation_on_new_user_turn(self):
        """Cache invalidates when user turns change."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "First goal")])
        sources1 = ge.get_sources(session)
        assert len(sources1) == 1

        # Add a new user turn
        session.turns.append(_make_turn(2, "user", "Second goal"))
        sources2 = ge.get_sources(session)
        # Cache should have been invalidated due to user_turn_count change
        assert sources2 is not sources1

    def test_cache_invalidation_on_plan_change(self):
        """Cache invalidates when plan content changes."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Goal")])
        sources1 = ge.get_sources(session)

        session.plan_content = "New plan"
        session.plan_updated_at = datetime.now()
        sources2 = ge.get_sources(session)
        plan_sources = [s for s in sources2 if s.source_type == "plan"]
        assert len(plan_sources) == 1

    def test_format_sources_labeled(self):
        """Sources are formatted with labels."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Fix bug")])
        result = ge.extract(session)
        assert "[User Request]" in result
        assert "Fix bug" in result

    def test_format_with_synthesized_goal(self):
        """Synthesized goal leads the formatted output."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Fix the auth bug")])
        # Manually set synthesized goal
        ge._synthesized[session.session_id] = "Fix authentication vulnerability in login flow"
        # Clear cache to force rebuild
        ge._cache.pop(session.session_id, None)
        ge._cache_version.pop(session.session_id, None)

        result = ge.extract(session)
        assert result.startswith("Synthesized goal: Fix authentication vulnerability")
        assert "Goal sources:" in result
        assert "[User Request]" in result


class TestGitHubIssueDetection:
    """Tests for GitHub issue parsing and fetching."""

    def test_detect_hash_reference(self):
        """Detects #123 pattern."""
        matches = list(_GH_ISSUE_PATTERN.finditer("Fix #123 please"))
        assert len(matches) == 1
        assert matches[0].group(1) == "123"

    def test_detect_url_reference(self):
        """Detects full GitHub issue URL."""
        text = "See https://github.com/owner/repo/issues/456"
        matches = list(_GH_ISSUE_PATTERN.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(2) == "owner/repo"
        assert matches[0].group(3) == "456"

    def test_no_issue_reference(self):
        """No false positives on normal text."""
        matches = list(_GH_ISSUE_PATTERN.finditer("Just regular text here"))
        assert len(matches) == 0

    def test_gh_fetch_success(self):
        """Mocked gh subprocess returns issue data."""
        ge = GoalExtractor()
        mock_output = json.dumps({
            "title": "Login broken",
            "body": "Users can't log in since yesterday",
            "labels": [{"name": "bug"}, {"name": "P1"}],
        })
        with patch("betty.goals.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_output
            )
            ge._fetch_gh_issue("sess:123", "123", None)

        with ge._gh_lock:
            result = ge._gh_cache["sess:123"]
        assert result is not None
        assert "#123: Login broken" in result
        assert "bug" in result
        assert "P1" in result
        assert "Users can't log in" in result

    def test_gh_fetch_failure(self):
        """Failed gh call handled silently."""
        ge = GoalExtractor()
        with patch("betty.goals.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
            ge._fetch_gh_issue("sess:999", "999", None)

        with ge._gh_lock:
            # Should remain None (in-flight marker set before call)
            assert ge._gh_cache.get("sess:999") is None

    def test_gh_not_installed(self):
        """Missing gh binary handled silently."""
        ge = GoalExtractor()
        with patch("betty.goals.subprocess.run", side_effect=FileNotFoundError("gh")):
            ge._fetch_gh_issue("sess:100", "100", None)
        # Should not raise, cache stays None
        with ge._gh_lock:
            assert ge._gh_cache.get("sess:100") is None

    def test_gh_result_cached(self):
        """Second call uses cache, doesn't re-fetch."""
        ge = GoalExtractor()
        with ge._gh_lock:
            ge._gh_cache["sess:42"] = "#42: Cached issue"

        # _detect_and_fetch_issues should skip already-cached keys
        with patch("betty.goals.subprocess.run") as mock_run:
            ge._detect_and_fetch_issues("sess", "Fix #42 please")
            mock_run.assert_not_called()

    def test_gh_fetch_with_repo(self):
        """Fetch with explicit repo passes --repo flag."""
        ge = GoalExtractor()
        mock_output = json.dumps({
            "title": "Issue title",
            "body": "Body text",
            "labels": [],
        })
        with patch("betty.goals.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=mock_output)
            ge._fetch_gh_issue("sess:77", "77", "owner/repo")

        call_args = mock_run.call_args[0][0]
        assert "--repo" in call_args
        assert "owner/repo" in call_args

    def test_issue_included_in_sources(self):
        """Fetched issue appears in goal sources."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Fix #42")])
        # Pre-populate gh cache
        with ge._gh_lock:
            ge._gh_cache[f"{session.session_id}:42"] = "#42: Bug title\nBug body"
        sources = ge.get_sources(session)
        gh_sources = [s for s in sources if s.source_type == "github_issue"]
        assert len(gh_sources) == 1
        assert "#42: Bug title" in gh_sources[0].content


class TestGoalSynthesis:
    """Tests for LLM-based goal synthesis."""

    def test_synthesis_called_async(self):
        """Synthesis runs in background thread."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Fix the auth bug")])

        call_event = threading.Event()

        def mock_llm(prompt: str, system: str) -> str:
            call_event.set()
            return "Fix authentication vulnerability"

        ge.synthesize_goal_async(session, mock_llm)
        assert call_event.wait(timeout=5)
        # Wait for thread to finish writing
        time.sleep(0.1)
        assert ge.get_synthesized_goal(session) == "Fix authentication vulnerability"

    def test_synthesis_result_cached(self):
        """Synthesized goal is cached per session."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Build API")])
        ge._synthesized[session.session_id] = "Build a REST API endpoint"

        callback_results = []
        ge.synthesize_goal_async(
            session, lambda p, s: "should not call",
            callback=lambda r: callback_results.append(r),
        )
        # Should immediately call callback with cached result
        assert len(callback_results) == 1
        assert callback_results[0] == "Build a REST API endpoint"

    def test_synthesis_failure_graceful(self):
        """LLM failure doesn't break extraction."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Fix bug")])

        done_event = threading.Event()

        def failing_llm(prompt: str, system: str) -> str:
            done_event.set()
            raise ConnectionError("no server")

        ge.synthesize_goal_async(session, failing_llm)
        done_event.wait(timeout=5)
        time.sleep(0.1)
        # Should not have a synthesized goal
        assert ge.get_synthesized_goal(session) is None
        # extract() should still work
        result = ge.extract(session)
        assert "Fix bug" in result

    def test_extract_includes_synthesized_when_available(self):
        """extract() prepends synthesized goal when available."""
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "user", "Fix the auth bug")])
        ge._synthesized[session.session_id] = "Fix login authentication"
        result = ge.extract(session)
        assert result.startswith("Synthesized goal: Fix login authentication")
        assert "Goal sources:" in result


class TestGoalSource:
    def test_construction(self):
        gs = GoalSource(
            source_type="user_request",
            label="User Request",
            content="Fix the bug",
        )
        assert gs.source_type == "user_request"
        assert gs.fresh is True  # Default

    def test_stale_source(self):
        gs = GoalSource(
            source_type="plan",
            label="Plan (from previous session)",
            content="Old plan",
            fresh=False,
        )
        assert gs.fresh is False
