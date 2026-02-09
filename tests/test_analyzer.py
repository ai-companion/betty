"""Tests for the on-demand turn analyzer."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from claude_companion.analyzer import (
    ANALYZER_SYSTEM_PROMPT,
    Analysis,
    Analyzer,
    ContextManager,
    make_analysis_cache_key,
)
from claude_companion.models import Session, TaskState, Turn


def _make_turn(
    turn_number: int = 1,
    role: str = "assistant",
    content: str = "Hello world",
    tool_name: str | None = None,
) -> Turn:
    """Create a Turn for testing."""
    from claude_companion.models import count_words

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


# --- Analysis dataclass tests ---


class TestAnalysis:
    def test_construction(self):
        a = Analysis(
            summary="Did something",
            critique="Looks good",
            sentiment="progress",
            word_count=10,
            context_word_count=100,
        )
        assert a.summary == "Did something"
        assert a.critique == "Looks good"
        assert a.sentiment == "progress"
        assert a.word_count == 10
        assert a.context_word_count == 100

    def test_all_sentiments(self):
        for sentiment in ["progress", "concern", "critical"]:
            a = Analysis(
                summary="test",
                critique="test",
                sentiment=sentiment,
                word_count=1,
                context_word_count=1,
            )
            assert a.sentiment == sentiment


# --- ContextManager tests ---


class TestContextManager:
    def setup_method(self):
        self.cm = ContextManager()

    def test_basic_context(self):
        turns = [
            _make_turn(1, "user", "Fix the login bug"),
            _make_turn(2, "assistant", "I'll look at the code"),
            _make_turn(3, "tool", "read src/auth.py", "Read"),
            _make_turn(4, "assistant", "Found the issue"),
        ]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[3])

        assert ctx["goal"] == "Fix the login bug"
        assert ctx["target_index"] == 3
        assert ctx["total_turns"] == 4
        assert ctx["target_turn"]["role"] == "assistant"
        assert ctx["target_turn"]["content"] == "Found the issue"

    def test_goal_from_first_user_turn(self):
        turns = [
            _make_turn(1, "assistant", "Welcome"),
            _make_turn(2, "user", "Build a REST API"),
            _make_turn(3, "user", "Add tests too"),
        ]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[2])
        assert ctx["goal"] == "Build a REST API"

    def test_goal_truncated_to_2000(self):
        long_msg = "x" * 3000
        turns = [_make_turn(1, "user", long_msg)]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[0])
        assert len(ctx["goal"]) == 2000

    def test_no_user_turns(self):
        turns = [_make_turn(1, "assistant", "Hello")]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[0])
        assert ctx["goal"] == "[No user message found]"

    def test_window_sizing(self):
        turns = [_make_turn(i, "assistant", f"Turn {i}") for i in range(20)]
        session = _make_session(turns)

        # Target at index 10 → window should be 5-14 (excluding target itself)
        ctx = self.cm.build_context(session, turns[10])
        window_indices = [w["index"] for w in ctx["window"]]
        assert 10 not in window_indices  # Target excluded from window
        assert 5 in window_indices
        assert 14 in window_indices

    def test_first_turn_window(self):
        turns = [_make_turn(i, "assistant", f"Turn {i}") for i in range(5)]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[0])
        # Should not include negative indices
        assert all(w["index"] >= 0 for w in ctx["window"])

    def test_last_turn_window(self):
        turns = [_make_turn(i, "assistant", f"Turn {i}") for i in range(5)]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[4])
        assert all(w["index"] < 5 for w in ctx["window"])

    def test_window_content_truncation(self):
        long_content = "word " * 200  # 1000 chars
        turns = [
            _make_turn(1, "user", "Goal"),
            _make_turn(2, "assistant", long_content),
            _make_turn(3, "assistant", "Target"),
        ]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[2])
        for w in ctx["window"]:
            assert len(w["content"]) <= 504  # 500 + "..."

    def test_active_tasks(self):
        turns = [_make_turn(1, "user", "Do something")]
        session = _make_session(turns)
        session.tasks["1"] = TaskState(
            task_id="1", subject="Fix bug", description="", status="in_progress"
        )
        session.tasks["2"] = TaskState(
            task_id="2", subject="Add tests", description="", status="pending"
        )
        session.tasks["3"] = TaskState(
            task_id="3", subject="Old task", description="", status="deleted"
        )
        ctx = self.cm.build_context(session, turns[0])
        assert len(ctx["active_tasks"]) == 2
        assert "[in_progress] Fix bug" in ctx["active_tasks"]
        assert "[pending] Add tests" in ctx["active_tasks"]

    def test_no_tasks(self):
        turns = [_make_turn(1, "user", "Hi")]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[0])
        assert ctx["active_tasks"] == []

    def test_tool_turn_context(self):
        turns = [
            _make_turn(1, "user", "Read the file"),
            _make_turn(2, "tool", "src/main.py", "Read"),
        ]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[1])
        assert ctx["target_turn"]["role"] == "tool"
        assert ctx["target_turn"]["tool_name"] == "Read"


# --- Analyzer._parse_response tests ---


class TestParseResponse:
    def setup_method(self):
        self.analyzer = Analyzer(model="test-model")

    def test_valid_json(self):
        raw = json.dumps({
            "summary": "Read the config file",
            "critique": "Good approach",
            "sentiment": "progress",
        })
        result = self.analyzer._parse_response(raw, 10, 100)
        assert result.summary == "Read the config file"
        assert result.critique == "Good approach"
        assert result.sentiment == "progress"
        assert result.word_count == 10
        assert result.context_word_count == 100

    def test_code_fenced_json(self):
        raw = '```json\n{"summary": "test", "critique": "ok", "sentiment": "concern"}\n```'
        result = self.analyzer._parse_response(raw, 5, 50)
        assert result.summary == "test"
        assert result.sentiment == "concern"

    def test_code_fenced_no_lang(self):
        raw = '```\n{"summary": "test", "critique": "ok", "sentiment": "progress"}\n```'
        result = self.analyzer._parse_response(raw, 5, 50)
        assert result.summary == "test"

    def test_malformed_json(self):
        raw = "This is not JSON at all"
        result = self.analyzer._parse_response(raw, 5, 50)
        assert result.summary.startswith("[parse error")

    def test_json_in_text(self):
        raw = 'Here is the analysis: {"summary": "found it", "critique": "nice", "sentiment": "progress"} end.'
        result = self.analyzer._parse_response(raw, 5, 50)
        assert result.summary == "found it"

    def test_missing_sentiment(self):
        raw = json.dumps({"summary": "test", "critique": "ok"})
        result = self.analyzer._parse_response(raw, 5, 50)
        assert result.sentiment == "progress"  # Default

    def test_unknown_sentiment(self):
        raw = json.dumps({
            "summary": "test",
            "critique": "ok",
            "sentiment": "unknown_value",
        })
        result = self.analyzer._parse_response(raw, 5, 50)
        assert result.sentiment == "progress"  # Fallback

    def test_missing_summary(self):
        raw = json.dumps({"critique": "ok", "sentiment": "progress"})
        result = self.analyzer._parse_response(raw, 5, 50)
        assert result.summary == "[no summary]"

    def test_missing_critique(self):
        raw = json.dumps({"summary": "test", "sentiment": "progress"})
        result = self.analyzer._parse_response(raw, 5, 50)
        assert result.critique == ""


# --- Cache key tests ---


class TestCacheKey:
    def test_same_input_same_key(self):
        turn = _make_turn(1, "assistant", "Hello world")
        context = {"goal": "Fix bug", "window": [{"content": "context"}]}
        key1 = make_analysis_cache_key(turn, context, "model-a", ANALYZER_SYSTEM_PROMPT)
        key2 = make_analysis_cache_key(turn, context, "model-a", ANALYZER_SYSTEM_PROMPT)
        assert key1 == key2

    def test_different_content_different_key(self):
        turn_a = _make_turn(1, "assistant", "Hello world")
        turn_b = _make_turn(1, "assistant", "Goodbye world")
        context = {"goal": "Fix bug", "window": []}
        key_a = make_analysis_cache_key(turn_a, context, "model-a", ANALYZER_SYSTEM_PROMPT)
        key_b = make_analysis_cache_key(turn_b, context, "model-a", ANALYZER_SYSTEM_PROMPT)
        assert key_a != key_b

    def test_different_model_different_key(self):
        turn = _make_turn(1, "assistant", "Hello world")
        context = {"goal": "Fix bug", "window": []}
        key_a = make_analysis_cache_key(turn, context, "model-a", ANALYZER_SYSTEM_PROMPT)
        key_b = make_analysis_cache_key(turn, context, "model-b", ANALYZER_SYSTEM_PROMPT)
        assert key_a != key_b

    def test_different_context_different_key(self):
        turn = _make_turn(1, "assistant", "Hello world")
        ctx_a = {"goal": "Fix bug", "window": [{"content": "context A"}]}
        ctx_b = {"goal": "Fix bug", "window": [{"content": "context B"}]}
        key_a = make_analysis_cache_key(turn, ctx_a, "model-a", ANALYZER_SYSTEM_PROMPT)
        key_b = make_analysis_cache_key(turn, ctx_b, "model-a", ANALYZER_SYSTEM_PROMPT)
        assert key_a != key_b

    def test_key_format(self):
        turn = _make_turn(1, "assistant", "Hello")
        context = {"goal": "Goal", "window": []}
        key = make_analysis_cache_key(turn, context, "test", ANALYZER_SYSTEM_PROMPT)
        assert key.startswith("ANALYSIS::")
        parts = key.split("::")
        assert len(parts) == 4


# --- Integration test with mocked LLM ---


class TestAnalyzerIntegration:
    def test_analyze_populates_turn(self):
        analyzer = Analyzer(model="test-model")

        turn = _make_turn(2, "assistant", "I fixed the bug by updating the auth module")
        session = _make_session([
            _make_turn(1, "user", "Fix the login bug"),
            turn,
        ])

        mock_response = json.dumps({
            "summary": "Fixed the authentication bug",
            "critique": "Good fix, addressed the root cause",
            "sentiment": "progress",
        })

        received = []

        def callback(analysis: Analysis, success: bool):
            received.append((analysis, success))

        with patch.object(analyzer, "_call_llm", return_value=mock_response):
            analyzer._analyze(session, turn, callback)

        assert len(received) == 1
        analysis, success = received[0]
        assert success is True
        assert analysis.summary == "Fixed the authentication bug"
        assert analysis.critique == "Good fix, addressed the root cause"
        assert analysis.sentiment == "progress"

    def test_analyze_handles_llm_error(self):
        analyzer = Analyzer(model="test-model")

        turn = _make_turn(1, "user", "Hello")
        session = _make_session([turn])

        received = []

        def callback(analysis: Analysis, success: bool):
            received.append((analysis, success))

        with patch.object(
            analyzer, "_call_llm", side_effect=ConnectionError("no server")
        ):
            analyzer._analyze(session, turn, callback)

        assert len(received) == 1
        analysis, success = received[0]
        assert success is False
        assert analysis.summary.startswith("[")

    def test_build_prompt_includes_all_sections(self):
        analyzer = Analyzer(model="test-model")

        turns = [
            _make_turn(1, "user", "Build a REST API"),
            _make_turn(2, "tool", "read server.py", "Read"),
            _make_turn(3, "assistant", "Here is the implementation"),
        ]
        session = _make_session(turns)
        session.tasks["1"] = TaskState(
            task_id="1", subject="Build API", description="", status="in_progress"
        )

        context = analyzer._context_manager.build_context(session, turns[2])
        prompt = analyzer._build_prompt(context)

        assert "## Goal" in prompt
        assert "Build a REST API" in prompt
        assert "## Active Tasks" in prompt
        assert "Build API" in prompt
        assert "## Surrounding Turns" in prompt
        assert "## Target Turn" in prompt
        assert "Here is the implementation" in prompt
