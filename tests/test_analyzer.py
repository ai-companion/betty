"""Tests for the on-demand turn analyzer."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from claude_companion.analyzer import (
    ANALYZER_SYSTEM_PROMPT,
    RANGE_SYSTEM_PROMPT,
    Analysis,
    Analyzer,
    ContextManager,
    GoalExtractor,
    make_analysis_cache_key,
    make_range_cache_key,
)
from claude_companion.models import Session, TaskState, Turn, compute_spans


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


# --- compute_spans tests ---


class TestComputeSpans:
    def test_empty(self):
        assert compute_spans([]) == []

    def test_single_user_turn(self):
        turns = [_make_turn(1, "user", "Hello")]
        assert compute_spans(turns) == [(0, 0)]

    def test_single_assistant_turn(self):
        turns = [_make_turn(1, "assistant", "Hello")]
        assert compute_spans(turns) == [(0, 0)]

    def test_basic_spans(self):
        turns = [
            _make_turn(1, "user", "Q1"),
            _make_turn(2, "assistant", "A1"),
            _make_turn(3, "tool", "T1", "Read"),
            _make_turn(4, "tool", "T2", "Write"),
            _make_turn(5, "user", "Q2"),
            _make_turn(6, "assistant", "A2"),
            _make_turn(7, "tool", "T3", "Bash"),
            _make_turn(8, "user", "Q3"),
            _make_turn(9, "assistant", "A3"),
        ]
        spans = compute_spans(turns)
        assert spans == [(0, 3), (4, 6), (7, 8)]

    def test_leading_non_user_turns(self):
        turns = [
            _make_turn(1, "assistant", "Welcome"),
            _make_turn(2, "tool", "T1", "Read"),
            _make_turn(3, "user", "Hello"),
            _make_turn(4, "assistant", "A1"),
        ]
        spans = compute_spans(turns)
        assert spans == [(0, 1), (2, 3)]

    def test_all_user_turns(self):
        turns = [
            _make_turn(1, "user", "Q1"),
            _make_turn(2, "user", "Q2"),
            _make_turn(3, "user", "Q3"),
        ]
        spans = compute_spans(turns)
        assert spans == [(0, 0), (1, 1), (2, 2)]

    def test_no_user_turns(self):
        turns = [
            _make_turn(1, "assistant", "A1"),
            _make_turn(2, "tool", "T1", "Read"),
            _make_turn(3, "assistant", "A2"),
        ]
        spans = compute_spans(turns)
        assert spans == [(0, 2)]


# --- GoalExtractor tests ---


class TestGoalExtractor:
    def test_basic_extraction(self):
        ge = GoalExtractor()
        turns = [
            _make_turn(1, "user", "Fix the login bug"),
            _make_turn(2, "assistant", "I'll look at it"),
        ]
        session = _make_session(turns)
        assert ge.extract(session) == "Fix the login bug"

    def test_no_user_turns(self):
        ge = GoalExtractor()
        session = _make_session([_make_turn(1, "assistant", "Hello")])
        assert ge.extract(session) == "[No user message found]"

    def test_truncation(self):
        ge = GoalExtractor()
        long_msg = "x" * 3000
        session = _make_session([_make_turn(1, "user", long_msg)])
        goal = ge.extract(session)
        assert len(goal) == 2000

    def test_caching(self):
        ge = GoalExtractor()
        turns = [_make_turn(1, "user", "Fix bug")]
        session = _make_session(turns)
        result1 = ge.extract(session)
        # Modify the turn content (shouldn't affect cached result)
        turns[0] = _make_turn(1, "user", "Different goal")
        session.turns = turns
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
        assert ge.extract(session) == "First goal"


# --- build_context_for_range tests ---


class TestBuildContextForRange:
    def setup_method(self):
        self.cm = ContextManager()

    def test_basic_range_context(self):
        turns = [
            _make_turn(1, "user", "Fix the bug"),
            _make_turn(2, "assistant", "Looking at code"),
            _make_turn(3, "tool", "read src/auth.py", "Read"),
            _make_turn(4, "assistant", "Found the issue"),
        ]
        session = _make_session(turns)
        ctx = self.cm.build_context_for_range(session, turns[1:3])

        assert ctx["goal"] == "Fix the bug"
        assert len(ctx["target_turns"]) == 2
        assert ctx["target_turns"][0]["role"] == "assistant"
        assert ctx["target_turns"][1]["role"] == "tool:Read"
        assert ctx["range_start"] == 1
        assert ctx["range_end"] == 2
        assert ctx["total_turns"] == 4

    def test_surrounding_context(self):
        turns = [
            _make_turn(1, "user", "Build API"),
            _make_turn(2, "assistant", "Sure"),
            _make_turn(3, "tool", "read file", "Read"),
            _make_turn(4, "assistant", "Done"),
            _make_turn(5, "user", "Add tests"),
        ]
        session = _make_session(turns)
        # Analyze turns 2-3 (the middle)
        ctx = self.cm.build_context_for_range(session, turns[1:3])

        # Surrounding should include turns outside the range
        surround_indices = [s["index"] for s in ctx["surrounding"]]
        assert 1 not in surround_indices  # In range
        assert 2 not in surround_indices  # In range
        # Turns 0, 3, 4 should be in surrounding
        assert 0 in surround_indices
        assert 3 in surround_indices
        assert 4 in surround_indices

    def test_full_session_range(self):
        turns = [
            _make_turn(1, "user", "Goal"),
            _make_turn(2, "assistant", "Response"),
        ]
        session = _make_session(turns)
        ctx = self.cm.build_context_for_range(session, turns)

        assert len(ctx["target_turns"]) == 2
        assert ctx["surrounding"] == []  # No turns outside range

    def test_active_tasks_included(self):
        turns = [_make_turn(1, "user", "Do something")]
        session = _make_session(turns)
        session.tasks["1"] = TaskState(
            task_id="1", subject="Task A", description="", status="in_progress"
        )
        ctx = self.cm.build_context_for_range(session, turns)
        assert len(ctx["active_tasks"]) == 1
        assert "[in_progress] Task A" in ctx["active_tasks"]


# --- make_range_cache_key tests ---


class TestMakeRangeCacheKey:
    def test_same_input_same_key(self):
        turns = [_make_turn(1, "user", "Hello"), _make_turn(2, "assistant", "World")]
        context = {"goal": "Fix bug", "surrounding": []}
        key1 = make_range_cache_key(turns, context, "model-a", RANGE_SYSTEM_PROMPT)
        key2 = make_range_cache_key(turns, context, "model-a", RANGE_SYSTEM_PROMPT)
        assert key1 == key2

    def test_different_turns_different_key(self):
        turns_a = [_make_turn(1, "user", "Hello")]
        turns_b = [_make_turn(1, "user", "Goodbye")]
        context = {"goal": "Fix bug", "surrounding": []}
        key_a = make_range_cache_key(turns_a, context, "model-a", RANGE_SYSTEM_PROMPT)
        key_b = make_range_cache_key(turns_b, context, "model-a", RANGE_SYSTEM_PROMPT)
        assert key_a != key_b

    def test_different_model_different_key(self):
        turns = [_make_turn(1, "user", "Hello")]
        context = {"goal": "Fix bug", "surrounding": []}
        key_a = make_range_cache_key(turns, context, "model-a", RANGE_SYSTEM_PROMPT)
        key_b = make_range_cache_key(turns, context, "model-b", RANGE_SYSTEM_PROMPT)
        assert key_a != key_b

    def test_different_context_different_key(self):
        turns = [_make_turn(1, "user", "Hello")]
        ctx_a = {"goal": "Fix bug", "surrounding": [{"content": "context A"}]}
        ctx_b = {"goal": "Fix bug", "surrounding": [{"content": "context B"}]}
        key_a = make_range_cache_key(turns, ctx_a, "model-a", RANGE_SYSTEM_PROMPT)
        key_b = make_range_cache_key(turns, ctx_b, "model-a", RANGE_SYSTEM_PROMPT)
        assert key_a != key_b

    def test_key_format(self):
        turns = [_make_turn(1, "user", "Hello")]
        context = {"goal": "Goal", "surrounding": []}
        key = make_range_cache_key(turns, context, "test", RANGE_SYSTEM_PROMPT)
        assert key.startswith("RANGE_ANALYSIS::")
        parts = key.split("::")
        assert len(parts) == 4

    def test_different_system_prompt_different_key(self):
        turns = [_make_turn(1, "user", "Hello")]
        context = {"goal": "Fix bug", "surrounding": []}
        key_a = make_range_cache_key(turns, context, "model-a", RANGE_SYSTEM_PROMPT)
        key_b = make_range_cache_key(turns, context, "model-a", ANALYZER_SYSTEM_PROMPT)
        assert key_a != key_b
