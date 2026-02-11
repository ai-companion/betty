"""Tests for the on-demand turn analyzer."""

import json
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_companion.analyzer import (
    ANALYZER_SYSTEM_PROMPT,
    GOAL_SYNTHESIS_PROMPT,
    LOCAL_GOAL_MIN_WORDS,
    RANGE_SYSTEM_PROMPT,
    Analysis,
    Analyzer,
    ContextManager,
    GoalExtractor,
    GoalSource,
    _GH_ISSUE_PATTERN,
    make_analysis_cache_key,
    make_range_cache_key,
)
from claude_companion.config import AnalyzerConfig
from claude_companion.models import Session, TaskState, Turn, compute_spans
from claude_companion.pricing import ModelPricing, get_pricing, estimate_cost, MODEL_PRICING


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

        assert "Fix the login bug" in ctx["goal"]
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
        assert "Build a REST API" in ctx["goal"]

    def test_goal_includes_label(self):
        turns = [_make_turn(1, "user", "Fix the bug")]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[0])
        assert "[User Request]" in ctx["goal"]

    def test_no_user_turns(self):
        turns = [_make_turn(1, "assistant", "Hello")]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[0])
        assert "[No user message found]" in ctx["goal"]

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

        assert "## Session Goal" in prompt
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

        assert "Fix the bug" in ctx["goal"]
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


# --- Pricing tests ---


class TestGetPricing:
    def test_exact_match(self):
        pricing = get_pricing("claude-sonnet-4")
        assert pricing is not None
        assert pricing.input_per_mtok == 3.0
        assert pricing.output_per_mtok == 15.0

    def test_prefix_match_with_version(self):
        pricing = get_pricing("claude-haiku-4-5-20251001")
        assert pricing is not None
        assert pricing.input_per_mtok == 0.80

    def test_prefix_match_opus(self):
        pricing = get_pricing("claude-opus-4-20250115")
        assert pricing is not None
        assert pricing.input_per_mtok == 15.0

    def test_unknown_model(self):
        assert get_pricing("unknown-model-123") is None

    def test_empty_string(self):
        assert get_pricing("") is None

    def test_none_returns_none(self):
        assert get_pricing(None) is None

    def test_all_known_models_have_pricing(self):
        for model_id in MODEL_PRICING:
            assert get_pricing(model_id) is not None


class TestEstimateCost:
    def test_basic_cost(self):
        pricing = ModelPricing(
            input_per_mtok=3.0,
            output_per_mtok=15.0,
            cache_write_per_mtok=3.75,
            cache_read_per_mtok=0.30,
        )
        cost = estimate_cost(1_000_000, 500_000, 0, 0, pricing)
        # 1M * 3.0/M + 500K * 15.0/M = 3.0 + 7.5 = 10.5
        assert cost == pytest.approx(10.5)

    def test_zero_tokens(self):
        pricing = ModelPricing(3.0, 15.0, 3.75, 0.30)
        assert estimate_cost(0, 0, 0, 0, pricing) == 0.0

    def test_with_cache_tokens(self):
        pricing = ModelPricing(3.0, 15.0, 3.75, 0.30)
        cost = estimate_cost(1000, 500, 2000, 3000, pricing)
        expected = (1000 * 3.0 + 500 * 15.0 + 2000 * 3.75 + 3000 * 0.30) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_haiku_pricing(self):
        pricing = get_pricing("claude-haiku-4-5-20251001")
        assert pricing is not None
        cost = estimate_cost(1000, 500, 2000, 0, pricing)
        expected = (1000 * 0.80 + 500 * 4.0 + 2000 * 1.0) / 1_000_000
        assert cost == pytest.approx(expected)


# --- Token parsing tests ---


class TestTokenParsing:
    def test_transcript_parse_entry_extracts_tokens(self):
        from claude_companion.transcript import _parse_entry

        entry = {
            "type": "assistant",
            "timestamp": "2025-01-01T12:00:00Z",
            "message": {
                "model": "claude-haiku-4-5-20251001",
                "content": [{"type": "text", "text": "Hello"}],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 300,
                },
            },
        }
        turns = _parse_entry(entry, 0)
        assert len(turns) == 1
        assert turns[0].input_tokens == 100
        assert turns[0].output_tokens == 50
        assert turns[0].cache_creation_tokens == 200
        assert turns[0].cache_read_tokens == 300
        assert turns[0].model_id == "claude-haiku-4-5-20251001"

    def test_transcript_parse_entry_first_turn_only(self):
        """Token data should only be on the first turn from an entry."""
        from claude_companion.transcript import _parse_entry

        entry = {
            "type": "assistant",
            "timestamp": "2025-01-01T12:00:00Z",
            "message": {
                "model": "claude-sonnet-4-20250115",
                "content": [
                    {"type": "text", "text": "I'll read the file"},
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "foo.py"}},
                ],
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 100,
                },
            },
        }
        turns = _parse_entry(entry, 0)
        assert len(turns) == 2
        # First turn gets the tokens
        assert turns[0].input_tokens == 500
        assert turns[0].output_tokens == 100
        assert turns[0].model_id == "claude-sonnet-4-20250115"
        # Second turn should NOT have tokens (avoid double-counting)
        assert turns[1].input_tokens is None
        assert turns[1].output_tokens is None
        assert turns[1].model_id is None

    def test_transcript_parse_entry_no_usage(self):
        """Entries without usage data should have None tokens."""
        from claude_companion.transcript import _parse_entry

        entry = {
            "type": "assistant",
            "timestamp": "2025-01-01T12:00:00Z",
            "message": {
                "content": [{"type": "text", "text": "Hello"}],
            },
        }
        turns = _parse_entry(entry, 0)
        assert len(turns) == 1
        assert turns[0].input_tokens is None
        assert turns[0].output_tokens is None
        assert turns[0].model_id is None

    def test_user_entry_no_tokens(self):
        """User entries should not have token data."""
        from claude_companion.transcript import _parse_entry

        entry = {
            "type": "user",
            "timestamp": "2025-01-01T12:00:00Z",
            "message": {"content": "Hello there"},
        }
        turns = _parse_entry(entry, 0)
        assert len(turns) == 1
        assert turns[0].input_tokens is None

    def test_watcher_parse_entry_extracts_tokens(self):
        from claude_companion.watcher import TranscriptWatcher

        watcher = TranscriptWatcher(on_turn=lambda t: None)
        entry = {
            "type": "assistant",
            "message": {
                "model": "claude-opus-4-20250115",
                "content": [{"type": "text", "text": "Hello"}],
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cache_creation_input_tokens": 5000,
                    "cache_read_input_tokens": 0,
                },
            },
        }
        turns = watcher._parse_entry(entry)
        assert len(turns) == 1
        assert turns[0].input_tokens == 1000
        assert turns[0].output_tokens == 200
        assert turns[0].cache_creation_tokens == 5000
        assert turns[0].cache_read_tokens == 0
        assert turns[0].model_id == "claude-opus-4-20250115"


# --- Session token properties tests ---


class TestSessionTokenProperties:
    def _make_token_turn(
        self,
        turn_number: int,
        role: str = "assistant",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_creation: int | None = None,
        cache_read: int | None = None,
        model_id: str | None = None,
    ) -> Turn:
        from claude_companion.models import count_words
        return Turn(
            turn_number=turn_number,
            role=role,
            content_preview="test",
            content_full="test content",
            word_count=2,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            model_id=model_id,
        )

    def test_total_tokens(self):
        session = Session(
            session_id="test",
            turns=[
                self._make_token_turn(1, input_tokens=100, output_tokens=50),
                self._make_token_turn(2, input_tokens=200, output_tokens=100),
            ],
        )
        assert session.total_input_tokens == 300
        assert session.total_output_tokens == 150

    def test_total_tokens_with_none(self):
        session = Session(
            session_id="test",
            turns=[
                self._make_token_turn(1, input_tokens=100, output_tokens=50),
                self._make_token_turn(2),  # No token data
            ],
        )
        assert session.total_input_tokens == 100
        assert session.total_output_tokens == 50

    def test_has_token_data_true(self):
        session = Session(
            session_id="test",
            turns=[self._make_token_turn(1, input_tokens=100)],
        )
        assert session.has_token_data is True

    def test_has_token_data_false(self):
        session = Session(
            session_id="test",
            turns=[self._make_token_turn(1)],
        )
        assert session.has_token_data is False

    def test_estimated_cost_with_known_model(self):
        session = Session(
            session_id="test",
            turns=[
                self._make_token_turn(
                    1,
                    input_tokens=1_000_000,
                    output_tokens=500_000,
                    cache_creation=0,
                    cache_read=0,
                    model_id="claude-sonnet-4-20250115",
                ),
            ],
        )
        cost = session.estimated_cost
        assert cost is not None
        # 1M * 3.0/M + 500K * 15.0/M = 3.0 + 7.5 = 10.5
        assert cost == pytest.approx(10.5)

    def test_estimated_cost_no_token_data(self):
        session = Session(
            session_id="test",
            turns=[self._make_token_turn(1)],
        )
        assert session.estimated_cost is None

    def test_estimated_cost_unknown_model(self):
        session = Session(
            session_id="test",
            turns=[
                self._make_token_turn(1, input_tokens=100, model_id="unknown-model"),
            ],
        )
        assert session.estimated_cost is None

    def test_cache_token_totals(self):
        session = Session(
            session_id="test",
            turns=[
                self._make_token_turn(1, cache_creation=1000, cache_read=2000),
                self._make_token_turn(2, cache_creation=500, cache_read=3000),
            ],
        )
        assert session.total_cache_creation_tokens == 1500
        assert session.total_cache_read_tokens == 5000


# --- AnalyzerConfig tests ---


class TestAnalyzerConfig:
    def test_defaults(self):
        config = AnalyzerConfig()
        assert config.context_budget == 20000
        assert config.small_range_max == 10
        assert config.large_range_min == 31
        assert config.per_turn_budget == 2000

    def test_custom_values(self):
        config = AnalyzerConfig(
            context_budget=30000,
            small_range_max=15,
            large_range_min=50,
            per_turn_budget=3000,
        )
        assert config.context_budget == 30000
        assert config.small_range_max == 15
        assert config.large_range_min == 50
        assert config.per_turn_budget == 3000

    def test_context_manager_uses_config(self):
        config = AnalyzerConfig(
            small_range_max=5,
            large_range_min=20,
            per_turn_budget=1000,
        )
        cm = ContextManager(analyzer_config=config)
        assert cm._config.small_range_max == 5
        assert cm._config.large_range_min == 20
        assert cm._config.per_turn_budget == 1000

    def test_context_manager_default_config(self):
        cm = ContextManager()
        assert cm._config.small_range_max == 10
        assert cm._config.large_range_min == 31

    def test_equality(self):
        a = AnalyzerConfig()
        b = AnalyzerConfig()
        assert a == b

    def test_inequality(self):
        a = AnalyzerConfig()
        b = AnalyzerConfig(context_budget=50000)
        assert a != b


# --- Multi-source GoalExtractor tests ---


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
        with patch("claude_companion.analyzer.subprocess.run") as mock_run:
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
        with patch("claude_companion.analyzer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
            ge._fetch_gh_issue("sess:999", "999", None)

        with ge._gh_lock:
            # Should remain None (in-flight marker set before call)
            assert ge._gh_cache.get("sess:999") is None

    def test_gh_not_installed(self):
        """Missing gh binary handled silently."""
        ge = GoalExtractor()
        with patch("claude_companion.analyzer.subprocess.run", side_effect=FileNotFoundError("gh")):
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
        with patch("claude_companion.analyzer.subprocess.run") as mock_run:
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
        with patch("claude_companion.analyzer.subprocess.run") as mock_run:
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


# --- GoalSource dataclass tests ---


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


# --- Analysis dataclass with new fields ---


class TestAnalysisGoalFields:
    def test_defaults_none(self):
        a = Analysis(
            summary="test", critique="ok", sentiment="progress",
            word_count=10, context_word_count=100,
        )
        assert a.goal_sources is None
        assert a.synthesized_goal is None

    def test_with_goal_sources(self):
        sources = [
            GoalSource("user_request", "User Request", "Fix bug"),
            GoalSource("plan", "Plan", "Step 1"),
        ]
        a = Analysis(
            summary="test", critique="ok", sentiment="progress",
            word_count=10, context_word_count=100,
            goal_sources=sources, synthesized_goal="Fix the authentication bug",
        )
        assert len(a.goal_sources) == 2
        assert a.synthesized_goal == "Fix the authentication bug"


# --- Local goal tests ---


class TestLocalGoal:
    """Tests for local goal (most recent substantive user message) feature."""

    def setup_method(self):
        self.cm = ContextManager()

    def test_local_goal_basic(self):
        """User("Build API"), asst, user("Add auth"), asst → local_goal = "Add auth"."""
        turns = [
            _make_turn(1, "user", "Build a REST API for the project"),
            _make_turn(2, "assistant", "I'll start building it"),
            _make_turn(3, "user", "Add auth to the endpoints"),
            _make_turn(4, "assistant", "Adding authentication now"),
        ]
        session = _make_session(turns)
        result = self.cm._find_local_goal(session, 3)  # analyzing turn at index 3
        assert result == "Add auth to the endpoints"

    def test_local_goal_is_first_message(self):
        """User("Build API"), asst → local_goal = None (avoid duplication)."""
        turns = [
            _make_turn(1, "user", "Build a REST API for the project"),
            _make_turn(2, "assistant", "I'll start building it"),
        ]
        session = _make_session(turns)
        result = self.cm._find_local_goal(session, 1)
        assert result is None

    def test_local_goal_short_filtered(self):
        """User("Build API"), asst, user("ok"), asst → local_goal = None."""
        turns = [
            _make_turn(1, "user", "Build a REST API for the project"),
            _make_turn(2, "assistant", "Working on it"),
            _make_turn(3, "user", "ok"),
            _make_turn(4, "assistant", "Done"),
        ]
        session = _make_session(turns)
        result = self.cm._find_local_goal(session, 3)
        # Walks past "ok" (< 4 words), hits first user, returns None
        assert result is None

    def test_local_goal_short_skipped_to_substantive(self):
        """User("Build API"), asst, user("Add error handling"), asst, user("ok"), asst → local_goal = "Add error handling"."""
        turns = [
            _make_turn(1, "user", "Build a REST API for the project"),
            _make_turn(2, "assistant", "Working on it"),
            _make_turn(3, "user", "Add error handling to all routes"),
            _make_turn(4, "assistant", "Adding error handling"),
            _make_turn(5, "user", "ok"),
            _make_turn(6, "assistant", "Done with error handling"),
        ]
        session = _make_session(turns)
        result = self.cm._find_local_goal(session, 5)
        assert result == "Add error handling to all routes"

    def test_local_goal_no_user_before(self):
        """asst, tool → local_goal = None."""
        turns = [
            _make_turn(1, "assistant", "I'll read the file"),
            _make_turn(2, "tool", "file contents", "Read"),
        ]
        session = _make_session(turns)
        result = self.cm._find_local_goal(session, 1)
        assert result is None

    def test_local_goal_in_context_dict(self):
        """Verify build_context() returns local_goal key."""
        turns = [
            _make_turn(1, "user", "Build a REST API for the project"),
            _make_turn(2, "assistant", "Starting"),
            _make_turn(3, "user", "Add auth to the endpoints"),
            _make_turn(4, "assistant", "Adding auth"),
        ]
        session = _make_session(turns)
        ctx = self.cm.build_context(session, turns[3])
        assert "local_goal" in ctx
        assert ctx["local_goal"] == "Add auth to the endpoints"

    def test_local_goal_range_starts_with_user(self):
        """Range starting with user turn → local_goal = None."""
        turns = [
            _make_turn(1, "user", "Build a REST API for the project"),
            _make_turn(2, "assistant", "Starting"),
            _make_turn(3, "user", "Add auth to the endpoints"),
            _make_turn(4, "assistant", "Adding auth"),
        ]
        session = _make_session(turns)
        # Range starts with user turn at index 2
        ctx = self.cm.build_context_for_range(session, turns[2:4])
        assert ctx["local_goal"] is None

    def test_local_goal_range_mid_span(self):
        """Range starting with asst turn → local_goal = preceding user message."""
        turns = [
            _make_turn(1, "user", "Build a REST API for the project"),
            _make_turn(2, "assistant", "Starting"),
            _make_turn(3, "user", "Add auth to the endpoints"),
            _make_turn(4, "assistant", "Adding auth"),
            _make_turn(5, "tool", "read auth.py", "Read"),
        ]
        session = _make_session(turns)
        # Range is [turns[3], turns[4]] — starts with assistant, not user
        ctx = self.cm.build_context_for_range(session, turns[3:5])
        assert ctx["local_goal"] == "Add auth to the endpoints"

    def test_local_goal_in_prompt(self):
        """Prompt contains '## Current Objective' when local_goal present, absent when not."""
        analyzer = Analyzer(model="test-model")

        # With local goal
        turns = [
            _make_turn(1, "user", "Build a REST API for the project"),
            _make_turn(2, "assistant", "Starting"),
            _make_turn(3, "user", "Add auth to the endpoints"),
            _make_turn(4, "assistant", "Adding auth"),
        ]
        session = _make_session(turns)
        ctx = analyzer._context_manager.build_context(session, turns[3])
        prompt = analyzer._build_prompt(ctx)
        assert "## Current Objective" in prompt
        assert "Add auth to the endpoints" in prompt

        # Without local goal (analyzing right after first user message)
        ctx2 = analyzer._context_manager.build_context(session, turns[1])
        prompt2 = analyzer._build_prompt(ctx2)
        assert "## Current Objective" not in prompt2

    def test_prompt_session_goal_header(self):
        """Prompt says '## Session Goal' not '## Goal (first user message)'."""
        analyzer = Analyzer(model="test-model")
        turns = [
            _make_turn(1, "user", "Build a REST API"),
            _make_turn(2, "assistant", "Working on it"),
        ]
        session = _make_session(turns)
        ctx = analyzer._context_manager.build_context(session, turns[1])
        prompt = analyzer._build_prompt(ctx)
        assert "## Session Goal" in prompt
        assert "## Goal (first user message)" not in prompt

        # Also check range prompt
        ctx_range = analyzer._context_manager.build_context_for_range(session, turns)
        prompt_range = analyzer._build_prompt_for_range(ctx_range)
        assert "## Session Goal" in prompt_range
        assert "## Goal (first user message)" not in prompt_range

    def test_local_goal_cache_key_differs(self):
        """Different local_goal → different cache key."""
        turn = _make_turn(1, "assistant", "Hello world")
        ctx_a = {"goal": "Fix bug", "window": [], "local_goal": "Add auth"}
        ctx_b = {"goal": "Fix bug", "window": [], "local_goal": "Add tests"}
        key_a = make_analysis_cache_key(turn, ctx_a, "model-a", ANALYZER_SYSTEM_PROMPT)
        key_b = make_analysis_cache_key(turn, ctx_b, "model-a", ANALYZER_SYSTEM_PROMPT)
        assert key_a != key_b

        # None vs present should also differ
        ctx_c = {"goal": "Fix bug", "window": [], "local_goal": None}
        key_c = make_analysis_cache_key(turn, ctx_c, "model-a", ANALYZER_SYSTEM_PROMPT)
        assert key_a != key_c

    def test_local_goal_truncation(self):
        """1000-char message truncated to 500 + '...'."""
        long_msg = "word " * 200  # 1000 chars
        turns = [
            _make_turn(1, "user", "Build an API for the project"),
            _make_turn(2, "assistant", "Starting"),
            _make_turn(3, "user", long_msg),
            _make_turn(4, "assistant", "Done"),
        ]
        session = _make_session(turns)
        result = self.cm._find_local_goal(session, 3)
        assert result is not None
        assert len(result) == 503  # 500 + "..."
        assert result.endswith("...")

    def test_analysis_local_goal_field(self):
        """Analysis dataclass preserves local_goal value."""
        a = Analysis(
            summary="test", critique="ok", sentiment="progress",
            word_count=10, context_word_count=100,
            local_goal="Add error handling to all routes",
        )
        assert a.local_goal == "Add error handling to all routes"

        # Default is None
        b = Analysis(
            summary="test", critique="ok", sentiment="progress",
            word_count=10, context_word_count=100,
        )
        assert b.local_goal is None
