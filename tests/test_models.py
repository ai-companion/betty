"""Tests for models, pricing, and token parsing."""

from datetime import datetime

import pytest

from betty.models import Session, Turn, compute_spans, count_words
from betty.pricing import ModelPricing, get_pricing, estimate_cost, MODEL_PRICING


def _make_turn(
    turn_number: int = 1,
    role: str = "assistant",
    content: str = "Hello world",
    tool_name: str | None = None,
) -> Turn:
    """Create a Turn for testing."""
    return Turn(
        turn_number=turn_number,
        role=role,
        content_preview=content[:80],
        content_full=content,
        word_count=count_words(content),
        tool_name=tool_name,
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
    )


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
        from betty.transcript import _parse_entry

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
        from betty.transcript import _parse_entry

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
        from betty.transcript import _parse_entry

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
        from betty.transcript import _parse_entry

        entry = {
            "type": "user",
            "timestamp": "2025-01-01T12:00:00Z",
            "message": {"content": "Hello there"},
        }
        turns = _parse_entry(entry, 0)
        assert len(turns) == 1
        assert turns[0].input_tokens is None

    def test_watcher_parse_entry_extracts_tokens(self):
        from betty.watcher import TranscriptWatcher

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
