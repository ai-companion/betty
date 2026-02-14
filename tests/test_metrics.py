"""Tests for session health metrics."""

from datetime import datetime, timedelta

from betty.metrics import (
    SessionMetrics,
    _compute_error_retry_count,
    _compute_repetitive_tool_score,
    _count_trailing_retries,
    _detect_output_shrinking,
    _extract_file_paths,
    _is_error_turn,
    compute_session_metrics,
)
from betty.models import Session, Turn


def _make_turn(
    turn_number: int = 1,
    role: str = "assistant",
    content: str = "",
    tool_name: str | None = None,
    timestamp: datetime | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> Turn:
    """Helper to create a Turn for testing."""
    return Turn(
        turn_number=turn_number,
        role=role,
        content_preview=content[:200],
        content_full=content,
        word_count=len(content.split()),
        tool_name=tool_name,
        timestamp=timestamp or datetime.now(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _make_session(turns: list[Turn] | None = None) -> Session:
    """Helper to create a Session for testing."""
    return Session(
        session_id="test-session",
        turns=turns or [],
    )


class TestErrorDetection:
    def test_exit_code_nonzero(self):
        turn = _make_turn(role="tool", content="exit code: 1", tool_name="Bash")
        assert _is_error_turn(turn)

    def test_exit_code_zero(self):
        turn = _make_turn(role="tool", content="exit code: 0", tool_name="Bash")
        assert not _is_error_turn(turn)

    def test_error_prefix(self):
        turn = _make_turn(role="tool", content="Error: file not found", tool_name="Bash")
        assert _is_error_turn(turn)

    def test_traceback(self):
        turn = _make_turn(
            role="tool",
            content="Traceback (most recent call last):\n  File...",
            tool_name="Bash",
        )
        assert _is_error_turn(turn)

    def test_failed(self):
        turn = _make_turn(role="tool", content="3 FAILED, 2 passed", tool_name="Bash")
        assert _is_error_turn(turn)

    def test_normal_output(self):
        turn = _make_turn(role="tool", content="5 tests passed", tool_name="Bash")
        assert not _is_error_turn(turn)

    def test_permission_denied(self):
        turn = _make_turn(role="tool", content="Permission denied", tool_name="Bash")
        assert _is_error_turn(turn)


class TestFilePathExtraction:
    def test_read_tool(self):
        turn = _make_turn(
            role="tool",
            content="/Users/user/project/src/main.py",
            tool_name="Read",
        )
        paths = _extract_file_paths(turn)
        assert "/Users/user/project/src/main.py" in paths

    def test_write_tool(self):
        turn = _make_turn(
            role="tool",
            content="/Users/user/project/src/main.py (25 lines)",
            tool_name="Write",
        )
        paths = _extract_file_paths(turn)
        assert "/Users/user/project/src/main.py" in paths

    def test_edit_tool(self):
        turn = _make_turn(
            role="tool",
            content="/Users/user/project/src/main.py (-3, +5)",
            tool_name="Edit",
        )
        paths = _extract_file_paths(turn)
        assert "/Users/user/project/src/main.py" in paths

    def test_no_tool_name(self):
        turn = _make_turn(role="tool", content="/some/path")
        assert _extract_file_paths(turn) == set()


class TestRetryDetection:
    def test_no_retries(self):
        turns = [
            _make_turn(role="tool", tool_name="Read"),
            _make_turn(role="tool", tool_name="Write"),
            _make_turn(role="tool", tool_name="Bash"),
        ]
        assert _count_trailing_retries(turns) == 0

    def test_two_retries(self):
        turns = [
            _make_turn(role="tool", tool_name="Read"),
            _make_turn(role="tool", tool_name="Bash"),
            _make_turn(role="tool", tool_name="Bash"),
        ]
        assert _count_trailing_retries(turns) == 2

    def test_three_retries(self):
        turns = [
            _make_turn(role="tool", tool_name="Read"),
            _make_turn(role="tool", tool_name="Bash"),
            _make_turn(role="tool", tool_name="Bash"),
            _make_turn(role="tool", tool_name="Bash"),
        ]
        assert _count_trailing_retries(turns) == 3

    def test_empty(self):
        assert _count_trailing_retries([]) == 0

    def test_single_tool(self):
        turns = [_make_turn(role="tool", tool_name="Read")]
        assert _count_trailing_retries(turns) == 0


class TestComputeSessionMetrics:
    def test_empty_session(self):
        session = _make_session([])
        metrics = compute_session_metrics(session)
        assert metrics.token_burn_rate == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.retry_count == 0
        assert metrics.turn_velocity == 0.0
        assert metrics.total_input_tokens == 0
        assert metrics.total_output_tokens == 0

    def test_basic_session(self):
        now = datetime.now()
        turns = [
            _make_turn(1, "user", "Fix the bug", timestamp=now - timedelta(minutes=10), input_tokens=100),
            _make_turn(2, "assistant", "I'll fix it", timestamp=now - timedelta(minutes=9), output_tokens=50),
            _make_turn(3, "tool", "/src/main.py", tool_name="Read", timestamp=now - timedelta(minutes=8)),
            _make_turn(4, "tool", "/src/main.py (-2, +3)", tool_name="Edit", timestamp=now - timedelta(minutes=7)),
            _make_turn(5, "assistant", "Fixed the bug", timestamp=now - timedelta(minutes=6), output_tokens=30),
        ]
        session = _make_session(turns)
        metrics = compute_session_metrics(session)

        assert metrics.total_input_tokens == 100
        assert metrics.total_output_tokens == 80
        assert metrics.turn_velocity > 0
        assert metrics.error_rate == 0.0
        assert "Read" in metrics.tool_distribution
        assert "Edit" in metrics.tool_distribution
        assert len(metrics.files_touched) > 0

    def test_error_heavy_session(self):
        now = datetime.now()
        turns = [
            _make_turn(1, "user", "Run tests", timestamp=now - timedelta(minutes=5)),
            _make_turn(2, "tool", "exit code: 1\n3 FAILED", tool_name="Bash", timestamp=now - timedelta(minutes=4)),
            _make_turn(3, "tool", "Error: module not found", tool_name="Bash", timestamp=now - timedelta(minutes=3)),
            _make_turn(4, "tool", "5 passed, 0 failed", tool_name="Bash", timestamp=now - timedelta(minutes=2)),
        ]
        session = _make_session(turns)
        metrics = compute_session_metrics(session)

        # 2 errors out of 3 tool turns
        assert abs(metrics.error_rate - 2 / 3) < 0.01

    def test_retry_loop(self):
        now = datetime.now()
        turns = [
            _make_turn(1, "user", "Fix it", timestamp=now - timedelta(minutes=5)),
            _make_turn(2, "tool", "ok", tool_name="Read", timestamp=now - timedelta(minutes=4)),
            _make_turn(3, "tool", "error", tool_name="Bash", timestamp=now - timedelta(minutes=3)),
            _make_turn(4, "tool", "error", tool_name="Bash", timestamp=now - timedelta(minutes=2)),
            _make_turn(5, "tool", "error", tool_name="Bash", timestamp=now - timedelta(minutes=1)),
        ]
        session = _make_session(turns)
        metrics = compute_session_metrics(session)
        assert metrics.retry_count == 3

    def test_tool_distribution(self):
        now = datetime.now()
        turns = [
            _make_turn(1, "tool", "ok", tool_name="Read", timestamp=now),
            _make_turn(2, "tool", "ok", tool_name="Read", timestamp=now),
            _make_turn(3, "tool", "ok", tool_name="Write", timestamp=now),
            _make_turn(4, "tool", "ok", tool_name="Edit", timestamp=now),
            _make_turn(5, "tool", "ok", tool_name="Bash", timestamp=now),
            _make_turn(6, "tool", "ok", tool_name="Bash", timestamp=now),
        ]
        session = _make_session(turns)
        metrics = compute_session_metrics(session)

        assert metrics.tool_distribution == {
            "Read": 2,
            "Write": 1,
            "Edit": 1,
            "Bash": 2,
        }


class TestRepetitiveToolScore:
    def test_all_same_tool(self):
        turns = [_make_turn(role="tool", tool_name="Bash") for _ in range(5)]
        assert _compute_repetitive_tool_score(turns) == 1.0

    def test_all_different(self):
        tools = ["Read", "Write", "Edit", "Bash", "Glob"]
        turns = [_make_turn(role="tool", tool_name=t) for t in tools]
        assert _compute_repetitive_tool_score(turns) == 0.2

    def test_too_few(self):
        turns = [_make_turn(role="tool", tool_name="Bash") for _ in range(2)]
        assert _compute_repetitive_tool_score(turns) == 0.0


class TestErrorRetryCount:
    def test_error_then_retry(self):
        turns = [
            _make_turn(role="tool", content="exit code: 1", tool_name="Bash"),
            _make_turn(role="tool", content="ok", tool_name="Bash"),
        ]
        assert _compute_error_retry_count(turns) == 1

    def test_error_different_tool(self):
        turns = [
            _make_turn(role="tool", content="exit code: 1", tool_name="Bash"),
            _make_turn(role="tool", content="ok", tool_name="Read"),
        ]
        assert _compute_error_retry_count(turns) == 0

    def test_multiple_error_retries(self):
        turns = [
            _make_turn(role="tool", content="exit code: 1", tool_name="Bash"),
            _make_turn(role="tool", content="exit code: 1", tool_name="Bash"),
            _make_turn(role="tool", content="ok", tool_name="Bash"),
        ]
        # First->second is error+same tool, second->third is error+same tool
        assert _compute_error_retry_count(turns) == 2


class TestOutputShrinking:
    def test_shrinking(self):
        now = datetime.now()
        turns = [
            _make_turn(role="assistant", content=" ".join(["word"] * 100), timestamp=now),
            _make_turn(role="assistant", content=" ".join(["word"] * 50), timestamp=now),
            _make_turn(role="assistant", content=" ".join(["word"] * 20), timestamp=now),
            _make_turn(role="assistant", content=" ".join(["word"] * 5), timestamp=now),
        ]
        assert _detect_output_shrinking(turns) is True

    def test_growing(self):
        now = datetime.now()
        turns = [
            _make_turn(role="assistant", content="short", timestamp=now),
            _make_turn(role="assistant", content="a longer message", timestamp=now),
            _make_turn(role="assistant", content="an even much longer message here", timestamp=now),
        ]
        assert _detect_output_shrinking(turns) is False

    def test_too_few(self):
        turns = [
            _make_turn(role="assistant", content="short", timestamp=datetime.now()),
        ]
        assert _detect_output_shrinking(turns) is False


class TestSpinMetricsInSession:
    def test_spin_session(self):
        """Session with heavy spin signals."""
        now = datetime.now()
        turns = [
            _make_turn(1, "user", "Fix it", timestamp=now - timedelta(minutes=5)),
        ]
        # 8 same-tool error turns
        for i in range(8):
            turns.append(_make_turn(
                i + 2, "tool", "exit code: 1", tool_name="Bash",
                timestamp=now - timedelta(minutes=4) + timedelta(seconds=i),
            ))
        session = _make_session(turns)
        metrics = compute_session_metrics(session)

        assert metrics.repetitive_tool_score > 0.7
        assert metrics.error_retry_count >= 1
        assert metrics.tool_diversity < 0.5
