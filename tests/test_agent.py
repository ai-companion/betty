"""Tests for Betty Agent observation engine."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from betty.agent import Agent
from betty.cache import AgentCache
from betty.config import AgentConfig
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


def _make_session(turns: list[Turn] | None = None, session_id: str = "test") -> Session:
    return Session(session_id=session_id, turns=turns or [])


def _make_agent(tmp_path: Path | None = None) -> Agent:
    """Create an agent with an isolated temp cache to avoid cross-test interference."""
    agent = Agent(AgentConfig(enabled=True))
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    agent._cache = AgentCache(cache_dir=tmp_path)
    return agent


class TestGoalSet:
    def test_first_user_turn_sets_goal(self):
        agent = _make_agent()
        session = _make_session()
        turn = _make_turn(1, "user", "Fix the authentication bug")
        session.turns.append(turn)

        agent.on_turn(turn, session)
        report = agent.get_report("test")

        assert report is not None
        assert report.goal == "Fix the authentication bug"
        obs = [o for o in report.observations if o.observation_type == "goal_set"]
        assert len(obs) == 1

    def test_second_user_turn_does_not_reset_goal(self):
        agent = _make_agent()
        session = _make_session()

        t1 = _make_turn(1, "user", "Fix the auth bug")
        session.turns.append(t1)
        agent.on_turn(t1, session)

        t2 = _make_turn(2, "user", "Actually fix the login page")
        session.turns.append(t2)
        agent.on_turn(t2, session)

        report = agent.get_report("test")
        assert report.goal == "Fix the auth bug"

    def test_non_user_turn_no_goal(self):
        agent = _make_agent()
        session = _make_session()

        t1 = _make_turn(1, "assistant", "I'll help you")
        session.turns.append(t1)
        agent.on_turn(t1, session)

        report = agent.get_report("test")
        assert report.goal is None


class TestErrorSpike:
    def test_high_error_rate(self):
        agent = _make_agent()
        session = _make_session()

        now = datetime.now()
        # 5 error tool turns (>40% of recent)
        for i in range(5):
            t = _make_turn(i + 1, "tool", "exit code: 1", tool_name="Bash",
                          timestamp=now + timedelta(seconds=i))
            session.turns.append(t)
            agent.on_turn(t, session)

        report = agent.get_report("test")
        error_obs = [o for o in report.observations if o.observation_type == "error_spike"]
        assert len(error_obs) > 0

    def test_low_error_rate(self):
        agent = _make_agent()
        session = _make_session()

        now = datetime.now()
        # 1 error out of 5 = 20% < 40%
        for i in range(4):
            t = _make_turn(i + 1, "tool", "ok", tool_name="Bash",
                          timestamp=now + timedelta(seconds=i))
            session.turns.append(t)
            agent.on_turn(t, session)

        t = _make_turn(5, "tool", "exit code: 1", tool_name="Bash",
                      timestamp=now + timedelta(seconds=5))
        session.turns.append(t)
        agent.on_turn(t, session)

        report = agent.get_report("test")
        error_obs = [o for o in report.observations if o.observation_type == "error_spike"]
        assert len(error_obs) == 0


class TestRetryLoop:
    def test_three_consecutive(self):
        agent = _make_agent()
        session = _make_session()

        now = datetime.now()
        for i in range(3):
            t = _make_turn(i + 1, "tool", "trying again", tool_name="Bash",
                          timestamp=now + timedelta(seconds=i))
            session.turns.append(t)
            agent.on_turn(t, session)

        report = agent.get_report("test")
        retry_obs = [o for o in report.observations if o.observation_type == "retry_loop"]
        assert len(retry_obs) > 0

    def test_different_tools_no_retry(self):
        agent = _make_agent()
        session = _make_session()

        now = datetime.now()
        tools = ["Read", "Write", "Bash"]
        for i, tool in enumerate(tools):
            t = _make_turn(i + 1, "tool", "ok", tool_name=tool,
                          timestamp=now + timedelta(seconds=i))
            session.turns.append(t)
            agent.on_turn(t, session)

        report = agent.get_report("test")
        retry_obs = [o for o in report.observations if o.observation_type == "retry_loop"]
        assert len(retry_obs) == 0


class TestStall:
    def test_long_gap(self):
        agent = _make_agent()
        session = _make_session()

        t1 = _make_turn(1, "user", "start", timestamp=datetime.now() - timedelta(minutes=5))
        session.turns.append(t1)
        agent.on_turn(t1, session)

        t2 = _make_turn(2, "assistant", "response", timestamp=datetime.now())
        session.turns.append(t2)
        agent.on_turn(t2, session)

        report = agent.get_report("test")
        stall_obs = [o for o in report.observations if o.observation_type == "stall"]
        assert len(stall_obs) == 1

    def test_no_stall_quick_turns(self):
        agent = _make_agent()
        session = _make_session()

        now = datetime.now()
        t1 = _make_turn(1, "user", "start", timestamp=now)
        session.turns.append(t1)
        agent.on_turn(t1, session)

        t2 = _make_turn(2, "assistant", "response", timestamp=now + timedelta(seconds=5))
        session.turns.append(t2)
        agent.on_turn(t2, session)

        report = agent.get_report("test")
        stall_obs = [o for o in report.observations if o.observation_type == "stall"]
        assert len(stall_obs) == 0


class TestFileChanges:
    def test_write_detected(self):
        agent = _make_agent()
        session = _make_session()

        t = _make_turn(1, "tool", "/src/main.py (25 lines)", tool_name="Write")
        session.turns.append(t)
        agent.on_turn(t, session)

        report = agent.get_report("test")
        file_obs = [o for o in report.observations if o.observation_type == "file_change"]
        assert len(file_obs) == 1
        assert "/src/main.py" in file_obs[0].content

    def test_edit_detected(self):
        agent = _make_agent()
        session = _make_session()

        t = _make_turn(1, "tool", "/src/main.py (-2, +3)", tool_name="Edit")
        session.turns.append(t)
        agent.on_turn(t, session)

        report = agent.get_report("test")
        file_obs = [o for o in report.observations if o.observation_type == "file_change"]
        assert len(file_obs) == 1

    def test_read_not_tracked(self):
        agent = _make_agent()
        session = _make_session()

        t = _make_turn(1, "tool", "/src/main.py", tool_name="Read")
        session.turns.append(t)
        agent.on_turn(t, session)

        report = agent.get_report("test")
        file_obs = [o for o in report.observations if o.observation_type == "file_change"]
        assert len(file_obs) == 0


class TestMilestone:
    def test_10th_tool_call(self):
        agent = _make_agent()
        session = _make_session()

        now = datetime.now()
        for i in range(10):
            t = _make_turn(i + 1, "tool", "ok", tool_name="Read",
                          timestamp=now + timedelta(seconds=i))
            session.turns.append(t)
            agent.on_turn(t, session)

        report = agent.get_report("test")
        milestone_obs = [o for o in report.observations if o.observation_type == "milestone"]
        assert any("10 tool calls" in o.content for o in milestone_obs)

    def test_5th_user_turn(self):
        agent = _make_agent()
        session = _make_session()

        now = datetime.now()
        for i in range(5):
            t = _make_turn(i + 1, "user", f"message {i + 1}",
                          timestamp=now + timedelta(seconds=i))
            session.turns.append(t)
            agent.on_turn(t, session)

        report = agent.get_report("test")
        milestone_obs = [o for o in report.observations if o.observation_type == "milestone"]
        assert any("5 user messages" in o.content for o in milestone_obs)


class TestObservationLimit:
    def test_max_observations_enforced(self):
        agent = Agent(AgentConfig(enabled=True, max_observations=5))
        session = _make_session()

        now = datetime.now()
        # Generate many file change observations
        for i in range(10):
            t = _make_turn(i + 1, "tool", f"/src/file{i}.py (10 lines)", tool_name="Write",
                          timestamp=now + timedelta(seconds=i))
            session.turns.append(t)
            agent.on_turn(t, session)

        report = agent.get_report("test")
        assert len(report.observations) <= 5


class TestNormalSession:
    def test_normal_flow(self):
        """Simulate a normal session: user asks, assistant responds, tools execute."""
        agent = _make_agent()
        session = _make_session()

        now = datetime.now()
        turns = [
            _make_turn(1, "user", "Add a login page", timestamp=now),
            _make_turn(2, "assistant", "I'll create a login page", timestamp=now + timedelta(seconds=2)),
            _make_turn(3, "tool", "/src/pages", tool_name="Read", timestamp=now + timedelta(seconds=3)),
            _make_turn(4, "tool", "/src/pages/login.py (50 lines)", tool_name="Write", timestamp=now + timedelta(seconds=5)),
            _make_turn(5, "assistant", "Created the login page", timestamp=now + timedelta(seconds=7)),
        ]
        for t in turns:
            session.turns.append(t)
            agent.on_turn(t, session)

        report = agent.get_report("test")
        assert report is not None
        assert report.goal is not None
        assert report.metrics is not None
        assert len(report.observations) > 0

        # Should have goal_set and file_change at minimum
        types = {o.observation_type for o in report.observations}
        assert "goal_set" in types
        assert "file_change" in types


class TestProgressAssessment:
    def test_healthy_session_on_track(self):
        agent = _make_agent()
        session = _make_session()

        now = datetime.now()
        turns = [
            _make_turn(1, "user", "Add a feature", timestamp=now),
            _make_turn(2, "tool", "/src/a.py", tool_name="Read", timestamp=now + timedelta(seconds=1)),
            _make_turn(3, "tool", "/src/b.py (10 lines)", tool_name="Write", timestamp=now + timedelta(seconds=2)),
            _make_turn(4, "tool", "5 passed", tool_name="Bash", timestamp=now + timedelta(seconds=3)),
            _make_turn(5, "assistant", "Done adding the feature", timestamp=now + timedelta(seconds=4)),
        ]
        for t in turns:
            session.turns.append(t)
            agent.on_turn(t, session)

        report = agent.get_report("test")
        assert report.progress_assessment == "on_track"

    def test_error_heavy_session_not_on_track(self):
        agent = _make_agent()
        session = _make_session()

        now = datetime.now()
        turns = [_make_turn(1, "user", "Fix bug", timestamp=now)]
        # Many same-tool error turns
        for i in range(10):
            turns.append(_make_turn(
                i + 2, "tool", "exit code: 1\nFAILED", tool_name="Bash",
                timestamp=now + timedelta(seconds=i + 1),
            ))
        for t in turns:
            session.turns.append(t)
            agent.on_turn(t, session)

        report = agent.get_report("test")
        assert report.progress_assessment in ("spinning", "stalled", "off_track")


class TestFileChangeTracking:
    def test_write_tracked(self):
        agent = _make_agent()
        session = _make_session()

        t = _make_turn(1, "tool", "/src/main.py (25 lines)", tool_name="Write")
        session.turns.append(t)
        agent.on_turn(t, session)

        report = agent.get_report("test")
        assert len(report.file_changes) == 1
        fc = report.file_changes[0]
        assert fc.file_path == "/src/main.py"
        assert fc.operation == "write"
        assert fc.lines_added == 25

    def test_edit_tracked_with_delta(self):
        agent = _make_agent()
        session = _make_session()

        t = _make_turn(1, "tool", "/src/main.py (-3, +5)", tool_name="Edit")
        session.turns.append(t)
        agent.on_turn(t, session)

        report = agent.get_report("test")
        assert len(report.file_changes) == 1
        fc = report.file_changes[0]
        assert fc.file_path == "/src/main.py"
        assert fc.operation == "edit"
        assert fc.lines_added == 5
        assert fc.lines_removed == 3

    def test_read_tracked(self):
        agent = _make_agent()
        session = _make_session()

        t = _make_turn(1, "tool", "/src/main.py", tool_name="Read")
        session.turns.append(t)
        agent.on_turn(t, session)

        report = agent.get_report("test")
        assert len(report.file_changes) == 1
        assert report.file_changes[0].operation == "read"

    def test_multiple_edits_to_same_file(self):
        agent = _make_agent()
        session = _make_session()

        now = datetime.now()
        for i in range(3):
            t = _make_turn(i + 1, "tool", f"/src/main.py (-1, +2)", tool_name="Edit",
                          timestamp=now + timedelta(seconds=i))
            session.turns.append(t)
            agent.on_turn(t, session)

        report = agent.get_report("test")
        # All 3 changes are tracked individually
        assert len(report.file_changes) == 3
        # All for same file
        assert all(fc.file_path == "/src/main.py" for fc in report.file_changes)


class TestMultiSession:
    def test_independent_sessions(self):
        agent = _make_agent()
        session_a = _make_session(session_id="a")
        session_b = _make_session(session_id="b")

        t1 = _make_turn(1, "user", "Fix bug A")
        session_a.turns.append(t1)
        agent.on_turn(t1, session_a)

        t2 = _make_turn(1, "user", "Fix bug B")
        session_b.turns.append(t2)
        agent.on_turn(t2, session_b)

        report_a = agent.get_report("a")
        report_b = agent.get_report("b")

        assert report_a.goal == "Fix bug A"
        assert report_b.goal == "Fix bug B"
