"""Tests for TUI ] / [ / Enter navigation (drill out/in with span highlighting)."""

import pytest
from datetime import datetime

from betty.models import Session, Turn, ToolGroup, SpanGroup, compute_spans
from betty.store import EventStore
from betty.tui_textual import (
    BettyApp,
    ConversationView,
    TurnWidget,
    ToolGroupWidget,
    SpanGroupWidget,
)


def _turn(num: int, role: str, tool_name: str | None = None) -> Turn:
    """Create a minimal Turn for testing."""
    content = f"{role} turn {num}"
    return Turn(
        turn_number=num,
        role=role,
        content_preview=content,
        content_full=content,
        word_count=3,
        tool_name=tool_name,
        timestamp=datetime(2025, 1, 1, 12, 0, num),
    )


def _make_session() -> Session:
    """Create a session with 2 spans, second span has tool turns.

    Span 1: user(1), assistant(2)
    Span 2: user(3), tool(4, Read), tool(5, Write), assistant(6)
    """
    session = Session(session_id="test-nav", project_path="/tmp/test")
    session.turns = [
        _turn(1, "user"),
        _turn(2, "assistant"),
        _turn(3, "user"),
        _turn(4, "tool", "Read"),
        _turn(5, "tool", "Write"),
        _turn(6, "assistant"),
    ]
    return session


def _make_app(session: Session) -> BettyApp:
    store = EventStore(enable_notifications=False)
    store._sessions[session.session_id] = session
    store._active_session_id = session.session_id
    return BettyApp(store=store, collapse_tools=True, ui_style="rich")


def _widgets(app: BettyApp):
    conv = app.query_one("#conversation", ConversationView)
    return list(conv.query("TurnWidget, ToolGroupWidget, SpanGroupWidget"))


# --- Tests ---


@pytest.mark.asyncio
async def test_initial_view_shows_expanded_turns():
    """Default view: all spans expanded, tools grouped."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        widgets = _widgets(app)

        # Span 1: user(1), assistant(2)
        # Span 2: user(3), ToolGroup(4,5), assistant(6)
        # = 5 widgets
        assert len(widgets) == 5, f"Expected 5 widgets, got {len(widgets)}: {[type(w).__name__ for w in widgets]}"
        assert isinstance(widgets[0], TurnWidget)  # user 1
        assert isinstance(widgets[1], TurnWidget)  # assistant 2
        assert isinstance(widgets[2], TurnWidget)  # user 3
        assert isinstance(widgets[3], ToolGroupWidget)  # tools 4,5
        assert isinstance(widgets[4], TurnWidget)  # assistant 6


@pytest.mark.asyncio
async def test_drill_out_highlights_span():
    """] on a turn highlights the parent span (all siblings selected)."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()

        # Select user turn 1 (index 0)
        conv = app.query_one("#conversation", ConversationView)
        conv.set_selected_index(0)
        await pilot.pause()

        # Press ] to highlight span
        await pilot.press("]")
        await pilot.pause()

        assert app._highlight == 1, f"Expected 1, got {app._highlight}"

        # Both widgets in span 1 should be selected
        widgets = _widgets(app)
        assert widgets[0].selected is True, "user turn should be selected"
        assert widgets[1].selected is True, "assistant turn should be selected"
        # Widgets outside span should not be selected
        assert widgets[2].selected is False, "user turn in span 2 should not be selected"


@pytest.mark.asyncio
async def test_drill_out_noop_when_highlighted():
    """] is a no-op when already highlighting."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation", ConversationView)
        conv.set_selected_index(0)

        await pilot.press("]")
        await pilot.pause()
        assert app._highlight == 1

        # Press ] again — should be no-op
        await pilot.press("]")
        await pilot.pause()
        assert app._highlight == 1


@pytest.mark.asyncio
async def test_drill_in_clears_highlight():
    """[ clears the highlight when one is active."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation", ConversationView)
        conv.set_selected_index(0)

        await pilot.press("]")
        await pilot.pause()
        assert app._highlight is not None

        await pilot.press("[")
        await pilot.pause()
        assert app._highlight is None

        # Selection should be back to single item
        widgets = _widgets(app)
        assert widgets[0].selected is True
        assert widgets[1].selected is False


@pytest.mark.asyncio
async def test_enter_collapses_highlighted_span():
    """Enter while span is highlighted collapses it into SpanGroup."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation", ConversationView)
        conv.set_selected_index(0)

        # Highlight span 1
        await pilot.press("]")
        await pilot.pause()

        # Enter to collapse
        await pilot.press("enter")
        await pilot.pause()

        assert app._highlight is None
        assert app._span_expanded_state.get(1) is False

        # Now first widget should be a SpanGroupWidget
        widgets = _widgets(app)
        assert isinstance(widgets[0], SpanGroupWidget), f"Expected SpanGroupWidget, got {type(widgets[0]).__name__}"


@pytest.mark.asyncio
async def test_bracket_expands_span_group_no_highlight():
    """[ on a SpanGroupWidget expands it, cursor on first child, no highlight."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation", ConversationView)
        conv.set_selected_index(0)

        # Highlight then collapse span 1
        await pilot.press("]")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        # Now cursor should be on the SpanGroup
        widgets = _widgets(app)
        assert isinstance(widgets[0], SpanGroupWidget)

        # Press [ to expand it back — no auto-highlight
        await pilot.press("[")
        await pilot.pause()

        assert app._span_expanded_state.get(1) is True
        widgets = _widgets(app)
        assert isinstance(widgets[0], TurnWidget)
        assert app._highlight is None  # [ does NOT auto-highlight
        assert widgets[0].selected is True   # cursor on first child
        assert widgets[1].selected is False  # single selection


@pytest.mark.asyncio
async def test_enter_expands_span_group_with_highlight():
    """Enter on a SpanGroupWidget expands it and auto-highlights the span."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation", ConversationView)
        conv.set_selected_index(0)

        # Highlight then collapse span 1
        await pilot.press("]")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        # Now cursor should be on the SpanGroup
        widgets = _widgets(app)
        assert isinstance(widgets[0], SpanGroupWidget)

        # Press Enter to expand it — auto-highlights
        await pilot.press("enter")
        await pilot.pause()

        assert app._span_expanded_state.get(1) is True
        widgets = _widgets(app)
        assert isinstance(widgets[0], TurnWidget)
        assert app._highlight == 1  # Enter auto-highlights
        assert widgets[0].selected is True
        assert widgets[1].selected is True
        assert widgets[2].selected is False  # different span


@pytest.mark.asyncio
async def test_jk_clears_highlight():
    """j/k movement clears any active highlight."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation", ConversationView)
        conv.set_selected_index(0)

        await pilot.press("]")
        await pilot.pause()
        assert app._highlight is not None

        await pilot.press("j")
        await pilot.pause()
        assert app._highlight is None


@pytest.mark.asyncio
async def test_escape_clears_highlight():
    """Escape clears highlight."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation", ConversationView)
        conv.set_selected_index(0)

        await pilot.press("]")
        await pilot.pause()
        assert app._highlight is not None

        await pilot.press("escape")
        await pilot.pause()
        assert app._highlight is None


@pytest.mark.asyncio
async def test_drill_out_on_tool_group_highlights_span():
    """] on a ToolGroupWidget highlights the parent span."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()

        # Select the ToolGroupWidget (index 3)
        conv = app.query_one("#conversation", ConversationView)
        conv.set_selected_index(3)
        await pilot.pause()

        await pilot.press("]")
        await pilot.pause()

        # Should highlight span 2 (starts at turn 3)
        assert app._highlight == 3, f"Expected 3, got {app._highlight}"

        # All widgets in span 2 should be selected
        widgets = _widgets(app)
        assert widgets[2].selected is True   # user 3
        assert widgets[3].selected is True   # ToolGroup
        assert widgets[4].selected is True   # assistant 6


@pytest.mark.asyncio
async def test_drill_in_tool_group():
    """[ on a ToolGroupWidget drills into individual tool turns."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation", ConversationView)
        conv.set_selected_index(3)  # ToolGroupWidget
        await pilot.pause()

        await pilot.press("[")
        await pilot.pause()

        # ToolGroup replaced by individual TurnWidgets
        widgets = _widgets(app)
        assert len(widgets) == 6  # was 5, now tool group split into 2 turns
        assert isinstance(widgets[3], TurnWidget)
        assert widgets[3].turn.tool_name == "Read"
        assert isinstance(widgets[4], TurnWidget)
        assert widgets[4].turn.tool_name == "Write"


@pytest.mark.asyncio
async def test_drill_out_tool_turn_undrills():
    """] on a drilled tool turn undrills back to ToolGroup."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation", ConversationView)

        # Drill into tool group
        conv.set_selected_index(3)
        await pilot.press("[")
        await pilot.pause()
        assert len(_widgets(app)) == 6

        # Select a tool turn and press ] to undrill
        conv.set_selected_index(3)  # tool turn Read
        await pilot.press("]")
        await pilot.pause()

        # Back to ToolGroupWidget
        widgets = _widgets(app)
        assert len(widgets) == 5
        assert isinstance(widgets[3], ToolGroupWidget)


@pytest.mark.asyncio
async def test_enter_toggles_tool_group_expand():
    """Enter on a ToolGroupWidget toggles its expanded state."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation", ConversationView)
        conv.set_selected_index(3)  # ToolGroupWidget
        await pilot.pause()

        widgets = _widgets(app)
        assert isinstance(widgets[3], ToolGroupWidget)
        was_expanded = widgets[3].expanded

        await pilot.press("enter")
        await pilot.pause()

        widgets = _widgets(app)
        assert isinstance(widgets[3], ToolGroupWidget)
        assert widgets[3].expanded is not was_expanded


@pytest.mark.asyncio
async def test_full_round_trip():
    """Full flow: drill tools, highlight span, collapse, expand back."""
    session = _make_session()
    app = _make_app(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        conv = app.query_one("#conversation", ConversationView)

        # 1. Start with 5 widgets (2 spans expanded)
        assert len(_widgets(app)) == 5

        # 2. Drill into tool group with [
        conv.set_selected_index(3)  # ToolGroupWidget
        await pilot.press("[")
        await pilot.pause()
        assert len(_widgets(app)) == 6  # 2 individual tool turns

        # 3. Undrill with ]
        conv.set_selected_index(3)
        await pilot.press("]")
        await pilot.pause()
        assert len(_widgets(app)) == 5  # Back to ToolGroupWidget

        # 4. Highlight span 2 from the tool group
        conv.set_selected_index(3)
        await pilot.press("]")
        await pilot.pause()
        assert app._highlight == 3

        # 5. Collapse span 2
        await pilot.press("enter")
        await pilot.pause()
        assert app._highlight is None
        widgets = _widgets(app)
        assert len(widgets) == 3
        assert isinstance(widgets[2], SpanGroupWidget)

        # 6. Expand span 2 back with [ — no auto-highlight
        conv.set_selected_index(2)
        await pilot.press("[")
        await pilot.pause()
        assert len(_widgets(app)) == 5
        assert app._highlight is None

        # 7. Highlight span 1 from user turn
        conv.set_selected_index(0)
        await pilot.press("]")
        await pilot.pause()
        assert app._highlight == 1

        # 8. Cancel with [
        await pilot.press("[")
        await pilot.pause()
        assert app._highlight is None
        assert len(_widgets(app)) == 5
