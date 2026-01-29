"""Rich TUI for Claude Companion."""

import sys
import threading
import time
from datetime import datetime

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .models import Event, Session, Turn
from .store import EventStore


# Styles for different turn types
STYLES = {
    "user": Style(color="blue"),
    "assistant": Style(color="green"),
    "tool": Style(color="cyan", dim=True),
    "tool_write": Style(color="yellow"),
    "tool_edit": Style(color="yellow"),
    "tool_bash": Style(color="magenta"),
    "tool_error": Style(color="red"),
    "stats": Style(dim=True),
    "header": Style(bold=True),
}

# Icons for tools
TOOL_ICONS = {
    "Read": "📄",
    "Write": "✏️ ",
    "Edit": "✏️ ",
    "Bash": "💻",
    "Glob": "🔍",
    "Grep": "🔍",
    "Task": "🤖",
    "WebFetch": "🌐",
    "WebSearch": "🌐",
    "default": "🔧",
}


def get_tool_style(tool_name: str | None) -> Style:
    """Get style for a tool."""
    if not tool_name:
        return STYLES["tool"]
    if tool_name in ("Write", "Edit"):
        return STYLES["tool_write"]
    if tool_name == "Bash":
        return STYLES["tool_bash"]
    return STYLES["tool"]


def get_tool_icon(tool_name: str | None) -> str:
    """Get icon for a tool."""
    if not tool_name:
        return TOOL_ICONS["default"]
    return TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])


class TUI:
    """Rich TUI for displaying Claude Code sessions."""

    def __init__(self, store: EventStore, console: Console | None = None):
        self.store = store
        self.console = console or Console()
        self._running = False
        self._refresh_event = threading.Event()

        # Register listener for store updates
        store.add_listener(self._on_event)

    def _on_event(self, event: Event) -> None:
        """Called when a new event arrives."""
        self._refresh_event.set()

    def _render_header(self, sessions: list[Session], active: Session | None) -> Panel:
        """Render the header with session selector and stats."""
        # Session selector
        session_parts = []
        for i, session in enumerate(sessions[:9], 1):  # Max 9 sessions
            if active and session.session_id == active.session_id:
                session_parts.append(f"[bold reverse] {i} [/] {session.display_name}")
            else:
                session_parts.append(f"[dim][{i}][/dim] {session.display_name}")

        sessions_text = "  ".join(session_parts) if session_parts else "[dim]No sessions[/dim]"

        # Stats
        if active:
            stats = (
                f"[dim]Model:[/dim] {active.model}  "
                f"[dim]Words:[/dim] ↓{active.total_input_words:,} ↑{active.total_output_words:,}  "
                f"[dim]Tools:[/dim] {active.total_tool_calls}"
            )
        else:
            stats = "[dim]Waiting for session...[/dim]"

        content = f"{sessions_text}\n{stats}"
        return Panel(
            content,
            title="[bold]Claude Companion[/bold]",
            title_align="left",
            border_style="blue",
        )

    def _render_turn(self, turn: Turn) -> Panel:
        """Render a single turn."""
        if turn.role == "user":
            title = f"Turn {turn.turn_number} │ User"
            border_style = "blue"
            content = turn.content_preview
        elif turn.role == "assistant":
            title = f"Turn {turn.turn_number} │ Assistant"
            border_style = "green"
            content = turn.content_preview
        else:  # tool
            icon = get_tool_icon(turn.tool_name)
            title = f"Turn {turn.turn_number} │ {icon} {turn.tool_name or 'Tool'}"
            border_style = get_tool_style(turn.tool_name)
            content = turn.content_full  # Show full content for tools

        # Add word count for non-tool turns
        subtitle = None
        if turn.role in ("user", "assistant"):
            subtitle = f"{turn.word_count:,} words"

        return Panel(
            content,
            title=title,
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style=border_style,
            padding=(0, 1),
        )

    def _render_turns(self, session: Session | None) -> Group:
        """Render all turns for a session."""
        if not session or not session.turns:
            return Group(
                Panel(
                    "[dim]Waiting for activity...[/dim]",
                    border_style="dim",
                )
            )

        # Show last N turns to fit screen
        max_turns = 20
        turns = session.turns[-max_turns:]

        panels = [self._render_turn(turn) for turn in turns]
        return Group(*panels)

    def _render_footer(self) -> Text:
        """Render the footer with keybindings."""
        return Text.from_markup(
            " [dim][1-9][/dim] Switch session  "
            "[dim][q][/dim] Quit  "
            "[dim][r][/dim] Refresh"
        )

    def _render(self) -> Group:
        """Render the full TUI."""
        sessions = self.store.get_sessions()
        active = self.store.get_active_session()

        return Group(
            self._render_header(sessions, active),
            self._render_turns(active),
            self._render_footer(),
        )

    def _handle_input(self) -> bool:
        """Handle keyboard input. Returns False to quit."""
        import select
        import sys
        import termios
        import tty

        # Check if input is available (non-blocking)
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                if char == "q":
                    return False
                elif char == "r":
                    self._refresh_event.set()
                elif char.isdigit() and char != "0":
                    self.store.set_active_session(int(char))
                    self._refresh_event.set()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        return True

    def run(self) -> None:
        """Run the TUI."""
        self._running = True

        with Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            screen=True,
        ) as live:
            while self._running:
                # Wait for event or timeout
                self._refresh_event.wait(timeout=0.25)
                self._refresh_event.clear()

                # Handle input
                if not self._handle_input():
                    break

                # Update display
                live.update(self._render())

    def stop(self) -> None:
        """Stop the TUI."""
        self._running = False
