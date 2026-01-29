"""Rich TUI for Claude Companion."""

import sys
import threading
from datetime import datetime

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
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
    "selected": Style(bold=True),
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
        self._selected_index: int | None = None  # Index into visible turns
        self._scroll_offset = 0  # For scrolling through turns
        self._max_visible_turns = 15

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

    def _render_turn(self, turn: Turn, is_selected: bool = False) -> Panel:
        """Render a single turn."""
        if turn.role == "user":
            title = f"Turn {turn.turn_number} │ User"
            border_style = "blue"
        elif turn.role == "assistant":
            title = f"Turn {turn.turn_number} │ Assistant"
            border_style = "green"
        else:  # tool
            icon = get_tool_icon(turn.tool_name)
            title = f"Turn {turn.turn_number} │ {icon} {turn.tool_name or 'Tool'}"
            border_style = get_tool_style(turn.tool_name)

        # Show full content if expanded, preview otherwise
        if turn.expanded:
            content = turn.content_full
            expand_indicator = "[dim][-][/dim] "
        else:
            content = turn.content_preview
            if turn.content_full != turn.content_preview:
                expand_indicator = "[dim][+][/dim] "
            else:
                expand_indicator = ""

        # Add selection indicator
        if is_selected:
            title = f"► {title}"
            border_style = "white"

        # Add word count for non-tool turns
        subtitle = None
        if turn.role in ("user", "assistant"):
            subtitle = f"{turn.word_count:,} words"

        return Panel(
            f"{expand_indicator}{content}",
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

        # Get visible turns with scrolling
        total_turns = len(session.turns)
        start_idx = self._scroll_offset
        end_idx = min(start_idx + self._max_visible_turns, total_turns)
        visible_turns = session.turns[start_idx:end_idx]

        # Render each turn
        panels = []
        for i, turn in enumerate(visible_turns):
            global_idx = start_idx + i
            is_selected = self._selected_index == global_idx
            panels.append(self._render_turn(turn, is_selected))

        # Add scroll indicators if needed
        if start_idx > 0:
            panels.insert(0, Text(f"  ↑ {start_idx} more above", style="dim"))
        if end_idx < total_turns:
            panels.append(Text(f"  ↓ {total_turns - end_idx} more below", style="dim"))

        return Group(*panels)

    def _render_footer(self) -> Text:
        """Render the footer with keybindings."""
        return Text.from_markup(
            " [dim][1-9][/dim] Session  "
            "[dim][↑↓][/dim] Navigate  "
            "[dim][Enter][/dim] Expand  "
            "[dim][e/c][/dim] Expand/Collapse all  "
            "[dim][q][/dim] Quit"
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

    def _get_active_turns(self) -> list[Turn]:
        """Get turns from active session."""
        session = self.store.get_active_session()
        return session.turns if session else []

    def _handle_input(self) -> bool:
        """Handle keyboard input. Returns False to quit."""
        import select
        import termios
        import tty

        # Check if input is available (non-blocking)
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)

                # Handle escape sequences (arrow keys)
                if char == "\x1b":
                    # Read the rest of the escape sequence
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        char += sys.stdin.read(2)

                return self._process_key(char)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        return True

    def _process_key(self, key: str) -> bool:
        """Process a key press. Returns False to quit."""
        turns = self._get_active_turns()
        total_turns = len(turns)

        if key == "q":
            return False

        elif key == "r":
            self._refresh_event.set()

        elif key.isdigit() and key != "0":
            self.store.set_active_session(int(key))
            self._selected_index = None
            self._scroll_offset = 0
            self._refresh_event.set()

        # Arrow keys
        elif key == "\x1b[A":  # Up arrow
            if total_turns > 0:
                if self._selected_index is None:
                    self._selected_index = total_turns - 1
                elif self._selected_index > 0:
                    self._selected_index -= 1
                # Adjust scroll to keep selection visible
                if self._selected_index < self._scroll_offset:
                    self._scroll_offset = self._selected_index
                self._refresh_event.set()

        elif key == "\x1b[B":  # Down arrow
            if total_turns > 0:
                if self._selected_index is None:
                    self._selected_index = 0
                elif self._selected_index < total_turns - 1:
                    self._selected_index += 1
                # Adjust scroll to keep selection visible
                if self._selected_index >= self._scroll_offset + self._max_visible_turns:
                    self._scroll_offset = self._selected_index - self._max_visible_turns + 1
                self._refresh_event.set()

        elif key == "\n" or key == "\r":  # Enter
            if self._selected_index is not None and 0 <= self._selected_index < total_turns:
                turns[self._selected_index].expanded = not turns[self._selected_index].expanded
                self._refresh_event.set()

        elif key == "e":  # Expand all
            for turn in turns:
                turn.expanded = True
            self._refresh_event.set()

        elif key == "c":  # Collapse all
            for turn in turns:
                turn.expanded = False
            self._refresh_event.set()

        elif key == "G":  # Go to end
            if total_turns > 0:
                self._selected_index = total_turns - 1
                self._scroll_offset = max(0, total_turns - self._max_visible_turns)
                self._refresh_event.set()

        elif key == "g":  # Go to beginning
            if total_turns > 0:
                self._selected_index = 0
                self._scroll_offset = 0
                self._refresh_event.set()

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
