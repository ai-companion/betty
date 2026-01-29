"""Rich TUI for Claude Companion."""

import os
import sys
import threading
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from .alerts import Alert, AlertLevel
from .export import export_session_markdown, get_export_filename
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

# Filter options
FILTERS = [
    ("all", "All"),
    ("tool", "Tools only"),
    ("Read", "📄 Read"),
    ("Write", "✏️ Write"),
    ("Edit", "✏️ Edit"),
    ("Bash", "💻 Bash"),
]


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


class KeyReader:
    """Non-blocking keyboard reader for Unix systems."""

    def __init__(self):
        self._old_settings = None
        self._fd = sys.stdin.fileno()

    def __enter__(self):
        import termios
        import tty
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *args):
        import termios
        if self._old_settings:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self) -> str | None:
        """Read a key if available, return None if no input."""
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            # Handle escape sequences (arrow keys)
            if char == "\x1b":
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    char += sys.stdin.read(2)
            return char
        return None


class TUI:
    """Rich TUI for displaying Claude Code sessions."""

    def __init__(self, store: EventStore, console: Console | None = None):
        self.store = store
        self.console = console or Console()
        self._running = False
        self._refresh_event = threading.Event()
        self._selected_index: int | None = None  # Index into filtered turns
        self._scroll_offset = 0  # For scrolling through turns
        self._max_visible_turns = 15
        self._filter_index = 0  # Index into FILTERS
        self._status_message: str | None = None  # Temporary status message
        self._status_until: float = 0  # When to clear status message
        self._show_alerts = True  # Whether to show alerts panel
        self._auto_scroll = True  # Auto-scroll to bottom on new events

        # Register listener for store updates
        store.add_listener(self._on_event)
        store.add_alert_listener(self._on_alert)

    def _on_event(self, event: Event) -> None:
        """Called when a new event arrives."""
        # Auto-scroll to show new content
        if self._auto_scroll:
            session = self.store.get_active_session()
            if session:
                filtered = self._get_filtered_turns(session)
                total = len(filtered)
                if total > self._max_visible_turns:
                    self._scroll_offset = total - self._max_visible_turns
        self._refresh_event.set()

    def _on_alert(self, alert: Alert) -> None:
        """Called when a new alert is triggered."""
        self._refresh_event.set()

    def _get_filtered_turns(self, session: Session | None) -> list[Turn]:
        """Get turns filtered by current filter."""
        if not session:
            return []

        filter_key, _ = FILTERS[self._filter_index]

        if filter_key == "all":
            return session.turns
        elif filter_key == "tool":
            return [t for t in session.turns if t.role == "tool"]
        else:
            return [t for t in session.turns if t.tool_name == filter_key]

    def _show_status(self, message: str, duration: float = 3.0) -> None:
        """Show a temporary status message."""
        self._status_message = message
        self._status_until = time.time() + duration
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

        # Stats and filter
        if active:
            _, filter_label = FILTERS[self._filter_index]
            scroll_indicator = "" if self._auto_scroll else " [dim](paused)[/dim]"
            stats = (
                f"[dim]Model:[/dim] {active.model}  "
                f"[dim]Words:[/dim] ↓{active.total_input_words:,} ↑{active.total_output_words:,}  "
                f"[dim]Tools:[/dim] {active.total_tool_calls}  "
                f"[dim]Filter:[/dim] {filter_label}{scroll_indicator}"
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
        filtered_turns = self._get_filtered_turns(session)

        if not filtered_turns:
            filter_key, filter_label = FILTERS[self._filter_index]
            if filter_key != "all" and session and session.turns:
                msg = f"[dim]No {filter_label} turns. Press [f] to change filter.[/dim]"
            else:
                msg = "[dim]Waiting for activity...[/dim]"
            return Group(Panel(msg, border_style="dim"))

        # Get visible turns with scrolling
        total_turns = len(filtered_turns)
        start_idx = self._scroll_offset
        end_idx = min(start_idx + self._max_visible_turns, total_turns)
        visible_turns = filtered_turns[start_idx:end_idx]

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

    def _render_alerts(self) -> Panel | None:
        """Render alerts panel if there are any warnings/dangers."""
        if not self._show_alerts:
            return None

        alerts = self.store.get_recent_alerts(3)
        # Only show warnings and dangers
        alerts = [a for a in alerts if a.level in (AlertLevel.WARNING, AlertLevel.DANGER)]

        if not alerts:
            return None

        lines = []
        for alert in alerts:
            if alert.level == AlertLevel.DANGER:
                icon = "🚨"
                style = "bold red"
            else:
                icon = "⚠️ "
                style = "yellow"

            lines.append(f"[{style}]{icon} {alert.title}[/{style}]: {alert.message}")

        content = "\n".join(lines)
        return Panel(
            content,
            title="[bold red]Alerts[/bold red]",
            title_align="left",
            border_style="red",
            padding=(0, 1),
        )

    def _render_footer(self) -> Text:
        """Render the footer with keybindings."""
        # Check if we should show status message
        if self._status_message and time.time() < self._status_until:
            return Text.from_markup(f" [bold green]{self._status_message}[/bold green]")

        self._status_message = None

        # Show alert count if any
        alerts = self.store.get_alerts()
        danger_count = sum(1 for a in alerts if a.level == AlertLevel.DANGER)
        warn_count = sum(1 for a in alerts if a.level == AlertLevel.WARNING)

        alert_indicator = ""
        if danger_count > 0:
            alert_indicator = f" [bold red]🚨 {danger_count}[/bold red]"
        elif warn_count > 0:
            alert_indicator = f" [yellow]⚠️  {warn_count}[/yellow]"

        return Text.from_markup(
            f"{alert_indicator} "
            "[dim][1-9][/dim] Session  "
            "[dim][jk][/dim] Navigate  "
            "[dim][f][/dim] Filter  "
            "[dim][x][/dim] Export  "
            "[dim][a][/dim] Alerts  "
            "[dim][q][/dim] Quit"
        )

    def _render(self) -> Group:
        """Render the full TUI."""
        sessions = self.store.get_sessions()
        active = self.store.get_active_session()

        parts = [
            self._render_header(sessions, active),
        ]

        # Add alerts panel if there are any
        alerts_panel = self._render_alerts()
        if alerts_panel:
            parts.append(alerts_panel)

        parts.append(self._render_turns(active))
        parts.append(self._render_footer())

        return Group(*parts)

    def _get_active_turns(self) -> list[Turn]:
        """Get filtered turns from active session."""
        session = self.store.get_active_session()
        return self._get_filtered_turns(session)

    def _process_key(self, key: str) -> bool:
        """Process a key press. Returns False to quit."""
        turns = self._get_active_turns()
        total_turns = len(turns)

        if key == "q" or key == "\x03":  # q or Ctrl+C
            return False

        elif key == "r":
            self._refresh_event.set()

        elif key.isdigit() and key != "0":
            self.store.set_active_session(int(key))
            self._selected_index = None
            self._scroll_offset = 0
            self._filter_index = 0  # Reset filter on session change
            self._auto_scroll = True
            self._refresh_event.set()

        # Vim keys for navigation
        elif key == "k" or key == "\x1b[A":  # k or Up arrow
            self._auto_scroll = False  # User is navigating, disable auto-scroll
            if total_turns > 0:
                if self._selected_index is None:
                    self._selected_index = total_turns - 1
                elif self._selected_index > 0:
                    self._selected_index -= 1
                # Adjust scroll to keep selection visible
                if self._selected_index < self._scroll_offset:
                    self._scroll_offset = self._selected_index
                self._refresh_event.set()

        elif key == "j" or key == "\x1b[B":  # j or Down arrow
            self._auto_scroll = False  # User is navigating, disable auto-scroll
            if total_turns > 0:
                if self._selected_index is None:
                    self._selected_index = 0
                elif self._selected_index < total_turns - 1:
                    self._selected_index += 1
                # Adjust scroll to keep selection visible
                if self._selected_index >= self._scroll_offset + self._max_visible_turns:
                    self._scroll_offset = self._selected_index - self._max_visible_turns + 1
                self._refresh_event.set()

        elif key == "\n" or key == "\r" or key == " " or key == "o":  # Enter, Space, or o
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

        elif key == "G":  # Go to end and re-enable auto-scroll
            if total_turns > 0:
                self._selected_index = total_turns - 1
                self._scroll_offset = max(0, total_turns - self._max_visible_turns)
                self._auto_scroll = True
                self._refresh_event.set()

        elif key == "g":  # Go to beginning
            self._auto_scroll = False
            if total_turns > 0:
                self._selected_index = 0
                self._scroll_offset = 0
                self._refresh_event.set()

        elif key == "f":  # Cycle filter
            self._filter_index = (self._filter_index + 1) % len(FILTERS)
            self._selected_index = None
            self._scroll_offset = 0
            self._auto_scroll = True
            self._refresh_event.set()

        elif key == "x":  # Export session
            session = self.store.get_active_session()
            if session:
                filename = get_export_filename(session, "markdown")
                output_path = Path.cwd() / filename
                export_session_markdown(session, output_path)
                self._show_status(f"Exported to {filename}")
            else:
                self._show_status("No session to export")

        elif key == "a":  # Toggle/clear alerts
            if self.store.get_alerts():
                self.store.clear_alerts()
                self._show_status("Alerts cleared")
            else:
                self._show_alerts = not self._show_alerts
                self._show_status(f"Alerts {'shown' if self._show_alerts else 'hidden'}")
            self._refresh_event.set()

        return True

    def run(self) -> None:
        """Run the TUI."""
        self._running = True

        with KeyReader() as keys:
            with Live(
                self._render(),
                console=self.console,
                refresh_per_second=2,
                screen=True,
                vertical_overflow="visible",
            ) as live:
                while self._running:
                    # Check for keyboard input
                    key = keys.read_key()
                    if key:
                        if not self._process_key(key):
                            break

                    # Wait for event or timeout
                    self._refresh_event.wait(timeout=0.2)
                    self._refresh_event.clear()

                    # Update display
                    live.update(self._render())

    def stop(self) -> None:
        """Stop the TUI."""
        self._running = False
