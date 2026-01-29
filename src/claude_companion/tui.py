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
from rich.table import Table

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

# Icons for roles and tools (2 chars wide for alignment)
ROLE_ICONS = {
    "user": "👤",
    "assistant": "🤖",
}

TOOL_ICONS = {
    "Read": "📄",
    "Write": "✏️",
    "Edit": "✏️",
    "Bash": "💻",
    "Glob": "🔍",
    "Grep": "🔍",
    "Task": "⚙️",
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
        self._max_visible_turns = 10
        self._filter_index = 0  # Index into FILTERS
        self._status_message: str | None = None  # Temporary status message
        self._status_until: float = 0  # When to clear status message
        self._show_alerts = True  # Whether to show alerts panel
        self._auto_scroll = True  # Auto-scroll to bottom on new events
        self._use_summary = True  # Use LLM summary vs first few chars for assistant preview

        # Text boxes content
        self._monitor_text = ""  # Monitoring instructions
        self._ask_text = ""  # Ask about trace query

        # Live display reference
        self._live: Live | None = None

        # Register listener for store updates
        store.add_listener(self._on_event)
        store.add_alert_listener(self._on_alert)
        store.add_turn_listener(self._on_turn)

    @property
    def monitor_instruction(self) -> str:
        """Get the current monitoring instruction."""
        return self._monitor_text

    def on_ask_submit(self, query: str) -> None:
        """Called when user submits an ask query. Override or set callback."""
        # Placeholder - will be used for LLM integration later
        self._show_status(f"Query submitted: {query[:30]}...")

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

    def _on_turn(self, turn: Turn) -> None:
        """Called when a new turn arrives from transcript watcher."""
        # Auto-scroll to show new content
        if self._auto_scroll:
            session = self.store.get_active_session()
            if session:
                filtered = self._get_filtered_turns(session)
                total = len(filtered)
                if total > self._max_visible_turns:
                    self._scroll_offset = total - self._max_visible_turns
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
        session_parts = []
        for i, session in enumerate(sessions[:9], 1):
            if active and session.session_id == active.session_id:
                session_parts.append(f"[bold reverse] {i} [/] {session.display_name}")
            else:
                session_parts.append(f"[dim][{i}][/dim] {session.display_name}")

        sessions_text = "  ".join(session_parts) if session_parts else "[dim]No sessions[/dim]"

        if active:
            _, filter_label = FILTERS[self._filter_index]
            scroll_indicator = "" if self._auto_scroll else " [dim](paused)[/dim]"
            historical_count = sum(1 for t in active.turns if t.is_historical)
            live_count = len(active.turns) - historical_count
            turns_info = f"◷{historical_count}+{live_count}" if historical_count else f"{live_count}"
            stats = (
                f"[dim]Model:[/dim] {active.model}  "
                f"[dim]Words:[/dim] ↓{active.total_input_words:,} ↑{active.total_output_words:,}  "
                f"[dim]Turns:[/dim] {turns_info}  "
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
        # Historical indicator for turns loaded from transcript
        history_prefix = "◷ " if turn.is_historical else ""

        if turn.role == "user":
            icon = ROLE_ICONS["user"]
            title = f"{history_prefix}Turn {turn.turn_number} │ {icon} User"
            border_style = "blue" if not turn.is_historical else "dim blue"
        elif turn.role == "assistant":
            icon = ROLE_ICONS["assistant"]
            title = f"{history_prefix}Turn {turn.turn_number} │ {icon} Assistant"
            border_style = "green" if not turn.is_historical else "dim green"
        else:
            icon = get_tool_icon(turn.tool_name)
            title = f"{history_prefix}Turn {turn.turn_number} │ {icon} {turn.tool_name or 'Tool'}"
            base_style = get_tool_style(turn.tool_name)
            border_style = base_style if not turn.is_historical else f"dim {base_style}"

        if turn.expanded:
            content = turn.content_full
            expand_indicator = "[dim]\\[-][/dim] "
        else:
            # For assistant turns, show summary if available and enabled
            if turn.role == "assistant" and self._use_summary:
                if turn.summary:
                    content = turn.summary
                    expand_indicator = "[dim]\\[tldr;][/dim] "
                else:
                    # Summary pending
                    content = f"{turn.content_preview} [dim italic]\\[summarizing...][/dim italic]"
                    expand_indicator = "[dim]\\[+][/dim] " if turn.content_full != turn.content_preview else ""
            else:
                content = turn.content_preview
                if turn.content_full != turn.content_preview:
                    expand_indicator = "[dim]\\[+][/dim] "
                else:
                    expand_indicator = ""

        if is_selected:
            title = f"► {title}"
            border_style = "white"

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

        total_turns = len(filtered_turns)
        start_idx = self._scroll_offset
        end_idx = min(start_idx + self._max_visible_turns, total_turns)
        visible_turns = filtered_turns[start_idx:end_idx]

        panels = []
        for i, turn in enumerate(visible_turns):
            global_idx = start_idx + i
            is_selected = self._selected_index == global_idx
            panels.append(self._render_turn(turn, is_selected))

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

        return Panel(
            "\n".join(lines),
            title="[bold red]Alerts[/bold red]",
            title_align="left",
            border_style="red",
            padding=(0, 1),
        )

    def _render_input_boxes(self) -> Table:
        """Render the two input boxes as a table (different from turn panels)."""
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)

        # Monitor box - use rounded style to differentiate
        monitor_content = self._monitor_text if self._monitor_text else "[dim]Press \\[m] to set monitoring rules[/dim]"
        monitor_box = Panel(
            monitor_content,
            title="[cyan]📋 Monitor[/cyan]",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
            style="on grey7",  # Subtle background
        )

        # Ask box
        ask_content = self._ask_text if self._ask_text else "[dim]Press \\[?] to ask about trace[/dim]"
        ask_box = Panel(
            ask_content,
            title="[magenta]❓ Ask[/magenta]",
            title_align="left",
            border_style="magenta",
            padding=(0, 1),
            style="on grey7",
        )

        table.add_row(monitor_box, ask_box)
        return table

    def _render_footer(self) -> Text:
        """Render the footer with keybindings."""
        if self._status_message and time.time() < self._status_until:
            return Text.from_markup(f" [bold green]{self._status_message}[/bold green]")

        self._status_message = None

        alerts = self.store.get_alerts()
        danger_count = sum(1 for a in alerts if a.level == AlertLevel.DANGER)
        warn_count = sum(1 for a in alerts if a.level == AlertLevel.WARNING)

        alert_indicator = ""
        if danger_count > 0:
            alert_indicator = f"[bold red]🚨{danger_count}[/bold red] "
        elif warn_count > 0:
            alert_indicator = f"[yellow]⚠️{warn_count}[/yellow] "

        return Text.from_markup(
            f"{alert_indicator}"
            "[dim]j/k[/dim]:nav "
            "[dim]o[/dim]:open "
            "[dim]s[/dim]:summary "
            "[dim]f[/dim]:filter "
            "[dim]m[/dim]:monitor "
            "[dim]?[/dim]:ask "
            "[dim]q[/dim]:quit"
        )

    def _render(self) -> Group:
        """Render the full TUI."""
        sessions = self.store.get_sessions()
        active = self.store.get_active_session()

        parts = [self._render_header(sessions, active)]

        alerts_panel = self._render_alerts()
        if alerts_panel:
            parts.append(alerts_panel)

        parts.append(self._render_turns(active))
        parts.append(self._render_input_boxes())
        parts.append(self._render_footer())

        return Group(*parts)

    def _get_active_turns(self) -> list[Turn]:
        """Get filtered turns from active session."""
        session = self.store.get_active_session()
        return self._get_filtered_turns(session)

    def _edit_text(self, title: str, initial: str) -> str:
        """Open a native text input prompt. Returns edited text."""
        import termios
        import tty

        # Stop live display temporarily
        if self._live:
            self._live.stop()

        # Clear screen and show input prompt
        self.console.clear()
        self.console.print(f"\n[bold]{title}[/bold]")
        self.console.print("[dim]Type your text, then press Enter to save (Esc to cancel)[/dim]\n")

        # Show current value
        text = initial
        self.console.print(f"> {text}", end="", highlight=False)

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            while True:
                char = sys.stdin.read(1)

                if char == "\x1b":  # Escape
                    text = initial  # Revert
                    break
                elif char == "\n" or char == "\r":  # Enter
                    break
                elif char == "\x7f" or char == "\x08":  # Backspace
                    if text:
                        text = text[:-1]
                        # Clear line and reprint
                        self.console.print("\r> " + text + " " * 10, end="", highlight=False)
                        self.console.print("\r> " + text, end="", highlight=False)
                elif char.isprintable():
                    text += char
                    self.console.print(char, end="", highlight=False)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # Restart live display
        self.console.clear()
        if self._live:
            self._live.start()

        return text

    def _process_key(self, key: str) -> bool:
        """Process a key press. Returns False to quit."""
        turns = self._get_active_turns()
        total_turns = len(turns)

        if key == "q" or key == "\x03":  # q or Ctrl+C
            return False

        elif key == "r":
            self._refresh_event.set()

        elif key == "m":  # Edit monitor text
            self._monitor_text = self._edit_text("Monitor Instructions", self._monitor_text)
            self._refresh_event.set()

        elif key == "?":  # Edit ask text
            new_text = self._edit_text("Ask About Trace", self._ask_text)
            if new_text and new_text != self._ask_text:
                self._ask_text = new_text
                self.on_ask_submit(new_text)
                self._ask_text = ""  # Clear after submit
            self._refresh_event.set()

        elif key.isdigit() and key != "0":
            self.store.set_active_session(int(key))
            self._selected_index = None
            self._scroll_offset = 0
            self._filter_index = 0
            self._auto_scroll = True
            self._refresh_event.set()

        elif key == "k" or key == "\x1b[A":  # k or Up arrow
            self._auto_scroll = False
            if total_turns > 0:
                if self._selected_index is None:
                    self._selected_index = total_turns - 1
                elif self._selected_index > 0:
                    self._selected_index -= 1
                if self._selected_index < self._scroll_offset:
                    self._scroll_offset = self._selected_index
                self._refresh_event.set()

        elif key == "j" or key == "\x1b[B":  # j or Down arrow
            self._auto_scroll = False
            if total_turns > 0:
                if self._selected_index is None:
                    self._selected_index = 0
                elif self._selected_index < total_turns - 1:
                    self._selected_index += 1
                if self._selected_index >= self._scroll_offset + self._max_visible_turns:
                    self._scroll_offset = self._selected_index - self._max_visible_turns + 1
                self._refresh_event.set()

        elif key == "\n" or key == "\r" or key == " " or key == "o":
            if self._selected_index is not None and 0 <= self._selected_index < total_turns:
                turns[self._selected_index].expanded = not turns[self._selected_index].expanded
                self._refresh_event.set()

        elif key == "e":
            for turn in turns:
                turn.expanded = True
            self._refresh_event.set()

        elif key == "c":
            for turn in turns:
                turn.expanded = False
            self._refresh_event.set()

        elif key == "G":
            if total_turns > 0:
                self._selected_index = total_turns - 1
                self._scroll_offset = max(0, total_turns - self._max_visible_turns)
                self._auto_scroll = True
                self._refresh_event.set()

        elif key == "g":
            self._auto_scroll = False
            if total_turns > 0:
                self._selected_index = 0
                self._scroll_offset = 0
                self._refresh_event.set()

        elif key == "f":
            self._filter_index = (self._filter_index + 1) % len(FILTERS)
            self._selected_index = None
            self._scroll_offset = 0
            self._auto_scroll = True
            self._refresh_event.set()

        elif key == "x":
            session = self.store.get_active_session()
            if session:
                filename = get_export_filename(session, "markdown")
                output_path = Path.cwd() / filename
                export_session_markdown(session, output_path)
                self._show_status(f"Exported to {filename}")
            else:
                self._show_status("No session to export")

        elif key == "a":
            if self.store.get_alerts():
                self.store.clear_alerts()
                self._show_status("Alerts cleared")
            else:
                self._show_alerts = not self._show_alerts
                self._show_status(f"Alerts {'shown' if self._show_alerts else 'hidden'}")
            self._refresh_event.set()

        elif key == "s":
            self._use_summary = not self._use_summary
            mode = "summary" if self._use_summary else "preview"
            self._show_status(f"Preview mode: {mode}")
            self._refresh_event.set()

        return True

    def run(self) -> None:
        """Run the TUI."""
        self._running = True

        with KeyReader() as keys:
            with Live(
                self._render(),
                console=self.console,
                refresh_per_second=4,
                screen=True,
                vertical_overflow="visible",
            ) as live:
                self._live = live
                while self._running:
                    key = keys.read_key()
                    if key:
                        if not self._process_key(key):
                            break

                    self._refresh_event.wait(timeout=0.2)
                    self._refresh_event.clear()

                    live.update(self._render())

                self._live = None

    def stop(self) -> None:
        """Stop the TUI."""
        self._running = False
