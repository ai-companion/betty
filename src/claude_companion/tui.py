"""Rich TUI for Claude Companion."""

import sys
import threading
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .alerts import Alert, AlertLevel
from .export import export_session_markdown, get_export_filename
from .models import Event, Session, Turn
from .store import EventStore


# Muted color palette - matches Claude Code aesthetic
STYLES = {
    "user": Style(color="white"),
    "assistant": Style(color="cyan"),
    "tool": Style(dim=True),
    "tool_write": Style(color="yellow", dim=True),
    "tool_edit": Style(color="yellow", dim=True),
    "tool_bash": Style(dim=True),
    "tool_error": Style(color="red"),
    "stats": Style(dim=True),
    "header": Style(bold=True),
    "selected": Style(bold=True, reverse=True),
}

# Left margin indicators (matching Claude Code)
BULLET = "⏺"
BULLET_STYLES = {
    "assistant": "white",
    "tool": "green",
}

TOOL_INDICATORS = {
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "Bash": "$",
    "Glob": "glob",
    "Grep": "grep",
    "Task": "task",
    "WebFetch": "fetch",
    "WebSearch": "search",
    "default": "tool",
}

# Filter options (no emojis)
FILTERS = [
    ("all", "All"),
    ("tool", "Tools"),
    ("Read", "Read"),
    ("Write", "Write"),
    ("Edit", "Edit"),
    ("Bash", "Bash"),
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


def get_tool_indicator(tool_name: str | None) -> str:
    """Get text indicator for a tool."""
    if not tool_name:
        return TOOL_INDICATORS["default"]
    return TOOL_INDICATORS.get(tool_name, TOOL_INDICATORS["default"])


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

        # Monitor instruction text
        self._monitor_text = ""  # Monitoring instructions

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
            turns_info = f"h{historical_count}+{live_count}" if historical_count else f"{live_count}"
            stats = (
                f"{active.model} | "
                f"in:{active.total_input_words:,} out:{active.total_output_words:,} | "
                f"turns:{turns_info} | "
                f"{filter_label}{scroll_indicator}"
            )
        else:
            stats = "[dim]Waiting for session...[/dim]"

        content = f"{sessions_text}\n[dim]{stats}[/dim]"
        return Panel(
            content,
            title="[bold]Claude Companion[/bold]",
            title_align="left",
            border_style="dim",
        )

    def _render_turn(self, turn: Turn, is_selected: bool = False, conv_turn: int = 0) -> Text | Group:
        """Render a single turn matching Claude Code style."""
        # User turns: "> message" with dim background
        if turn.role == "user":
            content = turn.content_full if turn.expanded else turn.content_preview
            text = Text()
            if is_selected:
                text.append("❯ ", style="bold white on grey23")
                text.append(content, style="on grey23")
            else:
                text.append("❯ ", style="dim on grey15")
                text.append(content, style="on grey15")
            return Group(text, Text(""))

        # Assistant turns: render as Markdown with colored bullet
        if turn.role == "assistant":
            bullet_style = BULLET_STYLES["assistant"]
            if turn.expanded:
                indicator = "[-]"
                content = turn.content_full
            elif self._use_summary and turn.summary:
                indicator = "[tldr]"
                content = turn.summary
            elif self._use_summary and not turn.summary:
                indicator = "[+]"
                content = f"{turn.content_preview} *summarizing...*"
            elif turn.content_full != turn.content_preview:
                indicator = "[+]"
                content = turn.content_preview
            else:
                indicator = ""
                content = turn.content_preview

            # Build markdown with indicator
            if indicator:
                md_content = f"**{indicator}** {content}"
            else:
                md_content = content

            # Use table grid to combine styled bullet with markdown
            row = Table.grid(padding=(0, 1))
            row.add_column(width=1)
            row.add_column()
            row.add_row(Text(BULLET, style=bullet_style), Markdown(md_content))

            parts = [row]
            if conv_turn > 0:
                status = f"── turn {conv_turn}, {turn.word_count} words ──"
                parts.append(Text(status, style="dim", justify="right"))
            return Group(*parts)

        # Tool turns: render as Markdown with colored bullet
        bullet_style = BULLET_STYLES["tool"]
        tool_indicator = get_tool_indicator(turn.tool_name)
        indicator = f"[{tool_indicator}]"
        content = turn.content_full if turn.expanded else turn.content_preview
        md_content = f"**{indicator}** {content}"

        row = Table.grid(padding=(0, 1))
        row.add_column(width=1)
        row.add_column()
        row.add_row(Text(BULLET, style=bullet_style), Markdown(md_content))

        return Group(row, Text(""))

    def _render_turns(self, session: Session | None) -> Group:
        """Render all turns for a session."""
        filtered_turns = self._get_filtered_turns(session)

        if not filtered_turns:
            filter_key, filter_label = FILTERS[self._filter_index]
            if filter_key != "all" and session and session.turns:
                msg = f"[dim]No {filter_label} turns. Press \\[f] to change filter.[/dim]"
            else:
                msg = "[dim]Waiting for activity...[/dim]"
            return Group(Panel(msg, border_style="dim"))

        total_turns = len(filtered_turns)
        start_idx = self._scroll_offset
        end_idx = min(start_idx + self._max_visible_turns, total_turns)
        visible_turns = filtered_turns[start_idx:end_idx]

        # Compute conversation turn numbers (only counting user+assistant)
        conv_turn_map: dict[int, int] = {}
        conv_turn = 0
        for turn in filtered_turns:
            if turn.role in ("user", "assistant"):
                conv_turn += 1
                conv_turn_map[id(turn)] = conv_turn

        panels = []
        for i, turn in enumerate(visible_turns):
            global_idx = start_idx + i
            is_selected = self._selected_index == global_idx
            panels.append(self._render_turn(turn, is_selected, conv_turn_map.get(id(turn), 0)))

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
                indicator = "[!]"
                style = "bold red"
            else:
                indicator = "[*]"
                style = "yellow"
            lines.append(f"[{style}]{indicator} {alert.title}[/{style}]: {alert.message}")

        return Panel(
            "\n".join(lines),
            title="[bold red]Alerts[/bold red]",
            title_align="left",
            border_style="red",
            padding=(0, 1),
        )

    def _render_status_line(self) -> Panel:
        """Render the status line showing monitor instruction if set."""
        if self._monitor_text:
            content = self._monitor_text
        else:
            content = "[dim]Press \\[m] to set[/dim]"
        return Panel(
            content,
            title="[bold]Monitor[/bold]",
            title_align="left",
            border_style="dim",
        )

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
            alert_indicator = f"[bold red][!{danger_count}][/bold red] "
        elif warn_count > 0:
            alert_indicator = f"[yellow][*{warn_count}][/yellow] "

        return Text.from_markup(
            f"{alert_indicator}"
            "[dim]j/k[/dim]:nav "
            "[dim]g/G[/dim]:end "
            "[dim]o[/dim]:open "
            "[dim]s/S[/dim]:summary "
            "[dim]f[/dim]:filter "
            "[dim]m[/dim]:monitor "
            "[dim]?[/dim]:ask "
            "[dim]x[/dim]:export "
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
        parts.append(self._render_status_line())
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

        elif key == "?":  # Ask about trace
            query = self._edit_text("Ask About Trace", "")
            if query:
                self.on_ask_submit(query)
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

        elif key == "S":  # Summarize all historical turns without summaries
            count = self.store.summarize_historical_turns()
            if count > 0:
                self._show_status(f"Summarizing {count} historical turns...")
            else:
                self._show_status("No historical turns need summarization")
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
