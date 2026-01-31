"""Rich TUI for Claude Companion."""

import sys
import threading
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .alerts import Alert, AlertLevel
from .config import DEFAULT_STYLE
from .export import export_session_markdown, get_export_filename
from .models import Event, Session, Turn
from .store import EventStore
from .styles import STYLES, get_style


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

    def __init__(
        self,
        store: EventStore,
        console: Console | None = None,
        ui_style: str = DEFAULT_STYLE,
    ):
        self.store = store
        self.console = console or Console()
        self._running = False
        self._refresh_event = threading.Event()
        self._selected_index: int | None = None  # Index into filtered turns
        self._scroll_offset = 0  # For scrolling through turns
        self._last_visible_count = 5  # Updated by _render_turns based on actual space
        self._filter_index = 0  # Index into filters
        self._status_message: str | None = None  # Temporary status message
        self._status_until: float = 0  # When to clear status message
        self._show_alerts = True  # Whether to show alerts panel
        self._auto_scroll = True  # Auto-scroll to bottom on new events
        self._use_summary = True  # Use LLM summary vs first few chars for assistant preview
        self._last_active_session_id: str | None = None  # Track session changes
        self._pending_scroll_to_bottom = False  # Flag for deferred scroll computation
        self._show_tasks = False  # Whether to show task list view
        self._show_plan = False  # Whether to show plan view

        # Style renderer
        self._style = get_style(ui_style if ui_style in STYLES else DEFAULT_STYLE)

        # Monitor instruction text
        self._monitor_text = ""  # Monitoring instructions

        # Live display reference
        self._live: Live | None = None

        # Register listener for store updates
        store.add_listener(self._on_event)
        store.add_alert_listener(self._on_alert)
        store.add_turn_listener(self._on_turn)

    def _measure_height(self, renderable) -> int:
        """Measure the height of a renderable in lines."""
        options = self.console.options
        lines = list(self.console.render_lines(renderable, options))
        return len(lines)

    def _get_available_height_for_turns(self) -> int:
        """Calculate available height for turns area."""
        session = self.store.get_active_session()
        sessions = self.store.get_sessions()

        # Measure actual overhead
        header_height = self._measure_height(self._render_header(sessions, session))
        status_height = self._measure_height(self._style.render_status_line(self._monitor_text))
        footer_height = self._measure_height(self._render_footer())

        alerts_panel = self._render_alerts()
        alerts_height = self._measure_height(alerts_panel) if alerts_panel else 0

        # +2 for potential scroll indicators
        overhead = header_height + alerts_height + status_height + footer_height + 2
        return self.console.size.height - overhead

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
        session = self.store.get_active_session()

        # Detect session change and reset scroll state
        current_session_id = session.session_id if session else None
        if current_session_id != self._last_active_session_id:
            self._last_active_session_id = current_session_id
            # Reset scroll state when session changes
            self._scroll_offset = 0
            self._selected_index = None
            # Request scroll to bottom on next render (deferred to avoid cross-thread rendering)
            if session:
                self._pending_scroll_to_bottom = True
        elif self._auto_scroll:
            # Auto-scroll to show new content in current session
            self._pending_scroll_to_bottom = True
        self._refresh_event.set()

    def _on_alert(self, alert: Alert) -> None:
        """Called when a new alert is triggered."""
        self._refresh_event.set()

    def _on_turn(self, turn: Turn) -> None:
        """Called when a new turn arrives from transcript watcher."""
        # Auto-scroll to show new content (deferred to avoid cross-thread rendering)
        if self._auto_scroll:
            self._pending_scroll_to_bottom = True
        self._refresh_event.set()

    def _get_filtered_turns(self, session: Session | None) -> list[Turn]:
        """Get turns filtered by current filter."""
        if not session:
            return []

        filter_key, _ = self._style.filters[self._filter_index]

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

        sessions_text = (
            "  ".join(session_parts) if session_parts else "[dim]No sessions[/dim]"
        )

        _, filter_label = self._style.filters[self._filter_index]

        return self._style.render_header(
            sessions_text, active, filter_label, self._auto_scroll
        )

    def _render_turns(self, session: Session | None) -> Group:
        """Render all turns for a session."""
        filtered_turns = self._get_filtered_turns(session)

        if not filtered_turns:
            filter_key, filter_label = self._style.filters[self._filter_index]
            if filter_key != "all" and session and session.turns:
                msg = f"[dim]No {filter_label} turns. Press \\[f] to change filter.[/dim]"
            else:
                msg = "[dim]Waiting for activity...[/dim]"
            return Group(Panel(msg, border_style="dim"))

        total_turns = len(filtered_turns)
        available_height = self._get_available_height_for_turns()

        # Compute conversation turn numbers (only counting user+assistant)
        conv_turn_map: dict[int, int] = {}
        conv_turn = 0
        for turn in filtered_turns:
            if turn.role in ("user", "assistant"):
                conv_turn += 1
                conv_turn_map[id(turn)] = conv_turn

        # Add turns one by one until we run out of space
        panels = []
        used_height = 0
        start_idx = self._scroll_offset
        end_idx = start_idx

        for i in range(start_idx, total_turns):
            turn = filtered_turns[i]
            is_selected = self._selected_index == i
            panel = self._style.render_turn(
                turn,
                is_selected,
                conv_turn_map.get(id(turn), 0),
                self._use_summary,
            )
            panel_height = self._measure_height(panel)

            if used_height + panel_height > available_height and panels:
                break  # Would overflow, stop here

            panels.append(panel)
            used_height += panel_height
            end_idx = i + 1

        # Store how many turns we actually showed (for scroll calculations)
        self._last_visible_count = end_idx - start_idx

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
            level = "danger" if alert.level == AlertLevel.DANGER else "warning"
            lines.append(self._style.format_alert_line(level, alert.title, alert.message))

        return Panel(
            "\n".join(lines),
            title="[bold red]Alerts[/bold red]",
            title_align="left",
            border_style="red",
            padding=(0, 1),
        )

    def _render_task_list(self, session: Session | None) -> Panel:
        """Render the task list panel."""
        if not session or not session.tasks:
            return Panel(
                "[dim]No tasks yet. Claude will create tasks when working on complex projects.[/dim]",
                title="[bold]Task List[/bold]",
                border_style="cyan",
                padding=(1, 2),
            )

        # Get non-deleted tasks sorted by ID
        tasks = [t for t in session.tasks.values() if not t.is_deleted]
        tasks.sort(key=lambda t: int(t.task_id) if t.task_id.isdigit() else 0)

        lines = []
        for task in tasks:
            # Status indicator
            if task.status == "completed":
                indicator = "[green]✓[/green]"
                status_style = "dim green"
            elif task.status == "in_progress":
                indicator = "[yellow]▶[/yellow]"
                status_style = "yellow"
            else:  # pending
                indicator = "[dim]○[/dim]"
                status_style = "dim"

            # Task subject (truncate if too long)
            subject = task.subject[:60] + "..." if len(task.subject) > 60 else task.subject

            # Blocking info
            blocking_info = ""
            if task.blockedBy:
                blocked_ids = ", ".join(f"#{bid}" for bid in task.blockedBy)
                blocking_info = f" [dim](blocked by {blocked_ids})[/dim]"

            # Format line
            line = f"{indicator} [{task.task_id}] {subject}{blocking_info}"
            if task.status != "pending":
                lines.append(f"[{status_style}]{line}[/{status_style}]")
            else:
                lines.append(f"[dim]{line}[/dim]")

        # Summary footer
        completed = sum(1 for t in tasks if t.status == "completed")
        in_progress = sum(1 for t in tasks if t.status == "in_progress")
        pending = sum(1 for t in tasks if t.status == "pending")

        lines.append(f"\n[dim]Total: {len(tasks)} ({completed} completed, {in_progress} in progress, {pending} pending)[/dim]")

        return Panel(
            "\n".join(lines),
            title="[bold cyan]Task List[/bold cyan]",
            subtitle="[dim]Press T or ESC to return to turns view[/dim]",
            subtitle_align="right",
            border_style="cyan",
            padding=(1, 2),
        )

    def _render_plan(self, session: Session | None) -> Panel:
        """Render the plan panel with markdown."""
        if not session or not session.plan_content:
            return Panel(
                "[dim]No active plan detected. Claude will create a plan when you enter plan mode.[/dim]",
                title="[bold]Plan[/bold]",
                border_style="magenta",
                padding=(1, 2),
            )

        # Format header with file path and update time
        from datetime import datetime

        file_path = session.plan_file_path or "Unknown"
        updated_str = "Unknown"
        if session.plan_updated_at:
            delta = datetime.now() - session.plan_updated_at
            if delta.total_seconds() < 60:
                updated_str = "Just now"
            elif delta.total_seconds() < 3600:
                minutes = int(delta.total_seconds() / 60)
                updated_str = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                hours = int(delta.total_seconds() / 3600)
                updated_str = f"{hours} hour{'s' if hours != 1 else ''} ago"

        header = f"[dim]File:[/dim] {file_path}\n[dim]Updated:[/dim] {updated_str}\n"

        # Render markdown content with Rich Markdown
        content = Markdown(session.plan_content)

        return Panel(
            Group(Text.from_markup(header), Text(""), content),
            title="[bold magenta]Plan[/bold magenta]",
            subtitle="[dim]Press P or ESC to return to turns view[/dim]",
            subtitle_align="right",
            border_style="magenta",
            padding=(1, 2),
        )

    def _render_footer(self) -> Text:
        """Render the footer with keybindings."""
        if self._status_message and time.time() < self._status_until:
            return Text.from_markup(
                f" [bold green]{self._status_message}[/bold green]"
            )

        self._status_message = None

        alerts = self.store.get_alerts()
        danger_count = sum(1 for a in alerts if a.level == AlertLevel.DANGER)
        warn_count = sum(1 for a in alerts if a.level == AlertLevel.WARNING)

        alert_indicator = self._style.get_alert_indicator(danger_count, warn_count)

        return Text.from_markup(
            f"{alert_indicator}"
            "[dim]j/k[/dim]:nav "
            "[dim]g/G[/dim]:end "
            "[dim]o[/dim]:open "
            "[dim]T[/dim]:tasks "
            "[dim]P[/dim]:plan "
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

        # Show plan, tasks, OR turns (mutually exclusive views)
        if self._show_plan:
            parts.append(self._render_plan(active))
        elif self._show_tasks:
            parts.append(self._render_task_list(active))
        else:
            parts.append(self._render_turns(active))

        parts.append(self._style.render_status_line(self._monitor_text))
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
        self.console.print(
            "[dim]Type your text, then press Enter to save (Esc to cancel)[/dim]\n"
        )

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
                        self.console.print(
                            "\r> " + text + " " * 10, end="", highlight=False
                        )
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

        elif key == "\x1b":  # Esc - unselect
            self._selected_index = None
            if self._show_tasks:
                self._show_tasks = False
            if self._show_plan:
                self._show_plan = False
            self._refresh_event.set()

        elif key == "T":
            self._show_tasks = not self._show_tasks
            self._show_plan = False
            self._show_status("Showing task list" if self._show_tasks else "Showing conversation turns")
            self._refresh_event.set()

        elif key == "P":
            self._show_plan = not self._show_plan
            self._show_tasks = False
            self._show_status("Showing plan" if self._show_plan else "Showing conversation turns")
            self._refresh_event.set()

        elif key == "r":
            self._refresh_event.set()

        elif key == "m":  # Edit monitor text
            self._monitor_text = self._edit_text(
                "Monitor Instructions", self._monitor_text
            )
            self._refresh_event.set()

        elif key == "?":  # Ask about trace
            query = self._edit_text("Ask About Trace", "")
            if query:
                self.on_ask_submit(query)
            self._refresh_event.set()

        elif key.isdigit() and key != "0":
            self.store.set_active_session(int(key))
            self._filter_index = 0
            self._auto_scroll = True
            self.scroll_to_bottom()
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
                if self._selected_index >= self._scroll_offset + self._last_visible_count:
                    self._scroll_offset = (
                        self._selected_index - self._last_visible_count + 1
                    )
                self._refresh_event.set()

        elif key == "\n" or key == "\r" or key == " " or key == "o":
            if (
                self._selected_index is not None
                and 0 <= self._selected_index < total_turns
            ):
                turns[self._selected_index].expanded = not turns[
                    self._selected_index
                ].expanded
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
                self._scroll_offset = max(0, total_turns - self._last_visible_count)
                self._auto_scroll = True
                self._refresh_event.set()

        elif key == "g":
            self._auto_scroll = False
            if total_turns > 0:
                self._selected_index = 0
                self._scroll_offset = 0
                self._refresh_event.set()

        elif key == "f":
            self._filter_index = (self._filter_index + 1) % len(self._style.filters)
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

    def scroll_to_bottom(self) -> None:
        """Scroll to show the most recent turns."""
        session = self.store.get_active_session()
        if session:
            filtered = self._get_filtered_turns(session)
            total = len(filtered)
            if total == 0:
                self._scroll_offset = 0
                self._selected_index = None
                return

            # Calculate how many turns fit by measuring from the end
            available_height = self._get_available_height_for_turns()
            conv_turn_map: dict[int, int] = {}
            conv_turn = 0
            for turn in filtered:
                if turn.role in ("user", "assistant"):
                    conv_turn += 1
                    conv_turn_map[id(turn)] = conv_turn

            # Measure turns from the end backwards
            used_height = 0
            visible_count = 0
            for i in range(total - 1, -1, -1):
                turn = filtered[i]
                panel = self._style.render_turn(turn, False, conv_turn_map.get(id(turn), 0), self._use_summary)
                panel_height = self._measure_height(panel)
                if used_height + panel_height > available_height and visible_count > 0:
                    break
                used_height += panel_height
                visible_count += 1

            self._last_visible_count = visible_count
            self._scroll_offset = max(0, total - visible_count)
            self._selected_index = None

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

                    # Handle pending scroll request (from listener threads)
                    if self._pending_scroll_to_bottom:
                        self._pending_scroll_to_bottom = False
                        self.scroll_to_bottom()

                    live.update(self._render())

                self._live = None

    def stop(self) -> None:
        """Stop the TUI."""
        self._running = False
