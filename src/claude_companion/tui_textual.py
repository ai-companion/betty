"""Textual TUI for Claude Companion."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Group as RichGroup
from rich.markdown import Markdown
from rich.table import Table
from rich.markup import escape as markup_escape
from rich.text import Text as RichText
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static, Footer, Input, Label

from .alerts import Alert, AlertLevel
from .export import export_session_markdown, get_export_filename
from .models import Session, Turn, ToolGroup, compute_spans
from .pricing import get_pricing, estimate_cost

if TYPE_CHECKING:
    from .store import EventStore


def group_turns_for_display(
    session: Session,
    turns: list[Turn],
    use_summary: bool,
    group_expanded_state: dict[int, bool] | None = None,
) -> list[Turn | ToolGroup]:
    """
    Group consecutive tool turns into ToolGroup, keep assistant as separate Turn.

    Tools are grouped together with their own summary (green bullet).
    Assistant turns are displayed separately with their own summary (white bullet).
    """
    if not use_summary:
        return turns

    if group_expanded_state is None:
        group_expanded_state = {}

    result: list[Turn | ToolGroup] = []
    tool_buffer: list[Turn] = []

    def flush_tool_buffer() -> None:
        """Flush accumulated tool turns as a ToolGroup."""
        nonlocal tool_buffer
        if tool_buffer:
            first_turn_num = tool_buffer[0].turn_number
            expanded = group_expanded_state.get(first_turn_num, False)
            summary = tool_buffer[0].summary
            result.append(ToolGroup(
                tool_turns=tool_buffer,
                summary=summary,
                expanded=expanded,
            ))
            tool_buffer = []

    for turn in turns:
        if turn.role == "tool":
            tool_buffer.append(turn)
        else:
            flush_tool_buffer()
            result.append(turn)

    flush_tool_buffer()
    return result


# CSS Styles
CSS = """
Screen {
    layout: vertical;
}

#header {
    height: auto;
    border: solid $primary;
    padding: 0 1;
    margin-bottom: 1;
}

#conversation {
    height: 1fr;
    overflow-y: auto;
    scrollbar-gutter: stable;
}

#tasks-view, #plan-view {
    height: 1fr;
    display: none;
    overflow-y: auto;
    border: solid $primary;
    padding: 1 2;
}

#tasks-view.visible, #plan-view.visible {
    display: block;
}

#conversation.hidden {
    display: none;
}

#alerts {
    height: auto;
    max-height: 5;
    display: none;
    border: solid $error;
    padding: 0 1;
    margin-top: 1;
}

#alerts.has-alerts {
    display: block;
}

#inputs {
    height: auto;
    margin-top: 1;
}

#inputs-row {
    height: auto;
}

.input-panel {
    width: 1fr;
    height: auto;
    border: solid $secondary;
    padding: 0 1;
    margin: 0 1;
}

.input-panel:focus-within {
    border: solid $accent;
}

.input-label {
    color: $text-muted;
    padding: 0;
}

.input-field {
    border: none;
    padding: 0;
    height: 1;
    background: transparent;
}

.input-field:focus {
    border: none;
}

#footer {
    dock: bottom;
    height: 1;
}

.turn-user {
    margin-bottom: 1;
}

.turn-assistant {
    margin-bottom: 1;
}

.turn-tool {
    margin-bottom: 1;
}

.turn-group {
    margin-bottom: 1;
}

/* Rich style - boxes around turns with reduced spacing */
.rich-style {
    margin-bottom: 0;
}

.rich-style.turn-user {
    background: $surface;
    border: solid $primary;
}

.rich-style.turn-assistant {
    border: solid $success;
}

.rich-style.turn-tool {
    border: solid $warning;
}

.rich-style.turn-group {
    border: solid $success;
}

/* Selection highlighting - only change background, no border to avoid layout shift */
TurnWidget.selected, ToolGroupWidget.selected {
    background: $primary 30%;
}

/* Span highlight for range analysis */
TurnWidget.span-highlight, ToolGroupWidget.span-highlight {
    background: $success 15%;
}

.status-line {
    height: 1;
    padding: 0 1;
    background: $surface;
}
"""


class TurnWidget(Static):
    """Widget for displaying a conversation turn."""

    DEFAULT_CSS = """
    TurnWidget {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    # Icons for rich style
    ROLE_ICONS = {"user": "👤", "assistant": "🤖"}
    TOOL_ICONS = {
        "Read": "📄", "Write": "✏️", "Edit": "✏️", "Bash": "💻",
        "Glob": "🔍", "Grep": "🔍", "Task": "⚙️",
        "WebFetch": "🌐", "WebSearch": "🌐", "default": "🔧",
    }
    # Indicators for claude-code style
    TOOL_INDICATORS = {
        "Read": "read", "Write": "write", "Edit": "edit", "Bash": "$",
        "Glob": "glob", "Grep": "grep", "Task": "task",
        "WebFetch": "fetch", "WebSearch": "search", "default": "tool",
    }
    BULLET = "⏺"

    expanded: reactive[bool] = reactive(False)
    selected: reactive[bool] = reactive(False)
    span_highlight: reactive[bool] = reactive(False)

    def __init__(
        self,
        turn: Turn,
        conv_turn: int = 0,
        use_summary: bool = True,
        ui_style: str = "rich",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.turn = turn
        self.conv_turn = conv_turn
        self.use_summary = use_summary
        self.ui_style = ui_style
        self.expanded = turn.expanded

    def compose(self) -> ComposeResult:
        yield Static(id="turn-content")

    def on_mount(self) -> None:
        self._update_content()
        self._update_classes()

    def watch_expanded(self, value: bool) -> None:
        self.turn.expanded = value
        self._update_content()

    def watch_selected(self, value: bool) -> None:
        self._update_classes()
        self._update_content()  # Re-render to update bullet color

    def watch_span_highlight(self, value: bool) -> None:
        self._update_classes()

    def _update_classes(self) -> None:
        self.remove_class("selected", "span-highlight", "turn-user", "turn-assistant", "turn-tool", "rich-style")
        if self.selected:
            self.add_class("selected")
        elif self.span_highlight:
            self.add_class("span-highlight")
        if self.ui_style == "rich":
            self.add_class("rich-style")
        if self.turn.role == "user":
            self.add_class("turn-user")
        elif self.turn.role == "assistant":
            self.add_class("turn-assistant")
        else:
            self.add_class("turn-tool")

    def _update_content(self) -> None:
        """Update the rendered content based on current state and style."""
        if self.ui_style == "rich":
            renderable = self._render_rich_style()
        else:
            renderable = self._render_claude_code_style()

        try:
            content_widget = self.query_one("#turn-content", Static)
            content_widget.update(renderable)
        except NoMatches:
            pass

    def _render_rich_style(self):
        """Render turn in rich style with emojis and boxes."""
        turn = self.turn
        history_prefix = "◷ " if turn.is_historical else ""
        timestamp_str = turn.timestamp.strftime("%H:%M:%S")
        select_prefix = "► " if self.selected else ""

        if turn.role == "user":
            icon = self.ROLE_ICONS["user"]
            content = turn.content_full if self.expanded else turn.content_preview
            header = f"{select_prefix}{history_prefix}Turn {turn.turn_number} │ {icon} User │ {timestamp_str}"
            parts = [
                RichText.from_markup(f"[bold blue]{header}[/bold blue]"),
                RichText(content),
            ]
            # Add annotation if available
            if turn.annotation:
                parts.append(RichText.from_markup(f"[yellow]📝 {turn.annotation}[/yellow]"))
            return RichGroup(*parts)
        elif turn.role == "assistant":
            icon = self.ROLE_ICONS["assistant"]
            if self.expanded:
                indicator = "[-]"
                content = turn.content_full
            elif self.use_summary and turn.summary:
                indicator = "[+]"
                content = turn.summary
            elif self.use_summary and not turn.summary:
                indicator = "[+]"
                content = f"{turn.content_preview} *\\[summarizing...]*"
            elif turn.content_full != turn.content_preview:
                indicator = "[+]"
                content = turn.content_preview
            else:
                indicator = ""
                content = turn.content_preview

            header = f"{select_prefix}{history_prefix}Turn {turn.turn_number} │ {icon} Assistant │ {timestamp_str}"
            word_info = f"{turn.word_count:,} words"

            # Build parts: header, content, optionally critic
            parts = [
                RichText.from_markup(f"[bold green]{header}[/bold green] [dim]{word_info}[/dim]"),
                Markdown(f"{indicator} {content}" if indicator else content),
            ]

            # Add critic if available and not expanded
            if turn.critic and not self.expanded:
                critic_color = "#5fd787" if turn.critic_sentiment == "progress" else (
                    "#ffaf00" if turn.critic_sentiment == "concern" else "#ff5f5f"
                )
                parts.append(RichText.from_markup(f"[{critic_color}]{turn.critic}[/{critic_color}]"))

            # Add annotation if available
            if turn.annotation:
                parts.append(RichText.from_markup(f"[yellow]📝 {turn.annotation}[/yellow]"))

            return RichGroup(*parts)
        else:
            # Tool turn
            tool_name = turn.tool_name or "Tool"
            icon = self.TOOL_ICONS.get(tool_name, self.TOOL_ICONS["default"])
            content = turn.content_full if self.expanded else turn.content_preview
            header = f"{select_prefix}{history_prefix}Turn {turn.turn_number} │ {icon} {tool_name} │ {timestamp_str}"
            parts = [
                RichText.from_markup(f"[bold yellow]{header}[/bold yellow]"),
                RichText(content, style="dim"),  # Use plain Text to preserve verbatim output
            ]
            # Add annotation if available
            if turn.annotation:
                parts.append(RichText.from_markup(f"[yellow]📝 {turn.annotation}[/yellow]"))
            return RichGroup(*parts)

    def _render_claude_code_style(self):
        """Render turn in minimal claude-code style with bullets in first column."""
        turn = self.turn
        timestamp_str = turn.timestamp.strftime("%H:%M")
        selected_style = "light_steel_blue" if self.selected else ""

        if turn.role == "user":
            content = turn.content_full if self.expanded else turn.content_preview
            content_style = selected_style or "on grey15"
            row = Table.grid(padding=(0, 0))
            row.add_column(width=2)
            row.add_column()
            row.add_row(
                RichText("❯ ", style=selected_style or "dim"),
                RichText(content, style=content_style),
            )
            parts = [row]
            # Add annotation if available
            if turn.annotation:
                annotation_row = Table.grid(padding=(0, 0))
                annotation_row.add_column(width=2)
                annotation_row.add_column()
                annotation_row.add_row(
                    RichText("  ", style="dim"),
                    RichText.from_markup(f"[yellow]📝 {turn.annotation}[/yellow]"),
                )
                parts.append(annotation_row)
            return RichGroup(*parts) if len(parts) > 1 else parts[0]
        elif turn.role == "assistant":
            bullet_style = selected_style or "white"
            if self.expanded:
                indicator = "[-]"
                content = turn.content_full
            elif self.use_summary and turn.summary:
                indicator = "[+]"
                content = turn.summary
            elif self.use_summary and not turn.summary:
                indicator = "[+]"
                content = f"{turn.content_preview} *summarizing...*"
            elif turn.content_full != turn.content_preview:
                indicator = "[+]"
                content = turn.content_preview
            else:
                indicator = ""
                content = turn.content_preview

            md_content = f"**{indicator}** {content}" if indicator else content
            status = f"── turn {self.conv_turn}, {turn.word_count} words, {timestamp_str} ──" if self.conv_turn > 0 else ""

            row = Table.grid(padding=(0, 0))
            row.add_column(width=2)
            row.add_column()
            row.add_row(
                RichText(f"{self.BULLET} ", style=bullet_style),
                Markdown(md_content),
            )

            # Build parts: row, status, optionally critic
            parts = [row]
            if status:
                parts.append(RichText(status, style="dim"))

            # Add critic if available and not expanded
            if turn.critic and not self.expanded:
                critic_color = "#5fd787" if turn.critic_sentiment == "progress" else (
                    "#ffaf00" if turn.critic_sentiment == "concern" else "#ff5f5f"
                )
                # Indent critic slightly
                critic_row = Table.grid(padding=(0, 0))
                critic_row.add_column(width=2)
                critic_row.add_column()
                critic_row.add_row(
                    RichText("  ", style="dim"),
                    RichText.from_markup(f"[{critic_color}]{turn.critic}[/{critic_color}]"),
                )
                parts.append(critic_row)

            # Add annotation if available
            if turn.annotation:
                annotation_row = Table.grid(padding=(0, 0))
                annotation_row.add_column(width=2)
                annotation_row.add_column()
                annotation_row.add_row(
                    RichText("  ", style="dim"),
                    RichText.from_markup(f"[yellow]📝 {turn.annotation}[/yellow]"),
                )
                parts.append(annotation_row)

            return RichGroup(*parts) if len(parts) > 1 else parts[0]
        else:
            # Tool turn - use green bullet
            tool_name = turn.tool_name or "Tool"
            indicator = self.TOOL_INDICATORS.get(tool_name, self.TOOL_INDICATORS["default"])
            bullet_style = selected_style or "#5fd787"
            content = turn.content_full if self.expanded else turn.content_preview

            row = Table.grid(padding=(0, 0))
            row.add_column(width=2)
            row.add_column()
            tool_text = RichText()
            tool_text.append(f"[{indicator}] ", style="bold" if not selected_style else selected_style)
            tool_text.append(content, style=selected_style or "dim")
            row.add_row(
                RichText(f"{self.BULLET} ", style=bullet_style),
                tool_text,
            )

            parts = [row]
            # Add annotation if available
            if turn.annotation:
                annotation_row = Table.grid(padding=(0, 0))
                annotation_row.add_column(width=2)
                annotation_row.add_column()
                annotation_row.add_row(
                    RichText("  ", style="dim"),
                    RichText.from_markup(f"[yellow]📝 {turn.annotation}[/yellow]"),
                )
                parts.append(annotation_row)

            return RichGroup(*parts) if len(parts) > 1 else parts[0]


class ToolGroupWidget(Static):
    """Widget for displaying a group of tool turns."""

    DEFAULT_CSS = """
    ToolGroupWidget {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    BULLET = "⏺"
    TOOL_ICONS = TurnWidget.TOOL_ICONS
    TOOL_INDICATORS = TurnWidget.TOOL_INDICATORS

    expanded: reactive[bool] = reactive(False)
    selected: reactive[bool] = reactive(False)
    span_highlight: reactive[bool] = reactive(False)

    def __init__(self, group: ToolGroup, ui_style: str = "rich", **kwargs) -> None:
        super().__init__(**kwargs)
        self.group = group
        self.ui_style = ui_style
        self.expanded = group.expanded

    def compose(self) -> ComposeResult:
        yield Static(id="group-content")

    def on_mount(self) -> None:
        self._update_content()
        self._update_classes()

    def watch_expanded(self, value: bool) -> None:
        self.group.expanded = value
        self._update_content()

    def watch_selected(self, value: bool) -> None:
        self._update_classes()
        self._update_content()  # Re-render to update bullet color

    def watch_span_highlight(self, value: bool) -> None:
        self._update_classes()

    def _update_classes(self) -> None:
        self.remove_class("selected", "span-highlight", "rich-style", "turn-group")
        if self.selected:
            self.add_class("selected")
        elif self.span_highlight:
            self.add_class("span-highlight")
        if self.ui_style == "rich":
            self.add_class("rich-style")
            self.add_class("turn-group")

    def _update_content(self) -> None:
        """Update the rendered content based on current state and style."""
        if self.ui_style == "rich":
            renderable = self._render_rich_style()
        else:
            renderable = self._render_claude_code_style()

        try:
            content_widget = self.query_one("#group-content", Static)
            content_widget.update(renderable)
        except NoMatches:
            pass

    def _render_rich_style(self):
        """Render tool group in rich style with emojis."""
        group = self.group
        first_tool = group.tool_turns[0] if group.tool_turns else None
        history_prefix = "◷ " if (first_tool and first_tool.is_historical) else ""
        timestamp_str = first_tool.timestamp.strftime("%H:%M:%S") if first_tool else ""
        select_prefix = "► " if self.selected else ""

        if not self.expanded and group.summary:
            # Collapsed with summary
            header = f"{select_prefix}{history_prefix}{group.tool_count} tools │ {timestamp_str}"
            md_content = Markdown(f"**tools:{group.tool_count}** {group.summary}")
            parts = [RichText.from_markup(f"[bold #5fd787]{header}[/bold #5fd787]"), md_content]
            # Add annotation if available (stored on first tool)
            if first_tool and first_tool.annotation:
                parts.append(RichText.from_markup(f"[yellow]📝 {first_tool.annotation}[/yellow]"))
            return RichGroup(*parts)
        else:
            # Expanded or no summary
            header = f"{select_prefix}{history_prefix}{group.tool_count} tools │ {timestamp_str}"
            parts = [RichText.from_markup(f"[bold #5fd787]{header}[/bold #5fd787]")]
            for tool_turn in group.tool_turns:
                tool_name = tool_turn.tool_name or "Tool"
                icon = self.TOOL_ICONS.get(tool_name, self.TOOL_ICONS["default"])
                content = tool_turn.content_full if tool_turn.expanded else tool_turn.content_preview
                # Build text without from_markup for content to preserve verbatim output
                tool_text = RichText(f"  {icon} ")
                tool_text.append(f"[{tool_name}]", style="bold yellow")
                tool_text.append(" ")
                tool_text.append(content, style="dim")
                parts.append(tool_text)
            # Add annotation if available (stored on first tool)
            if first_tool and first_tool.annotation:
                parts.append(RichText.from_markup(f"[yellow]📝 {first_tool.annotation}[/yellow]"))
            return RichGroup(*parts)

    def _render_claude_code_style(self):
        """Render tool group in minimal claude-code style with two-column layout."""
        group = self.group
        first_tool = group.tool_turns[0] if group.tool_turns else None
        selected_style = "light_steel_blue" if self.selected else ""
        bullet_style = selected_style or "#5fd787"

        if not self.expanded and group.summary:
            # Collapsed with summary - two column layout
            row = Table.grid(padding=(0, 0))
            row.add_column(width=2)
            row.add_column()
            row.add_row(
                RichText(f"{self.BULLET} ", style=bullet_style),
                Markdown(f"**tools:{group.tool_count}** {group.summary}"),
            )
            parts = [row]
            # Add annotation if available (stored on first tool)
            if first_tool and first_tool.annotation:
                annotation_row = Table.grid(padding=(0, 0))
                annotation_row.add_column(width=2)
                annotation_row.add_column()
                annotation_row.add_row(
                    RichText("  ", style="dim"),
                    RichText.from_markup(f"[yellow]📝 {first_tool.annotation}[/yellow]"),
                )
                parts.append(annotation_row)
            return RichGroup(*parts) if len(parts) > 1 else parts[0]
        else:
            # Expanded or no summary - each tool in two columns
            parts = []
            for tool_turn in group.tool_turns:
                tool_name = tool_turn.tool_name or "Tool"
                indicator = self.TOOL_INDICATORS.get(tool_name, self.TOOL_INDICATORS["default"])
                content = tool_turn.content_full if tool_turn.expanded else tool_turn.content_preview

                row = Table.grid(padding=(0, 0))
                row.add_column(width=2)
                row.add_column()
                tool_text = RichText()
                tool_text.append(f"[{indicator}] ", style="bold" if not selected_style else selected_style)
                tool_text.append(content, style=selected_style or "dim")
                row.add_row(
                    RichText(f"{self.BULLET} ", style=bullet_style),
                    tool_text,
                )
                parts.append(row)
            # Add annotation if available (stored on first tool)
            if first_tool and first_tool.annotation:
                annotation_row = Table.grid(padding=(0, 0))
                annotation_row.add_column(width=2)
                annotation_row.add_column()
                annotation_row.add_row(
                    RichText("  ", style="dim"),
                    RichText.from_markup(f"[yellow]📝 {first_tool.annotation}[/yellow]"),
                )
                parts.append(annotation_row)
            return RichGroup(*parts)


class HeaderPanel(Static):
    """Header panel with session tabs and stats."""

    DEFAULT_CSS = """
    HeaderPanel {
        height: auto;
        border: solid $primary;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sessions: list[Session] = []
        self._active: Session | None = None
        self._filter_label: str = "All"
        self._auto_scroll: bool = True

    def update_header(
        self,
        sessions: list[Session],
        active: Session | None,
        filter_label: str,
        auto_scroll: bool,
    ) -> None:
        self._sessions = sessions
        self._active = active
        self._filter_label = filter_label
        self._auto_scroll = auto_scroll
        self._render_header()

    def _render_header(self) -> None:
        session_parts = []
        for i, session in enumerate(self._sessions[:9], 1):
            if self._active and session.session_id == self._active.session_id:
                session_parts.append(f"[bold reverse] {i} [/] {session.display_name}")
            else:
                session_parts.append(f"[dim]\\[{i}][/dim] {session.display_name}")

        sessions_text = "  ".join(session_parts) if session_parts else "[dim]No sessions[/dim]"

        if self._active:
            scroll_indicator = "" if self._auto_scroll else " [dim](paused)[/dim]"
            historical_count = sum(1 for t in self._active.turns if t.is_historical)
            live_count = len(self._active.turns) - historical_count
            turns_info = f"h{historical_count}+{live_count}" if historical_count else str(live_count)

            if self._active.has_token_data:
                token_stats = f"in:{self._active.total_input_tokens:,}tok out:{self._active.total_output_tokens:,}tok"
                cost = self._active.estimated_cost
                if cost is not None:
                    token_stats += f" ~${cost:.2f}"
            else:
                token_stats = f"in:{self._active.total_input_words:,} out:{self._active.total_output_words:,}"

            stats = (
                f"{self._active.model} | "
                f"{token_stats} | "
                f"turns:{turns_info} | "
                f"{self._filter_label}{scroll_indicator}"
            )
        else:
            stats = "[dim]Waiting for session...[/dim]"

        content = f"[bold]Claude Companion[/bold]\n{sessions_text}\n[dim]{stats}[/dim]"
        self.update(RichText.from_markup(content))


class AlertsPanel(Static):
    """Panel showing recent alerts."""

    DEFAULT_CSS = """
    AlertsPanel {
        height: auto;
        max-height: 5;
        display: none;
        border: solid $error;
        padding: 0 1;
        margin-top: 1;
    }

    AlertsPanel.has-alerts {
        display: block;
    }
    """

    def update_alerts(self, alerts: list[Alert]) -> None:
        """Update displayed alerts."""
        filtered = [a for a in alerts if a.level in (AlertLevel.WARNING, AlertLevel.DANGER)]

        if not filtered:
            self.remove_class("has-alerts")
            self.update("")
            return

        self.add_class("has-alerts")

        lines = []
        for alert in filtered[-3:]:  # Show last 3
            if alert.level == AlertLevel.DANGER:
                indicator = "[bold red]\\[!][/bold red]"
            else:
                indicator = "[yellow]\\[*][/yellow]"
            lines.append(f"{indicator} {alert.title}: {alert.message}")

        self.update(RichText.from_markup("\n".join(lines)))


class TaskListView(Static):
    """View for displaying task list."""

    DEFAULT_CSS = """
    TaskListView {
        height: 1fr;
        display: none;
        overflow-y: auto;
        border: solid $primary;
        padding: 1 2;
    }

    TaskListView.visible {
        display: block;
    }
    """

    def update_tasks(self, session: Session | None) -> None:
        """Update task list display."""
        if not session or not session.tasks:
            self.update(RichText.from_markup(
                "[dim]No tasks yet. Claude will create tasks when working on complex projects.[/dim]"
            ))
            return

        tasks = [t for t in session.tasks.values() if not t.is_deleted]
        tasks.sort(key=lambda t: int(t.task_id) if t.task_id.isdigit() else 0)

        lines = ["[bold cyan]Task List[/bold cyan]\n"]
        for task in tasks:
            if task.status == "completed":
                indicator = "[green]v[/green]"
            elif task.status == "in_progress":
                indicator = "[yellow]>[/yellow]"
            else:
                indicator = "[dim]o[/dim]"

            subject = task.subject[:60] + "..." if len(task.subject) > 60 else task.subject
            blocking_info = ""
            if task.blockedBy:
                blocked_ids = ", ".join(f"#{bid}" for bid in task.blockedBy)
                blocking_info = f" [dim](blocked by {blocked_ids})[/dim]"

            lines.append(f"{indicator} \\[{task.task_id}] {subject}{blocking_info}")

        completed = sum(1 for t in tasks if t.status == "completed")
        in_progress = sum(1 for t in tasks if t.status == "in_progress")
        pending = sum(1 for t in tasks if t.status == "pending")
        lines.append(f"\n[dim]Total: {len(tasks)} ({completed} completed, {in_progress} in progress, {pending} pending)[/dim]")
        lines.append("\n[dim]Press T or ESC to return to conversation[/dim]")

        self.update(RichText.from_markup("\n".join(lines)))


class PlanView(Static):
    """View for displaying plan content."""

    DEFAULT_CSS = """
    PlanView {
        height: 1fr;
        display: none;
        overflow-y: auto;
        border: solid $primary;
        padding: 1 2;
    }

    PlanView.visible {
        display: block;
    }
    """

    def update_plan(self, session: Session | None) -> None:
        """Update plan display."""
        if not session or not session.plan_content:
            self.update(RichText.from_markup(
                "[dim]No active plan detected. Claude will create a plan when you enter plan mode.[/dim]"
            ))
            return

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
        md_content = Markdown(session.plan_content)

        self.update(RichGroup(RichText.from_markup(header), md_content))


class AnalysisPanel(Static):
    """Right-side panel for displaying turn analysis results."""

    DEFAULT_CSS = """
    AnalysisPanel {
        dock: right;
        width: 40;
        height: 1fr;
        display: none;
        border: solid $success;
        padding: 1 2;
        overflow-y: auto;
    }

    AnalysisPanel.visible {
        display: block;
    }
    """

    def update_analysis(self, turn: Turn, analysis, label: str = "") -> None:
        """Update the panel with analysis for the given turn."""
        color = "#5fd787" if analysis.sentiment == "progress" else (
            "#ffaf00" if analysis.sentiment == "concern" else "#ff5f5f"
        )
        indicator = {"progress": "✓", "concern": "⚠", "critical": "✗"}.get(
            analysis.sentiment, "?"
        )

        role = turn.role.capitalize()
        header_label = label or f"Turn {turn.turn_number}"
        summary = markup_escape(analysis.summary)
        critique = markup_escape(analysis.critique)

        lines = [
            f"[bold]Analysis[/bold]  [dim]{header_label} · {role}[/dim]\n",
            f"[{color}]{indicator} Sentiment: {analysis.sentiment}[/{color}]\n",
            f"[bold]Summary[/bold]\n{summary}\n",
            f"[bold]Critique[/bold]\n{critique}\n",
            f"[dim]Turn words: {analysis.word_count:,}[/dim]",
            f"[dim]Context words: {analysis.context_word_count:,}[/dim]",
        ]

        if turn.input_tokens is not None:
            token_line = f"[dim]Tokens: {turn.input_tokens:,} in / {turn.output_tokens or 0:,} out[/dim]"
            lines.append(token_line)
            # Cost for this turn
            if turn.model_id:
                pricing = get_pricing(turn.model_id)
                if pricing:
                    cost = estimate_cost(
                        turn.input_tokens or 0,
                        turn.output_tokens or 0,
                        turn.cache_creation_tokens or 0,
                        turn.cache_read_tokens or 0,
                        pricing,
                    )
                    lines.append(f"[dim]Cost: ${cost:.4f}[/dim]")

        self._append_goal_sources(lines, analysis)
        self.update(RichText.from_markup("\n".join(lines)))

    def update_range_analysis(self, turns: list[Turn], analysis, label: str) -> None:
        """Update panel for multi-turn range analysis."""
        color = "#5fd787" if analysis.sentiment == "progress" else (
            "#ffaf00" if analysis.sentiment == "concern" else "#ff5f5f"
        )
        indicator = {"progress": "✓", "concern": "⚠", "critical": "✗"}.get(
            analysis.sentiment, "?"
        )

        total_words = sum(t.word_count for t in turns)
        summary = markup_escape(analysis.summary)
        critique = markup_escape(analysis.critique)

        lines = [
            f"[bold]Analysis[/bold]  [dim]{label} · {len(turns)} turns[/dim]\n",
            f"[{color}]{indicator} Sentiment: {analysis.sentiment}[/{color}]\n",
            f"[bold]Summary[/bold]\n{summary}\n",
            f"[bold]Critique[/bold]\n{critique}\n",
            f"[dim]Total words: {total_words:,}[/dim]",
            f"[dim]Context words: {analysis.context_word_count:,}[/dim]",
        ]

        # Show token totals and cost if any turns have token data
        has_tokens = any(t.input_tokens is not None for t in turns)
        if has_tokens:
            total_in = sum(t.input_tokens or 0 for t in turns)
            total_out = sum(t.output_tokens or 0 for t in turns)
            total_cache_create = sum(t.cache_creation_tokens or 0 for t in turns)
            total_cache_read = sum(t.cache_read_tokens or 0 for t in turns)
            lines.append(f"[dim]Tokens: {total_in:,} in / {total_out:,} out[/dim]")

            # Find model from first turn with model_id
            model_id = next((t.model_id for t in turns if t.model_id), None)
            if model_id:
                pricing = get_pricing(model_id)
                if pricing:
                    cost = estimate_cost(
                        total_in, total_out, total_cache_create, total_cache_read, pricing,
                    )
                    lines.append(f"[dim]Cost: ${cost:.4f}[/dim]")

        self._append_goal_sources(lines, analysis)
        self.update(RichText.from_markup("\n".join(lines)))

    def _append_goal_sources(self, lines: list[str], analysis) -> None:
        """Append goal source information to display lines."""
        if not getattr(analysis, "goal_sources", None):
            return
        lines.append("")
        if analysis.synthesized_goal:
            lines.append(
                f"[bold]Goal[/bold]\n[dim]{markup_escape(analysis.synthesized_goal)}[/dim]"
            )
        lines.append(f"[bold]Goal Sources[/bold] ({len(analysis.goal_sources)})")
        for gs in analysis.goal_sources:
            indicator = "" if gs.fresh else " [dim italic](stale)[/dim italic]"
            preview = gs.content[:80].replace("\n", " ")
            if len(gs.content) > 80:
                preview += "..."
            lines.append(
                f"  [dim]{markup_escape(gs.label)}{indicator}: {markup_escape(preview)}[/dim]"
            )


class ConversationView(ScrollableContainer):
    """Scrollable container for conversation turns."""

    DEFAULT_CSS = """
    ConversationView {
        height: 1fr;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }

    ConversationView.hidden {
        display: none;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._selected_index: int | None = None
        self._auto_scroll: bool = True
        self._span_start: int | None = None
        self._span_end: int | None = None

    def get_selected_index(self) -> int | None:
        return self._selected_index

    def set_selected_index(self, index: int | None) -> None:
        self._selected_index = index
        self._update_selection()

    def set_span_range(self, start: int | None, end: int | None) -> None:
        """Set the span highlight range (inclusive widget indices)."""
        self._span_start = start
        self._span_end = end
        self._update_selection()

    def _update_selection(self) -> None:
        """Update selection and span highlight state on all turn widgets."""
        widgets = list(self.query("TurnWidget, ToolGroupWidget"))
        for i, widget in enumerate(widgets):
            widget.selected = (i == self._selected_index)
            # Span highlight: in range but not the primary selected widget
            in_span = (
                self._span_start is not None
                and self._span_end is not None
                and self._span_start <= i <= self._span_end
                and i != self._selected_index
            )
            widget.span_highlight = in_span

    def scroll_to_selected(self) -> None:
        """Scroll to keep selected item visible."""
        if self._selected_index is None:
            return
        widgets = list(self.query("TurnWidget, ToolGroupWidget"))
        if 0 <= self._selected_index < len(widgets):
            widgets[self._selected_index].scroll_visible()

    def scroll_to_end_if_auto(self) -> None:
        """Scroll to end if auto-scroll is enabled."""
        if self._auto_scroll:
            self.scroll_end(animate=False)


class InputPanel(Horizontal):
    """Unified input panel for Monitor and Ask."""

    DEFAULT_CSS = """
    InputPanel {
        height: auto;
        margin-top: 1;
    }

    InputPanel .input-box {
        width: 1fr;
        height: auto;
        border: solid $secondary;
        padding: 0 1;
        margin: 0 1;
    }

    InputPanel .input-box:focus-within {
        border: solid $accent;
    }

    InputPanel .input-label {
        color: $text-muted;
    }

    InputPanel Input {
        border: none;
        background: transparent;
        height: 1;
        padding: 0;
    }

    InputPanel Input:focus {
        border: none;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._monitor_text: str = ""

    def compose(self) -> ComposeResult:
        with Vertical(classes="input-box", id="monitor-box"):
            yield Label("[cyan]Monitor[/cyan]", classes="input-label")
            yield Input(placeholder="Press [m] to set monitoring rules", id="monitor-input")
        with Vertical(classes="input-box", id="ask-box"):
            yield Label("[magenta]Ask[/magenta]", classes="input-label")
            yield Input(placeholder="Press [?] to ask about trace", id="ask-input")
        with Vertical(classes="input-box", id="annotate-box"):
            yield Label("[yellow]Annotate[/yellow]", classes="input-label")
            yield Input(placeholder="Press [n] to annotate selected turn", id="annotate-input")

    @property
    def monitor_text(self) -> str:
        return self._monitor_text

    @monitor_text.setter
    def monitor_text(self, value: str) -> None:
        self._monitor_text = value
        try:
            monitor_input = self.query_one("#monitor-input", Input)
            monitor_input.value = value
        except NoMatches:
            pass

    def focus_monitor(self) -> None:
        try:
            self.query_one("#monitor-input", Input).focus()
        except NoMatches:
            pass

    def focus_ask(self) -> None:
        try:
            self.query_one("#ask-input", Input).focus()
        except NoMatches:
            pass

    def focus_annotate(self) -> None:
        try:
            self.query_one("#annotate-input", Input).focus()
        except NoMatches:
            pass

    def set_annotate_placeholder(self, text: str) -> None:
        try:
            self.query_one("#annotate-input", Input).placeholder = text
        except NoMatches:
            pass

    def set_annotate_value(self, value: str) -> None:
        try:
            self.query_one("#annotate-input", Input).value = value
        except NoMatches:
            pass

    def clear_annotate(self) -> None:
        try:
            inp = self.query_one("#annotate-input", Input)
            inp.value = ""
            inp.placeholder = "Press [n] to annotate selected turn"
        except NoMatches:
            pass


class CompanionApp(App):
    """Textual TUI for Claude Companion."""

    CSS = CSS

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j", "nav_down", "Down", show=False),
        Binding("k", "nav_up", "Up", show=False),
        Binding("down", "nav_down", "Down", show=False),
        Binding("up", "nav_up", "Up", show=False),
        Binding("g", "go_top", "Top", show=False),
        Binding("G", "go_bottom", "Bottom", show=False),
        Binding("o", "toggle_expand", "Expand", show=False),
        Binding("space", "toggle_expand", "Expand", show=False),
        Binding("enter", "toggle_expand", "Expand", show=False),
        Binding("e", "expand_all", "Expand All", show=False),
        Binding("c", "collapse_all", "Collapse All", show=False),
        Binding("f", "cycle_filter", "Filter", show=False),
        Binding("T", "toggle_tasks", "Tasks", show=False),
        Binding("P", "toggle_plan", "Plan", show=False),
        Binding("x", "export", "Export", show=False),
        Binding("n", "annotate", "Annotate", show=False),
        Binding("a", "toggle_alerts", "Alerts", show=False),
        Binding("s", "toggle_summary", "Summary", show=False),
        Binding("S", "summarize_all", "Summarize All", show=False),
        Binding("A", "analyze_turn", "Analyze", show=False),
        Binding("]", "zoom_out", "Zoom Out", show=False),
        Binding("[", "zoom_in", "Zoom In", show=False),
        Binding("D", "delete_session", "Delete", show=False),
        Binding("m", "focus_monitor", "Monitor", show=False),
        Binding("?", "focus_ask", "Ask", show=False),
        Binding("escape", "escape", "Escape", show=False),
        Binding("1", "select_session_1", "Session 1", show=False),
        Binding("2", "select_session_2", "Session 2", show=False),
        Binding("3", "select_session_3", "Session 3", show=False),
        Binding("4", "select_session_4", "Session 4", show=False),
        Binding("5", "select_session_5", "Session 5", show=False),
        Binding("6", "select_session_6", "Session 6", show=False),
        Binding("7", "select_session_7", "Session 7", show=False),
        Binding("8", "select_session_8", "Session 8", show=False),
        Binding("9", "select_session_9", "Session 9", show=False),
    ]

    # Custom messages for thread-safe updates
    class TurnAdded(Message):
        """Message posted when a new turn arrives."""
        def __init__(self, turn: Turn) -> None:
            super().__init__()
            self.turn = turn

    class AlertAdded(Message):
        """Message posted when a new alert arrives."""
        def __init__(self, alert: Alert) -> None:
            super().__init__()
            self.alert = alert

    def __init__(self, store: "EventStore", collapse_tools: bool = True, ui_style: str = "rich") -> None:
        super().__init__()
        self.store = store
        self._collapse_tools = collapse_tools
        self._use_summary = True
        self._filter_index = 0
        self._ui_style = ui_style  # "rich" or "claude-code"
        # Filters differ by style
        if ui_style == "rich":
            self._turn_filters = [
                ("all", "All"),
                ("tool", "Tools only"),
                ("Read", "📄 Read"),
                ("Write", "✏️ Write"),
                ("Edit", "✏️ Edit"),
                ("Bash", "💻 Bash"),
            ]
        else:
            self._turn_filters = [
                ("all", "All"),
                ("tool", "Tools"),
                ("Read", "Read"),
                ("Write", "Write"),
                ("Edit", "Edit"),
                ("Bash", "Bash"),
            ]
        self._show_tasks = False
        self._show_plan = False
        self._show_alerts = True
        self._auto_scroll = True
        self._group_expanded_state: dict[int, bool] = {}
        self._status_message: str | None = None
        self._refresh_counter: int = 0  # Used to generate unique widget IDs
        self._annotating_turn_number: int | None = None  # Turn being annotated
        self._annotating_scroll_y: float = 0  # Scroll position before annotating
        self._analysis_level: str = "turn"  # "turn" | "span" | "session"
        self._analysis_radius: int = 0  # 0 = single span, 1 = ±1, etc.

    def compose(self) -> ComposeResult:
        yield HeaderPanel(id="header")
        yield ConversationView(id="conversation")
        yield TaskListView(id="tasks-view")
        yield PlanView(id="plan-view")
        yield AnalysisPanel(id="analysis-panel")
        yield AlertsPanel(id="alerts")
        yield InputPanel(id="inputs")
        yield Footer()

    def on_mount(self) -> None:
        """Wire up store listeners when app mounts."""
        self.store.add_turn_listener(self._on_store_turn)
        self.store.add_alert_listener(self._on_store_alert)
        # Initial render
        self._refresh_all()
        # Set up periodic refresh for summaries
        self.set_interval(0.5, self._check_for_updates)

    def _on_store_turn(self, turn: Turn) -> None:
        """Called from watcher thread - post message for thread safety."""
        self.post_message(self.TurnAdded(turn))

    def _on_store_alert(self, alert: Alert) -> None:
        """Called from watcher thread - post message for thread safety."""
        self.post_message(self.AlertAdded(alert))

    @on(TurnAdded)
    def handle_turn_added(self, message: TurnAdded) -> None:
        """Handle new turn on main thread."""
        self._refresh_conversation()
        if self._auto_scroll:
            self.query_one("#conversation", ConversationView).scroll_to_end_if_auto()
        self._refresh_analysis_panel()

    @on(AlertAdded)
    def handle_alert_added(self, message: AlertAdded) -> None:
        """Handle new alert on main thread."""
        self._refresh_alerts()

    def _check_for_updates(self) -> None:
        """Periodic check for updates (summaries, plan changes, etc.)."""
        self._refresh_header()
        self._refresh_views()  # Refresh plan/tasks in case they changed

    def _refresh_all(self) -> None:
        """Refresh all UI components."""
        self._refresh_header()
        self._refresh_conversation()
        self._refresh_alerts()
        self._refresh_views()

    def _refresh_header(self) -> None:
        """Refresh header panel."""
        sessions = self.store.get_sessions()
        active = self.store.get_active_session()
        _, filter_label = self._turn_filters[self._filter_index]
        header = self.query_one("#header", HeaderPanel)
        header.update_header(sessions, active, filter_label, self._auto_scroll)

    def _refresh_conversation(self) -> None:
        """Refresh conversation view."""
        conversation = self.query_one("#conversation", ConversationView)
        session = self.store.get_active_session()

        # Get current selection
        selected_index = conversation.get_selected_index()

        # Increment counter for unique IDs
        self._refresh_counter += 1
        rc = self._refresh_counter

        # Clear and rebuild
        conversation.remove_children()

        if not session:
            conversation.mount(Static("[dim]Waiting for activity...[/dim]"))
            return

        # Get filtered and grouped turns
        items = self._get_filtered_turns(session)

        if not items:
            filter_key, filter_label = self._turn_filters[self._filter_index]
            if filter_key != "all" and session.turns:
                msg = f"[dim]No {filter_label} turns. Press \\[f] to change filter.[/dim]"
            else:
                msg = "[dim]Waiting for activity...[/dim]"
            conversation.mount(Static(msg))
            return

        # Calculate conversation turn numbers
        conv_turn_map: dict[int, int] = {}
        conv_turn = 0
        for item in items:
            if isinstance(item, ToolGroup):
                pass
            elif item.role in ("user", "assistant"):
                conv_turn += 1
                conv_turn_map[id(item)] = conv_turn

        # Create widgets with unique IDs using refresh counter
        for i, item in enumerate(items):
            if isinstance(item, ToolGroup):
                widget = ToolGroupWidget(
                    item,
                    ui_style=self._ui_style,
                    id=f"group-{rc}-{item.first_turn_number}",
                )
                widget.selected = (i == selected_index)
            else:
                widget = TurnWidget(
                    item,
                    conv_turn=conv_turn_map.get(id(item), 0),
                    use_summary=self._use_summary,
                    ui_style=self._ui_style,
                    id=f"turn-{rc}-{item.turn_number}",
                )
                widget.selected = (i == selected_index)
            conversation.mount(widget)

        # Scroll to end if auto-scroll
        if self._auto_scroll:
            conversation.call_after_refresh(conversation.scroll_end)

    def _get_filtered_turns(self, session: Session | None) -> list[Turn | ToolGroup]:
        """Get turns filtered by current filter, with optional grouping."""
        if not session:
            return []

        filter_key, _ = self._turn_filters[self._filter_index]

        if self._collapse_tools and self._use_summary and filter_key == "all":
            return group_turns_for_display(
                session, session.turns, use_summary=True,
                group_expanded_state=self._group_expanded_state,
            )

        if filter_key == "all":
            return session.turns
        elif filter_key == "tool":
            return [t for t in session.turns if t.role == "tool"]
        else:
            return [t for t in session.turns if t.tool_name == filter_key]

    def _refresh_alerts(self) -> None:
        """Refresh alerts panel."""
        if not self._show_alerts:
            return
        alerts = self.store.get_recent_alerts(5)
        alerts_panel = self.query_one("#alerts", AlertsPanel)
        alerts_panel.update_alerts(alerts)

    def _refresh_views(self) -> None:
        """Refresh tasks and plan views."""
        session = self.store.get_active_session()

        tasks_view = self.query_one("#tasks-view", TaskListView)
        plan_view = self.query_one("#plan-view", PlanView)
        conversation = self.query_one("#conversation", ConversationView)

        # Update visibility
        tasks_view.set_class(self._show_tasks, "visible")
        plan_view.set_class(self._show_plan, "visible")
        conversation.set_class(self._show_tasks or self._show_plan, "hidden")

        if self._show_tasks:
            tasks_view.update_tasks(session)
        if self._show_plan:
            plan_view.update_plan(session)

    def _refresh_analysis_panel(self) -> None:
        """Update the analysis panel based on the currently selected turn and analysis level."""
        panel = self.query_one("#analysis-panel", AnalysisPanel)
        conversation = self.query_one("#conversation", ConversationView)
        index = conversation.get_selected_index()

        if index is None:
            panel.remove_class("visible")
            return

        result = self._get_analysis_turn_range()
        if result is None:
            panel.remove_class("visible")
            return

        turns, label = result

        if len(turns) == 1 and turns[0].analysis:
            # Single turn — existing behavior
            panel.update_analysis(turns[0], turns[0].analysis, label)
            panel.add_class("visible")
        elif len(turns) > 1:
            # Multi-turn: check if store has a range analysis
            range_key = (turns[0].turn_number, turns[-1].turn_number)
            range_analysis = self.store.get_range_analysis(range_key)
            if range_analysis:
                panel.update_range_analysis(turns, range_analysis, label)
                panel.add_class("visible")
            else:
                panel.remove_class("visible")
        else:
            panel.remove_class("visible")

        # Update span highlighting in conversation
        self._update_span_highlighting()

    def _get_analysis_turn_range(self) -> tuple[list[Turn], str] | None:
        """Get turns and label for the current analysis level.

        Returns:
            (turns_in_range, label) or None if no selection
        """
        conversation = self.query_one("#conversation", ConversationView)
        index = conversation.get_selected_index()
        if index is None:
            return None

        widgets = list(conversation.query("TurnWidget, ToolGroupWidget"))
        if not (0 <= index < len(widgets)):
            return None

        widget = widgets[index]
        if isinstance(widget, TurnWidget):
            selected_turn = widget.turn
        elif isinstance(widget, ToolGroupWidget):
            selected_turn = widget.group.tool_turns[0] if widget.group.tool_turns else None
        else:
            selected_turn = None

        if not selected_turn:
            return None

        if self._analysis_level == "turn":
            return [selected_turn], f"Turn {selected_turn.turn_number}"

        # For span/session levels, we need the session's turns
        session = self.store.get_active_session()
        if not session or not session.turns:
            return [selected_turn], f"Turn {selected_turn.turn_number}"

        spans = compute_spans(session.turns)
        if not spans:
            return [selected_turn], f"Turn {selected_turn.turn_number}"

        # Find which span contains the selected turn
        selected_span_idx = 0
        for si, (start, end) in enumerate(spans):
            for ti in range(start, end + 1):
                if session.turns[ti] is selected_turn:
                    selected_span_idx = si
                    break

        if self._analysis_level == "session":
            return list(session.turns), "Session"

        # Span level with radius
        span_start = max(0, selected_span_idx - self._analysis_radius)
        span_end = min(len(spans) - 1, selected_span_idx + self._analysis_radius)

        # Collect all turns in the span range
        turn_start = spans[span_start][0]
        turn_end = spans[span_end][1]
        range_turns = session.turns[turn_start:turn_end + 1]

        if self._analysis_radius == 0:
            label = f"Span {selected_span_idx + 1}"
        else:
            label = f"Spans {span_start + 1}\u2013{span_end + 1}"

        return range_turns, label

    def _get_analysis_widget_range(self) -> tuple[int, int] | None:
        """Get widget index range for the current analysis level.

        Returns the (start_widget_idx, end_widget_idx) inclusive, or None.
        """
        result = self._get_analysis_turn_range()
        if result is None:
            return None

        turns, _ = result
        if len(turns) <= 1:
            return None

        # Map turns to widget indices
        conversation = self.query_one("#conversation", ConversationView)
        widgets = list(conversation.query("TurnWidget, ToolGroupWidget"))

        turn_set = set(id(t) for t in turns)
        widget_indices = []

        for i, widget in enumerate(widgets):
            if isinstance(widget, TurnWidget):
                if id(widget.turn) in turn_set:
                    widget_indices.append(i)
            elif isinstance(widget, ToolGroupWidget):
                for tt in widget.group.tool_turns:
                    if id(tt) in turn_set:
                        widget_indices.append(i)
                        break

        if not widget_indices:
            return None

        return (min(widget_indices), max(widget_indices))

    def _update_span_highlighting(self) -> None:
        """Update span highlighting on conversation widgets."""
        conversation = self.query_one("#conversation", ConversationView)

        if self._analysis_level == "turn":
            conversation.set_span_range(None, None)
            return

        widget_range = self._get_analysis_widget_range()
        if widget_range:
            conversation.set_span_range(widget_range[0], widget_range[1])
        else:
            conversation.set_span_range(None, None)

    def _reset_analysis_level(self) -> None:
        """Reset analysis level to turn."""
        self._analysis_level = "turn"
        self._analysis_radius = 0
        conversation = self.query_one("#conversation", ConversationView)
        conversation.set_span_range(None, None)

    def action_zoom_out(self) -> None:
        """Zoom out analysis level: turn → span → span±1 → ... → session."""
        conversation = self.query_one("#conversation", ConversationView)
        if conversation.get_selected_index() is None:
            return

        session = self.store.get_active_session()
        if not session or not session.turns:
            return

        spans = compute_spans(session.turns)
        max_radius = len(spans) - 1  # Max meaningful radius

        if self._analysis_level == "turn":
            self._analysis_level = "span"
            self._analysis_radius = 0
        elif self._analysis_level in ("span",):
            self._analysis_radius += 1
            # Check if radius covers all spans
            if self._analysis_radius >= max_radius:
                self._analysis_level = "session"
                self._analysis_radius = 0
        elif self._analysis_level == "session":
            return  # Already at max

        self._update_span_highlighting()
        self._refresh_analysis_panel()
        # Show level label
        result = self._get_analysis_turn_range()
        if result:
            _, label = result
            self._show_status(f"Analysis level: {label}")

    def action_zoom_in(self) -> None:
        """Zoom in analysis level: session → span±N → span → turn."""
        conversation = self.query_one("#conversation", ConversationView)
        if conversation.get_selected_index() is None:
            return

        if self._analysis_level == "session":
            session = self.store.get_active_session()
            if session and session.turns:
                spans = compute_spans(session.turns)
                # Set to max non-session radius
                self._analysis_level = "span"
                self._analysis_radius = max(0, len(spans) - 2)
            else:
                self._analysis_level = "turn"
                self._analysis_radius = 0
        elif self._analysis_level == "span":
            if self._analysis_radius > 0:
                self._analysis_radius -= 1
            else:
                self._analysis_level = "turn"
                self._analysis_radius = 0
        elif self._analysis_level == "turn":
            return  # Already at min

        self._update_span_highlighting()
        self._refresh_analysis_panel()
        # Show level label
        result = self._get_analysis_turn_range()
        if result:
            _, label = result
            self._show_status(f"Analysis level: {label}")

    def _show_status(self, message: str) -> None:
        """Show a status message in the footer."""
        self._status_message = message
        self.notify(message, timeout=3)

    # Action handlers
    def action_nav_down(self) -> None:
        """Navigate down in conversation."""
        self._auto_scroll = False
        self._reset_analysis_level()
        conversation = self.query_one("#conversation", ConversationView)
        widgets = list(conversation.query("TurnWidget, ToolGroupWidget"))
        if not widgets:
            return
        current = conversation.get_selected_index()
        if current is None:
            new_index = 0
        elif current < len(widgets) - 1:
            new_index = current + 1
        else:
            new_index = current
        conversation.set_selected_index(new_index)
        conversation.scroll_to_selected()
        self._refresh_analysis_panel()

    def action_nav_up(self) -> None:
        """Navigate up in conversation."""
        self._auto_scroll = False
        self._reset_analysis_level()
        conversation = self.query_one("#conversation", ConversationView)
        widgets = list(conversation.query("TurnWidget, ToolGroupWidget"))
        if not widgets:
            return
        current = conversation.get_selected_index()
        if current is None:
            new_index = len(widgets) - 1
        elif current > 0:
            new_index = current - 1
        else:
            new_index = current
        conversation.set_selected_index(new_index)
        conversation.scroll_to_selected()
        self._refresh_analysis_panel()

    def action_go_top(self) -> None:
        """Go to top of conversation."""
        self._auto_scroll = False
        self._reset_analysis_level()
        conversation = self.query_one("#conversation", ConversationView)
        widgets = list(conversation.query("TurnWidget, ToolGroupWidget"))
        if widgets:
            conversation.set_selected_index(0)
            conversation.scroll_home()
        self._refresh_analysis_panel()

    def action_go_bottom(self) -> None:
        """Go to bottom of conversation."""
        self._auto_scroll = True
        self._reset_analysis_level()
        conversation = self.query_one("#conversation", ConversationView)
        widgets = list(conversation.query("TurnWidget, ToolGroupWidget"))
        if widgets:
            conversation.set_selected_index(len(widgets) - 1)
            conversation.scroll_end()
        self._refresh_analysis_panel()

    def action_toggle_expand(self) -> None:
        """Toggle expand on selected item."""
        conversation = self.query_one("#conversation", ConversationView)
        index = conversation.get_selected_index()
        if index is None:
            return
        widgets = list(conversation.query("TurnWidget, ToolGroupWidget"))
        if 0 <= index < len(widgets):
            widget = widgets[index]
            # Trigger summarization for unsummarized tool groups
            if isinstance(widget, ToolGroupWidget) and not widget.group.summary:
                self.store.summarize_tool_group(widget.group.tool_turns)
            widget.expanded = not widget.expanded
            # Persist group state
            if isinstance(widget, ToolGroupWidget):
                self._group_expanded_state[widget.group.first_turn_number] = widget.expanded

    def action_expand_all(self) -> None:
        """Expand all items."""
        conversation = self.query_one("#conversation", ConversationView)
        for widget in conversation.query("TurnWidget, ToolGroupWidget"):
            widget.expanded = True
            if isinstance(widget, ToolGroupWidget):
                self._group_expanded_state[widget.group.first_turn_number] = True

    def action_collapse_all(self) -> None:
        """Collapse all items."""
        conversation = self.query_one("#conversation", ConversationView)
        for widget in conversation.query("TurnWidget, ToolGroupWidget"):
            widget.expanded = False
            if isinstance(widget, ToolGroupWidget):
                self._group_expanded_state[widget.group.first_turn_number] = False

    def action_cycle_filter(self) -> None:
        """Cycle through filters."""
        self._filter_index = (self._filter_index + 1) % len(self._turn_filters)
        self._auto_scroll = True
        conversation = self.query_one("#conversation", ConversationView)
        conversation.set_selected_index(None)
        self._refresh_conversation()
        self._refresh_header()

    def action_toggle_tasks(self) -> None:
        """Toggle task list view."""
        self._show_tasks = not self._show_tasks
        self._show_plan = False
        self._refresh_views()
        self._show_status("Showing task list" if self._show_tasks else "Showing conversation")

    def action_toggle_plan(self) -> None:
        """Toggle plan view."""
        self._show_plan = not self._show_plan
        self._show_tasks = False
        self._refresh_views()
        self._show_status("Showing plan" if self._show_plan else "Showing conversation")

    def action_export(self) -> None:
        """Export current session."""
        session = self.store.get_active_session()
        if session:
            filename = get_export_filename(session, "markdown")
            output_path = Path.cwd() / filename
            export_session_markdown(session, output_path)
            self._show_status(f"Exported to {filename}")
        else:
            self._show_status("No session to export")

    def action_annotate(self) -> None:
        """Annotate the selected turn."""
        conversation = self.query_one("#conversation", ConversationView)

        # Save scroll position before annotating
        self._annotating_scroll_y = conversation.scroll_y

        index = conversation.get_selected_index()
        if index is None:
            self._show_status("Select a turn first (j/k to navigate)")
            return

        widgets = list(conversation.query("TurnWidget, ToolGroupWidget"))
        if not (0 <= index < len(widgets)):
            return

        widget = widgets[index]
        # Get the turn from the widget
        if isinstance(widget, TurnWidget):
            turn = widget.turn
        elif isinstance(widget, ToolGroupWidget):
            # For tool groups, annotate the first tool turn
            turn = widget.group.tool_turns[0] if widget.group.tool_turns else None
        else:
            turn = None

        if not turn:
            self._show_status("Cannot annotate this item")
            return

        # Store the turn number we're annotating
        self._annotating_turn_number = turn.turn_number

        # Set up the annotation input
        inputs = self.query_one("#inputs", InputPanel)
        existing = turn.annotation or ""
        inputs.set_annotate_value(existing)
        inputs.set_annotate_placeholder(f"Annotate turn {turn.turn_number} (Enter to save, empty to clear)")
        inputs.focus_annotate()

    def action_toggle_alerts(self) -> None:
        """Toggle alerts panel or clear alerts."""
        alerts = self.store.get_alerts()
        if alerts:
            self.store.clear_alerts()
            self._refresh_alerts()
            self._show_status("Alerts cleared")
        else:
            self._show_alerts = not self._show_alerts
            alerts_panel = self.query_one("#alerts", AlertsPanel)
            if not self._show_alerts:
                alerts_panel.remove_class("has-alerts")
            else:
                self._refresh_alerts()
            self._show_status(f"Alerts {'shown' if self._show_alerts else 'hidden'}")

    def action_toggle_summary(self) -> None:
        """Toggle summary mode."""
        self._use_summary = not self._use_summary
        mode = "summary" if self._use_summary else "preview"
        self._show_status(f"Preview mode: {mode}")
        self._refresh_conversation()

    def action_summarize_all(self) -> None:
        """Summarize all historical turns."""
        count = self.store.summarize_historical_turns()
        if count > 0:
            self._show_status(f"Summarizing {count} historical turns...")
        else:
            self._show_status("No historical turns need summarization")

    def action_analyze_turn(self) -> None:
        """Analyze the selected turn or range, or toggle the panel if already analyzed."""
        panel = self.query_one("#analysis-panel", AnalysisPanel)
        conversation = self.query_one("#conversation", ConversationView)
        index = conversation.get_selected_index()
        if index is None:
            self._show_status("Select a turn first (j/k to navigate)")
            return

        result = self._get_analysis_turn_range()
        if result is None:
            return

        turns, label = result

        if len(turns) == 1:
            # Single-turn analysis (existing logic)
            turn = turns[0]

            # Toggle panel off if already showing analysis for this turn
            if turn.analysis and panel.has_class("visible"):
                panel.remove_class("visible")
                return

            submitted = self.store.analyze_turn(turn)
            if submitted:
                self._show_status(f"Analyzing {label}...")
            else:
                if turn.analysis and not turn.analysis.summary.startswith("["):
                    self._show_status(f"{label} already analyzed")
                    self._refresh_conversation()
                else:
                    self._show_status(f"Analysis loaded for {label}")
                    self._refresh_conversation()
        else:
            # Multi-turn range analysis
            range_key = (turns[0].turn_number, turns[-1].turn_number)
            existing = self.store.get_range_analysis(range_key)
            if existing and panel.has_class("visible"):
                panel.remove_class("visible")
                return

            submitted = self.store.analyze_range(turns)
            if submitted:
                self._show_status(f"Analyzing {label}...")
            else:
                self._show_status(f"Analysis loaded for {label}")

        self._refresh_analysis_panel()

    def action_delete_session(self) -> None:
        """Delete active session."""
        session = self.store.get_active_session()
        if session:
            session_id = session.session_id
            if self.store.delete_session(session_id):
                self._show_status(f"Deleted session {session_id[:8]}...")
                self._group_expanded_state.clear()
                self._refresh_all()
        else:
            self._show_status("No session to delete")

    def action_focus_monitor(self) -> None:
        """Focus monitor input."""
        inputs = self.query_one("#inputs", InputPanel)
        inputs.focus_monitor()

    def action_focus_ask(self) -> None:
        """Focus ask input."""
        inputs = self.query_one("#inputs", InputPanel)
        inputs.focus_ask()

    def action_escape(self) -> None:
        """Handle escape key."""
        self._reset_analysis_level()
        # Clear selection or close views
        if self._show_tasks:
            self._show_tasks = False
            self._refresh_views()
        elif self._show_plan:
            self._show_plan = False
            self._refresh_views()
        else:
            conversation = self.query_one("#conversation", ConversationView)
            conversation.set_selected_index(None)
        self._refresh_analysis_panel()

    def _select_session(self, index: int) -> None:
        """Select session by index (1-based)."""
        if self.store.set_active_session(index):
            self._filter_index = 0
            self._auto_scroll = True
            self._group_expanded_state.clear()
            self._reset_analysis_level()
            self.store.clear_range_analyses()
            self._refresh_all()
            self._refresh_analysis_panel()

    def action_select_session_1(self) -> None:
        self._select_session(1)

    def action_select_session_2(self) -> None:
        self._select_session(2)

    def action_select_session_3(self) -> None:
        self._select_session(3)

    def action_select_session_4(self) -> None:
        self._select_session(4)

    def action_select_session_5(self) -> None:
        self._select_session(5)

    def action_select_session_6(self) -> None:
        self._select_session(6)

    def action_select_session_7(self) -> None:
        self._select_session(7)

    def action_select_session_8(self) -> None:
        self._select_session(8)

    def action_select_session_9(self) -> None:
        self._select_session(9)

    @on(Input.Submitted, "#monitor-input")
    def handle_monitor_submit(self, event: Input.Submitted) -> None:
        """Handle monitor input submission."""
        inputs = self.query_one("#inputs", InputPanel)
        inputs.monitor_text = event.value
        self._show_status(f"Monitor set: {event.value[:30]}..." if event.value else "Monitor cleared")
        # Blur the input
        self.set_focus(None)

    @on(Input.Submitted, "#ask-input")
    def handle_ask_submit(self, event: Input.Submitted) -> None:
        """Handle ask input submission."""
        if event.value:
            self._show_status(f"Query: {event.value[:30]}...")
            # TODO: Integrate with LLM for actual query handling
        # Clear and blur
        event.input.value = ""
        self.set_focus(None)

    @on(Input.Submitted, "#annotate-input")
    def handle_annotate_submit(self, event: Input.Submitted) -> None:
        """Handle annotation input submission."""
        if self._annotating_turn_number is not None:
            annotation = event.value.strip()
            if self.store.set_annotation(self._annotating_turn_number, annotation):
                if annotation:
                    self._show_status(f"Annotation saved for turn {self._annotating_turn_number}")
                else:
                    self._show_status(f"Annotation cleared for turn {self._annotating_turn_number}")
                # Disable auto-scroll to preserve position
                self._auto_scroll = False
                self._refresh_conversation()
                # Restore scroll position from before annotation started
                conversation = self.query_one("#conversation", ConversationView)
                saved_y = self._annotating_scroll_y
                conversation.call_after_refresh(lambda: conversation.scroll_to(y=saved_y, animate=False))
            else:
                self._show_status("Failed to save annotation")
            self._annotating_turn_number = None

        # Clear and blur
        inputs = self.query_one("#inputs", InputPanel)
        inputs.clear_annotate()
        self.set_focus(None)

    @property
    def monitor_instruction(self) -> str:
        """Get current monitor instruction text."""
        try:
            inputs = self.query_one("#inputs", InputPanel)
            return inputs.monitor_text
        except NoMatches:
            return ""
