"""Textual TUI for Betty."""

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
from textual.containers import Container, ScrollableContainer, Horizontal, Vertical
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

#main-content {
    height: 1fr;
}

#detail-pane {
    height: 1fr;
    width: 1fr;
}

#detail-pane.hidden {
    display: none;
}

ManagerView.expand-visible {
    width: 27%;
    border: solid $surface;
}

ManagerView.expand-visible.panel-focused {
    border: solid $accent;
}

ManagerView.expand-visible #manager-grid {
    grid-size: 1;
}

#detail-pane.expand-active {
    border: solid $surface;
}

#detail-pane.expand-active.panel-focused {
    border: solid $accent;
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
        self._manager_expanded: bool = False
        self._focus_panel: str = "manager"

    def update_header(
        self,
        sessions: list[Session],
        active: Session | None,
        filter_label: str,
        auto_scroll: bool,
        manager_active: bool = False,
        manager_expanded: bool = False,
        focus_panel: str = "manager",
    ) -> None:
        self._sessions = sessions
        self._active = active
        self._filter_label = filter_label
        self._auto_scroll = auto_scroll
        self._manager_active = manager_active
        self._manager_expanded = manager_expanded
        self._focus_panel = focus_panel
        self._render_header()

    def _render_header(self) -> None:
        if self._manager_active and self._manager_expanded:
            n = sum(1 for s in self._sessions if s.turns)
            focus_label = "manager" if self._focus_panel == "manager" else "detail"
            content = (
                f"[bold]Betty[/bold] — Manager \\[Expanded] focus:{focus_label}\n"
                f"[dim]{n} session{'s' if n != 1 else ''} | "
                f"h/l:switch panel  j/k:navigate  enter:select  Esc:collapse[/dim]"
            )
            self.update(RichText.from_markup(content))
            return
        if self._manager_active:
            n = sum(1 for s in self._sessions if s.turns)
            content = (
                f"[bold]Betty[/bold] — Manager View\n"
                f"[dim]{n} session{'s' if n != 1 else ''} | "
                f"j/k/h/l:navigate  enter:open  M:toggle  q:quit[/dim]"
            )
            self.update(RichText.from_markup(content))
            return

        session_parts = []
        for i, session in enumerate(self._sessions[:9], 1):
            safe_name = session.display_name.replace("[", "\\[")
            if self._active and session.session_id == self._active.session_id:
                session_parts.append(f"[bold reverse] {i} [/] {safe_name}")
            else:
                session_parts.append(f"[dim]\\[{i}][/dim] {safe_name}")

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

    def show_empty(self) -> None:
        """Display an empty state placeholder when no analysis is available."""
        lines = [
            "[bold]Analysis[/bold]\n",
            "[dim]No analysis for this selection.[/dim]",
            "[dim]Press A to analyze.[/dim]",
        ]
        self.update(RichText.from_markup("\n".join(lines)))

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
        """Append goal and objective information to display lines."""
        goal_sources = getattr(analysis, "goal_sources", None)
        synthesized_goal = getattr(analysis, "synthesized_goal", None)
        local_goal = getattr(analysis, "local_goal", None)

        has_content = goal_sources or synthesized_goal or local_goal
        if not has_content:
            return

        lines.append("")

        # Session Goal: synthesized if available, otherwise first user request
        if synthesized_goal:
            lines.append(
                f"[bold]Session Goal[/bold]\n[dim]{markup_escape(synthesized_goal)}[/dim]"
            )
        elif goal_sources:
            user_sources = [s for s in goal_sources if s.source_type == "user_request"]
            if user_sources and user_sources[0].content != "[No user message found]":
                preview = user_sources[0].content[:200].replace("\n", " ")
                if len(user_sources[0].content) > 200:
                    preview += "..."
                lines.append(
                    f"[bold]Session Goal[/bold]\n[dim]{markup_escape(preview)}[/dim]"
                )

        # Current Objective (local goal)
        if local_goal:
            lines.append(
                f"[bold]Current Objective[/bold]\n[dim]{markup_escape(local_goal)}[/dim]"
            )

        # Detailed goal sources (when available)
        if goal_sources:
            lines.append(f"[bold]Goal Sources[/bold] ({len(goal_sources)})")
            for gs in goal_sources:
                indicator = "" if gs.fresh else " [dim italic](stale)[/dim italic]"
                preview = gs.content[:80].replace("\n", " ")
                if len(gs.content) > 80:
                    preview += "..."
                lines.append(
                    f"  [dim]{markup_escape(gs.label)}{indicator}: {markup_escape(preview)}[/dim]"
                )


class SessionCard(Static):
    """Widget for displaying a session card in the manager view."""

    DEFAULT_CSS = """
    SessionCard {
        height: auto;
        min-height: 5;
        padding: 1 2;
        border: solid $primary;
        margin: 0;
    }

    SessionCard.selected {
        border: solid $accent;
        background: $primary 30%;
    }

    SessionCard.active-session {
        border: solid $success;
    }

    SessionCard.selected.active-session {
        border: solid $accent;
        background: $success 20%;
    }
    """

    selected: reactive[bool] = reactive(False)

    def __init__(self, session: Session, is_active: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session = session
        self.is_active = is_active

    def on_mount(self) -> None:
        self._update_content()
        self._update_classes()

    def watch_selected(self, value: bool) -> None:
        self._update_classes()

    def _update_classes(self) -> None:
        self.remove_class("selected", "active-session")
        if self.selected:
            self.add_class("selected")
        if self.is_active:
            self.add_class("active-session")

    def _update_content(self) -> None:
        session = self.session
        last = session.last_activity
        # Show relative time for last activity
        delta = (datetime.now() - last).total_seconds()
        if delta < 60:
            time_str = "just now"
        elif delta < 3600:
            time_str = f"{int(delta / 60)}m ago"
        elif delta < 86400:
            time_str = f"{int(delta / 3600)}h ago"
        else:
            time_str = last.strftime("%b %d")

        turn_count = len(session.turns)
        tool_calls = session.total_tool_calls
        words_in = session.total_input_words
        words_out = session.total_output_words
        active_marker = " [green]*[/green]" if self.is_active else ""

        # Escape Rich markup chars in display name (branch names may contain [])
        safe_name = session.display_name.replace("[", "\\[")
        lines = [
            f"[bold]{safe_name}[/bold]{active_marker}",
            f"[dim]{session.model} | {time_str}[/dim]",
            f"turns:{turn_count} tools:{tool_calls}",
            f"[dim]in:{words_in:,} out:{words_out:,}[/dim]",
        ]

        # Show PR link if available
        if session.pr_info:
            pr = session.pr_info
            pr_title = pr.title[:30] + ("..." if len(pr.title) > 30 else "")
            # Escape Rich markup in PR title
            pr_title = pr_title.replace("[", "\\[")
            state_color = {"OPEN": "green", "MERGED": "magenta", "CLOSED": "red"}.get(pr.state, "dim")
            lines.append(f"[{state_color}]#{pr.number}[/{state_color}] {pr_title}")

        self.update(RichText.from_markup("\n".join(lines)))


class ManagerView(ScrollableContainer):
    """Grid container for session cards in manager view."""

    DEFAULT_CSS = """
    ManagerView {
        height: 1fr;
        display: none;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }

    ManagerView.visible {
        display: block;
    }

    #manager-grid {
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
        padding: 1 2;
        height: auto;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._selected_index: int = 0
        self._session_ids: list[str] = []
        self._refresh_counter: int = 0
        self._last_snapshot: str = ""  # fingerprint to skip no-op rebuilds

    def compose(self) -> ComposeResult:
        yield Container(id="manager-grid")

    def refresh_cards(self, sessions: list[Session], active_session_id: str | None) -> None:
        """Rebuild session cards from current sessions."""
        # Build a fingerprint of relevant data to skip no-op rebuilds.
        # Include minute-resolution timestamp so relative time labels refresh.
        # Include display_name/branch so cards update promptly when detected.
        now_minute = int(datetime.now().timestamp() // 60)
        snapshot = "|".join(
            f"{s.session_id}:{len(s.turns)}:{s.total_tool_calls}:{s.model}:{s.display_name}:{s.branch or ''}:{s.pr_info.number if s.pr_info else ''}"
            for s in sessions
        ) + f"||{active_session_id}||{now_minute}"
        if snapshot == self._last_snapshot:
            return
        self._last_snapshot = snapshot

        # Remember which session was previously selected so we can restore
        # the selection even if the list order changes.
        previous_selected_id = self.get_selected_session_id()

        self._session_ids = [s.session_id for s in sessions]

        # Restore selection by session_id if possible; otherwise clamp index.
        if self._session_ids:
            if previous_selected_id is not None and previous_selected_id in self._session_ids:
                self._selected_index = self._session_ids.index(previous_selected_id)
            else:
                self._selected_index = min(self._selected_index, len(self._session_ids) - 1)
        else:
            self._selected_index = 0

        self._refresh_counter += 1
        rc = self._refresh_counter

        # Clear children of the grid and rebuild
        try:
            grid = self.query_one("#manager-grid", Container)
        except NoMatches:
            return

        grid.remove_children()

        if not sessions:
            grid.mount(Static("[dim]No sessions found. Waiting...[/dim]"))
            return

        for i, session in enumerate(sessions):
            is_active = (session.session_id == active_session_id)
            card = SessionCard(
                session,
                is_active=is_active,
                id=f"card-{rc}-{i}",
            )
            card.selected = (i == self._selected_index)
            grid.mount(card)

    def get_selected_session_id(self) -> str | None:
        """Get the session ID of the currently selected card."""
        if self._session_ids and 0 <= self._selected_index < len(self._session_ids):
            return self._session_ids[self._selected_index]
        return None

    def move_selection(self, delta: int) -> None:
        """Move selection by delta (positive=right/down, negative=left/up)."""
        if not self._session_ids:
            return
        new_index = self._selected_index + delta
        new_index = max(0, min(new_index, len(self._session_ids) - 1))
        if new_index != self._selected_index:
            self._selected_index = new_index
            self._update_card_selection()

    def _update_card_selection(self) -> None:
        """Update selected state on all cards."""
        cards = list(self.query("SessionCard"))
        for i, card in enumerate(cards):
            card.selected = (i == self._selected_index)
        # Scroll selected card into view
        if 0 <= self._selected_index < len(cards):
            cards[self._selected_index].scroll_visible()


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


class BettyApp(App):
    """Textual TUI for Betty."""

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
        Binding("I", "toggle_analysis_panel", "Insights", show=False),
        Binding("D", "delete_session", "Delete", show=False),
        Binding("M", "toggle_manager", "Manager", show=False),
        Binding("O", "open_pr", "Open PR", show=False),
        Binding("h", "nav_left", "Left", show=False),
        Binding("l", "nav_right", "Right", show=False),
        Binding("left", "nav_left", "Left", show=False),
        Binding("right", "nav_right", "Right", show=False),
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

    def __init__(self, store: "EventStore", collapse_tools: bool = True, ui_style: str = "rich", manager_mode: bool = False, manager_open_mode: str = "auto") -> None:
        super().__init__()
        self.store = store
        self._collapse_tools = collapse_tools
        self._use_summary = True
        self._filter_index = 0
        self._ui_style = ui_style  # "rich" or "claude-code"
        self._manager_mode = manager_mode  # Started with --manager flag
        self._manager_view_active = manager_mode  # Currently showing manager view
        self._manager_open_mode = manager_open_mode  # "swap" | "expand" | "auto"
        self._manager_expanded = False  # True = side-by-side mode active
        self._focus_panel = "manager"  # "manager" | "detail" — which panel j/k controls
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
        self._show_analysis_panel: bool = False

    def _resolve_open_mode(self) -> str:
        """Resolve 'auto' manager open mode based on terminal width."""
        if self._manager_open_mode != "auto":
            return self._manager_open_mode
        return "expand" if self.size.width >= 120 else "swap"

    def compose(self) -> ComposeResult:
        yield HeaderPanel(id="header")
        with Horizontal(id="main-content"):
            yield ManagerView(id="manager-view")
            with Vertical(id="detail-pane"):
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
        # Show manager view if started with --manager
        if self._manager_mode:
            self._show_manager_view()
        # Set up periodic refresh for summaries
        self.set_interval(0.5, self._check_for_updates)

    def on_resize(self, event) -> None:
        """Handle terminal resize — collapse expand mode if too narrow in auto mode."""
        if self._manager_expanded and self._manager_open_mode == "auto" and self.size.width < 120:
            self._exit_expand_mode()

    def _on_store_turn(self, turn: Turn) -> None:
        """Called from watcher thread - post message for thread safety."""
        self.post_message(self.TurnAdded(turn))

    def _on_store_alert(self, alert: Alert) -> None:
        """Called from watcher thread - post message for thread safety."""
        self.post_message(self.AlertAdded(alert))

    @on(TurnAdded)
    def handle_turn_added(self, message: TurnAdded) -> None:
        """Handle new turn on main thread."""
        if self._manager_expanded:
            # Expand mode: refresh both manager cards and conversation
            self._refresh_manager()
            self._refresh_conversation()
            if self._auto_scroll:
                self.query_one("#conversation", ConversationView).scroll_to_end_if_auto()
            self._refresh_analysis_panel()
        elif self._manager_view_active:
            self._refresh_manager()
        else:
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
        self.store.update_pr_info()
        self._refresh_header()
        if self._manager_expanded:
            self._refresh_manager()
            self._refresh_views()
        elif self._manager_view_active:
            self._refresh_manager()
        else:
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
        if self._manager_view_active:
            # In manager mode, filter out empty sessions (e.g. from claude -r)
            sessions = [s for s in sessions if s.turns]
        active = self.store.get_active_session()
        _, filter_label = self._turn_filters[self._filter_index]
        header = self.query_one("#header", HeaderPanel)
        header.update_header(
            sessions, active, filter_label, self._auto_scroll,
            manager_active=self._manager_view_active,
            manager_expanded=self._manager_expanded,
            focus_panel=self._focus_panel,
        )

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
        manager_view = self.query_one("#manager-view", ManagerView)
        conversation = self.query_one("#conversation", ConversationView)
        detail_pane = self.query_one("#detail-pane", Vertical)

        if self._manager_expanded:
            # Side-by-side: manager (left) + detail pane (right)
            manager_view.set_class(True, "visible")
            manager_view.add_class("expand-visible")
            detail_pane.remove_class("hidden")
            detail_pane.add_class("expand-active")
            conversation.remove_class("hidden")
            tasks_view.set_class(False, "visible")
            plan_view.set_class(False, "visible")
            # Focus indicator (border color changes, no layout shift)
            manager_view.set_class(self._focus_panel == "manager", "panel-focused")
            detail_pane.set_class(self._focus_panel == "detail", "panel-focused")
        elif self._manager_view_active:
            # Full-screen manager
            manager_view.set_class(True, "visible")
            manager_view.remove_class("expand-visible")
            manager_view.remove_class("panel-focused")
            detail_pane.add_class("hidden")
            detail_pane.remove_class("expand-active")
            detail_pane.remove_class("panel-focused")
            conversation.remove_class("hidden")
            tasks_view.set_class(False, "visible")
            plan_view.set_class(False, "visible")
        else:
            # Normal conversation view (no manager)
            manager_view.set_class(False, "visible")
            manager_view.remove_class("expand-visible")
            manager_view.remove_class("panel-focused")
            detail_pane.remove_class("hidden")
            detail_pane.remove_class("expand-active")
            detail_pane.remove_class("panel-focused")
            tasks_view.set_class(self._show_tasks, "visible")
            plan_view.set_class(self._show_plan, "visible")
            conversation.set_class(
                self._show_tasks or self._show_plan, "hidden"
            )

        if self._show_tasks and not self._manager_view_active:
            tasks_view.update_tasks(session)
        if self._show_plan and not self._manager_view_active:
            plan_view.update_plan(session)

    def _refresh_analysis_panel(self) -> None:
        """Update the analysis panel based on toggle state and current selection."""
        panel = self.query_one("#analysis-panel", AnalysisPanel)

        if not self._show_analysis_panel:
            panel.remove_class("visible")
            self._update_span_highlighting()
            return

        panel.add_class("visible")

        conversation = self.query_one("#conversation", ConversationView)
        index = conversation.get_selected_index()

        if index is None:
            panel.show_empty()
            self._update_span_highlighting()
            return

        result = self._get_analysis_turn_range()
        if result is None:
            panel.show_empty()
            self._update_span_highlighting()
            return

        turns, label = result

        if len(turns) == 1 and turns[0].analysis:
            panel.update_analysis(turns[0], turns[0].analysis, label)
        elif len(turns) > 1:
            range_key = (turns[0].turn_number, turns[-1].turn_number)
            range_analysis = self.store.get_range_analysis(range_key)
            if range_analysis:
                panel.update_range_analysis(turns, range_analysis, label)
            else:
                panel.show_empty()
        else:
            panel.show_empty()

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

        self._show_analysis_panel = True
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

        self._show_analysis_panel = True
        self._update_span_highlighting()
        self._refresh_analysis_panel()
        # Show level label
        result = self._get_analysis_turn_range()
        if result:
            _, label = result
            self._show_status(f"Analysis level: {label}")

    def _refresh_manager(self) -> None:
        """Refresh the manager view with current sessions."""
        if not self._manager_view_active:
            return
        sessions = self.store.get_sessions()
        # Filter out empty sessions (e.g. claude -r creates sessions with only
        # file-history-snapshot entries before a conversation actually starts)
        sessions = [s for s in sessions if s.turns]
        active = self.store.get_active_session()
        active_id = active.session_id if active else None
        manager_view = self.query_one("#manager-view", ManagerView)
        manager_view.refresh_cards(sessions, active_id)

    def _show_manager_view(self) -> None:
        """Show the manager view."""
        self._manager_view_active = True
        self._show_tasks = False
        self._show_plan = False
        self._refresh_manager()
        self._refresh_views()
        self._refresh_header()

    def _hide_manager_view(self) -> None:
        """Hide the manager view and return to conversation."""
        self._manager_view_active = False
        self._manager_expanded = False
        self._focus_panel = "manager"
        self._refresh_views()
        self._refresh_header()
        self._refresh_conversation()

    def _enter_expand_mode(self) -> None:
        """Enter side-by-side expand mode (manager + detail)."""
        self._manager_expanded = True
        self._manager_view_active = True
        self._focus_panel = "detail"
        self._show_tasks = False
        self._show_plan = False
        self._refresh_manager()
        self._refresh_views()
        self._refresh_header()
        self._refresh_conversation()

    def _exit_expand_mode(self) -> None:
        """Exit expand mode. Goes to manager or conversation based on focus."""
        was_detail_focused = self._focus_panel == "detail"
        self._manager_expanded = False
        self._focus_panel = "manager"
        if was_detail_focused:
            # Detail was focused — switch to full conversation view
            self._hide_manager_view()
        else:
            # Manager was focused — return to full-width manager
            self._refresh_views()
            self._refresh_header()

    def _show_status(self, message: str) -> None:
        """Show a status message in the footer."""
        self._status_message = message
        self.notify(message, timeout=3)

    # Action handlers
    def _nav_conversation_down(self) -> None:
        """Navigate down in conversation view."""
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

    def _nav_conversation_up(self) -> None:
        """Navigate up in conversation view."""
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

    def action_nav_down(self) -> None:
        """Navigate down in conversation or manager grid."""
        if self._manager_expanded and self._focus_panel == "detail":
            self._nav_conversation_down()
            return
        if self._manager_view_active:
            manager = self.query_one("#manager-view", ManagerView)
            cols = 1 if self._manager_expanded else 3
            manager.move_selection(cols)
            return
        self._nav_conversation_down()

    def action_nav_up(self) -> None:
        """Navigate up in conversation or manager grid."""
        if self._manager_expanded and self._focus_panel == "detail":
            self._nav_conversation_up()
            return
        if self._manager_view_active:
            manager = self.query_one("#manager-view", ManagerView)
            cols = 1 if self._manager_expanded else 3
            manager.move_selection(-cols)
            return
        self._nav_conversation_up()

    def action_nav_left(self) -> None:
        """Navigate left in manager grid or switch focus panel."""
        if self._manager_expanded:
            if self._focus_panel == "detail":
                self._focus_panel = "manager"
                self._refresh_header()
                self._refresh_views()
            else:
                manager = self.query_one("#manager-view", ManagerView)
                manager.move_selection(-1)
            return
        if self._manager_view_active:
            manager = self.query_one("#manager-view", ManagerView)
            manager.move_selection(-1)

    def action_nav_right(self) -> None:
        """Navigate right in manager grid or switch focus panel."""
        if self._manager_expanded:
            if self._focus_panel == "manager":
                self._focus_panel = "detail"
                self._refresh_header()
                self._refresh_views()
            else:
                # Already in detail, no-op for right
                pass
            return
        if self._manager_view_active:
            manager = self.query_one("#manager-view", ManagerView)
            manager.move_selection(1)

    def action_go_top(self) -> None:
        """Go to top of conversation or manager grid."""
        if self._manager_expanded and self._focus_panel == "manager":
            manager = self.query_one("#manager-view", ManagerView)
            if manager._session_ids:
                manager._selected_index = 0
                manager._update_card_selection()
            return
        if self._manager_view_active and not self._manager_expanded:
            manager = self.query_one("#manager-view", ManagerView)
            if manager._session_ids:
                manager._selected_index = 0
                manager._update_card_selection()
            return
        self._auto_scroll = False
        self._reset_analysis_level()
        conversation = self.query_one("#conversation", ConversationView)
        widgets = list(conversation.query("TurnWidget, ToolGroupWidget"))
        if widgets:
            conversation.set_selected_index(0)
            conversation.scroll_home()
        self._refresh_analysis_panel()

    def action_go_bottom(self) -> None:
        """Go to bottom of conversation or manager grid."""
        if self._manager_expanded and self._focus_panel == "manager":
            manager = self.query_one("#manager-view", ManagerView)
            if manager._session_ids:
                manager._selected_index = len(manager._session_ids) - 1
                manager._update_card_selection()
            return
        if self._manager_view_active and not self._manager_expanded:
            manager = self.query_one("#manager-view", ManagerView)
            if manager._session_ids:
                manager._selected_index = len(manager._session_ids) - 1
                manager._update_card_selection()
            return
        self._auto_scroll = True
        self._reset_analysis_level()
        conversation = self.query_one("#conversation", ConversationView)
        widgets = list(conversation.query("TurnWidget, ToolGroupWidget"))
        if widgets:
            conversation.set_selected_index(len(widgets) - 1)
            conversation.scroll_end()
        self._refresh_analysis_panel()

    def action_toggle_expand(self) -> None:
        """Toggle expand on selected item, or open session in manager view."""
        if self._manager_view_active and not (self._manager_expanded and self._focus_panel == "detail"):
            self._open_selected_session()
            return
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

    def _open_selected_session(self) -> None:
        """Open the selected session from manager view."""
        manager = self.query_one("#manager-view", ManagerView)
        session_id = manager.get_selected_session_id()
        if not session_id:
            return

        if self._manager_expanded and self._focus_panel == "manager":
            # In expand mode with manager focus: switch active session in place
            if self.store.set_active_session_by_id(session_id):
                self._auto_scroll = True
                self._group_expanded_state.clear()
                self._refresh_conversation()
                self._refresh_manager()
                self._refresh_header()
            return

        if not self.store.set_active_session_by_id(session_id):
            return

        self._auto_scroll = True
        self._group_expanded_state.clear()

        mode = self._resolve_open_mode()
        if mode == "expand":
            self._enter_expand_mode()
        else:
            self._hide_manager_view()

    def action_toggle_manager(self) -> None:
        """Toggle between manager view and conversation view."""
        if self._manager_view_active:
            self._hide_manager_view()
        else:
            self._show_manager_view()

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
        if self._manager_view_active:
            return
        self._show_tasks = not self._show_tasks
        self._show_plan = False
        self._refresh_views()
        self._show_status("Showing task list" if self._show_tasks else "Showing conversation")

    def action_toggle_plan(self) -> None:
        """Toggle plan view."""
        if self._manager_view_active:
            return
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
        """Analyze the selected turn or range."""
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
            # Single-turn analysis
            turn = turns[0]
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
            submitted = self.store.analyze_range(turns)
            if submitted:
                self._show_status(f"Analyzing {label}...")
            else:
                self._show_status(f"Analysis loaded for {label}")

        self._show_analysis_panel = True
        self._refresh_analysis_panel()

    def action_toggle_analysis_panel(self) -> None:
        """Toggle the insights panel on/off."""
        self._show_analysis_panel = not self._show_analysis_panel
        if not self._show_analysis_panel:
            self._reset_analysis_level()
        self._refresh_analysis_panel()
        label = "shown" if self._show_analysis_panel else "hidden"
        self._show_status(f"Insights panel {label}")

    def action_open_pr(self) -> None:
        """Open the PR associated with the selected/active session in a browser."""
        import webbrowser

        session = None
        if self._manager_view_active and not self._manager_expanded:
            # Manager mode: use selected card's session
            manager_view = self.query_one("#manager-view", ManagerView)
            sid = manager_view.get_selected_session_id()
            if sid:
                for s in self.store.get_sessions():
                    if s.session_id == sid:
                        session = s
                        break
        else:
            session = self.store.get_active_session()

        if not session:
            self._show_status("No session selected")
            return
        if not session.pr_info:
            self._show_status("No PR linked to this session")
            return

        webbrowser.open(session.pr_info.url)
        self._show_status(f"Opened PR #{session.pr_info.number}")

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
        # Close analysis panel first
        if self._show_analysis_panel:
            self._show_analysis_panel = False
            self._refresh_analysis_panel()
            return
        # Always close tasks/plan overlays first, regardless of mode
        if self._show_tasks:
            self._show_tasks = False
            self._refresh_views()
            return
        if self._show_plan:
            self._show_plan = False
            self._refresh_views()
            return
        # Exit expand mode back to full-width manager
        if self._manager_expanded:
            self._exit_expand_mode()
            return
        # Return to manager view if started with --manager and viewing a session
        if not self._manager_view_active and self._manager_mode:
            self._show_manager_view()
            return
        # Close manager view if it's open (toggled via M key)
        if self._manager_view_active:
            self._hide_manager_view()
            return
        # Clear selection
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
