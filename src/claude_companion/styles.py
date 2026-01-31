"""UI style renderers for Claude Companion TUI."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from .models import Session, Turn, TurnGroup


class StyleRenderer(ABC):
    """Base class for UI style renderers."""

    name: str
    filters: list[tuple[str, str]]

    @abstractmethod
    def render_turn(
        self, turn: "Turn", is_selected: bool, conv_turn: int, use_summary: bool
    ) -> Text | Group | Panel:
        """Render a single turn."""
        ...

    @abstractmethod
    def render_turn_group(
        self, group: "TurnGroup", is_selected: bool, conv_turn: int
    ) -> Panel | Group:
        """Render a turn group (assistant + tools)."""
        ...

    @abstractmethod
    def render_header(
        self,
        sessions_text: str,
        active: "Session | None",
        filter_label: str,
        auto_scroll: bool,
    ) -> Panel:
        """Render the header panel."""
        ...

    @abstractmethod
    def render_status_line(self, monitor_text: str) -> Panel | Table:
        """Render the status/monitor line."""
        ...

    @abstractmethod
    def get_alert_indicator(self, danger_count: int, warn_count: int) -> str:
        """Get the alert indicator for the footer."""
        ...

    @abstractmethod
    def format_alert_line(self, level: str, title: str, message: str) -> str:
        """Format a single alert line."""
        ...


class RichStyle(StyleRenderer):
    """Rich style with boxes and emojis."""

    name = "rich"
    filters = [
        ("all", "All"),
        ("tool", "Tools only"),
        ("Read", "📄 Read"),
        ("Write", "✏️ Write"),
        ("Edit", "✏️ Edit"),
        ("Bash", "💻 Bash"),
    ]

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

    def _get_tool_icon(self, tool_name: str | None) -> str:
        """Get icon for a tool."""
        if not tool_name:
            return self.TOOL_ICONS["default"]
        return self.TOOL_ICONS.get(tool_name, self.TOOL_ICONS["default"])

    def render_turn_group(
        self, group: "TurnGroup", is_selected: bool, conv_turn: int
    ) -> Panel:
        """Render a turn group in rich boxed style."""
        history_prefix = "◷ " if group.assistant_turn.is_historical else ""
        timestamp_str = group.assistant_turn.timestamp.strftime("%H:%M:%S")

        # Title with tools count (tools first)
        icon = self.ROLE_ICONS["assistant"]
        title = f"{history_prefix}Turn {group.assistant_turn.turn_number} │ {group.tool_count} tools + {icon} │ {timestamp_str}"

        border_style = "green" if not group.assistant_turn.is_historical else "dim green"
        if is_selected:
            title = f"► {title}"
            border_style = "white"

        # Content
        if group.expanded:
            # Show tools first, then full assistant text
            content_parts = [
                Text(f"Tools used ({group.tool_count}):", style="dim"),
            ]
            for tool_turn in group.tool_turns:
                icon = self._get_tool_icon(tool_turn.tool_name)
                content_parts.append(Text(f"  {icon} [{tool_turn.tool_name}] {tool_turn.content_preview}", style="dim"))
            content_parts.extend([
                Text(""),
                Text("Full response:", style="dim"),
                Markdown(group.assistant_turn.content_full),
            ])
        else:
            # Show ONLY the summary (which covers both assistant text and tools)
            content_parts = [Markdown(f"**\\[tldr]** {group.assistant_turn.summary}")]

        subtitle = f"{group.assistant_turn.word_count:,} words"

        return Panel(
            Group(*content_parts),
            title=title,
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style=border_style,
            padding=(0, 1),
        )

    def render_turn(
        self, turn: "Turn", is_selected: bool, conv_turn: int, use_summary: bool
    ) -> Panel:
        """Render a turn in default style (boxes with emojis)."""
        history_prefix = "◷ " if turn.is_historical else ""
        timestamp_str = turn.timestamp.strftime("%H:%M:%S")

        if turn.role == "user":
            icon = self.ROLE_ICONS["user"]
            title = f"{history_prefix}Turn {turn.turn_number} │ {icon} User │ {timestamp_str}"
            border_style = "blue" if not turn.is_historical else "dim blue"
        elif turn.role == "assistant":
            icon = self.ROLE_ICONS["assistant"]
            title = f"{history_prefix}Turn {turn.turn_number} │ {icon} Assistant │ {timestamp_str}"
            border_style = "green" if not turn.is_historical else "dim green"
        else:
            icon = self._get_tool_icon(turn.tool_name)
            title = f"{history_prefix}Turn {turn.turn_number} │ {icon} {turn.tool_name or 'Tool'} │ {timestamp_str}"
            if turn.tool_name in ("Write", "Edit"):
                border_style = "yellow" if not turn.is_historical else "dim yellow"
            elif turn.tool_name == "Bash":
                border_style = "magenta" if not turn.is_historical else "dim magenta"
            else:
                border_style = "cyan" if not turn.is_historical else "dim cyan"

        if turn.expanded:
            content = turn.content_full
            expand_indicator = "\\[-] "
        else:
            if turn.role == "assistant" and use_summary:
                if turn.summary:
                    content = turn.summary
                    expand_indicator = "\\[tldr;] "
                else:
                    content = f"{turn.content_preview} *\\[summarizing...]*"
                    expand_indicator = (
                        "\\[+] " if turn.content_full != turn.content_preview else ""
                    )
            else:
                content = turn.content_preview
                expand_indicator = (
                    "\\[+] " if turn.content_full != turn.content_preview else ""
                )

        if is_selected:
            title = f"► {title}"
            border_style = "white"

        subtitle = None
        if turn.role in ("user", "assistant"):
            subtitle = f"{turn.word_count:,} words"

        if turn.role == "assistant":
            panel_content = Markdown(f"{expand_indicator}{content}")
        else:
            panel_content = Text(f"{expand_indicator}{content}")

        return Panel(
            panel_content,
            title=title,
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style=border_style,
            padding=(0, 1),
        )

    def render_header(
        self,
        sessions_text: str,
        active: "Session | None",
        filter_label: str,
        auto_scroll: bool,
    ) -> Panel:
        """Render the header panel."""
        if active:
            scroll_indicator = "" if auto_scroll else " [dim](paused)[/dim]"
            historical_count = sum(1 for t in active.turns if t.is_historical)
            live_count = len(active.turns) - historical_count
            turns_info = (
                f"◷{historical_count}+{live_count}" if historical_count else f"{live_count}"
            )
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

    def render_status_line(self, monitor_text: str) -> Table:
        """Render two input boxes side by side."""
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)

        monitor_content = (
            monitor_text if monitor_text else "[dim]Press \\[m] to set monitoring rules[/dim]"
        )
        monitor_box = Panel(
            monitor_content,
            title="[cyan]📋 Monitor[/cyan]",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
            style="on grey7",
        )

        ask_box = Panel(
            "[dim]Press \\[?] to ask about trace[/dim]",
            title="[magenta]❓ Ask[/magenta]",
            title_align="left",
            border_style="magenta",
            padding=(0, 1),
            style="on grey7",
        )

        table.add_row(monitor_box, ask_box)
        return table

    def get_alert_indicator(self, danger_count: int, warn_count: int) -> str:
        """Get the alert indicator for the footer."""
        if danger_count > 0:
            return f"[bold red]🚨{danger_count}[/bold red] "
        elif warn_count > 0:
            return f"[yellow]⚠️{warn_count}[/yellow] "
        return ""

    def format_alert_line(self, level: str, title: str, message: str) -> str:
        """Format a single alert line."""
        if level == "danger":
            indicator = "🚨"
            style = "bold red"
        else:
            indicator = "⚠️ "
            style = "yellow"
        return f"[{style}]{indicator} {title}[/{style}]: {message}"


class ClaudeCodeStyle(StyleRenderer):
    """Minimal style matching Claude Code aesthetic."""

    name = "claude-code"
    filters = [
        ("all", "All"),
        ("tool", "Tools"),
        ("Read", "Read"),
        ("Write", "Write"),
        ("Edit", "Edit"),
        ("Bash", "Bash"),
    ]

    BULLET = "⏺"
    BULLET_STYLES = {
        "assistant": "white",
        "tool": "#5fd787",  # Matches Claude Code's green bullet
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

    def _get_tool_indicator(self, tool_name: str | None) -> str:
        """Get text indicator for a tool."""
        if not tool_name:
            return self.TOOL_INDICATORS["default"]
        return self.TOOL_INDICATORS.get(tool_name, self.TOOL_INDICATORS["default"])

    def render_turn_group(
        self, group: "TurnGroup", is_selected: bool, conv_turn: int
    ) -> Group:
        """Render a turn group in minimal style."""
        selected_style = "light_steel_blue" if is_selected else ""
        bullet_style = selected_style or self.BULLET_STYLES["assistant"]

        parts = []

        if group.expanded:
            # Header line (bullet at top)
            header_row = Table.grid(padding=(0, 0))
            header_row.add_column(width=2)
            header_row.add_column()
            header_row.add_row(
                Text(f"{self.BULLET} ", style=bullet_style),
                Markdown(f"tools:{group.tool_count}", style=selected_style),
            )
            parts.append(header_row)

            # Tool turns (indented)
            for tool_turn in group.tool_turns:
                tool_indicator = self._get_tool_indicator(tool_turn.tool_name)
                tool_row = Table.grid(padding=(0, 0))
                tool_row.add_column(width=4)  # Extra indent
                tool_row.add_column()
                tool_text = Text()
                tool_text.append(f"[{tool_indicator}] ", style="bold" if not selected_style else selected_style)
                tool_text.append(tool_turn.content_preview, style=selected_style or "dim")
                tool_row.add_row(Text("  "), tool_text)
                parts.append(tool_row)

            # Full assistant content with [-] indicator (in second column)
            content_row = Table.grid(padding=(0, 0))
            content_row.add_column(width=2)
            content_row.add_column()
            content_row.add_row(
                Text(""),
                Markdown(f"**[-]** {group.assistant_turn.content_full}", style=selected_style),
            )
            parts.append(content_row)
        else:
            # When collapsed: show summary only
            content = group.assistant_turn.summary or group.assistant_turn.content_preview
            md_content = f"tools:{group.tool_count} + **[tldr]** {content}"
            row = Table.grid(padding=(0, 0))
            row.add_column(width=2)
            row.add_column()
            row.add_row(
                Text(f"{self.BULLET} ", style=bullet_style),
                Markdown(md_content, style=selected_style),
            )
            parts.append(row)

        # Add timestamp/metadata line
        if conv_turn > 0:
            timestamp_str = group.assistant_turn.timestamp.strftime("%H:%M")
            status = f"── turn {conv_turn}, {group.assistant_turn.word_count} words, {timestamp_str} ──"
            parts.append(Text(status, style="dim", justify="right"))

        return Group(*parts)

    def render_turn(
        self, turn: "Turn", is_selected: bool, conv_turn: int, use_summary: bool
    ) -> Text | Group:
        """Render a turn in Claude Code style (minimal with bullets)."""
        selected_text_style = "light_steel_blue" if is_selected else ""

        # User turns: "❯ message" with dim background
        if turn.role == "user":
            content = turn.content_full if turn.expanded else turn.content_preview
            indicator_style = selected_text_style or "dim"
            content_style = selected_text_style or "on grey15"
            row = Table.grid(padding=(0, 0))
            row.add_column(width=2)
            row.add_column()
            row.add_row(
                Text("❯ ", style=indicator_style),
                Text(content, style=content_style),
            )
            return Group(row, Text(""))

        # Assistant turns: render as Markdown with colored bullet
        if turn.role == "assistant":
            bullet_style = selected_text_style or self.BULLET_STYLES["assistant"]
            if turn.expanded:
                indicator = "\\[-]"
                content = turn.content_full
            elif use_summary and turn.summary:
                indicator = "\\[tldr]"
                content = turn.summary
            elif use_summary and not turn.summary:
                indicator = "\\[+]"
                content = f"{turn.content_preview} *summarizing...*"
            elif turn.content_full != turn.content_preview:
                indicator = "\\[+]"
                content = turn.content_preview
            else:
                indicator = ""
                content = turn.content_preview

            md_content = f"**{indicator}** {content}" if indicator else content
            row = Table.grid(padding=(0, 0))
            row.add_column(width=2)
            row.add_column()
            row.add_row(
                Text(f"{self.BULLET} ", style=bullet_style),
                Markdown(md_content, style=selected_text_style),
            )

            parts = [row]
            if conv_turn > 0:
                timestamp_str = turn.timestamp.strftime("%H:%M")
                status = f"── turn {conv_turn}, {turn.word_count} words, {timestamp_str} ──"
                parts.append(Text(status, style="dim", justify="right"))
            return Group(*parts)

        # Tool turns: render as Text to preserve verbatim output
        bullet_style = selected_text_style or self.BULLET_STYLES["tool"]
        tool_indicator = self._get_tool_indicator(turn.tool_name)
        content = turn.content_full if turn.expanded else turn.content_preview

        row = Table.grid(padding=(0, 0))
        row.add_column(width=2)
        row.add_column()

        tool_text = Text()
        tool_text.append(
            f"[{tool_indicator}] ",
            style="bold" if not selected_text_style else selected_text_style,
        )
        tool_text.append(content, style=selected_text_style or "dim")

        row.add_row(
            Text(f"{self.BULLET} ", style=bullet_style),
            tool_text,
        )
        return Group(row, Text(""))

    def render_header(
        self,
        sessions_text: str,
        active: "Session | None",
        filter_label: str,
        auto_scroll: bool,
    ) -> Panel:
        """Render the header panel."""
        if active:
            scroll_indicator = "" if auto_scroll else " [dim](paused)[/dim]"
            historical_count = sum(1 for t in active.turns if t.is_historical)
            live_count = len(active.turns) - historical_count
            turns_info = (
                f"h{historical_count}+{live_count}" if historical_count else f"{live_count}"
            )
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

    def render_status_line(self, monitor_text: str) -> Panel:
        """Render a single status line."""
        if monitor_text:
            content = monitor_text
        else:
            content = "[dim]Press \\[m] to set[/dim]"
        return Panel(
            content,
            title="[bold]Monitor[/bold]",
            title_align="left",
            border_style="dim",
        )

    def get_alert_indicator(self, danger_count: int, warn_count: int) -> str:
        """Get the alert indicator for the footer."""
        if danger_count > 0:
            return f"[bold red]\\[!{danger_count}][/bold red] "
        elif warn_count > 0:
            return f"[yellow]\\[*{warn_count}][/yellow] "
        return ""

    def format_alert_line(self, level: str, title: str, message: str) -> str:
        """Format a single alert line."""
        if level == "danger":
            indicator = "\\[!]"
            style = "bold red"
        else:
            indicator = "\\[*]"
            style = "yellow"
        return f"[{style}]{indicator} {title}[/{style}]: {message}"


# Style registry
STYLES: dict[str, type[StyleRenderer]] = {
    "rich": RichStyle,
    "claude-code": ClaudeCodeStyle,
}


def get_style(name: str) -> StyleRenderer:
    """Get a style renderer instance by name."""
    style_class = STYLES.get(name, RichStyle)
    return style_class()
