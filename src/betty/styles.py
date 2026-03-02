"""UI style renderers for Betty TUI."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from .models import Session, Turn, ToolGroup


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
    def render_tool_group(
        self, group: "ToolGroup", is_selected: bool
    ) -> Panel | Group:
        """Render a tool group (consecutive tools with their own summary)."""
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

    def render_tool_group(
        self, group: "ToolGroup", is_selected: bool
    ) -> Panel | Group:
        """Render a tool group in rich boxed style (green border, separate from assistant)."""
        first_tool = group.tool_turns[0] if group.tool_turns else None

        # If collapsed and has summary, show as a single panel with summary
        if not group.expanded and group.summary:
            history_prefix = "◷ " if (first_tool and first_tool.is_historical) else ""
            timestamp_str = first_tool.timestamp.strftime("%H:%M:%S") if first_tool else ""

            title = f"{history_prefix}{group.tool_count} tools │ {timestamp_str}"
            border_style = "#5fd787" if not (first_tool and first_tool.is_historical) else "dim green"
            if is_selected:
                title = f"► {title}"
                border_style = "white"

            return Panel(
                Markdown(f"**tools:{group.tool_count}** {group.summary}"),
                title=title,
                title_align="left",
                border_style=border_style,
                padding=(0, 1),
            )

        # Otherwise (expanded or no summary yet), show individual tools in one panel
        content_parts = []
        for i, tool_turn in enumerate(group.tool_turns):
            icon = self._get_tool_icon(tool_turn.tool_name)
            content = tool_turn.content_full if tool_turn.expanded else tool_turn.content_preview
            content_parts.append(Text(f"{icon} [{tool_turn.tool_name}] {content}", style="dim"))
            if i < len(group.tool_turns) - 1:
                content_parts.append(Text(""))  # Blank line between tools

        history_prefix = "◷ " if (first_tool and first_tool.is_historical) else ""
        timestamp_str = first_tool.timestamp.strftime("%H:%M:%S") if first_tool else ""
        title = f"{history_prefix}{group.tool_count} tools │ {timestamp_str}"
        border_style = "#5fd787" if not (first_tool and first_tool.is_historical) else "dim green"
        if is_selected:
            title = f"► {title}"
            border_style = "white"

        return Panel(
            Group(*content_parts),
            title=title,
            title_align="left",
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
                    expand_indicator = "\\[+] "
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

        # Add critic sentiment indicator for assistant turns
        if turn.role == "assistant" and turn.critic_sentiment:
            sentiment_map = {
                "progress": "[green]✓[/green]",
                "concern": "[yellow]⚠[/yellow]",
                "critical": "[red]✗[/red]",
            }
            indicator = sentiment_map.get(turn.critic_sentiment, "")
            if indicator and subtitle:
                subtitle = f"{subtitle} {indicator}"

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

    def render_tool_group(
        self, group: "ToolGroup", is_selected: bool
    ) -> Group:
        """Render a tool group in minimal style (green bullet, separate from assistant)."""
        selected_style = "light_steel_blue" if is_selected else ""
        bullet_style = selected_style or self.BULLET_STYLES["tool"]

        # If collapsed and has summary, show as single line with summary
        if not group.expanded and group.summary:
            md_content = f"**tools:{group.tool_count}** {group.summary}"
            row = Table.grid(padding=(0, 0))
            row.add_column(width=2)
            row.add_column()
            row.add_row(
                Text(f"{self.BULLET} ", style=bullet_style),
                Markdown(md_content, style=selected_style),
            )
            return Group(row, Text(""))

        # Otherwise (expanded or no summary yet), show individual tools with spacing
        parts = []
        for tool_turn in group.tool_turns:
            tool_indicator = self._get_tool_indicator(tool_turn.tool_name)
            content = tool_turn.content_full if tool_turn.expanded else tool_turn.content_preview

            row = Table.grid(padding=(0, 0))
            row.add_column(width=2)
            row.add_column()
            tool_text = Text()
            tool_text.append(f"[{tool_indicator}] ", style="bold" if not selected_style else selected_style)
            tool_text.append(content, style=selected_style or "dim")
            row.add_row(
                Text(f"{self.BULLET} ", style=bullet_style),
                tool_text,
            )
            parts.append(row)
            parts.append(Text(""))  # Blank line after each tool

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
                indicator = "\\[+]"
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
                sentiment_indicator = ""
                if turn.critic_sentiment:
                    sentiment_map = {
                        "progress": " ✓",
                        "concern": " ⚠",
                        "critical": " ✗",
                    }
                    sentiment_indicator = sentiment_map.get(turn.critic_sentiment, "")
                sentiment_style = {
                    "progress": "green",
                    "concern": "yellow",
                    "critical": "red",
                }.get(turn.critic_sentiment or "", "dim")
                status_text = Text(justify="right")
                # Split to color only the sentiment indicator
                base = f"── turn {conv_turn}, {turn.word_count} words, {timestamp_str}"
                status_text.append(base, style="dim")
                if sentiment_indicator:
                    status_text.append(sentiment_indicator, style=sentiment_style)
                status_text.append(" ──", style="dim")
                parts.append(status_text)
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


class BerryStyle(StyleRenderer):
    """Berry-themed style with purple/pink palette and blueberry character."""

    name = "berry"
    filters = [
        ("all", "All"),
        ("tool", "Tools"),
        ("Read", "📖 Read"),
        ("Write", "✏️ Write"),
        ("Edit", "✏️ Edit"),
        ("Bash", "💻 Bash"),
    ]

    # Re-use RichStyle's rendering logic for turns/tool groups
    _rich = None

    def _get_rich(self) -> RichStyle:
        if self._rich is None:
            self._rich = RichStyle()
        return self._rich

    def render_turn(
        self, turn: "Turn", is_selected: bool, conv_turn: int, use_summary: bool
    ) -> Text | Group | Panel:
        return self._get_rich().render_turn(turn, is_selected, conv_turn, use_summary)

    def render_tool_group(
        self, group: "ToolGroup", is_selected: bool
    ) -> Panel | Group:
        return self._get_rich().render_tool_group(group, is_selected)

    def render_header(
        self,
        sessions_text: str,
        active: "Session | None",
        filter_label: str,
        auto_scroll: bool,
    ) -> Panel:
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
            title="[bold #c084fc]🫐 Berry[/bold #c084fc]",
            title_align="left",
            border_style="#9333ea",
        )

    def render_status_line(self, monitor_text: str) -> Table:
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)

        monitor_content = (
            monitor_text if monitor_text else "[dim]Press \\[m] to set monitoring rules[/dim]"
        )
        monitor_box = Panel(
            monitor_content,
            title="[#c084fc]📋 Monitor[/#c084fc]",
            title_align="left",
            border_style="#a78bfa",
            padding=(0, 1),
        )

        ask_box = Panel(
            "[dim]Press \\[?] to ask about trace[/dim]",
            title="[#f472b6]❓ Ask[/#f472b6]",
            title_align="left",
            border_style="#f472b6",
            padding=(0, 1),
        )

        table.add_row(monitor_box, ask_box)
        return table

    def get_alert_indicator(self, danger_count: int, warn_count: int) -> str:
        if danger_count > 0:
            return f"[bold red]🚨{danger_count}[/bold red] "
        elif warn_count > 0:
            return f"[yellow]⚠️{warn_count}[/yellow] "
        return ""

    def format_alert_line(self, level: str, title: str, message: str) -> str:
        if level == "danger":
            indicator = "🚨"
            style = "bold red"
        else:
            indicator = "⚠️ "
            style = "yellow"
        return f"[{style}]{indicator} {title}[/{style}]: {message}"


# Style registry
STYLES: dict[str, type[StyleRenderer]] = {
    "rich": RichStyle,
    "claude-code": ClaudeCodeStyle,
    "berry": BerryStyle,
}


def get_style(name: str) -> StyleRenderer:
    """Get a style renderer instance by name."""
    style_class = STYLES.get(name, RichStyle)
    return style_class()
