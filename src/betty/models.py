"""Data models for Betty."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .analyzer import Analysis
    from .github import PRInfo


class DetailLevel(Enum):
    """Detail level for conversation display."""
    OVERVIEW = "overview"
    DEFAULT = "default"
    DETAILED = "detailed"


def count_words(text: str) -> int:
    """Count words in text as a proxy for tokens."""
    if not text:
        return 0
    return len(text.split())


@dataclass
class Turn:
    """A turn in the conversation (user message, assistant response, or tool use)."""

    turn_number: int
    role: str  # "user" | "assistant" | "tool"
    content_preview: str
    content_full: str
    word_count: int
    input_tokens: int | None = None       # From usage.input_tokens
    output_tokens: int | None = None      # From usage.output_tokens
    cache_creation_tokens: int | None = None  # From usage.cache_creation_input_tokens
    cache_read_tokens: int | None = None  # From usage.cache_read_input_tokens
    model_id: str | None = None           # From message.model
    tool_name: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    expanded: bool = False
    is_historical: bool = False  # True if loaded from transcript history
    summary: str | None = None  # LLM-generated summary for assistant turns
    critic: str | None = None  # LLM-generated critique for assistant turns
    critic_sentiment: str | None = None  # "progress" | "concern" | "critical"
    task_operation: tuple[str, dict[str, Any]] | None = None  # (operation_type, data) for task tools
    annotation: str | None = None  # User-provided annotation
    analysis: "Analysis | None" = None  # On-demand LLM analysis


@dataclass
class TaskState:
    """State of a task from TaskCreate/TaskUpdate operations."""

    task_id: str
    subject: str
    description: str
    status: str  # "pending" | "in_progress" | "completed" | "deleted"
    activeForm: str | None = None
    owner: str | None = None
    blockedBy: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_deleted(self) -> bool:
        """Check if task is deleted."""
        return self.status == "deleted"


def parse_task_operation(tool_name: str, tool_input: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Parse Task tool operations and return (operation_type, data)."""
    if tool_name == "TaskCreate":
        return ("create", {
            "subject": tool_input.get("subject", ""),
            "description": tool_input.get("description", ""),
            "activeForm": tool_input.get("activeForm"),
        })
    elif tool_name == "TaskUpdate":
        return ("update", {
            "taskId": tool_input.get("taskId", ""),
            "status": tool_input.get("status"),
            "subject": tool_input.get("subject"),
            "description": tool_input.get("description"),
            "activeForm": tool_input.get("activeForm"),
            "owner": tool_input.get("owner"),
            "addBlockedBy": tool_input.get("addBlockedBy", []),
            "addBlocks": tool_input.get("addBlocks", []),
        })
    return None  # TaskGet, TaskList are read-only


def _extract_tool_content(tool_name: str | None, tool_input: dict[str, Any]) -> str:
    """Extract displayable content from tool input."""
    if not tool_name:
        return str(tool_input)

    if tool_name == "Read":
        return tool_input.get("file_path", str(tool_input))
    elif tool_name == "Write":
        path = tool_input.get("file_path", "")
        content = tool_input.get("content", "")
        lines = len(content.splitlines()) if content else 0
        return f"{path} ({lines} lines)"
    elif tool_name == "Edit":
        path = tool_input.get("file_path", "")
        old = tool_input.get("old_string", "")
        new = tool_input.get("new_string", "")
        return f"{path} (-{len(old.splitlines())}, +{len(new.splitlines())})"
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        desc = tool_input.get("description", "")
        return f"{cmd}" + (f" # {desc}" if desc else "")
    elif tool_name == "Glob":
        return tool_input.get("pattern", str(tool_input))
    elif tool_name == "Grep":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", ".")
        return f"/{pattern}/ in {path}"
    elif tool_name == "TaskCreate":
        subject = tool_input.get("subject", "")
        return f"Create task: {subject}"
    elif tool_name == "TaskUpdate":
        task_id = tool_input.get("taskId", "")
        status = tool_input.get("status", "")
        return f"Update task {task_id}: {status}" if status else f"Update task {task_id}"
    elif tool_name in ("Task", "TaskGet", "TaskList"):
        return tool_input.get("description", str(tool_input))
    else:
        return str(tool_input)[:200]


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max length."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


@dataclass
class ToolGroup:
    """A display group for consecutive tool turns with a summary.

    Separate from assistant turns - tools have their own identity and summary.
    """

    tool_turns: list[Turn]  # Consecutive tool turns
    summary: str | None = None  # LLM summary for these tools
    expanded: bool = False  # Group-level expand state

    @property
    def tool_count(self) -> int:
        return len(self.tool_turns)

    @property
    def first_turn_number(self) -> int:
        """Get turn number of first tool (used for tracking expanded state)."""
        return self.tool_turns[0].turn_number if self.tool_turns else 0

    @property
    def tool_names_preview(self) -> str:
        """Get comma-separated tool names for preview."""
        names = [t.tool_name or "tool" for t in self.tool_turns]
        return ", ".join(names[:3]) + ("..." if len(names) > 3 else "")


@dataclass
class SpanGroup:
    """A display group for a user turn and all its response turns (assistant + tool)."""

    user_turn: Turn | None          # The user turn that starts this span (None if span starts with non-user turns)
    response_turns: list[Turn]      # All assistant + tool turns in this span
    expanded: bool = False

    @property
    def first_turn_number(self) -> int:
        if self.user_turn:
            return self.user_turn.turn_number
        return self.response_turns[0].turn_number if self.response_turns else 0

    @property
    def response_summary(self) -> str:
        """One-line summary: assistant preview + tool count."""
        assistant_turns = [t for t in self.response_turns if t.role == "assistant"]
        tool_turns = [t for t in self.response_turns if t.role == "tool"]
        parts = []
        if assistant_turns:
            preview = assistant_turns[0].content_preview[:80]
            parts.append(preview)
        if tool_turns:
            names = [t.tool_name or "tool" for t in tool_turns]
            unique = list(dict.fromkeys(names))  # preserve order, dedupe
            tool_str = ", ".join(unique[:3]) + ("..." if len(unique) > 3 else "")
            parts.append(f"{len(tool_turns)} tools ({tool_str})")
        return " | ".join(parts) if parts else "no response"

    @property
    def total_turns(self) -> int:
        return (1 if self.user_turn else 0) + len(self.response_turns)


def compute_spans(turns: list[Turn]) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx) spans.

    A span starts at each user turn and extends to just before the next user turn.
    If the session starts with non-user turns, they form a span of their own.
    Returns inclusive ranges into session.turns.
    """
    if not turns:
        return []

    spans: list[tuple[int, int]] = []
    current_start = 0

    for i, turn in enumerate(turns):
        if turn.role == "user" and i > current_start:
            # Close previous span
            spans.append((current_start, i - 1))
            current_start = i

    # Close final span
    spans.append((current_start, len(turns) - 1))

    return spans


@dataclass
class Session:
    """A Claude Code session."""

    session_id: str
    project_path: str = ""
    model: str = "unknown"
    started_at: datetime = field(default_factory=datetime.now)
    turns: list[Turn] = field(default_factory=list)
    tasks: dict[str, TaskState] = field(default_factory=dict)
    plan_content: str | None = None  # Full markdown content of plan
    plan_file_path: str | None = None  # Absolute path to plan file
    plan_updated_at: datetime | None = None  # Last modification time
    active: bool = True
    branch: str | None = None  # Git branch name if detected
    pr_info: "PRInfo | None" = None  # GitHub PR linked to this branch
    _display_name_from_path: str | None = field(default=None, init=False, repr=False)

    @property
    def display_name(self) -> str:
        """Short display name for the session.

        Prefers branch name if available, otherwise shows last 2 path segments
        of the decoded project path (cached to avoid repeated filesystem access).
        """
        if self.branch:
            return self.branch
        if self._display_name_from_path is not None:
            return self._display_name_from_path
        if self.project_path:
            from .utils import decode_project_path
            decoded = decode_project_path(self.project_path)
            if decoded:
                parts = [p for p in decoded.split("/") if p]
                if len(parts) >= 2:
                    self._display_name_from_path = "/".join(parts[-2:])
                else:
                    self._display_name_from_path = parts[-1] if parts else self.session_id[:8]
                return self._display_name_from_path
        return self.session_id[:8]

    @property
    def last_activity(self) -> datetime:
        """Timestamp of the most recent turn, or started_at if no turns."""
        if self.turns:
            return self.turns[-1].timestamp
        return self.started_at

    @property
    def total_input_words(self) -> int:
        """Total words from user turns."""
        return sum(t.word_count for t in self.turns if t.role == "user")

    @property
    def total_output_words(self) -> int:
        """Total words from assistant turns."""
        return sum(t.word_count for t in self.turns if t.role == "assistant")

    @property
    def total_tool_calls(self) -> int:
        """Total tool calls."""
        return sum(1 for t in self.turns if t.role == "tool")

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all turns with usage data."""
        return sum(t.input_tokens or 0 for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all turns with usage data."""
        return sum(t.output_tokens or 0 for t in self.turns)

    @property
    def total_cache_creation_tokens(self) -> int:
        """Total cache creation tokens across all turns."""
        return sum(t.cache_creation_tokens or 0 for t in self.turns)

    @property
    def total_cache_read_tokens(self) -> int:
        """Total cache read tokens across all turns."""
        return sum(t.cache_read_tokens or 0 for t in self.turns)

    @property
    def has_token_data(self) -> bool:
        """Whether any turns have real token usage data."""
        return any(t.input_tokens is not None for t in self.turns)

    @property
    def estimated_cost(self) -> float | None:
        """Estimated session cost in USD, or None if no pricing data."""
        if not self.has_token_data:
            return None
        # Find model from first turn with model_id
        model_id = None
        for t in self.turns:
            if t.model_id:
                model_id = t.model_id
                break
        if not model_id:
            return None
        from .pricing import get_pricing, estimate_cost
        pricing = get_pricing(model_id)
        if not pricing:
            return None
        return estimate_cost(
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_cache_creation_tokens,
            self.total_cache_read_tokens,
            pricing,
        )
