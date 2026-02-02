"""Data models for Claude Companion."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
    tool_name: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    expanded: bool = False
    is_historical: bool = False  # True if loaded from transcript history
    summary: str | None = None  # LLM-generated summary for assistant turns
    critic: str | None = None  # LLM-generated critique for assistant turns
    critic_sentiment: str | None = None  # "progress" | "concern" | "critical"
    task_operation: tuple[str, dict[str, Any]] | None = None  # (operation_type, data) for task tools


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

    @property
    def display_name(self) -> str:
        """Short display name for the session."""
        if self.project_path:
            # Extract last component of path
            parts = self.project_path.replace("-", "/").split("/")
            return parts[-1] if parts else self.session_id[:8]
        return self.session_id[:8]

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
