"""Data models for Betty Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import SessionMetrics


@dataclass
class AgentObservation:
    """A single observation made by the agent."""

    turn_number: int
    timestamp: datetime
    observation_type: str  # "goal_set" | "goal_drift" | "progress" | "stall" | "error_spike" | "retry_loop" | "file_change" | "milestone"
    content: str  # Human-readable text
    severity: str  # "info" | "warning" | "critical"
    metadata: dict = field(default_factory=dict)


@dataclass
class FileChange:
    """A tracked file operation within a session."""

    file_path: str
    operation: str      # "read" | "write" | "edit" | "create" | "delete"
    turn_number: int
    timestamp: datetime
    lines_added: int = 0
    lines_removed: int = 0


@dataclass
class SessionReport:
    """Running situation report for a session."""

    session_id: str
    goal: str | None = None
    current_objective: str | None = None
    narrative: str = ""
    progress_assessment: str = "on_track"  # "on_track" | "slow" | "stalled" | "spinning" | "off_track"
    observations: list[AgentObservation] = field(default_factory=list)
    file_changes: list[FileChange] = field(default_factory=list)
    metrics: SessionMetrics | None = None
    updated_at: datetime = field(default_factory=datetime.now)
