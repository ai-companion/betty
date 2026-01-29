"""Alert detection and notification system for Claude Companion."""

import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .models import Event


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"


@dataclass
class Alert:
    """An alert triggered by session activity."""
    level: AlertLevel
    title: str
    message: str
    event: Event


# Patterns for detecting potentially dangerous operations
DANGER_PATTERNS = [
    # Destructive git commands
    (r"git\s+push\s+.*--force", "Force push detected", AlertLevel.DANGER),
    (r"git\s+reset\s+--hard", "Hard reset detected", AlertLevel.DANGER),
    (r"git\s+clean\s+-f", "Git clean -f detected", AlertLevel.WARNING),
    (r"git\s+checkout\s+\.", "Checkout . (discard changes)", AlertLevel.WARNING),

    # Destructive file operations
    (r"rm\s+-rf?\s+/", "Recursive delete from root", AlertLevel.DANGER),
    (r"rm\s+-rf?\s+~", "Recursive delete from home", AlertLevel.DANGER),
    (r"rm\s+-rf?\s+\*", "Recursive delete with wildcard", AlertLevel.WARNING),
    (r">\s*/dev/null\s+2>&1\s*&", "Background with suppressed output", AlertLevel.INFO),

    # System modifications
    (r"chmod\s+777", "chmod 777 (world writable)", AlertLevel.WARNING),
    (r"sudo\s+", "sudo command", AlertLevel.INFO),

    # Network operations
    (r"curl\s+.*\|\s*sh", "Piping curl to shell", AlertLevel.DANGER),
    (r"wget\s+.*\|\s*sh", "Piping wget to shell", AlertLevel.DANGER),

    # Database operations
    (r"DROP\s+TABLE", "DROP TABLE statement", AlertLevel.DANGER),
    (r"DROP\s+DATABASE", "DROP DATABASE statement", AlertLevel.DANGER),
    (r"TRUNCATE\s+TABLE", "TRUNCATE TABLE statement", AlertLevel.WARNING),
    (r"DELETE\s+FROM\s+\w+\s*;?\s*$", "DELETE without WHERE", AlertLevel.WARNING),
]

# Patterns for tracking intent/progress
INFO_PATTERNS = [
    (r"npm\s+install|yarn\s+add|pip\s+install", "Installing dependencies", AlertLevel.INFO),
    (r"npm\s+test|pytest|jest|cargo\s+test", "Running tests", AlertLevel.INFO),
    (r"npm\s+run\s+build|cargo\s+build|make\s+build", "Building project", AlertLevel.INFO),
]


def check_event_for_alerts(event: Event) -> list[Alert]:
    """Check an event for potential alerts."""
    alerts = []

    if event.event_type != "PreToolUse":
        return alerts

    # Check Bash commands
    if event.tool_name == "Bash" and event.tool_input:
        command = event.tool_input.get("command", "")
        alerts.extend(_check_command(command, event))

    # Check file writes for sensitive paths
    if event.tool_name in ("Write", "Edit") and event.tool_input:
        file_path = event.tool_input.get("file_path", "")
        alerts.extend(_check_file_operation(file_path, event))

    return alerts


def _check_command(command: str, event: Event) -> list[Alert]:
    """Check a bash command for dangerous patterns."""
    alerts = []

    for pattern, title, level in DANGER_PATTERNS + INFO_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            alerts.append(Alert(
                level=level,
                title=title,
                message=f"Command: {command[:100]}{'...' if len(command) > 100 else ''}",
                event=event,
            ))

    return alerts


def _check_file_operation(file_path: str, event: Event) -> list[Alert]:
    """Check file operations for sensitive paths."""
    alerts = []

    sensitive_paths = [
        (".env", "Modifying .env file", AlertLevel.WARNING),
        ("credentials", "Modifying credentials file", AlertLevel.WARNING),
        ("secret", "Modifying file with 'secret' in name", AlertLevel.WARNING),
        (".ssh/", "Modifying SSH config", AlertLevel.DANGER),
        (".aws/", "Modifying AWS config", AlertLevel.WARNING),
        ("id_rsa", "Modifying SSH key", AlertLevel.DANGER),
        ("/etc/", "Modifying system file", AlertLevel.DANGER),
    ]

    for pattern, title, level in sensitive_paths:
        if pattern.lower() in file_path.lower():
            alerts.append(Alert(
                level=level,
                title=title,
                message=f"File: {file_path}",
                event=event,
            ))

    return alerts


def send_system_notification(alert: Alert) -> bool:
    """Send a system notification for an alert."""
    try:
        if sys.platform == "darwin":
            # macOS notification using osascript
            title = f"Claude Companion: {alert.level.value.upper()}"
            message = f"{alert.title}\n{alert.message}"

            script = f'''
            display notification "{message}" with title "{title}"
            '''
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
            )
            return True

        elif sys.platform == "linux":
            # Linux notification using notify-send
            subprocess.run(
                [
                    "notify-send",
                    f"Claude Companion: {alert.level.value.upper()}",
                    f"{alert.title}\n{alert.message}",
                ],
                capture_output=True,
                timeout=5,
            )
            return True

    except Exception:
        pass

    return False
