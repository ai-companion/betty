"""Hook installation and management for Claude Companion."""

import json
from pathlib import Path
from typing import Any

CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

# Hook events we want to listen to
# Only SessionStart is needed - we get everything else from watching the transcript
HOOK_EVENTS = [
    "SessionStart",
]

# Marker to identify our hooks
HOOK_MARKER = "claude-companion"


def _make_hook_command(port: int) -> str:
    """Create the curl command for a hook. Fails silently if server not running."""
    return (
        f"curl -s --connect-timeout 1 -X POST http://localhost:{port}/event "
        f"-H 'Content-Type: application/json' -d \"$(cat)\" "
        f">/dev/null 2>&1 || true "
        f"# {HOOK_MARKER}"
    )


def _make_hook_entry(port: int) -> dict[str, Any]:
    """Create a hook entry."""
    return {
        "type": "command",
        "command": _make_hook_command(port),
    }


def _is_our_hook(hook: dict[str, Any]) -> bool:
    """Check if a hook entry is ours."""
    command = hook.get("command", "")
    return HOOK_MARKER in command


def load_settings() -> dict[str, Any]:
    """Load Claude settings from file."""
    if not CLAUDE_SETTINGS_PATH.exists():
        return {}

    try:
        with open(CLAUDE_SETTINGS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_settings(settings: dict[str, Any]) -> None:
    """Save Claude settings to file."""
    CLAUDE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CLAUDE_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def install_hooks(port: int = 5432) -> dict[str, str]:
    """Install Claude Companion hooks. Returns status for each event."""
    settings = load_settings()
    hooks = settings.get("hooks", {})
    results = {}

    for event in HOOK_EVENTS:
        event_hooks = hooks.get(event, [])

        # Check if we already have a hook for this event
        has_our_hook = False
        for hook_group in event_hooks:
            inner_hooks = hook_group.get("hooks", [])
            for hook in inner_hooks:
                if _is_our_hook(hook):
                    has_our_hook = True
                    break
            if has_our_hook:
                break

        if has_our_hook:
            results[event] = "already installed"
        else:
            # Add our hook
            hook_group: dict[str, Any] = {"hooks": [_make_hook_entry(port)]}

            # Add matcher for PreToolUse and PostToolUse
            if event in ("PreToolUse", "PostToolUse"):
                hook_group["matcher"] = "*"

            event_hooks.append(hook_group)
            hooks[event] = event_hooks
            results[event] = "installed"

    settings["hooks"] = hooks
    save_settings(settings)

    return results


def uninstall_hooks() -> dict[str, str]:
    """Remove Claude Companion hooks. Returns status for each event."""
    settings = load_settings()
    hooks = settings.get("hooks", {})
    results = {}

    for event in HOOK_EVENTS:
        event_hooks = hooks.get(event, [])
        original_count = len(event_hooks)

        # Filter out our hooks
        filtered_hooks = []
        for hook_group in event_hooks:
            inner_hooks = hook_group.get("hooks", [])
            filtered_inner = [h for h in inner_hooks if not _is_our_hook(h)]

            if filtered_inner:
                hook_group["hooks"] = filtered_inner
                filtered_hooks.append(hook_group)

        if len(filtered_hooks) < original_count:
            results[event] = "removed"
        else:
            results[event] = "not found"

        if filtered_hooks:
            hooks[event] = filtered_hooks
        elif event in hooks:
            del hooks[event]

    settings["hooks"] = hooks
    save_settings(settings)

    return results


def check_hooks_status(port: int = 5432) -> dict[str, str]:
    """Check installation status of hooks. Returns status for each event."""
    settings = load_settings()
    hooks = settings.get("hooks", {})
    results = {}

    expected_marker = f"localhost:{port}"

    for event in HOOK_EVENTS:
        event_hooks = hooks.get(event, [])
        status = "not installed"

        for hook_group in event_hooks:
            inner_hooks = hook_group.get("hooks", [])
            for hook in inner_hooks:
                if _is_our_hook(hook):
                    command = hook.get("command", "")
                    if expected_marker in command:
                        status = "installed"
                    else:
                        status = "installed (different port)"
                    break
            if status != "not installed":
                break

        results[event] = status

    return results
