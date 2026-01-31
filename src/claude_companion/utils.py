"""Utility functions for Claude Companion."""

from pathlib import Path


def decode_project_path(encoded_path: str) -> str:
    """Decode project path from Claude's encoded format.

    Example: -Users-akash-src-foo → /Users/akash/src/foo
    Example: -Users-akash-cursor-projects-claude-companion → /Users/akash/cursor-projects/claude-companion

    Claude encodes paths by replacing '/' with '-' and prepending '-'.
    Decoding is tricky because dashes in directory names must be preserved.
    We try simple decode first, then fall back to incremental matching.

    Args:
        encoded_path: Encoded path from Session.project_path

    Returns:
        Decoded absolute file system path
    """
    if not encoded_path or not encoded_path.startswith("-"):
        return ""

    # Fast path: Try simple decode first (works for 99% of paths without dashes)
    simple_decode = "/" + encoded_path[1:].replace("-", "/")
    if Path(simple_decode).exists():
        return simple_decode

    # Slow path: Only for paths with dashes in directory names
    # Build path incrementally, checking what exists on filesystem
    parts = encoded_path[1:].split("-")
    current = ""
    remaining = parts[:]

    # Limit iterations to prevent infinite loops or excessive filesystem calls
    max_iterations = len(parts) * 2  # Safety limit
    iteration = 0

    while remaining and iteration < max_iterations:
        iteration += 1
        # Try combining next N parts with dashes
        for n in range(len(remaining), 0, -1):
            candidate = "/" + "/".join(current.split("/") + ["-".join(remaining[:n])])
            candidate = candidate.replace("//", "/")

            if Path(candidate).exists():
                current = candidate
                remaining = remaining[n:]
                break
        else:
            # No match found, fall back to simple decode for remaining parts
            if current:
                current = current + "/" + "-".join(remaining)
            else:
                current = "/" + "/".join(remaining)
            break

    return current if current else "/" + "/".join(parts)


def find_plan_file(project_dir: str) -> str | None:
    """Find plan file in project directory.

    Searches in order of preference:
    1. PLAN.md
    2. .claude/plan.md
    3. .claude/PLAN.md
    4. plan.md

    Args:
        project_dir: Absolute path to project directory

    Returns:
        Absolute path to plan file if found, None otherwise
    """
    if not project_dir:
        return None

    base = Path(project_dir)
    if not base.exists() or not base.is_dir():
        return None

    # Search in priority order
    candidates = [
        base / "PLAN.md",
        base / ".claude" / "plan.md",
        base / ".claude" / "PLAN.md",
        base / "plan.md",
    ]

    for path in candidates:
        if path.exists() and path.is_file():
            return str(path)

    return None
