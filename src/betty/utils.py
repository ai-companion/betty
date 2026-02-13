"""Utility functions for Betty."""

from pathlib import Path


def decode_project_path(encoded_path: str) -> str:
    """Decode project path from Claude's encoded format.

    Example: -Users-akash-src-foo → /Users/akash/src/foo
    Example: -Users-akash-cursor-projects-my-app → /Users/akash/cursor-projects/my-app

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


def is_claude_plan_file(file_path: str) -> bool:
    """Check if a file path is a Claude Code plan file in ~/.claude/plans/.

    Args:
        file_path: File path string to check

    Returns:
        True if the path matches ~/.claude/plans/*.md
    """
    return "/.claude/plans/" in file_path and file_path.endswith(".md")
