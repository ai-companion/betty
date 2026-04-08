"""Tmux-based session discovery backend.

Discovers Claude Code sessions by inspecting tmux panes for running
``claude`` processes, then locates their transcript files in
``~/.claude/projects/``.
"""

import logging
import subprocess
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_tmux_not_found_warned = False


def _run_tmux(args: list[str], socket: str | None = None, timeout: float = 5.0) -> str | None:
    """Run a tmux command and return stdout, or None on failure."""
    global _tmux_not_found_warned
    cmd = ["tmux"]
    if socket:
        cmd.extend(["-L", socket])
    cmd.extend(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout
        logger.debug("tmux %s returned %d: %s", args, result.returncode, result.stderr.strip())
    except FileNotFoundError:
        if not _tmux_not_found_warned:
            logger.warning("tmux not found on PATH")
            _tmux_not_found_warned = True
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("tmux command failed: %s", exc)
    return None


def _get_pane_pids(socket: str | None = None) -> list[tuple[str, str]]:
    """Get list of (pane_id, pane_pid) for all tmux panes."""
    output = _run_tmux(["list-panes", "-a", "-F", "#{pane_id} #{pane_pid}"], socket=socket)
    if not output:
        return []
    panes = []
    for line in output.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            panes.append((parts[0], parts[1]))
    return panes


def _find_claude_processes(pane_pid: str) -> list[dict]:
    """Find claude processes that are children of the given pane PID.

    Returns list of dicts with 'pid' and 'cwd' keys.
    """
    # Use ps to find child processes that look like claude
    try:
        # Get all descendants' command lines and CWDs
        result = subprocess.run(
            ["ps", "-e", "-o", "pid,ppid,comm"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []

    # Build parent-child tree to find descendants of pane_pid
    children: dict[str, list[str]] = {}
    comm_by_pid: dict[str, str] = {}
    for line in result.stdout.strip().splitlines()[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 3:
            pid, ppid, comm = parts[0], parts[1], parts[2]
            children.setdefault(ppid, []).append(pid)
            comm_by_pid[pid] = comm

    # BFS to find all descendants
    descendants = set()
    queue = [pane_pid]
    while queue:
        p = queue.pop(0)
        for child in children.get(p, []):
            if child not in descendants:
                descendants.add(child)
                queue.append(child)

    # Find claude processes among descendants
    claude_procs = []
    for pid in descendants:
        comm = comm_by_pid.get(pid, "")
        if "claude" in comm.lower():
            # Get CWD from /proc (Linux) or lsof (macOS)
            cwd = _get_process_cwd(pid)
            if cwd:
                claude_procs.append({"pid": pid, "cwd": cwd})

    return claude_procs


def _get_process_cwd(pid: str) -> str | None:
    """Get the current working directory of a process."""
    # Try /proc first (Linux)
    proc_cwd = Path(f"/proc/{pid}/cwd")
    try:
        if proc_cwd.is_symlink():
            return str(proc_cwd.resolve())
    except (PermissionError, OSError):
        pass

    # Fallback: lsof (macOS/other)
    try:
        result = subprocess.run(
            ["lsof", "-p", pid, "-Fn", "-a", "-d", "cwd"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("n/"):
                    return line[1:]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return None


def _encode_project_path(path: str) -> str:
    """Encode an absolute path to Claude's project directory name."""
    return "-" + path.lstrip("/").replace("/", "-")


def discover_sessions_from_tmux(
    socket: str | None = None,
) -> list[tuple[str, Path]]:
    """Discover Claude Code sessions from tmux panes.

    Scans all tmux panes for running claude processes, determines their
    working directories, and locates corresponding session transcript
    files in ~/.claude/projects/.

    Args:
        socket: Optional tmux socket name (``-L`` flag).

    Returns:
        List of (session_id, transcript_path) tuples.
    """
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return []

    discovered: list[tuple[str, Path]] = []
    seen_sessions: set[str] = set()

    panes = _get_pane_pids(socket=socket)
    if not panes:
        return discovered

    # Build the process tree once for all panes (single ps invocation).
    children: dict[str, list[str]] = {}
    comm_by_pid: dict[str, str] = {}
    try:
        ps_result = subprocess.run(
            ["ps", "-e", "-o", "pid,ppid,comm"],
            capture_output=True, text=True, timeout=5,
        )
        if ps_result.returncode == 0:
            for line in ps_result.stdout.strip().splitlines()[1:]:  # skip header
                parts = line.split()
                if len(parts) >= 3:
                    pid, ppid, comm = parts[0], parts[1], parts[2]
                    children.setdefault(ppid, []).append(pid)
                    comm_by_pid[pid] = comm
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    for _, pane_pid in panes:
        # BFS to find all descendants of this pane process.
        descendants: set[str] = set()
        queue = [pane_pid]
        while queue:
            p = queue.pop(0)
            for child in children.get(p, []):
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)

        for pid in descendants:
            comm = comm_by_pid.get(pid, "")
            if "claude" not in comm.lower():
                continue
            cwd = _get_process_cwd(pid)
            if not cwd:
                continue

            encoded = _encode_project_path(cwd)
            project_path = projects_dir / encoded

            if not project_path.exists():
                continue

            # Find session files, sorted by mtime (most recent first)
            try:
                session_files = sorted(
                    [f for f in project_path.iterdir() if f.suffix == ".jsonl" and f.is_file()],
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
            except (PermissionError, OSError):
                continue

            for sf in session_files:
                sid = sf.stem
                if sid not in seen_sessions:
                    seen_sessions.add(sid)
                    discovered.append((sid, sf))

    return discovered


class TmuxProjectWatcher:
    """Watch tmux panes for Claude Code sessions.

    Drop-in replacement for ProjectWatcher that discovers sessions
    by inspecting tmux panes instead of scanning directories.
    """

    def __init__(
        self,
        on_session_discovered: Callable[[str, Path], None],
        socket: str | None = None,
        poll_interval: float = 5.0,
        on_initial_load_done: Callable[[], None] | None = None,
    ):
        self._on_session_discovered = on_session_discovered
        self._socket = socket
        self._poll_interval = poll_interval
        self._on_initial_load_done = on_initial_load_done
        self._known_sessions: set[str] = set()
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Initial scan and start polling."""
        # Initial discovery
        sessions = discover_sessions_from_tmux(socket=self._socket)
        for session_id, transcript_path in sessions:
            if session_id not in self._known_sessions:
                self._known_sessions.add(session_id)
                try:
                    if transcript_path.stat().st_size > 0:
                        self._on_session_discovered(session_id, transcript_path)
                except (PermissionError, OSError):
                    pass

        if self._on_initial_load_done:
            self._on_initial_load_done()

        # Start polling thread
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_once(self) -> None:
        """Discover sessions and report any that are new. Called by the poll loop."""
        sessions = discover_sessions_from_tmux(socket=self._socket)
        for session_id, transcript_path in sessions:
            if session_id not in self._known_sessions:
                try:
                    if transcript_path.stat().st_size > 0:
                        self._known_sessions.add(session_id)
                        self._on_session_discovered(session_id, transcript_path)
                except (PermissionError, OSError):
                    pass

    def _poll_loop(self) -> None:
        while self._running:
            self._stop_event.wait(timeout=self._poll_interval)
            if not self._running:
                break
            self._poll_once()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=6.0)
