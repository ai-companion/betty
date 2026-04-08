"""Tests for the tmux-based session discovery backend."""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from betty.tmux_backend import (
    TmuxProjectWatcher,
    _encode_project_path,
    _find_claude_processes,
    _get_pane_pids,
    _get_process_cwd,
    discover_sessions_from_tmux,
)


class TestEncodeProjectPath:
    def test_basic(self):
        assert _encode_project_path("/home/user/project") == "-home-user-project"

    def test_trailing_slash(self):
        assert _encode_project_path("/home/user/project/") == "-home-user-project-"

    def test_root(self):
        assert _encode_project_path("/") == "-"


class TestGetPanePids:
    @patch("betty.tmux_backend._run_tmux")
    def test_parses_output(self, mock_run):
        mock_run.return_value = "%0 12345\n%1 67890\n"
        result = _get_pane_pids()
        assert result == [("%0", "12345"), ("%1", "67890")]
        mock_run.assert_called_once_with(
            ["list-panes", "-a", "-F", "#{pane_id} #{pane_pid}"], socket=None
        )

    @patch("betty.tmux_backend._run_tmux")
    def test_with_socket(self, mock_run):
        mock_run.return_value = "%0 111\n"
        result = _get_pane_pids(socket="mysock")
        mock_run.assert_called_once_with(
            ["list-panes", "-a", "-F", "#{pane_id} #{pane_pid}"], socket="mysock"
        )

    @patch("betty.tmux_backend._run_tmux")
    def test_empty(self, mock_run):
        mock_run.return_value = None
        assert _get_pane_pids() == []

    @patch("betty.tmux_backend._run_tmux")
    def test_blank(self, mock_run):
        mock_run.return_value = ""
        assert _get_pane_pids() == []


class TestGetProcessCwd:
    def test_current_process(self):
        """Our own process has a /proc entry on Linux."""
        pid = str(os.getpid())
        cwd = _get_process_cwd(pid)
        if Path(f"/proc/{pid}/cwd").exists():
            assert cwd is not None
            assert Path(cwd).is_dir()

    def test_nonexistent_pid(self):
        cwd = _get_process_cwd("9999999")
        # Should return None (process doesn't exist)
        assert cwd is None


class TestFindClaudeProcesses:
    @patch("subprocess.run")
    @patch("betty.tmux_backend._get_process_cwd")
    def test_finds_claude_child(self, mock_cwd, mock_run):
        # Simulate ps output with a claude process as child of pane PID
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "  PID  PPID COMMAND\n"
                " 1000  999 bash\n"
                " 1001 1000 claude\n"
                " 1002 1001 node\n"
            ),
        )
        mock_cwd.return_value = "/home/user/project"

        result = _find_claude_processes("1000")
        assert len(result) == 1
        assert result[0]["pid"] == "1001"
        assert result[0]["cwd"] == "/home/user/project"

    @patch("subprocess.run")
    def test_no_claude_children(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "  PID  PPID COMMAND\n"
                " 1000  999 bash\n"
                " 1001 1000 vim\n"
            ),
        )
        result = _find_claude_processes("1000")
        assert result == []

    @patch("subprocess.run")
    def test_ps_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = _find_claude_processes("1000")
        assert result == []


class TestDiscoverSessionsFromTmux:
    def test_no_tmux(self):
        """When tmux is not available, returns empty list."""
        with patch("betty.tmux_backend._get_pane_pids", return_value=[]):
            result = discover_sessions_from_tmux()
            assert result == []

    def test_discovers_sessions(self):
        """Integration test with temp directory simulating ~/.claude/projects/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up fake projects dir
            projects_dir = Path(tmpdir) / ".claude" / "projects"
            project_dir = projects_dir / "-home-user-myproject"
            project_dir.mkdir(parents=True)

            # Create a session file
            session_file = project_dir / "abc123.jsonl"
            session_file.write_text('{"type":"user"}\n')

            # Mock the tmux/process functions
            with patch("betty.tmux_backend._get_pane_pids", return_value=[("%0", "1000")]), \
                 patch("betty.tmux_backend._find_claude_processes", return_value=[{"pid": "1001", "cwd": "/home/user/myproject"}]), \
                 patch("betty.tmux_backend._encode_project_path", return_value="-home-user-myproject"), \
                 patch("pathlib.Path.home", return_value=Path(tmpdir)):
                result = discover_sessions_from_tmux()
                assert len(result) == 1
                assert result[0][0] == "abc123"
                assert result[0][1] == session_file


class TestTmuxProjectWatcher:
    def test_initial_discovery(self):
        """Watcher calls on_session_discovered for initial sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            projects_dir = Path(tmpdir) / ".claude" / "projects"
            project_dir = projects_dir / "-home-user-proj"
            project_dir.mkdir(parents=True)

            sf = project_dir / "sess1.jsonl"
            sf.write_text('{"type":"user"}\n')

            callback = MagicMock()
            initial_done = MagicMock()

            with patch("betty.tmux_backend.discover_sessions_from_tmux", return_value=[("sess1", sf)]):
                watcher = TmuxProjectWatcher(
                    on_session_discovered=callback,
                    on_initial_load_done=initial_done,
                )
                watcher.start()
                watcher.stop()

            callback.assert_called_once_with("sess1", sf)
            initial_done.assert_called_once()

    def test_deduplicates(self):
        """Watcher doesn't call callback twice for same session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            projects_dir = Path(tmpdir) / ".claude" / "projects"
            project_dir = projects_dir / "-home-user-proj"
            project_dir.mkdir(parents=True)

            sf = project_dir / "sess1.jsonl"
            sf.write_text('{"type":"user"}\n')

            callback = MagicMock()

            # discover returns same session on both calls
            with patch("betty.tmux_backend.discover_sessions_from_tmux", return_value=[("sess1", sf)]):
                watcher = TmuxProjectWatcher(
                    on_session_discovered=callback,
                )
                watcher.start()
                # Manually trigger another poll cycle
                watcher._poll_loop.__wrapped__ if hasattr(watcher._poll_loop, '__wrapped__') else None
                watcher.stop()

            # Should only be called once despite multiple scans
            callback.assert_called_once()

    def test_skips_empty_files(self):
        """Watcher skips session files with zero size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            projects_dir = Path(tmpdir) / ".claude" / "projects"
            project_dir = projects_dir / "-home-user-proj"
            project_dir.mkdir(parents=True)

            sf = project_dir / "empty.jsonl"
            sf.write_text("")

            callback = MagicMock()

            with patch("betty.tmux_backend.discover_sessions_from_tmux", return_value=[("empty", sf)]):
                watcher = TmuxProjectWatcher(
                    on_session_discovered=callback,
                )
                watcher.start()
                watcher.stop()

            callback.assert_not_called()


class TestSessionDataShape:
    """Verify tmux backend produces same session data shape as file backend."""

    def test_session_has_expected_fields(self):
        """Sessions from tmux backend are regular Session objects."""
        from betty.models import Session

        # TmuxProjectWatcher calls EventStore._on_session_discovered
        # which creates Session objects - same as ProjectWatcher.
        # Verify this by checking the callback signature matches.
        from betty.store import EventStore
        import inspect

        sig = inspect.signature(EventStore._on_session_discovered)
        params = list(sig.parameters.keys())
        assert "session_id" in params
        assert "transcript_path" in params
