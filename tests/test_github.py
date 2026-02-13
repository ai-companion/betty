"""Tests for betty.github module."""

import json
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from betty.github import PRDetector, PRInfo, _REMOTE_URL_PATTERN


# --- Regex tests ---

class TestRemoteURLPattern:
    """Test _REMOTE_URL_PATTERN against various git remote URL formats."""

    def test_ssh_url(self):
        m = _REMOTE_URL_PATTERN.search("git@github.com:owner/repo.git")
        assert m is not None
        assert m.group("owner") == "owner"
        assert m.group("repo") == "repo"

    def test_ssh_url_no_dot_git(self):
        m = _REMOTE_URL_PATTERN.search("git@github.com:owner/repo")
        assert m is not None
        assert m.group("owner") == "owner"
        assert m.group("repo") == "repo"

    def test_https_url(self):
        m = _REMOTE_URL_PATTERN.search("https://github.com/owner/repo.git")
        assert m is not None
        assert m.group("owner") == "owner"
        assert m.group("repo") == "repo"

    def test_https_url_no_dot_git(self):
        m = _REMOTE_URL_PATTERN.search("https://github.com/owner/repo")
        assert m is not None
        assert m.group("owner") == "owner"
        assert m.group("repo") == "repo"

    def test_non_github_url(self):
        m = _REMOTE_URL_PATTERN.search("https://gitlab.com/owner/repo.git")
        assert m is None

    def test_ssh_with_org_and_dashes(self):
        m = _REMOTE_URL_PATTERN.search("git@github.com:my-org/my-repo.git")
        assert m is not None
        assert m.group("owner") == "my-org"
        assert m.group("repo") == "my-repo"


# --- PRDetector tests ---

class TestPRDetector:
    """Test PRDetector cache and fetch logic."""

    def test_get_pr_returns_none_when_not_cached(self):
        detector = PRDetector()
        assert detector.get_pr("/some/dir", "main") is None

    def test_get_pr_returns_cached_value(self):
        detector = PRDetector()
        pr = PRInfo(number=42, title="Fix bug", url="https://github.com/o/r/pull/42", state="OPEN")
        detector._cache[("/some/dir", "fix-branch")] = pr
        assert detector.get_pr("/some/dir", "fix-branch") == pr

    def test_get_pr_returns_none_for_cached_none(self):
        """None is cached to avoid retries on branches without PRs."""
        detector = PRDetector()
        detector._cache[("/some/dir", "no-pr")] = None
        assert detector.get_pr("/some/dir", "no-pr") is None

    @patch("betty.github.subprocess.run")
    def test_fetch_pr_success(self, mock_run):
        pr_data = [{"number": 7, "title": "Add feature", "url": "https://github.com/o/r/pull/7", "state": "OPEN"}]
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_data))

        result = PRDetector._fetch_pr("feature-branch", "owner/repo")
        assert result is not None
        assert result.number == 7
        assert result.title == "Add feature"
        assert result.state == "OPEN"

    @patch("betty.github.subprocess.run")
    def test_fetch_pr_empty_results(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="[]")

        result = PRDetector._fetch_pr("no-pr-branch", "owner/repo")
        assert result is None

    @patch("betty.github.subprocess.run")
    def test_fetch_pr_gh_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("gh not found")

        result = PRDetector._fetch_pr("branch", "owner/repo")
        assert result is None

    @patch("betty.github.subprocess.run")
    def test_fetch_pr_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gh", timeout=10)

        result = PRDetector._fetch_pr("branch", "owner/repo")
        assert result is None

    @patch("betty.github.subprocess.run")
    def test_fetch_pr_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")

        result = PRDetector._fetch_pr("branch", "owner/repo")
        assert result is None

    @patch("betty.github.subprocess.run")
    def test_get_github_remote_ssh(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="git@github.com:owner/repo.git\n")

        detector = PRDetector()
        result = detector._get_github_remote("/some/dir")
        assert result == "owner/repo"

    @patch("betty.github.subprocess.run")
    def test_get_github_remote_https(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="https://github.com/owner/repo.git\n")

        detector = PRDetector()
        result = detector._get_github_remote("/some/dir")
        assert result == "owner/repo"

    @patch("betty.github.subprocess.run")
    def test_get_github_remote_caches_result(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="git@github.com:owner/repo.git\n")

        detector = PRDetector()
        result1 = detector._get_github_remote("/some/dir")
        result2 = detector._get_github_remote("/some/dir")
        assert result1 == result2 == "owner/repo"
        # Second call should use cache, not subprocess
        assert mock_run.call_count == 1

    @patch("betty.github.subprocess.run")
    def test_get_github_remote_non_github(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="git@gitlab.com:owner/repo.git\n")

        detector = PRDetector()
        result = detector._get_github_remote("/some/dir")
        assert result is None

    def test_detect_async_skips_if_cached(self):
        detector = PRDetector()
        detector._cache[("/dir", "branch")] = None
        # Should not start a thread
        detector.detect_async("/dir", "branch")
        assert ("/dir", "branch") not in detector._in_flight
