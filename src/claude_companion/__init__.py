"""Claude Companion - A CLI supervisor for Claude Code sessions."""

try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import version

    __version__ = version("claude-companion")
