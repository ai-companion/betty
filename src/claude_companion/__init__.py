"""Claude Companion - A CLI supervisor for Claude Code sessions."""

try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import version

        __version__ = version("claude-companion")
    except Exception:
        __version__ = "0.0.0+unknown"
