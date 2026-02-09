"""CLI entry point for Claude Companion."""

import logging
import subprocess
from logging.handlers import RotatingFileHandler
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import CONFIG_FILE, get_example_configs, load_config, save_config, Config, LLMConfig
from .store import EventStore
from .tui_textual import CompanionApp

console = Console()


def encode_project_path(path: str) -> str:
    """Encode an absolute path to Claude's project directory name.

    Strips the leading ``/`` and joins components with ``-``.

    e.g., /Users/kai/src/foo -> -Users-kai-src-foo
    """
    return "-" + path.lstrip("/").replace("/", "-")


def cwd_to_project_path() -> str:
    """Encode CWD to Claude's project directory name."""
    return encode_project_path(str(Path.cwd()))


class GitNotFoundError(Exception):
    """Raised when the git executable is not found on PATH."""


class GitCommandError(Exception):
    """Raised when the git command fails to execute (timeout, OS error)."""


class NotAGitRepoError(Exception):
    """Raised when the current directory is not inside a git repository."""


def get_worktree_paths() -> list[str]:
    """Get all worktree paths for the current git repository.

    Runs ``git worktree list --porcelain`` and extracts the worktree paths.

    Returns:
        List of absolute paths for each worktree.

    Raises:
        GitNotFoundError: If the git executable is not found on PATH.
        GitCommandError: If the git command fails to execute (timeout, OS error).
        NotAGitRepoError: If the current directory is not inside a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        raise GitNotFoundError("git is not installed or not on PATH")
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise GitCommandError(f"failed to run git: {exc}")

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise NotAGitRepoError(
            stderr or "current directory is not inside a git repository"
        )

    paths: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            paths.append(line[len("worktree "):])

    if not paths:
        raise NotAGitRepoError("git worktree list returned no worktrees")

    return paths


def get_project_paths(global_mode: bool, worktree_mode: bool = False) -> list[Path]:
    """Get project directories to watch.

    Args:
        global_mode: If True, return all project directories.
        worktree_mode: If True, return project directories for all git worktrees
            of the current repository.

    Returns:
        List of project directories to watch (may include non-existent paths that will be watched)
    """
    projects_dir = Path.home() / ".claude" / "projects"

    if global_mode:
        # All project directories (those starting with "-")
        if not projects_dir.exists():
            return []
        return [p for p in projects_dir.iterdir()
                if p.is_dir() and p.name.startswith("-")]
    elif worktree_mode:
        # All worktrees of the current git repo
        worktrees = get_worktree_paths()
        return [projects_dir / encode_project_path(wt) for wt in worktrees]
    else:
        # Current directory only - return path even if it doesn't exist yet
        # The watcher will poll until sessions appear
        encoded = cwd_to_project_path()
        return [projects_dir / encoded]


@click.group(invoke_without_command=True)
@click.option("--global", "-g", "global_mode", is_flag=True, help="Watch all projects (not just current directory)")
@click.option("--worktree", "-w", "worktree_mode", is_flag=True, help="Watch all git worktrees of the current repository")
@click.option("--style", type=click.Choice(["rich", "claude-code"]), default=None, help="UI style override for this run")
@click.option("--collapse-tools/--no-collapse-tools", default=None, help="Collapse tool turns into groups")
@click.option("--debug-logging/--no-debug-logging", default=None, help="Enable debug logging to file")
@click.option("--version", "-v", is_flag=True, help="Show version")
@click.pass_context
def main(ctx: click.Context, global_mode: bool, worktree_mode: bool, style: str | None, collapse_tools: bool | None, debug_logging: bool | None, version: bool) -> None:
    """Claude Companion - A CLI supervisor for Claude Code sessions."""
    if version:
        console.print(f"claude-companion v{__version__}")
        return

    if global_mode and worktree_mode:
        raise click.UsageError("--global and --worktree are mutually exclusive")

    # If no subcommand, run the main TUI
    if ctx.invoked_subcommand is None:
        config = load_config()
        # Apply CLI overrides (not saved to config file)
        if style is not None:
            config.style = style
        if collapse_tools is not None:
            config.collapse_tools = collapse_tools
        if debug_logging is not None:
            config.debug_logging = debug_logging
        run_companion(global_mode=global_mode, worktree_mode=worktree_mode, config=config)


def run_companion(global_mode: bool = False, worktree_mode: bool = False, config: Config | None = None) -> None:
    """Run the main companion TUI with directory-based session discovery."""
    if config is None:
        config = load_config()

    # Configure logging based on config (opt-in debug logging)
    if config.debug_logging:
        # Debug logging enabled - use rotating file handler
        log_file = Path.home() / ".cache" / "claude-companion" / "debug.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
        )
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[handler],
        )
        logging.info("Claude Companion starting (debug logging enabled)")
    else:
        # Default: only warn+ so TUI stays clean
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    projects_dir = Path.home() / ".claude" / "projects"
    try:
        project_paths = get_project_paths(global_mode, worktree_mode=worktree_mode)
    except GitNotFoundError:
        console.print("[red]Error:[/red] --worktree requires git, but git is not installed or not on PATH")
        raise SystemExit(1)
    except GitCommandError as exc:
        console.print(f"[red]Error:[/red] --worktree failed: {exc}")
        raise SystemExit(1)
    except NotAGitRepoError as exc:
        console.print(
            f"[red]Error:[/red] --worktree requires a git repository; git reported: {exc}"
        )
        raise SystemExit(1)

    # Create store and start watching
    # Load only most recent session by default; new sessions auto-detected while running
    max_sessions = 1
    store = EventStore()
    store.start_watching(
        project_paths,
        max_sessions=max_sessions,
        projects_dir=projects_dir,
        global_mode=global_mode,
    )

    if global_mode:
        scope = "all projects"
    elif worktree_mode:
        scope = f"git worktrees ({len(project_paths)} found)"
    else:
        scope = "current directory"
    console.print(f"[dim]Watching {scope} for Claude sessions...[/dim]")

    try:
        # Run Textual TUI
        app = CompanionApp(store, collapse_tools=config.collapse_tools, ui_style=config.style)
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        store.stop()
        console.print("\n[dim]Goodbye![/dim]")


@main.command()
@click.option("--style", type=click.Choice(["rich", "claude-code"]), help="UI style")
@click.option("--url", help="LLM server API base URL (e.g., http://localhost:1234/v1)")
@click.option("--model", help="LLM model name with litellm prefix (e.g., openai/gpt-4o-mini)")
@click.option("--llm-preset", "preset", type=click.Choice(["claude-code", "vllm", "lm-studio", "ollama", "openai", "openrouter", "anthropic"]), help="Use preset LLM configuration")
@click.option("--collapse-tools/--no-collapse-tools", default=None, help="Collapse tool turns into groups")
@click.option("--debug-logging/--no-debug-logging", default=None, help="Enable debug logging to file")
@click.option("--show", is_flag=True, help="Show current configuration")
def config(style: str | None, url: str | None, model: str | None, preset: str | None, collapse_tools: bool | None, debug_logging: bool | None, show: bool) -> None:
    """Configure Claude Companion settings.

    Examples:
      claude-companion config --style claude-code        # Set UI style
      claude-companion config --llm-preset openrouter    # Use OpenRouter API
      claude-companion config --llm-preset lm-studio     # Use LM Studio
      claude-companion config --collapse-tools           # Enable tool collapsing
      claude-companion config --debug-logging            # Enable debug logging
      claude-companion config --show                     # Show current config
    """
    if show:
        # Show current configuration
        current_config = load_config()

        console.print("\n[bold]Current Configuration:[/bold]")
        console.print(f"  Style:          [cyan]{current_config.style}[/cyan]")
        console.print(f"  Collapse Tools: [cyan]{current_config.collapse_tools}[/cyan]")
        console.print(f"  Debug Logging:  [cyan]{current_config.debug_logging}[/cyan]")

        console.print("\n[bold]LLM Configuration:[/bold]")
        console.print(f"  Model:    [cyan]{current_config.llm.model}[/cyan]")
        if current_config.llm.api_base:
            console.print(f"  API Base: [cyan]{current_config.llm.api_base}[/cyan]")

        # Show API key reminder based on model prefix
        model_str = current_config.llm.model
        if model_str.startswith("openai/"):
            import os
            has_key = "set" if os.getenv("OPENAI_API_KEY") else "not set"
            console.print(f"  API Key:  [{'green' if os.getenv('OPENAI_API_KEY') else 'red'}]{has_key}[/] (OPENAI_API_KEY)")
        elif model_str.startswith("openrouter/"):
            import os
            has_key = "set" if os.getenv("OPENROUTER_API_KEY") else "not set"
            console.print(f"  API Key:  [{'green' if os.getenv('OPENROUTER_API_KEY') else 'red'}]{has_key}[/] (OPENROUTER_API_KEY)")
        elif model_str.startswith("anthropic/"):
            import os
            has_key = "set" if os.getenv("ANTHROPIC_API_KEY") else "not set"
            console.print(f"  API Key:  [{'green' if os.getenv('ANTHROPIC_API_KEY') else 'red'}]{has_key}[/] (ANTHROPIC_API_KEY)")

        console.print(f"\nConfig file: [dim]{CONFIG_FILE}[/dim]")

        # Show environment variable overrides if set
        import os
        if os.getenv("CLAUDE_COMPANION_STYLE"):
            console.print(f"\n[yellow]Note:[/yellow] CLAUDE_COMPANION_STYLE is set: {os.getenv('CLAUDE_COMPANION_STYLE')}")
        if os.getenv("CLAUDE_COMPANION_COLLAPSE_TOOLS"):
            console.print(f"[yellow]Note:[/yellow] CLAUDE_COMPANION_COLLAPSE_TOOLS is set: {os.getenv('CLAUDE_COMPANION_COLLAPSE_TOOLS')}")
        if os.getenv("CLAUDE_COMPANION_DEBUG_LOGGING"):
            console.print(f"[yellow]Note:[/yellow] CLAUDE_COMPANION_DEBUG_LOGGING is set: {os.getenv('CLAUDE_COMPANION_DEBUG_LOGGING')}")
        if os.getenv("CLAUDE_COMPANION_LLM_PROVIDER"):
            console.print(f"[yellow]Note:[/yellow] CLAUDE_COMPANION_LLM_PROVIDER is set (deprecated): {os.getenv('CLAUDE_COMPANION_LLM_PROVIDER')}")
        if os.getenv("CLAUDE_COMPANION_LLM_API_BASE"):
            console.print(f"[yellow]Note:[/yellow] CLAUDE_COMPANION_LLM_API_BASE is set: {os.getenv('CLAUDE_COMPANION_LLM_API_BASE')}")
        if os.getenv("CLAUDE_COMPANION_LLM_URL"):
            console.print(f"[yellow]Note:[/yellow] CLAUDE_COMPANION_LLM_URL is set: {os.getenv('CLAUDE_COMPANION_LLM_URL')}")
        if os.getenv("CLAUDE_COMPANION_LLM_MODEL"):
            console.print(f"[yellow]Note:[/yellow] CLAUDE_COMPANION_LLM_MODEL is set: {os.getenv('CLAUDE_COMPANION_LLM_MODEL')}")

        return

    # Load current config to preserve values not being changed
    current_config = load_config()
    new_style = style if style else current_config.style
    new_collapse_tools = collapse_tools if collapse_tools is not None else current_config.collapse_tools
    new_debug_logging = debug_logging if debug_logging is not None else current_config.debug_logging

    if preset:
        # Use preset configuration
        examples = get_example_configs()
        if preset not in examples:
            console.print(f"[red]Error:[/red] Unknown preset '{preset}'")
            return

        preset_config = examples[preset]
        preset_model = preset_config["model"]
        preset_api_base = preset_config.get("api_base")
        console.print(f"Using [cyan]{preset_config['description']}[/cyan] preset")

        # Create config from preset, preserving style, collapse_tools, and debug_logging
        new_config = Config(
            llm=LLMConfig(
                model=preset_model,
                api_base=preset_api_base,
            ),
            style=new_style,
            collapse_tools=new_collapse_tools,
            debug_logging=new_debug_logging,
        )

        save_config(new_config)

        console.print("\n[green]Configuration saved![/green]")
        console.print(f"  Style:    [cyan]{new_style}[/cyan]")
        console.print(f"  Model:    [cyan]{preset_model}[/cyan]")
        if preset_api_base:
            console.print(f"  API Base: [cyan]{preset_api_base}[/cyan]")

        # Show API key reminder based on model prefix
        if preset_model.startswith("openai/") and not preset_api_base:
            console.print("\n[yellow]Remember:[/yellow] Set OPENAI_API_KEY environment variable")
        elif preset_model.startswith("openrouter/"):
            console.print("\n[yellow]Remember:[/yellow] Set OPENROUTER_API_KEY environment variable")
        elif preset_model.startswith("anthropic/"):
            console.print("\n[yellow]Remember:[/yellow] Set ANTHROPIC_API_KEY environment variable")

        console.print(f"\nSaved to: [dim]{CONFIG_FILE}[/dim]")
        return

    # Non-LLM config change (style, collapse_tools, debug_logging only)
    if not url and not model and not preset:
        if style or collapse_tools is not None or debug_logging is not None:
            new_config = Config(
                llm=current_config.llm,
                style=new_style,
                collapse_tools=new_collapse_tools,
                debug_logging=new_debug_logging,
            )
            save_config(new_config)

            console.print("\n[green]Configuration saved![/green]")
            console.print(f"  Style:          [cyan]{new_style}[/cyan]")
            console.print(f"  Collapse Tools: [cyan]{new_collapse_tools}[/cyan]")
            console.print(f"  Debug Logging:  [cyan]{new_debug_logging}[/cyan]")
            console.print(f"\nSaved to: [dim]{CONFIG_FILE}[/dim]")
            return

        # No flags at all — show help
        console.print("\n[bold]UI Styles:[/bold]")
        console.print("  [cyan]rich[/cyan]        Boxes with emojis (default)")
        console.print("  [cyan]claude-code[/cyan]  Minimal style matching Claude Code")

        console.print("\n[bold]LLM Presets:[/bold]\n")
        examples = get_example_configs()
        for name, cfg in examples.items():
            console.print(f"  [cyan]{name:12}[/cyan] {cfg['description']}")
            if "api_base" in cfg:
                console.print(f"               API Base: {cfg['api_base']}")
            console.print(f"               Model:    {cfg['model']}\n")

        console.print("Use --style to set UI style.")
        console.print("Use --llm-preset to apply an LLM preset.")
        console.print("Use --show to view current configuration.")
        return

    # Custom LLM configuration (advanced users)
    new_api_base = url if url else current_config.llm.api_base
    new_model = model if model else current_config.llm.model

    new_config = Config(
        llm=LLMConfig(
            model=new_model,
            api_base=new_api_base,
        ),
        style=new_style,
        collapse_tools=new_collapse_tools,
        debug_logging=new_debug_logging,
    )

    save_config(new_config)

    console.print("\n[green]Configuration saved![/green]")
    console.print(f"  Style:    [cyan]{new_style}[/cyan]")
    console.print(f"  Model:    [cyan]{new_model}[/cyan]")
    if new_config.llm.api_base:
        console.print(f"  API Base: [cyan]{new_config.llm.api_base}[/cyan]")
    console.print(f"\nSaved to: [dim]{CONFIG_FILE}[/dim]")


@main.command()
@click.option("--demo", is_flag=True, help="Auto-play a sample conversation")
@click.option("--project", "-p", help="Simulated project path (default: /tmp/mock-project)")
@click.option("--delay", "-d", type=click.FloatRange(min=0.0, min_open=True), default=1.5, help="Delay between messages in demo mode (default: 1.5s)")
def mock(demo: bool, project: str | None, delay: float) -> None:
    """Generate mock Claude Code sessions for development.

    This is useful for testing and development in environments without
    access to Claude Code (e.g., cloud-based development).

    Examples:
      claude-companion mock                # Interactive mode
      claude-companion mock --demo         # Auto-play sample conversation
      claude-companion mock --demo -d 2.0  # Slower demo (2s between messages)
    """
    from .mock_session import run_demo, run_interactive

    if demo:
        run_demo(project_path=project, delay=delay)
    else:
        run_interactive(project_path=project)


if __name__ == "__main__":
    main()
