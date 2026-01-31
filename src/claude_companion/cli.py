"""CLI entry point for Claude Companion."""

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import CONFIG_FILE, DEFAULT_STYLE, get_example_configs, load_config, save_config, Config, LLMConfig
from .history import cwd_to_project_path, get_history, get_most_recent
from .hooks import check_hooks_status, install_hooks, uninstall_hooks
from .server import ServerThread
from .store import EventStore
from .tui import TUI

console = Console()


@click.group(invoke_without_command=True)
@click.option("--port", "-p", default=5433, help="Port for the HTTP server")
@click.option("--version", "-v", is_flag=True, help="Show version")
@click.option("--continue", "-c", "continue_session", is_flag=True, help="Continue the most recent session (current dir)")
@click.option("--resume", "-r", is_flag=True, help="Select a session to resume (current dir)")
@click.option("-C", "continue_global", is_flag=True, help="Continue the most recent session (global)")
@click.option("-R", "resume_global", is_flag=True, help="Select a session to resume (global)")
@click.pass_context
def main(ctx: click.Context, port: int, version: bool, continue_session: bool, resume: bool, continue_global: bool, resume_global: bool) -> None:
    """Claude Companion - A CLI supervisor for Claude Code sessions."""
    if version:
        console.print(f"claude-companion v{__version__}")
        return

    # If no subcommand, run the main TUI
    if ctx.invoked_subcommand is None:
        # Check for mutually exclusive options
        resume_flags = sum([continue_session, resume, continue_global, resume_global])
        if resume_flags > 1:
            console.print("[red]Error:[/red] Options -c, -r, -C, -R are mutually exclusive.")
            raise SystemExit(1)

        config = load_config()

        # Determine project scope: -c/-r filter by current dir, -C/-R are global
        project_path = None
        if continue_session or resume:
            project_path = cwd_to_project_path()

        # Handle -c/--continue or -C: load most recent session
        initial_transcript = None
        if continue_session or continue_global:
            recent = get_most_recent(project_path=project_path)
            if recent:
                initial_transcript = recent.transcript_path
                console.print(f"[dim]Continuing session:[/dim] {recent.project_name}")
            else:
                scope = "current directory" if project_path else "global"
                console.print(f"[yellow]No session history found ({scope}).[/yellow]")

        # Handle -r/--resume or -R: show session picker
        elif resume or resume_global:
            initial_transcript = _pick_session(project_path=project_path)
            if initial_transcript is None:
                return  # User cancelled or no history

        run_companion(port, config.style, initial_transcript)


def _pick_session(project_path: str | None = None) -> str | None:
    """Show interactive session picker for -r/--resume. Returns transcript path or None."""
    import select
    import sys
    import termios
    import tty

    from rich.live import Live
    from rich.text import Text

    # Check for TTY - picker requires interactive terminal
    if not sys.stdin.isatty():
        console.print("[red]Error:[/red] Session picker requires an interactive terminal.")
        raise SystemExit(1)

    history = get_history(limit=20, project_path=project_path)
    if not history:
        console.print("[yellow]No session history found.[/yellow]")
        return None

    selected_idx = 0

    def render_picker() -> Table:
        """Render the session picker table with current selection highlighted."""
        table = Table(show_header=True, header_style="bold", highlight=False)
        table.add_column("", width=2)  # Selection indicator
        table.add_column("Project", style="cyan")
        table.add_column("Session", style="dim", width=8)
        table.add_column("Last Accessed", style="dim")

        for i, record in enumerate(history):
            last_accessed = record.last_accessed_dt.strftime("%Y-%m-%d %H:%M")
            session_short = record.session_id[:8]

            if i == selected_idx:
                # Highlighted row
                table.add_row(
                    "[bold cyan]>[/bold cyan]",
                    f"[bold reverse] {record.project_name} [/bold reverse]",
                    f"[reverse] {session_short} [/reverse]",
                    f"[reverse] {last_accessed} [/reverse]",
                )
            else:
                table.add_row("", record.project_name, session_short, last_accessed)

        return table

    def read_key() -> str | None:
        """Read a key if available."""
        if select.select([sys.stdin], [], [], 0.1)[0]:
            char = sys.stdin.read(1)
            # Handle escape sequences (arrow keys)
            if char == "\x1b":
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    char += sys.stdin.read(2)
            return char
        return None

    # Set up terminal for raw input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        with Live(render_picker(), console=console, refresh_per_second=10, screen=False) as live:
            # Show header and footer
            console.print("\n[bold]Select session:[/bold]")
            console.print("[dim]j/k or ↑/↓: navigate | Enter: select | q/Esc: cancel[/dim]\n")

            while True:
                key = read_key()
                if key is None:
                    continue

                if key in ("q", "\x1b", "\x03"):  # q, Esc, Ctrl+C
                    return None

                elif key in ("\n", "\r", " "):  # Enter or Space
                    selected = history[selected_idx]
                    console.print(f"\n[dim]Resuming session:[/dim] {selected.project_name}")
                    return selected.transcript_path

                elif key in ("k", "\x1b[A"):  # k or Up arrow
                    if selected_idx > 0:
                        selected_idx -= 1
                        live.update(render_picker())

                elif key in ("j", "\x1b[B"):  # j or Down arrow
                    if selected_idx < len(history) - 1:
                        selected_idx += 1
                        live.update(render_picker())

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_companion(port: int, ui_style: str = DEFAULT_STYLE, initial_transcript: str | None = None) -> None:
    """Run the main companion TUI and server."""
    # Check if hooks are installed
    status = check_hooks_status(port)
    not_installed = [e for e, s in status.items() if s == "not installed"]

    if not_installed:
        console.print(
            f"[yellow]Warning:[/yellow] Hooks not installed for: {', '.join(not_installed)}"
        )
        console.print("Run [bold]claude-companion install[/bold] to set up hooks.\n")

    # Create store, server, and TUI
    store = EventStore()
    server = ServerThread(store, port=port)

    # Load initial session if provided (from -c or -r)
    if initial_transcript:
        if not store.load_session_from_transcript(initial_transcript):
            console.print(f"[yellow]Warning:[/yellow] Could not load session from {initial_transcript}")

    console.print(f"[dim]Starting server on http://127.0.0.1:{port}[/dim]")

    # Start server in background
    server.start()

    # Wait for server to be ready
    if not server.wait_ready(timeout=2.0):
        console.print("[yellow]Warning:[/yellow] Server may not have started properly")

    try:
        # Run TUI in main thread
        tui = TUI(store, console, ui_style=ui_style)
        # Scroll to bottom if resuming a session
        if initial_transcript:
            tui.scroll_to_bottom()
        tui.run()
    except KeyboardInterrupt:
        pass
    finally:
        store.stop()
        server.shutdown()
        console.print("\n[dim]Goodbye![/dim]")


@main.command()
@click.option("--port", "-p", default=5433, help="Port for hooks to connect to")
def install(port: int) -> None:
    """Install Claude Code hooks."""
    console.print(f"Installing hooks for port {port}...")

    results = install_hooks(port)

    table = Table(title="Hook Installation Results")
    table.add_column("Event", style="cyan")
    table.add_column("Status")

    for event, status in results.items():
        if status == "installed":
            style = "green"
        elif status == "already installed":
            style = "yellow"
        else:
            style = "red"
        table.add_row(event, f"[{style}]{status}[/{style}]")

    console.print(table)
    console.print("\n[green]Done![/green] Hooks will send events to Claude Companion.")
    console.print("Run [bold]claude-companion[/bold] to start monitoring.")


@main.command()
def uninstall() -> None:
    """Remove Claude Code hooks."""
    console.print("Removing hooks...")

    results = uninstall_hooks()

    table = Table(title="Hook Removal Results")
    table.add_column("Event", style="cyan")
    table.add_column("Status")

    for event, status in results.items():
        if status == "removed":
            style = "green"
        else:
            style = "dim"
        table.add_row(event, f"[{style}]{status}[/{style}]")

    console.print(table)
    console.print("\n[green]Done![/green] Claude Companion hooks removed.")


@main.command()
@click.option("--port", "-p", default=5433, help="Port to check")
def status(port: int) -> None:
    """Check hook installation status."""
    results = check_hooks_status(port)

    table = Table(title=f"Hook Status (port {port})")
    table.add_column("Event", style="cyan")
    table.add_column("Status")

    all_installed = True
    for event, hook_status in results.items():
        if hook_status == "installed":
            style = "green"
        elif hook_status == "installed (different port)":
            style = "yellow"
            all_installed = False
        else:
            style = "red"
            all_installed = False
        table.add_row(event, f"[{style}]{hook_status}[/{style}]")

    console.print(table)

    if all_installed:
        console.print("\n[green]All hooks installed![/green]")
    else:
        console.print("\nRun [bold]claude-companion install[/bold] to install missing hooks.")


@main.command()
@click.option("--style", type=click.Choice(["rich", "claude-code"]), help="UI style")
@click.option("--url", help="LLM server base URL (e.g., http://localhost:1234/v1, for local only)")
@click.option("--model", help="LLM model name (e.g., gpt-4o-mini)")
@click.option("--preset", type=click.Choice(["vllm", "lm-studio", "ollama", "openai", "openrouter", "anthropic"]), help="Use preset LLM configuration")
@click.option("--show", is_flag=True, help="Show current configuration")
def config(style: str | None, url: str | None, model: str | None, preset: str | None, show: bool) -> None:
    """Configure Claude Companion settings.

    Examples:
      claude-companion config --style claude-code    # Set UI style
      claude-companion config --preset openrouter    # Use OpenRouter API
      claude-companion config --preset lm-studio     # Use LM Studio
      claude-companion config --show                 # Show current config
    """
    if show:
        # Show current configuration
        current_config = load_config()

        console.print("\n[bold]Current Configuration:[/bold]")
        console.print(f"  Style:    [cyan]{current_config.style}[/cyan]")

        console.print("\n[bold]LLM Configuration:[/bold]")
        console.print(f"  Provider: [cyan]{current_config.llm.provider}[/cyan]")
        if current_config.llm.provider == "local":
            console.print(f"  Base URL: [cyan]{current_config.llm.base_url}[/cyan]")
        console.print(f"  Model:    [cyan]{current_config.llm.model}[/cyan]")

        # Show API key status
        if current_config.llm.provider in ("openai", "openrouter", "anthropic"):
            has_key = "✓ set" if current_config.llm.api_key else "✗ not set"
            console.print(f"  API Key:  [{'green' if current_config.llm.api_key else 'red'}]{has_key}[/]")

        console.print(f"\nConfig file: [dim]{CONFIG_FILE}[/dim]")

        # Show environment variable overrides if set
        import os
        if os.getenv("CLAUDE_COMPANION_STYLE"):
            console.print(f"\n[yellow]Note:[/yellow] CLAUDE_COMPANION_STYLE is set: {os.getenv('CLAUDE_COMPANION_STYLE')}")
        if os.getenv("CLAUDE_COMPANION_LLM_PROVIDER"):
            console.print(f"[yellow]Note:[/yellow] CLAUDE_COMPANION_LLM_PROVIDER is set: {os.getenv('CLAUDE_COMPANION_LLM_PROVIDER')}")
        if os.getenv("CLAUDE_COMPANION_LLM_URL"):
            console.print(f"[yellow]Note:[/yellow] CLAUDE_COMPANION_LLM_URL is set: {os.getenv('CLAUDE_COMPANION_LLM_URL')}")
        if os.getenv("CLAUDE_COMPANION_LLM_MODEL"):
            console.print(f"[yellow]Note:[/yellow] CLAUDE_COMPANION_LLM_MODEL is set: {os.getenv('CLAUDE_COMPANION_LLM_MODEL')}")

        return

    # Load current config to preserve values not being changed
    current_config = load_config()
    new_style = style if style else current_config.style

    if preset:
        # Use preset configuration
        examples = get_example_configs()
        if preset not in examples:
            console.print(f"[red]Error:[/red] Unknown preset '{preset}'")
            return

        preset_config = examples[preset]
        provider = preset_config["provider"]
        model = preset_config["model"]
        url = preset_config.get("base_url")  # Only for local providers
        console.print(f"Using [cyan]{preset_config['description']}[/cyan] preset")

        # Create config from preset, preserving style
        new_config = Config(
            llm=LLMConfig(
                provider=provider,
                model=model,
                base_url=url,
            ),
            style=new_style,
        )

        save_config(new_config)

        console.print("\n[green]Configuration saved![/green]")
        console.print(f"  Style:    [cyan]{new_style}[/cyan]")
        console.print(f"  Provider: [cyan]{provider}[/cyan]")
        if provider == "local":
            console.print(f"  Base URL: [cyan]{url}[/cyan]")
        console.print(f"  Model:    [cyan]{model}[/cyan]")

        # Show API key reminder for API providers
        if provider == "openai":
            console.print("\n[yellow]Remember:[/yellow] Set OPENAI_API_KEY environment variable")
        elif provider == "openrouter":
            console.print("\n[yellow]Remember:[/yellow] Set OPENROUTER_API_KEY environment variable")
        elif provider == "anthropic":
            console.print("\n[yellow]Remember:[/yellow] Set ANTHROPIC_API_KEY environment variable")

        console.print(f"\nSaved to: [dim]{CONFIG_FILE}[/dim]")
        return

    # Style-only change
    if style and not url and not model:
        new_config = Config(
            llm=current_config.llm,
            style=style,
        )
        save_config(new_config)

        console.print("\n[green]Configuration saved![/green]")
        console.print(f"  Style: [cyan]{style}[/cyan]")
        console.print(f"\nSaved to: [dim]{CONFIG_FILE}[/dim]")
        return

    if not url and not model and not preset and not style:
        # Show help
        console.print("\n[bold]UI Styles:[/bold]")
        console.print("  [cyan]rich[/cyan]        Boxes with emojis (default)")
        console.print("  [cyan]claude-code[/cyan]  Minimal style matching Claude Code")

        console.print("\n[bold]LLM Presets:[/bold]\n")
        examples = get_example_configs()
        for name, cfg in examples.items():
            console.print(f"  [cyan]{name:12}[/cyan] {cfg['description']}")
            if cfg["provider"] == "local":
                console.print(f"               URL:   {cfg['base_url']}")
            console.print(f"               Model: {cfg['model']}\n")

        console.print("Use --style to set UI style.")
        console.print("Use --preset to apply an LLM preset.")
        console.print("Use --show to view current configuration.")
        return

    # Custom LLM configuration (advanced users)
    new_url = url if url else current_config.llm.base_url
    new_model = model if model else current_config.llm.model

    new_config = Config(
        llm=LLMConfig(
            provider="local",  # Assume local for custom config
            base_url=new_url,
            model=new_model,
        ),
        style=new_style,
    )

    save_config(new_config)

    console.print("\n[green]Configuration saved![/green]")
    console.print(f"  Style:    [cyan]{new_style}[/cyan]")
    console.print(f"  Provider: [cyan]local[/cyan]")
    console.print(f"  Base URL: [cyan]{new_config.llm.base_url}[/cyan]")
    console.print(f"  Model:    [cyan]{new_config.llm.model}[/cyan]")
    console.print(f"\nSaved to: [dim]{CONFIG_FILE}[/dim]")


if __name__ == "__main__":
    main()
