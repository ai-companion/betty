"""CLI entry point for Claude Companion."""

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import CONFIG_FILE, get_example_configs, load_config, save_config, Config, LLMConfig
from .hooks import check_hooks_status, install_hooks, uninstall_hooks
from .server import ServerThread
from .store import EventStore
from .tui import TUI

console = Console()


@click.group(invoke_without_command=True)
@click.option("--port", "-p", default=5432, help="Port for the HTTP server")
@click.option("--version", "-v", is_flag=True, help="Show version")
@click.pass_context
def main(ctx: click.Context, port: int, version: bool) -> None:
    """Claude Companion - A CLI supervisor for Claude Code sessions."""
    if version:
        console.print(f"claude-companion v{__version__}")
        return

    # If no subcommand, run the main TUI
    if ctx.invoked_subcommand is None:
        run_companion(port)


def run_companion(port: int) -> None:
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

    console.print(f"[dim]Starting server on http://127.0.0.1:{port}[/dim]")

    # Start server in background
    server.start()

    try:
        # Run TUI in main thread
        tui = TUI(store, console)
        tui.run()
    except KeyboardInterrupt:
        pass
    finally:
        store.stop()
        server.shutdown()
        console.print("\n[dim]Goodbye![/dim]")


@main.command()
@click.option("--port", "-p", default=5432, help="Port for hooks to connect to")
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
@click.option("--port", "-p", default=5432, help="Port to check")
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
@click.option("--url", help="LLM server base URL (e.g., http://localhost:1234/v1)")
@click.option("--model", help="LLM model name (e.g., openai/gpt-oss-20b)")
@click.option("--preset", type=click.Choice(["vllm", "lm-studio", "ollama"]), help="Use preset configuration")
@click.option("--show", is_flag=True, help="Show current configuration")
def config(url: str | None, model: str | None, preset: str | None, show: bool) -> None:
    """Configure LLM server for summarization.

    Examples:
      claude-companion config --preset lm-studio
      claude-companion config --url http://localhost:1234/v1 --model openai/gpt-oss-20b
      claude-companion config --show
    """
    if show:
        # Show current configuration
        current_config = load_config()
        console.print("\n[bold]Current LLM Configuration:[/bold]")
        console.print(f"  Base URL: [cyan]{current_config.llm.base_url}[/cyan]")
        console.print(f"  Model:    [cyan]{current_config.llm.model}[/cyan]")
        console.print(f"\nConfig file: [dim]{CONFIG_FILE}[/dim]")

        # Show environment variable overrides if set
        import os
        if os.getenv("CLAUDE_COMPANION_LLM_URL"):
            console.print(f"\n[yellow]Note:[/yellow] CLAUDE_COMPANION_LLM_URL is set: {os.getenv('CLAUDE_COMPANION_LLM_URL')}")
        if os.getenv("CLAUDE_COMPANION_LLM_MODEL"):
            console.print(f"[yellow]Note:[/yellow] CLAUDE_COMPANION_LLM_MODEL is set: {os.getenv('CLAUDE_COMPANION_LLM_MODEL')}")

        return

    if preset:
        # Use preset configuration
        examples = get_example_configs()
        if preset not in examples:
            console.print(f"[red]Error:[/red] Unknown preset '{preset}'")
            return

        preset_config = examples[preset]
        url = preset_config["base_url"]
        model = preset_config["model"]
        console.print(f"Using [cyan]{preset_config['description']}[/cyan] preset")

    if not url and not model:
        # Show available presets
        console.print("\n[bold]Available Presets:[/bold]\n")
        examples = get_example_configs()
        for name, cfg in examples.items():
            console.print(f"  [cyan]{name}[/cyan]: {cfg['description']}")
            console.print(f"    URL:   {cfg['base_url']}")
            console.print(f"    Model: {cfg['model']}\n")

        console.print("Use --preset to apply a preset, or --url/--model to set custom values.")
        console.print("Use --show to view current configuration.")
        return

    # Load current config
    current_config = load_config()

    # Update with new values
    new_url = url if url else current_config.llm.base_url
    new_model = model if model else current_config.llm.model

    new_config = Config(llm=LLMConfig(base_url=new_url, model=new_model))

    # Save configuration
    save_config(new_config)

    console.print("\n[green]Configuration saved![/green]")
    console.print(f"  Base URL: [cyan]{new_config.llm.base_url}[/cyan]")
    console.print(f"  Model:    [cyan]{new_config.llm.model}[/cyan]")
    console.print(f"\nSaved to: [dim]{CONFIG_FILE}[/dim]")


if __name__ == "__main__":
    main()
