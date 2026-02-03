"""Mock session generator for development without Claude Code.

This module creates fake Claude Code sessions for testing and development,
particularly useful in cloud-based environments without access to Claude Code.
"""

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Sample conversation data for demo mode
DEMO_CONVERSATION = [
    {
        "type": "user",
        "message": {"content": "Can you help me create a simple Python script that reads a CSV file and prints the first 5 rows?"}
    },
    {
        "type": "assistant",
        "content": [
            {"type": "text", "text": "I'll help you create a Python script to read and display CSV data. Let me first check if there are any existing CSV files in the project."}
        ]
    },
    {
        "type": "assistant",
        "content": [
            {"type": "tool_use", "name": "Glob", "input": {"pattern": "**/*.csv"}}
        ]
    },
    {
        "type": "assistant",
        "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/mock-project/data/sample.csv"}}
        ]
    },
    {
        "type": "assistant",
        "content": [
            {"type": "text", "text": "I found a sample CSV file. Now let me create a script to read and display it."}
        ]
    },
    {
        "type": "assistant",
        "content": [
            {"type": "tool_use", "name": "Write", "input": {
                "file_path": "/tmp/mock-project/read_csv.py",
                "content": """import csv
import sys

def read_csv(filename, num_rows=5):
    \"\"\"Read a CSV file and print the first num_rows rows.\"\"\"
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= num_rows:
                break
            print(', '.join(row))

if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv) > 1 else 'data/sample.csv'
    read_csv(filename)
"""
            }}
        ]
    },
    {
        "type": "assistant",
        "content": [
            {"type": "tool_use", "name": "Bash", "input": {
                "command": "python /tmp/mock-project/read_csv.py",
                "description": "Run the CSV reader script"
            }}
        ]
    },
    {
        "type": "assistant",
        "content": [
            {"type": "text", "text": "I've created the script and tested it. The `read_csv.py` script:\n\n1. Takes an optional filename argument (defaults to `data/sample.csv`)\n2. Opens the CSV file and reads it using Python's csv module\n3. Prints the first 5 rows (configurable via the `num_rows` parameter)\n\nYou can run it with:\n```bash\npython read_csv.py           # Uses default file\npython read_csv.py myfile.csv # Uses specified file\n```"}
        ]
    },
    {
        "type": "user",
        "message": {"content": "Can you add error handling for when the file doesn't exist?"}
    },
    {
        "type": "assistant",
        "content": [
            {"type": "text", "text": "Good idea! Let me add proper error handling to the script."}
        ]
    },
    {
        "type": "assistant",
        "content": [
            {"type": "tool_use", "name": "Edit", "input": {
                "file_path": "/tmp/mock-project/read_csv.py",
                "old_string": """def read_csv(filename, num_rows=5):
    \"\"\"Read a CSV file and print the first num_rows rows.\"\"\"
    with open(filename, 'r', newline='') as f:""",
                "new_string": """def read_csv(filename, num_rows=5):
    \"\"\"Read a CSV file and print the first num_rows rows.\"\"\"
    try:
        with open(filename, 'r', newline='') as f:"""
            }}
        ]
    },
    {
        "type": "assistant",
        "content": [
            {"type": "tool_use", "name": "Edit", "input": {
                "file_path": "/tmp/mock-project/read_csv.py",
                "old_string": """            print(', '.join(row))""",
                "new_string": """            print(', '.join(row))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied reading '{filename}'.", file=sys.stderr)
        sys.exit(1)"""
            }}
        ]
    },
    {
        "type": "assistant",
        "content": [
            {"type": "text", "text": "Done! The script now handles:\n\n- **FileNotFoundError**: When the specified file doesn't exist\n- **PermissionError**: When you don't have read access to the file\n\nBoth errors print a helpful message to stderr and exit with code 1."}
        ]
    },
]


class MockSession:
    """Generate mock Claude Code sessions for testing."""

    def __init__(self, project_path: str | None = None, session_id: str | None = None):
        """Initialize mock session.

        Args:
            project_path: Simulated project path (default: /tmp/mock-project)
            session_id: Custom session ID (default: auto-generated UUID)
        """
        self.project_path = project_path or "/tmp/mock-project"
        self.session_id = session_id or str(uuid.uuid4())
        self._entry_count = 0

        # Encode project path like Claude Code does
        encoded_path = self.project_path.replace("/", "-")
        if not encoded_path.startswith("-"):
            encoded_path = "-" + encoded_path.lstrip("-")

        self.project_dir = Path.home() / ".claude" / "projects" / encoded_path
        self.session_file = self.project_dir / f"{self.session_id}.jsonl"

    def setup(self) -> Path:
        """Create the project directory and session file.

        Returns:
            Path to the session file
        """
        self.project_dir.mkdir(parents=True, exist_ok=True)
        # Create empty file
        self.session_file.touch()
        return self.session_file

    def cleanup(self) -> None:
        """Remove the mock session file."""
        if self.session_file.exists():
            self.session_file.unlink()

    def _timestamp(self) -> str:
        """Generate ISO 8601 timestamp."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def add_user_message(self, content: str) -> None:
        """Add a user message to the session."""
        entry = {
            "type": "user",
            "timestamp": self._timestamp(),
            "message": {"content": content}
        }
        self._write_entry(entry)

    def add_assistant_text(self, text: str) -> None:
        """Add an assistant text response."""
        entry = {
            "type": "assistant",
            "timestamp": self._timestamp(),
            "message": {
                "content": [{"type": "text", "text": text}]
            }
        }
        self._write_entry(entry)

    def add_tool_use(self, tool_name: str, tool_input: dict) -> None:
        """Add a tool use entry."""
        entry = {
            "type": "assistant",
            "timestamp": self._timestamp(),
            "message": {
                "content": [{
                    "type": "tool_use",
                    "id": f"tool_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": tool_input
                }]
            }
        }
        self._write_entry(entry)

    def add_entry(self, entry_data: dict) -> None:
        """Add a raw entry (from DEMO_CONVERSATION format).

        Handles both user and assistant message formats.
        """
        entry = {"timestamp": self._timestamp()}

        if entry_data.get("type") == "user":
            entry["type"] = "user"
            entry["message"] = entry_data.get("message", {})
        elif entry_data.get("type") == "assistant":
            entry["type"] = "assistant"
            content = entry_data.get("content", [])
            # Add IDs to tool_use blocks
            for block in content:
                if block.get("type") == "tool_use" and "id" not in block:
                    block["id"] = f"tool_{uuid.uuid4().hex[:8]}"
            entry["message"] = {"content": content}

        self._write_entry(entry)

    def _write_entry(self, entry: dict) -> None:
        """Write an entry to the session file."""
        with open(self.session_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._entry_count += 1


def run_interactive(project_path: str | None = None) -> None:
    """Run an interactive mock session.

    Press Enter to add the next message, 'q' to quit.
    """
    session = MockSession(project_path=project_path)
    session_file = session.setup()

    print(f"Mock session created: {session_file}")
    print(f"Session ID: {session.session_id}")
    print(f"Project path: {session.project_path}")
    print("\nRun 'claude-companion' in another terminal to watch this session.")
    print("\nPress Enter to add messages, 'u' for user message, 'a' for assistant, 'q' to quit.\n")

    message_index = 0

    try:
        while True:
            cmd = input(f"[{message_index}] > ").strip().lower()

            if cmd == 'q':
                break
            elif cmd == 'u':
                text = input("User message: ")
                if text:
                    session.add_user_message(text)
                    print(f"  Added user message")
                    message_index += 1
            elif cmd == 'a':
                text = input("Assistant response: ")
                if text:
                    session.add_assistant_text(text)
                    print(f"  Added assistant response")
                    message_index += 1
            elif cmd == '' and message_index < len(DEMO_CONVERSATION):
                # Add next demo message
                entry = DEMO_CONVERSATION[message_index]
                session.add_entry(entry)
                entry_type = entry.get("type", "unknown")
                if entry_type == "user":
                    preview = entry.get("message", {}).get("content", "")[:50]
                else:
                    content = entry.get("content", [])
                    if content and content[0].get("type") == "text":
                        preview = content[0].get("text", "")[:50]
                    elif content and content[0].get("type") == "tool_use":
                        preview = f"[{content[0].get('name')}]"
                    else:
                        preview = "..."
                print(f"  Added {entry_type}: {preview}...")
                message_index += 1
            elif cmd == '':
                print("  (end of demo conversation, use 'u' or 'a' to add custom messages)")

    except (KeyboardInterrupt, EOFError):
        pass

    print(f"\nSession file: {session_file}")
    print("Session will persist until manually deleted.")


def run_demo(project_path: str | None = None, delay: float = 1.5, on_entry: Callable[[], None] | None = None) -> None:
    """Run a demo that auto-plays a sample conversation.

    Args:
        project_path: Simulated project path
        delay: Seconds between messages
        on_entry: Optional callback after each entry is added
    """
    session = MockSession(project_path=project_path)
    session_file = session.setup()

    print(f"Mock session created: {session_file}")
    print(f"Session ID: {session.session_id}")
    print(f"\nPlaying demo conversation with {delay}s delay...")
    print("Press Ctrl+C to stop.\n")

    try:
        for i, entry in enumerate(DEMO_CONVERSATION):
            entry_type = entry.get("type", "unknown")

            # Show what we're adding
            if entry_type == "user":
                preview = entry.get("message", {}).get("content", "")[:60]
                print(f"[{i+1}/{len(DEMO_CONVERSATION)}] User: {preview}...")
            else:
                content = entry.get("content", [])
                if content and content[0].get("type") == "text":
                    preview = content[0].get("text", "")[:60]
                    print(f"[{i+1}/{len(DEMO_CONVERSATION)}] Assistant: {preview}...")
                elif content and content[0].get("type") == "tool_use":
                    tool_name = content[0].get("name", "unknown")
                    print(f"[{i+1}/{len(DEMO_CONVERSATION)}] Tool: {tool_name}")

            session.add_entry(entry)

            if on_entry:
                on_entry()

            time.sleep(delay)

    except KeyboardInterrupt:
        print("\n\nDemo stopped.")

    print(f"\nSession file: {session_file}")
    print("Run 'claude-companion' to view the session.")


def main() -> None:
    """CLI entry point for mock session."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate mock Claude Code sessions for development")
    parser.add_argument("--demo", action="store_true", help="Run demo with auto-playing conversation")
    parser.add_argument("--project", "-p", help="Simulated project path (default: /tmp/mock-project)")
    parser.add_argument("--delay", "-d", type=float, default=1.5, help="Delay between messages in demo mode (default: 1.5s)")

    args = parser.parse_args()

    if args.demo:
        run_demo(project_path=args.project, delay=args.delay)
    else:
        run_interactive(project_path=args.project)


if __name__ == "__main__":
    main()
