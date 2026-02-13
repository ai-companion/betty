"""Tests for transcript and watcher parsing logic.

Covers issue #130: empty turns from whitespace text blocks and dropped user list-content.
"""

import json
import tempfile
from pathlib import Path

import pytest

from betty.transcript import _parse_entry, parse_transcript
from betty.watcher import TranscriptWatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_entry(content):
    """Build a user JSONL entry."""
    return {"type": "user", "message": {"content": content}}


def _assistant_entry(blocks, **kwargs):
    """Build an assistant JSONL entry with content blocks."""
    msg = {"content": blocks, **kwargs}
    return {"type": "assistant", "message": msg}


# ---------------------------------------------------------------------------
# transcript._parse_entry  — user entries
# ---------------------------------------------------------------------------

class TestTranscriptParseUserString:
    """User entries where content is a plain string."""

    def test_basic_user_message(self):
        turns = _parse_entry(_user_entry("Hello, world!"), 0)
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content_full == "Hello, world!"
        assert turns[0].word_count == 2

    def test_empty_string_produces_no_turns(self):
        turns = _parse_entry(_user_entry(""), 0)
        assert turns == []

    def test_none_content_produces_no_turns(self):
        turns = _parse_entry(_user_entry(None), 0)
        assert turns == []

    def test_missing_message(self):
        turns = _parse_entry({"type": "user"}, 0)
        assert turns == []

    def test_normal_xml_like_message_kept(self):
        """Messages that happen to contain angle brackets but aren't command metadata."""
        content = "Please update the <title> tag in the HTML"
        turns = _parse_entry(_user_entry(content), 0)
        assert len(turns) == 1


class TestTranscriptCommandMerge:
    """Command metadata + expanded content merged into one turn."""

    def _write_jsonl(self, path, entries):
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_command_merged_with_expansion(self):
        """Command metadata + list-content expansion → single merged turn."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        entries = [
            _user_entry(
                "<command-message>agent-review</command-message>\n"
                "<command-name>/agent-review</command-name>"
            ),
            _user_entry([{"type": "text", "text": "Run the agent code review script"}]),
        ]
        self._write_jsonl(path, entries)

        turns, _, _ = parse_transcript(path)
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content_full.startswith("/agent-review\n\n")
        assert "Run the agent code review script" in turns[0].content_full

    def test_command_without_name_not_merged(self):
        """Command metadata without <command-name> passes through as normal."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        entries = [
            _user_entry("<command-message>commit</command-message>"),
            _user_entry([{"type": "text", "text": "Expanded content"}]),
        ]
        self._write_jsonl(path, entries)

        turns, _, _ = parse_transcript(path)
        # No command name → first entry is a normal turn, not merged
        assert len(turns) == 2

    def test_command_followed_by_non_user_not_merged(self):
        """Command metadata followed by assistant → command is consumed, assistant unaffected."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        entries = [
            _user_entry(
                "<command-message>review</command-message>\n"
                "<command-name>/review</command-name>"
            ),
            _assistant_entry([{"type": "text", "text": "I'll review"}]),
        ]
        self._write_jsonl(path, entries)

        turns, _, _ = parse_transcript(path)
        assert len(turns) == 1
        assert turns[0].role == "assistant"


class TestTranscriptParseUserList:
    """User entries where content is a list (e.g. slash command expansions)."""

    def test_single_text_block(self):
        content = [{"type": "text", "text": "Fix the login bug"}]
        turns = _parse_entry(_user_entry(content), 0)
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content_full == "Fix the login bug"

    def test_tool_result_produces_no_turns(self):
        content = [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]
        turns = _parse_entry(_user_entry(content), 0)
        assert turns == []

    def test_image_block_ignored(self):
        content = [{"type": "image", "source": {"data": "base64..."}}]
        turns = _parse_entry(_user_entry(content), 0)
        assert turns == []

    def test_mixed_text_and_tool_result(self):
        content = [
            {"type": "text", "text": "Here is context"},
            {"type": "tool_result", "tool_use_id": "x", "content": "ok"},
        ]
        turns = _parse_entry(_user_entry(content), 0)
        assert len(turns) == 1
        assert turns[0].content_full == "Here is context"

    def test_multiple_text_blocks_concatenated(self):
        content = [
            {"type": "text", "text": "First part."},
            {"type": "text", "text": "Second part."},
        ]
        turns = _parse_entry(_user_entry(content), 0)
        assert len(turns) == 1
        assert "First part." in turns[0].content_full
        assert "Second part." in turns[0].content_full

    def test_empty_list_produces_no_turns(self):
        turns = _parse_entry(_user_entry([]), 0)
        assert turns == []

    def test_whitespace_only_text_block_ignored(self):
        content = [{"type": "text", "text": "  \n  "}]
        turns = _parse_entry(_user_entry(content), 0)
        assert turns == []

    def test_null_text_in_block_ignored(self):
        """text value of None should not crash."""
        content = [{"type": "text", "text": None}]
        turns = _parse_entry(_user_entry(content), 0)
        assert turns == []

    def test_non_string_text_in_block_ignored(self):
        """text value of non-string (int) should not crash."""
        content = [{"type": "text", "text": 42}]
        turns = _parse_entry(_user_entry(content), 0)
        assert turns == []


# ---------------------------------------------------------------------------
# transcript._parse_entry  — assistant entries
# ---------------------------------------------------------------------------

class TestTranscriptParseAssistant:
    """Assistant entry parsing, focusing on whitespace filtering."""

    def test_basic_text_block(self):
        entry = _assistant_entry([{"type": "text", "text": "I'll help you."}])
        turns = _parse_entry(entry, 0)
        assert len(turns) == 1
        assert turns[0].role == "assistant"
        assert turns[0].content_full == "I'll help you."

    def test_whitespace_newlines_produce_no_turns(self):
        """The main bug: '\\n\\n' prefill tokens should be filtered."""
        entry = _assistant_entry([{"type": "text", "text": "\n\n"}])
        turns = _parse_entry(entry, 0)
        assert turns == []

    def test_whitespace_spaces_produce_no_turns(self):
        entry = _assistant_entry([{"type": "text", "text": "   "}])
        turns = _parse_entry(entry, 0)
        assert turns == []

    def test_whitespace_tabs_produce_no_turns(self):
        entry = _assistant_entry([{"type": "text", "text": "\t\n"}])
        turns = _parse_entry(entry, 0)
        assert turns == []

    def test_tool_use_block(self):
        entry = _assistant_entry([
            {"type": "tool_use", "name": "Read", "input": {"file_path": "foo.py"}},
        ])
        turns = _parse_entry(entry, 0)
        assert len(turns) == 1
        assert turns[0].role == "tool"
        assert turns[0].tool_name == "Read"

    def test_thinking_block_ignored(self):
        entry = _assistant_entry([{"type": "thinking", "thinking": "Let me think..."}])
        turns = _parse_entry(entry, 0)
        assert turns == []

    def test_mixed_thinking_text_tool(self):
        entry = _assistant_entry([
            {"type": "thinking", "thinking": "Hmm"},
            {"type": "text", "text": "Here's my answer"},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        ])
        turns = _parse_entry(entry, 0)
        assert len(turns) == 2
        assert turns[0].role == "assistant"
        assert turns[1].role == "tool"

    def test_whitespace_text_followed_by_tool(self):
        """Whitespace prefill before thinking — only tool should survive."""
        entry = _assistant_entry([
            {"type": "text", "text": "\n\n"},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "x.py"}},
        ])
        turns = _parse_entry(entry, 0)
        assert len(turns) == 1
        assert turns[0].role == "tool"

    def test_empty_text_block_filtered(self):
        entry = _assistant_entry([{"type": "text", "text": ""}])
        turns = _parse_entry(entry, 0)
        assert turns == []

    def test_empty_content_list(self):
        entry = _assistant_entry([])
        turns = _parse_entry(entry, 0)
        assert turns == []

    def test_null_text_in_block_no_crash(self):
        """Assistant text block with None text should not crash."""
        entry = _assistant_entry([{"type": "text", "text": None}])
        turns = _parse_entry(entry, 0)
        assert turns == []

    def test_non_string_text_in_block_no_crash(self):
        """Assistant text block with non-string text should not crash."""
        entry = _assistant_entry([{"type": "text", "text": 123}])
        turns = _parse_entry(entry, 0)
        assert turns == []

    def test_token_data_on_first_non_whitespace_turn(self):
        """Token data should attach to first real turn, not be lost on whitespace."""
        entry = _assistant_entry(
            [
                {"type": "text", "text": "\n\n"},
                {"type": "text", "text": "Real content here"},
            ],
            model="claude-sonnet-4-20250115",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        turns = _parse_entry(entry, 0)
        assert len(turns) == 1
        assert turns[0].content_full == "Real content here"
        assert turns[0].input_tokens == 100
        assert turns[0].output_tokens == 50
        assert turns[0].model_id == "claude-sonnet-4-20250115"


# ---------------------------------------------------------------------------
# transcript.parse_transcript — full file parsing
# ---------------------------------------------------------------------------

class TestParseTranscript:
    def _write_jsonl(self, path: Path, entries: list[dict]):
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_multi_entry_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        entries = [
            _user_entry("Hello"),
            _assistant_entry([{"type": "text", "text": "Hi there!"}]),
            _user_entry("Do something"),
            _assistant_entry([{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]),
        ]
        self._write_jsonl(path, entries)

        turns, pos, _ = parse_transcript(path)
        assert len(turns) == 4
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"
        assert turns[2].role == "user"
        assert turns[3].role == "tool"
        # Turn numbers are sequential starting at 1
        assert [t.turn_number for t in turns] == [1, 2, 3, 4]
        assert pos > 0

    def test_whitespace_entries_filtered_in_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        entries = [
            _user_entry("Hello"),
            _assistant_entry([{"type": "text", "text": "\n\n"}]),  # ghost turn
            _assistant_entry([{"type": "text", "text": "Real response"}]),
        ]
        self._write_jsonl(path, entries)

        turns, pos, _ = parse_transcript(path)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"
        assert turns[1].content_full == "Real response"
        # Turn numbers should be contiguous (no gaps from filtered entries)
        assert [t.turn_number for t in turns] == [1, 2]

    def test_user_list_content_in_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        entries = [
            _user_entry([{"type": "text", "text": "Init issue context"}]),
            _assistant_entry([{"type": "text", "text": "Got it"}]),
        ]
        self._write_jsonl(path, entries)

        turns, pos, _ = parse_transcript(path)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[0].content_full == "Init issue context"

    def test_malformed_json_skipped(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        with open(path, "w") as f:
            f.write(json.dumps(_user_entry("Before")) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps(_user_entry("After")) + "\n")

        turns, pos, _ = parse_transcript(path)
        assert len(turns) == 2
        assert turns[0].content_full == "Before"
        assert turns[1].content_full == "After"

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        turns, pos, _ = parse_transcript(path)
        assert turns == []
        assert pos == 0


# ---------------------------------------------------------------------------
# watcher._parse_entry
# ---------------------------------------------------------------------------

class TestWatcherParseUserString:
    def _make_watcher(self):
        return TranscriptWatcher(on_turn=lambda t: None)

    def test_basic_user_message(self):
        w = self._make_watcher()
        turns = w._parse_entry(_user_entry("Hello!"))
        assert len(turns) == 1
        assert turns[0].role == "user"

    def test_empty_string_no_turns(self):
        w = self._make_watcher()
        turns = w._parse_entry(_user_entry(""))
        assert turns == []


class TestWatcherParseUserList:
    def _make_watcher(self):
        return TranscriptWatcher(on_turn=lambda t: None)

    def test_single_text_block(self):
        w = self._make_watcher()
        content = [{"type": "text", "text": "Fix the bug"}]
        turns = w._parse_entry(_user_entry(content))
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content_full == "Fix the bug"

    def test_tool_result_no_turns(self):
        w = self._make_watcher()
        content = [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]
        turns = w._parse_entry(_user_entry(content))
        assert turns == []

    def test_mixed_text_and_tool_result(self):
        w = self._make_watcher()
        content = [
            {"type": "text", "text": "Context here"},
            {"type": "tool_result", "tool_use_id": "x", "content": "ok"},
        ]
        turns = w._parse_entry(_user_entry(content))
        assert len(turns) == 1
        assert turns[0].content_full == "Context here"


class TestWatcherParseAssistant:
    def _make_watcher(self):
        return TranscriptWatcher(on_turn=lambda t: None)

    def test_whitespace_text_filtered(self):
        w = self._make_watcher()
        entry = _assistant_entry([{"type": "text", "text": "\n\n"}])
        turns = w._parse_entry(entry)
        assert turns == []

    def test_real_text_passes(self):
        w = self._make_watcher()
        entry = _assistant_entry([{"type": "text", "text": "Hello"}])
        turns = w._parse_entry(entry)
        assert len(turns) == 1

    def test_spaces_only_filtered(self):
        w = self._make_watcher()
        entry = _assistant_entry([{"type": "text", "text": "   "}])
        turns = w._parse_entry(entry)
        assert turns == []

    def test_null_text_no_crash(self):
        w = self._make_watcher()
        entry = _assistant_entry([{"type": "text", "text": None}])
        turns = w._parse_entry(entry)
        assert turns == []


# ---------------------------------------------------------------------------
# watcher file polling integration
# ---------------------------------------------------------------------------

class TestWatcherCommandMerge:
    def test_command_merged_via_polling(self):
        """Watcher merges command metadata + expansion into one turn."""
        received = []
        watcher = TranscriptWatcher(on_turn=lambda t: received.append(t))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)
            f.write(json.dumps(_user_entry(
                "<command-message>commit</command-message>\n"
                "<command-name>/commit</command-name>"
            )) + "\n")
            f.write(json.dumps(_user_entry(
                [{"type": "text", "text": "commit the staged changes"}]
            )) + "\n")

        watcher._current_path = path
        watcher._last_position = 0
        watcher._turn_number = 0

        watcher._check_for_updates()
        assert len(received) == 1
        assert received[0].content_full.startswith("/commit\n\n")
        assert "commit the staged changes" in received[0].content_full


class TestWatcherFilePoll:
    def test_new_content_detected(self):
        """Watcher detects and parses new lines appended to file."""
        received = []
        watcher = TranscriptWatcher(on_turn=lambda t: received.append(t))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)
            f.write(json.dumps(_user_entry("First")) + "\n")

        # Set up watcher state to read from beginning
        watcher._current_path = path
        watcher._last_position = 0
        watcher._turn_number = 0

        watcher._check_for_updates()
        assert len(received) == 1
        assert received[0].content_full == "First"

        # Append more content
        with open(path, "a") as f:
            f.write(json.dumps(_assistant_entry([{"type": "text", "text": "Response"}])) + "\n")

        watcher._check_for_updates()
        assert len(received) == 2
        assert received[1].content_full == "Response"

    def test_partial_line_retried(self):
        """Incomplete JSON line (no newline) is retried on next poll."""
        received = []
        watcher = TranscriptWatcher(on_turn=lambda t: received.append(t))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)
            # Write complete line + incomplete line (no newline)
            f.write(json.dumps(_user_entry("Complete")) + "\n")
            f.write('{"type": "user", "message": {"content": "Incom')

        watcher._current_path = path
        watcher._last_position = 0
        watcher._turn_number = 0

        watcher._check_for_updates()
        assert len(received) == 1  # Only the complete line

        # Now complete the line
        with open(path, "a") as f:
            f.write('plete"}}\n')

        watcher._check_for_updates()
        assert len(received) == 2
        assert received[1].content_full == "Incomplete"

    def test_malformed_json_skipped(self):
        """Complete but malformed JSON lines are skipped."""
        received = []
        watcher = TranscriptWatcher(on_turn=lambda t: received.append(t))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)
            f.write(json.dumps(_user_entry("Before")) + "\n")
            f.write("not valid json at all\n")
            f.write(json.dumps(_user_entry("After")) + "\n")

        watcher._current_path = path
        watcher._last_position = 0
        watcher._turn_number = 0

        watcher._check_for_updates()
        assert len(received) == 2
        assert received[0].content_full == "Before"
        assert received[1].content_full == "After"
