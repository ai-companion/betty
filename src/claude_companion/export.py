"""Export session data to various formats."""

import json
from datetime import datetime
from pathlib import Path

from .models import Session, Turn


def export_session_markdown(session: Session, output_path: Path | None = None) -> str:
    """Export session to Markdown format."""
    lines = [
        f"# Claude Code Session: {session.display_name}",
        "",
        f"- **Session ID**: `{session.session_id}`",
        f"- **Model**: {session.model}",
        f"- **Started**: {session.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Status**: {'Active' if session.active else 'Ended'}",
        "",
        "## Statistics",
        "",
        f"- Input words: {session.total_input_words:,}",
        f"- Output words: {session.total_output_words:,}",
        f"- Tool calls: {session.total_tool_calls}",
        "",
        "## Turns",
        "",
    ]

    for turn in session.turns:
        if turn.role == "user":
            lines.append(f"### Turn {turn.turn_number} - User ({turn.word_count} words)")
        elif turn.role == "assistant":
            lines.append(f"### Turn {turn.turn_number} - Assistant ({turn.word_count} words)")
        else:
            lines.append(f"### Turn {turn.turn_number} - Tool: {turn.tool_name}")

        lines.append("")
        if turn.summary:
            lines.append(f"**Summary**: {turn.summary}")
            lines.append("")
        if turn.critic:
            lines.append(f"**Critic**: {turn.critic}")
            lines.append("")
        if turn.annotation:
            lines.append(f"**Annotation**: {turn.annotation}")
            lines.append("")
        if turn.analysis:
            sentiment_indicator = {"progress": "✓", "concern": "⚠", "critical": "✗"}.get(turn.analysis.sentiment, "?")
            lines.append(f"**Analysis** ({sentiment_indicator} {turn.analysis.sentiment}): {turn.analysis.summary}")
            if turn.analysis.critique:
                lines.append(f"**Critique**: {turn.analysis.critique}")
            lines.append("")
        lines.append(turn.content_full)
        lines.append("")

    content = "\n".join(lines)

    if output_path:
        output_path.write_text(content)

    return content


def export_session_json(session: Session, output_path: Path | None = None) -> str:
    """Export session to JSON format."""
    data = {
        "session_id": session.session_id,
        "display_name": session.display_name,
        "model": session.model,
        "started_at": session.started_at.isoformat(),
        "active": session.active,
        "stats": {
            "total_input_words": session.total_input_words,
            "total_output_words": session.total_output_words,
            "total_tool_calls": session.total_tool_calls,
        },
        "turns": [
            {
                "turn_number": turn.turn_number,
                "role": turn.role,
                "tool_name": turn.tool_name,
                "content": turn.content_full,
                "summary": turn.summary,
                "critic": turn.critic,
                "critic_sentiment": turn.critic_sentiment,
                "annotation": turn.annotation,
                "analysis": {
                    "summary": turn.analysis.summary,
                    "critique": turn.analysis.critique,
                    "sentiment": turn.analysis.sentiment,
                    "word_count": turn.analysis.word_count,
                    "context_word_count": turn.analysis.context_word_count,
                } if turn.analysis else None,
                "word_count": turn.word_count,
                "timestamp": turn.timestamp.isoformat(),
            }
            for turn in session.turns
        ],
    }

    content = json.dumps(data, indent=2)

    if output_path:
        output_path.write_text(content)

    return content


def get_export_filename(session: Session, format: str) -> str:
    """Generate export filename based on session and format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = session.display_name.replace("/", "-").replace(" ", "_")
    ext = "md" if format == "markdown" else "json"
    return f"claude_session_{name}_{timestamp}.{ext}"
