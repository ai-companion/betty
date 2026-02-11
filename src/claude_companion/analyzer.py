"""On-demand turn analyzer for Claude Companion.

Provides structured LLM analysis (summary + critique + sentiment) for any turn.
Coexists with the existing auto-summarizer — no existing functionality is changed.
"""

import hashlib
import json
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import litellm
import openai

from .models import Turn, count_words, compute_spans

if TYPE_CHECKING:
    from .models import Session

logger = logging.getLogger(__name__)

VALID_SENTIMENTS = {"progress", "concern", "critical"}

ANALYZER_SYSTEM_PROMPT = (
    "You are a supervisor analyzing an AI coding assistant's conversation. "
    "You will be given a target turn (user message, assistant response, or tool action) "
    "with surrounding context.\n\n"
    "Analyze the target turn and return valid JSON with exactly these fields:\n"
    '- "summary": 1-2 sentence summary of what happened in this turn\n'
    '- "critique": 1-2 sentence evaluation of this turn relative to the goal\n'
    '- "sentiment": one of "progress", "concern", or "critical"\n\n'
    "Sentiment guide:\n"
    '- "progress": on track, making forward progress, successful actions\n'
    '- "concern": minor issues, suboptimal approach, partial progress\n'
    '- "critical": serious problems, repeated failures, off-track, destructive actions\n\n'
    "Role-specific guidance:\n"
    "- For user turns: summarize the request/instruction, evaluate clarity and scope\n"
    "- For assistant turns: summarize the response, evaluate correctness and helpfulness\n"
    "- For tool turns: summarize the action taken, evaluate whether it succeeded and was appropriate\n\n"
    "Return ONLY valid JSON, no markdown fences, no extra text."
)

RANGE_SYSTEM_PROMPT = (
    "You are a supervisor analyzing a sequence of turns from an AI coding assistant's conversation. "
    "You will be given multiple turns (a conversation segment) with context.\n\n"
    "Analyze the segment as a whole and return valid JSON with exactly these fields:\n"
    '- "summary": 2-3 sentence summary of what happened in this segment\n'
    '- "critique": 1-2 sentence evaluation of progress relative to the goal\n'
    '- "sentiment": one of "progress", "concern", or "critical"\n\n'
    "Sentiment guide:\n"
    '- "progress": on track, making forward progress, successful actions\n'
    '- "concern": minor issues, suboptimal approach, partial progress\n'
    '- "critical": serious problems, repeated failures, off-track, destructive actions\n\n'
    "Return ONLY valid JSON, no markdown fences, no extra text."
)


@dataclass
class Analysis:
    """Structured analysis result for a turn."""

    summary: str
    critique: str
    sentiment: str  # "progress" | "concern" | "critical"
    word_count: int  # word count of analyzed turn
    context_word_count: int  # word count of context sent to LLM


class GoalExtractor:
    """Extract and cache session goal from first user message."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}  # session_id -> goal

    def extract(self, session: "Session") -> str:
        """Return the session goal text (first user message, truncated to 2000 chars).

        Cached per session to avoid repeated scanning.
        """
        if session.session_id in self._cache:
            return self._cache[session.session_id]

        goal = "[No user message found]"
        for t in session.turns:
            if t.role == "user":
                goal = t.content_full[:2000]
                break

        self._cache[session.session_id] = goal
        return goal


class ContextManager:
    """Builds context for turn analysis."""

    def __init__(self, analyzer_config: "AnalyzerConfig | None" = None) -> None:
        from .config import AnalyzerConfig
        self._goal_extractor = GoalExtractor()
        self._config = analyzer_config or AnalyzerConfig()

    def build_context(self, session: "Session", target_turn: Turn) -> dict:
        """Assemble context for analyzing a turn.

        Args:
            session: The session containing the turn
            target_turn: The turn to analyze

        Returns:
            Context dict with goal, window, active_tasks, target_turn, and positional info
        """
        # Find target index
        target_index = -1
        for i, t in enumerate(session.turns):
            if t is target_turn:
                target_index = i
                break

        if target_index == -1:
            target_index = len(session.turns) - 1

        # Goal: first user turn's content
        goal = self._goal_extractor.extract(session)

        # Sliding window: ~5 before, ~4 after target
        window_start = max(0, target_index - 5)
        window_end = min(len(session.turns), target_index + 5)
        window = []
        for i in range(window_start, window_end):
            t = session.turns[i]
            if t is target_turn:
                continue  # Skip target itself, it gets full content
            truncated = t.content_full[:500]
            if len(t.content_full) > 500:
                truncated += "..."
            window.append({
                "index": i,
                "role": t.role,
                "tool_name": t.tool_name,
                "content": truncated,
            })

        # Active tasks
        active_tasks = []
        for task in session.tasks.values():
            if not task.is_deleted:
                active_tasks.append(f"[{task.status}] {task.subject}")

        return {
            "goal": goal,
            "window": window,
            "active_tasks": active_tasks,
            "target_turn": {
                "role": target_turn.role,
                "tool_name": target_turn.tool_name,
                "content": target_turn.content_full,
            },
            "target_index": target_index,
            "total_turns": len(session.turns),
        }

    def _condensed_turn(self, t: Turn, budget: int) -> str:
        """Return a condensed representation of a turn.

        Uses existing summaries when available, otherwise truncates content.
        User turns get more budget since they carry the intent.
        """
        if t.role == "user":
            content = t.content_full[:budget]
            if len(t.content_full) > budget:
                content += "..."
            return content

        # For assistant turns: prefer existing summary
        if t.role == "assistant" and t.summary:
            return t.summary

        # For tool turns: use tool name + preview
        if t.role == "tool":
            tool = t.tool_name or "tool"
            preview = t.content_preview[:budget]
            return f"[{tool}] {preview}"

        # Fallback: truncate
        content = t.content_full[:budget]
        if len(t.content_full) > budget:
            content += "..."
        return content

    def _build_span_summary(
        self,
        span_turns: list[Turn],
        span_analyses: dict[tuple[int, int], "Analysis"] | None,
    ) -> str:
        """Build a one-paragraph summary for a span.

        Uses existing span analysis if available, otherwise creates
        a condensed description from the span's turns.
        """
        if not span_turns:
            return "[empty span]"

        # Check if we have an existing analysis for this exact span
        if span_analyses:
            span_key = (span_turns[0].turn_number, span_turns[-1].turn_number)
            existing = span_analyses.get(span_key)
            if existing and not existing.summary.startswith("["):
                return existing.summary

        # No existing analysis — build condensed description
        parts = []
        # User message (the span's intent)
        for t in span_turns:
            if t.role == "user":
                user_msg = t.content_full[:300]
                if len(t.content_full) > 300:
                    user_msg += "..."
                parts.append(f"User: {user_msg}")
                break

        # Count assistant/tool activity
        asst_count = sum(1 for t in span_turns if t.role == "assistant")
        tool_count = sum(1 for t in span_turns if t.role == "tool")

        # Use assistant summaries if available
        asst_summaries = []
        for t in span_turns:
            if t.role == "assistant" and t.summary:
                asst_summaries.append(t.summary)

        if asst_summaries:
            parts.append("Assistant: " + " | ".join(asst_summaries[:3]))
        elif asst_count:
            parts.append(f"{asst_count} assistant response(s)")

        if tool_count:
            tool_names = [t.tool_name or "tool" for t in span_turns if t.role == "tool"]
            unique_tools = list(dict.fromkeys(tool_names))[:5]  # Deduplicate, keep order
            parts.append(f"{tool_count} tool call(s): {', '.join(unique_tools)}")

        return " — ".join(parts)

    def build_context_for_range(
        self,
        session: "Session",
        turns: list[Turn],
        span_analyses: dict[tuple[int, int], "Analysis"] | None = None,
    ) -> dict:
        """Build context for analyzing a range of turns.

        Uses a three-tier strategy based on range size:
        - Small (<=10 turns): per-turn content, truncated to 2000 chars
        - Medium (11-30 turns): condensed per-turn (summaries for assistant/tool)
        - Large (>30 turns): span-level summaries (hierarchical/nested approach)

        Args:
            session: The session containing the turns
            turns: The turns in the range to analyze
            span_analyses: Existing span-level analyses for nested summarization

        Returns:
            Context dict with goal, target_turns or span_summaries, surrounding, etc.
        """
        goal = self._goal_extractor.extract(session)

        # Find indices of range in session.turns
        range_indices: list[int] = []
        for t in turns:
            for i, st in enumerate(session.turns):
                if st is t:
                    range_indices.append(i)
                    break

        if not range_indices:
            range_indices = [0]

        range_start = min(range_indices)
        range_end = max(range_indices)

        n_turns = len(turns)

        # Active tasks
        active_tasks = []
        for task in session.tasks.values():
            if not task.is_deleted:
                active_tasks.append(f"[{task.status}] {task.subject}")

        # === Large range (>=large_range_min turns): span-level summaries ===
        if n_turns >= self._config.large_range_min:
            spans = compute_spans(turns)
            span_summaries = []
            for si, (s_start, s_end) in enumerate(spans):
                span_turns = turns[s_start:s_end + 1]
                summary = self._build_span_summary(span_turns, span_analyses)
                span_summaries.append({
                    "span_index": si + 1,
                    "turn_range": f"{span_turns[0].turn_number}-{span_turns[-1].turn_number}",
                    "turn_count": len(span_turns),
                    "summary": summary,
                })

            return {
                "goal": goal,
                "target_turns": [],  # Empty — using span_summaries instead
                "span_summaries": span_summaries,
                "surrounding": [],  # No surrounding for session-level
                "active_tasks": active_tasks,
                "range_start": range_start,
                "range_end": range_end,
                "total_turns": len(session.turns),
            }

        # === Medium range (>small_range_max turns): condensed per-turn ===
        if n_turns > self._config.small_range_max:
            user_count = sum(1 for t in turns if t.role == "user")
            other_count = n_turns - user_count
            total_shares = user_count * 3 + other_count
            medium_budget = self._config.context_budget // 2
            per_share = max(50, medium_budget // max(total_shares, 1))
            user_budget = per_share * 3
            other_budget = per_share
        else:
            user_budget = other_budget = 0  # Not used for small ranges

        # === Small (<=10) and medium (11-30): per-turn content ===
        target_turns = []
        for t in turns:
            role = t.role
            if role == "tool" and t.tool_name:
                role = f"tool:{t.tool_name}"

            if n_turns > self._config.small_range_max:
                budget = user_budget if t.role == "user" else other_budget
                content = self._condensed_turn(t, budget)
            else:
                ptb = self._config.per_turn_budget
                content = t.content_full[:ptb]
                if len(t.content_full) > ptb:
                    content += "..."

            target_turns.append({
                "role": role,
                "tool_name": t.tool_name,
                "content": content,
                "turn_number": t.turn_number,
            })

        # Surrounding: turns just outside the range (truncated, for context)
        surrounding = []
        surround_start = max(0, range_start - 3)
        surround_end = min(len(session.turns), range_end + 4)
        for i in range(surround_start, surround_end):
            if range_start <= i <= range_end:
                continue
            t = session.turns[i]
            truncated = t.content_full[:500]
            if len(t.content_full) > 500:
                truncated += "..."
            surrounding.append({
                "index": i,
                "role": t.role,
                "tool_name": t.tool_name,
                "content": truncated,
            })

        return {
            "goal": goal,
            "target_turns": target_turns,
            "span_summaries": [],  # Empty — using target_turns instead
            "surrounding": surrounding,
            "active_tasks": active_tasks,
            "range_start": range_start,
            "range_end": range_end,
            "total_turns": len(session.turns),
        }


class Analyzer:
    """On-demand turn analyzer using LLM."""

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_workers: int = 1,
        analyzer_config: "AnalyzerConfig | None" = None,
    ):
        from .config import AnalyzerConfig
        self.model = model
        self.api_base = api_base.rstrip("/") if api_base else None
        self.api_key = api_key
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._context_manager = ContextManager(analyzer_config=analyzer_config)

        if self.model.startswith("claude-code/") and not shutil.which("claude"):
            logger.warning(
                "claude-code provider selected but 'claude' CLI not found on PATH. "
                "Analysis will fail. Install Claude Code or switch provider."
            )

    def analyze_async(
        self,
        session: "Session",
        turn: Turn,
        callback: Callable[["Analysis", bool], None],
    ) -> None:
        """Submit analysis job, calls callback with (analysis, success)."""
        self._executor.submit(self._analyze, session, turn, callback)

    def analyze_range_async(
        self,
        session: "Session",
        turns: list[Turn],
        callback: Callable[["Analysis", bool], None],
        span_analyses: dict[tuple[int, int], "Analysis"] | None = None,
    ) -> None:
        """Submit range analysis job, calls callback with (analysis, success)."""
        self._executor.submit(self._analyze_range, session, turns, callback, span_analyses)

    def _analyze(
        self,
        session: "Session",
        turn: Turn,
        callback: Callable[["Analysis", bool], None],
    ) -> None:
        """Build context, call LLM, parse response, call callback."""
        try:
            context = self._context_manager.build_context(session, turn)
            prompt = self._build_prompt(context)
            context_word_count = count_words(prompt)

            raw = self._call_llm(prompt, ANALYZER_SYSTEM_PROMPT)
            analysis = self._parse_response(raw, turn.word_count, context_word_count)
            callback(analysis, True)

        except (litellm.exceptions.APIConnectionError, openai.APIConnectionError, ConnectionError):
            logger.debug(f"Analyzer: {self.model} server not available")
            callback(
                Analysis(
                    summary="[server unavailable]",
                    critique="",
                    sentiment="concern",
                    word_count=turn.word_count,
                    context_word_count=0,
                ),
                False,
            )
        except (litellm.exceptions.Timeout, openai.APITimeoutError, subprocess.TimeoutExpired):
            logger.debug(f"Analyzer: {self.model} request timed out")
            callback(
                Analysis(
                    summary="[timeout]",
                    critique="",
                    sentiment="concern",
                    word_count=turn.word_count,
                    context_word_count=0,
                ),
                False,
            )
        except Exception as e:
            logger.debug(f"Analyzer error ({self.model}): {e}")
            callback(
                Analysis(
                    summary=f"[error: {type(e).__name__}]",
                    critique="",
                    sentiment="concern",
                    word_count=turn.word_count,
                    context_word_count=0,
                ),
                False,
            )

    def _analyze_range(
        self,
        session: "Session",
        turns: list[Turn],
        callback: Callable[["Analysis", bool], None],
        span_analyses: dict[tuple[int, int], "Analysis"] | None = None,
    ) -> None:
        """Build context for range, call LLM, parse response, call callback."""
        total_words = sum(t.word_count for t in turns)
        try:
            context = self._context_manager.build_context_for_range(
                session, turns, span_analyses=span_analyses
            )
            prompt = self._build_prompt_for_range(context)
            context_word_count = count_words(prompt)

            raw = self._call_llm(prompt, RANGE_SYSTEM_PROMPT)
            analysis = self._parse_response(raw, total_words, context_word_count)
            callback(analysis, True)

        except (litellm.exceptions.APIConnectionError, openai.APIConnectionError, ConnectionError):
            logger.debug(f"Analyzer: {self.model} server not available")
            callback(
                Analysis(
                    summary="[server unavailable]",
                    critique="",
                    sentiment="concern",
                    word_count=total_words,
                    context_word_count=0,
                ),
                False,
            )
        except (litellm.exceptions.Timeout, openai.APITimeoutError, subprocess.TimeoutExpired):
            logger.debug(f"Analyzer: {self.model} request timed out")
            callback(
                Analysis(
                    summary="[timeout]",
                    critique="",
                    sentiment="concern",
                    word_count=total_words,
                    context_word_count=0,
                ),
                False,
            )
        except Exception as e:
            logger.debug(f"Analyzer range error ({self.model}): {e}")
            callback(
                Analysis(
                    summary=f"[error: {type(e).__name__}]",
                    critique="",
                    sentiment="concern",
                    word_count=total_words,
                    context_word_count=0,
                ),
                False,
            )

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Call LLM with same routing as Summarizer but different params."""
        if self.model.startswith("claude-code/"):
            return self._call_claude_code(prompt, system_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        if self.api_base:
            from openai import OpenAI

            client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key or "no-key-required",
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.2,
            )
        else:
            kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.2,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            response = litellm.completion(**kwargs)

        return response.choices[0].message.content.strip()

    def _call_claude_code(self, prompt: str, system_prompt: str) -> str:
        """Call claude CLI in single-prompt mode."""
        claude_model = self.model.split("/", 1)[1] if "/" in self.model else self.model

        result = subprocess.run(
            [
                "claude", "-p", "--no-session-persistence", "--model", claude_model,
                "--disable-slash-commands", "--tools", "", "--setting-sources", "",
                "--system-prompt", system_prompt,
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"claude exited with code {result.returncode}")
        return result.stdout.strip()

    def _build_prompt(self, context: dict) -> str:
        """Render context dict into user prompt for LLM."""
        parts = []

        # Goal
        parts.append(f"## Goal (first user message)\n{context['goal']}")

        # Active tasks
        if context["active_tasks"]:
            tasks_str = "\n".join(f"- {t}" for t in context["active_tasks"])
            parts.append(f"## Active Tasks\n{tasks_str}")

        # Surrounding window
        if context["window"]:
            window_lines = []
            for w in context["window"]:
                role = w["role"]
                if role == "tool" and w.get("tool_name"):
                    role = f"tool:{w['tool_name']}"
                window_lines.append(f"[Turn {w['index'] + 1}] ({role}) {w['content']}")
            parts.append(f"## Surrounding Turns\n" + "\n\n".join(window_lines))

        # Target turn
        target = context["target_turn"]
        role = target["role"]
        if role == "tool" and target.get("tool_name"):
            role = f"tool:{target['tool_name']}"
        parts.append(
            f"## Target Turn (turn {context['target_index'] + 1} of {context['total_turns']})\n"
            f"Role: {role}\n\n"
            f"{target['content']}"
        )

        parts.append("Analyze the target turn. Return JSON with summary, critique, and sentiment.")

        return "\n\n".join(parts)

    def _build_prompt_for_range(self, context: dict) -> str:
        """Build prompt for multi-turn range analysis.

        Handles two formats:
        - Per-turn: context has target_turns with individual turn content
        - Span-level: context has span_summaries with nested summaries (for large ranges)
        """
        parts = []

        # Goal
        parts.append(f"## Goal (first user message)\n{context['goal']}")

        # Active tasks
        if context["active_tasks"]:
            tasks_str = "\n".join(f"- {t}" for t in context["active_tasks"])
            parts.append(f"## Active Tasks\n{tasks_str}")

        # Surrounding context
        if context.get("surrounding"):
            surround_lines = []
            for w in context["surrounding"]:
                role = w["role"]
                if role == "tool" and w.get("tool_name"):
                    role = f"tool:{w['tool_name']}"
                surround_lines.append(f"[Turn {w['index'] + 1}] ({role}) {w['content']}")
            parts.append("## Surrounding Context\n" + "\n\n".join(surround_lines))

        range_desc = f"turns {context['range_start'] + 1}\u2013{context['range_end'] + 1} of {context['total_turns']}"

        # Span-level summaries (large ranges / session)
        if context.get("span_summaries"):
            span_lines = []
            for s in context["span_summaries"]:
                span_lines.append(
                    f"Span {s['span_index']} (turns {s['turn_range']}, {s['turn_count']} turns): {s['summary']}"
                )
            parts.append(
                f"## Session Overview ({range_desc}, {len(context['span_summaries'])} spans)\n"
                + "\n\n".join(span_lines)
            )
        # Per-turn content (small/medium ranges)
        elif context.get("target_turns"):
            turn_lines = []
            for t in context["target_turns"]:
                role = t["role"]
                turn_lines.append(
                    f"[Turn {t['turn_number']}] ({role})\n{t['content']}"
                )
            parts.append(f"## Segment ({range_desc})\n" + "\n\n".join(turn_lines))

        parts.append("Analyze this conversation segment as a whole. Return JSON with summary, critique, and sentiment.")

        return "\n\n".join(parts)

    def _parse_response(self, raw: str, word_count: int, context_word_count: int) -> Analysis:
        """Parse JSON response from LLM, handling code fences and malformed JSON."""
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            # Remove first line (```json or ```)
            lines = text.split("\n", 1)
            if len(lines) > 1:
                text = lines[1]
            # Remove trailing ```
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return Analysis(
                        summary="[parse error: invalid JSON]",
                        critique="",
                        sentiment="concern",
                        word_count=word_count,
                        context_word_count=context_word_count,
                    )
            else:
                return Analysis(
                    summary="[parse error: no JSON found]",
                    critique="",
                    sentiment="concern",
                    word_count=word_count,
                    context_word_count=context_word_count,
                )

        summary = str(data.get("summary", "[no summary]"))
        critique = str(data.get("critique", ""))
        sentiment = str(data.get("sentiment", "progress")).lower()

        if sentiment not in VALID_SENTIMENTS:
            sentiment = "progress"

        return Analysis(
            summary=summary,
            critique=critique,
            sentiment=sentiment,
            word_count=word_count,
            context_word_count=context_word_count,
        )

    def shutdown(self) -> None:
        """Shutdown the executor, cancelling pending tasks."""
        self._executor.shutdown(wait=False, cancel_futures=True)


def make_analysis_cache_key(
    turn: Turn,
    context: dict,
    model: str,
    system_prompt: str,
) -> str:
    """Create cache key for an analysis result.

    Pattern: ANALYSIS::{fingerprint}::{content_sig}::{context_sig}

    Args:
        turn: The turn being analyzed
        context: Context dict from ContextManager
        model: Model name
        system_prompt: System prompt text

    Returns:
        Cache key string
    """
    from .summarizer import _prompt_fingerprint

    fingerprint = _prompt_fingerprint(model, system_prompt)
    content_sig = hashlib.sha256(turn.content_full.encode()).hexdigest()[:12]

    # Context signature from goal + window content
    context_parts = [context["goal"]]
    for w in context.get("window", []):
        context_parts.append(w["content"])
    context_str = "|".join(context_parts)
    context_sig = hashlib.sha256(context_str.encode()).hexdigest()[:8]

    return f"ANALYSIS::{fingerprint}::{content_sig}::{context_sig}"


def make_range_cache_key(
    turns: list[Turn],
    context: dict,
    model: str,
    system_prompt: str,
) -> str:
    """Create cache key for a range analysis result.

    Pattern: RANGE_ANALYSIS::{fingerprint}::{content_sig}::{context_sig}

    Args:
        turns: The turns in the range being analyzed
        context: Context dict from ContextManager.build_context_for_range
        model: Model name
        system_prompt: System prompt text

    Returns:
        Cache key string
    """
    from .summarizer import _prompt_fingerprint

    fingerprint = _prompt_fingerprint(model, system_prompt)

    # Content signature from concatenated turn contents
    content_parts = [t.content_full for t in turns]
    content_str = "|".join(content_parts)
    content_sig = hashlib.sha256(content_str.encode()).hexdigest()[:12]

    # Context signature from goal + surrounding content + span summaries
    context_parts = [context["goal"]]
    for w in context.get("surrounding", []):
        context_parts.append(w["content"])
    for s in context.get("span_summaries", []):
        context_parts.append(s["summary"])
    context_str = "|".join(context_parts)
    context_sig = hashlib.sha256(context_str.encode()).hexdigest()[:8]

    return f"RANGE_ANALYSIS::{fingerprint}::{content_sig}::{context_sig}"
