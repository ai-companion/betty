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

from .models import Turn, count_words

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


@dataclass
class Analysis:
    """Structured analysis result for a turn."""

    summary: str
    critique: str
    sentiment: str  # "progress" | "concern" | "critical"
    word_count: int  # word count of analyzed turn
    context_word_count: int  # word count of context sent to LLM


class ContextManager:
    """Builds context for turn analysis."""

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
        goal = "[No user message found]"
        for t in session.turns:
            if t.role == "user":
                goal = t.content_full[:2000]
                break

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


class Analyzer:
    """On-demand turn analyzer using LLM."""

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_workers: int = 1,
    ):
        self.model = model
        self.api_base = api_base.rstrip("/") if api_base else None
        self.api_key = api_key
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._context_manager = ContextManager()

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
