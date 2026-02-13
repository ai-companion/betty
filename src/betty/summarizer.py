"""LLM summarization for assistant turns using litellm, openai SDK, or claude-code subprocess."""

import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable

import litellm
import openai

from .models import Turn

if TYPE_CHECKING:
    from .models import Session

logger = logging.getLogger(__name__)

# Suppress litellm's verbose debug/info logging
litellm.suppress_debug_info = True

# System prompt for assistant text summarization (no tool context)
SYSTEM_PROMPT = (
    "You are a supervisor observing an AI coding assistant at work. "
    "Summarize what the assistant communicated in 1 sentence. "
    "Focus on the key point or explanation given to the user. "
    "Use third person (e.g., 'Explained the authentication flow'). "
    "Be concise."
)

# System prompt for tool summarization
TOOL_SYSTEM_PROMPT = (
    "You are a supervisor observing an AI coding assistant at work. "
    "Summarize the tool actions taken in 1 sentence. "
    "Focus on what was done (files read/written, commands run). "
    "Use third person (e.g., 'Read config.py, edited server.py to fix the bug'). "
    "Be concise and action-oriented."
)

# System prompt for critic evaluation
CRITIC_SYSTEM_PROMPT = (
    "You are a supervisor evaluating an AI coding assistant's work. "
    "Assess the assistant's response AND tool actions together. Consider: "
    "Did commands succeed or fail? Were files modified correctly? Is the approach sound?\n\n"
    "Start with exactly one indicator:\n"
    "✓ = making progress, tools succeeded, on track\n"
    "⚠ = partial progress, minor concerns (e.g. retry needed, suboptimal approach)\n"
    "✗ = critical issue needing human attention (destructive action, repeated failures, "
    "off-track from request, or errors the assistant hasn't addressed)\n\n"
    "Follow with a 1-2 sentence evaluation. Be specific."
)


def _prompt_fingerprint(model: str, system_prompt: str) -> str:
    """Create a short fingerprint from model name and system prompt.

    This ensures cache keys are invalidated when the model or prompt changes.

    Args:
        model: Model name/identifier
        system_prompt: System prompt text

    Returns:
        12-char hex digest uniquely identifying the model+prompt combination
    """
    import hashlib
    return hashlib.sha256(f"{model}|{system_prompt}".encode()).hexdigest()[:12]


def make_tool_cache_key(tool_turns: list[Turn], model: str, system_prompt: str) -> str:
    """Create cache key for a sequence of tool turns.

    Args:
        tool_turns: List of tool turns
        model: Model name (for cache invalidation on model change)
        system_prompt: System prompt (for cache invalidation on prompt change)

    Returns:
        Cache key that uniquely identifies this tool sequence
    """
    import hashlib
    fingerprint = _prompt_fingerprint(model, system_prompt)
    # use hash of full content to avoid collisions from truncated previews
    tool_sig = "|".join(
        f"{t.tool_name}:{hashlib.sha256(t.content_full.encode()).hexdigest()[:12]}"
        for t in tool_turns
    )
    return f"TOOLS::{fingerprint}::{tool_sig}"


def make_critic_cache_key(assistant_turn: Turn, context: dict, model: str, system_prompt: str) -> str:
    """Create cache key for critic evaluation.

    Args:
        assistant_turn: The assistant turn being critiqued
        context: Context dictionary with user_message and active_tasks
        model: Model name (for cache invalidation on model change)
        system_prompt: System prompt (for cache invalidation on prompt change)

    Returns:
        Cache key that uniquely identifies this critique
    """
    import hashlib
    fingerprint = _prompt_fingerprint(model, system_prompt)
    content_sig = hashlib.sha256(assistant_turn.content_full.encode()).hexdigest()[:12]
    context_str = f"{context['user_message']}|{context['active_tasks']}|{context.get('tool_summary', '')}"
    context_sig = hashlib.sha256(context_str.encode()).hexdigest()[:8]
    return f"CRITIC::{fingerprint}::{content_sig}::{context_sig}"


def _get_turn_context(session: "Session", assistant_turn: Turn, max_tools: int = 10) -> tuple[Turn | None, list[Turn]]:
    """Get context for summarizing an assistant turn.

    Looks backward from the assistant turn to find the preceding user message
    and consecutive tool calls.

    Args:
        session: The session containing all turns
        assistant_turn: The assistant turn being summarized
        max_tools: Maximum number of tools to include (prevents huge context)

    Returns:
        Tuple of (user_turn, tool_turns):
        - user_turn: The user message that triggered this sequence (or None)
        - tool_turns: List of tool turns in chronological order (oldest first)
    """
    tools = []
    user_turn = None

    try:
        turn_idx = session.turns.index(assistant_turn)
    except ValueError:
        # Turn not in session (shouldn't happen, but handle gracefully)
        return None, []

    # Walk backward from assistant turn
    for i in range(turn_idx - 1, -1, -1):
        prev_turn = session.turns[i]

        if prev_turn.role == "user":
            user_turn = prev_turn
            break  # Found the user message that started this sequence

        elif prev_turn.role == "assistant":
            break  # Hit previous assistant response, stop here

        elif prev_turn.role == "tool":
            tools.insert(0, prev_turn)  # Insert at start to maintain chronological order

            # Limit context size
            if len(tools) >= max_tools:
                break

    return user_turn, tools


def _get_critic_context(session: "Session", assistant_turn: Turn) -> dict:
    """Get context for critiquing an assistant turn.

    Args:
        session: The session containing all turns
        assistant_turn: The assistant turn being critiqued

    Returns:
        Dictionary with user_message, assistant_response, active_tasks,
        turn_number, total_turns, and tool_summary
    """
    # Find preceding user message and tool turns
    user_turn, tool_turns = _get_turn_context(session, assistant_turn)
    user_message = user_turn.content_full if user_turn else "[No user message]"

    # Get active tasks
    active_tasks = [
        f"- [{task.status}] {task.subject}"
        for task in session.tasks.values()
        if not task.is_deleted
    ]
    tasks_summary = "\n".join(active_tasks) if active_tasks else "[No active tasks]"

    # Build tool summary from preceding tool turns
    max_tool_content = 300
    tool_lines = []
    for t in tool_turns[:10]:
        content = t.content_full
        if len(content) > max_tool_content:
            content = content[:max_tool_content] + "..."
        tool_lines.append(f"  - {t.tool_name}: {content}")
    tool_summary = "\n".join(tool_lines) if tool_lines else "  [No tools called]"

    return {
        "user_message": user_message,
        "assistant_response": assistant_turn.content_full,
        "active_tasks": tasks_summary,
        "tool_summary": tool_summary,
        "turn_number": assistant_turn.turn_number,
        "total_turns": len(session.turns),
    }


class Summarizer:
    """Summarizes assistant turns using litellm (any provider) or claude-code subprocess."""

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_workers: int = 2,
    ):
        self.model = model
        self.api_base = api_base.rstrip("/") if api_base else None
        self.api_key = api_key
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Preflight check for claude-code provider
        if self.model.startswith("claude-code/") and not shutil.which("claude"):
            logger.warning(
                "claude-code provider selected but 'claude' CLI not found on PATH. "
                "Summarization will fail. Install Claude Code or switch provider: "
                "betty config --llm-preset <provider>"
            )

    def summarize_async(
        self,
        turn: Turn,
        callback: Callable[[str, bool], None]
    ) -> None:
        """Submit assistant summarization job, calls callback with (result, success)."""
        self._executor.submit(self._summarize, turn, callback)

    def summarize_tools_async(
        self,
        tool_turns: list[Turn],
        callback: Callable[[str, bool], None]
    ) -> None:
        """Submit tool summarization job, calls callback with (result, success)."""
        self._executor.submit(self._summarize_tools, tool_turns, callback)

    def critique_async(
        self,
        turn: Turn,
        context: dict,
        callback: Callable[[str, str, bool], None]
    ) -> None:
        """Submit critique job, calls callback with (critique, sentiment, success)."""
        self._executor.submit(self._critique, turn, context, callback)

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Call LLM via litellm, openai SDK, or claude-code subprocess.

        Routing:
        - claude-code/* → subprocess
        - api_base set  → openai SDK direct (local/custom servers)
        - otherwise     → litellm (cloud providers with auto-routing)

        Args:
            prompt: User prompt to send
            system_prompt: System prompt for context

        Returns:
            LLM response text
        """
        if self.model.startswith("claude-code/"):
            return self._call_claude_code(prompt, system_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        if self.api_base:
            # Local/custom OpenAI-compatible server: use openai SDK directly
            # to avoid litellm model-name parsing and auth issues.
            from openai import OpenAI
            client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key or "no-key-required",
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.3,
            )
        else:
            # Cloud provider: use litellm for routing (openai/, anthropic/, etc.)
            kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 150,
                "temperature": 0.3,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            response = litellm.completion(**kwargs)

        return response.choices[0].message.content.strip()

    def _call_claude_code(self, prompt: str, system_prompt: str) -> str:
        """Call claude CLI in single-prompt mode.

        Prompt is piped via stdin to avoid OS ARG_MAX limits on long prompts.
        """
        # Extract model name from "claude-code/<model>" prefix
        claude_model = self.model.split("/", 1)[1] if "/" in self.model else self.model

        result = subprocess.run(
            ["claude", "-p", "--no-session-persistence", "--model", claude_model,
             "--disable-slash-commands", "--tools", "", "--setting-sources", "",
             "--system-prompt", system_prompt],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"claude exited with code {result.returncode}")
        return result.stdout.strip()

    def _summarize_tools(
        self,
        tool_turns: list[Turn],
        callback: Callable[[str, bool], None]
    ) -> None:
        """Call LLM to summarize tool actions."""
        try:
            if not tool_turns:
                callback("[no tools]", False)
                return

            # Build tool actions prompt (truncate each tool to avoid huge prompts)
            max_tool_len = 500
            tool_lines = []
            for t in tool_turns:
                content = t.content_full
                if len(content) > max_tool_len:
                    content = content[:max_tool_len] + "..."
                tool_lines.append(f"- {t.tool_name}: {content}")
            prompt = "Actions taken:\n" + "\n".join(tool_lines) + "\n\nSummarize these actions."

            summary = self._call_llm(prompt, TOOL_SYSTEM_PROMPT)
            callback(summary, True)

        except (litellm.exceptions.APIConnectionError, openai.APIConnectionError, ConnectionError):
            logger.debug(f"Summarizer: {self.model} server not available")
            callback("[server unavailable]", False)
        except (litellm.exceptions.Timeout, openai.APITimeoutError, subprocess.TimeoutExpired):
            logger.debug(f"Summarizer: {self.model} request timed out")
            callback("[timeout]", False)
        except Exception as e:
            logger.debug(f"Summarizer error ({self.model}): {e}")
            callback(f"[error: {type(e).__name__}]", False)

    def _critique(
        self,
        turn: Turn,
        context: dict,
        callback: Callable[[str, str, bool], None]
    ) -> None:
        """Call LLM to critique assistant turn."""
        try:
            # Truncate content to avoid token limits
            max_content_len = 6000
            assistant_response = context["assistant_response"]
            if len(assistant_response) > max_content_len:
                assistant_response = assistant_response[:max_content_len] + "..."

            max_user_len = 2000
            user_message = context["user_message"]
            if len(user_message) > max_user_len:
                user_message = user_message[:max_user_len] + "..."

            # Build prompt
            prompt = f"""Context:
- User's request: {user_message}
- Turn {context['turn_number']} of {context['total_turns']}
- Active tasks:
{context['active_tasks']}
- Tools called:
{context['tool_summary']}

Assistant's response:
{assistant_response}

Evaluate: Is the assistant making progress? Did tool actions succeed? Any critical issues?"""

            critique = self._call_llm(prompt, CRITIC_SYSTEM_PROMPT)

            # Extract sentiment
            sentiment = "progress"
            if critique.startswith("✓"):
                sentiment = "progress"
            elif critique.startswith("⚠"):
                sentiment = "concern"
            elif critique.startswith("✗"):
                sentiment = "critical"

            callback(critique, sentiment, True)

        except (litellm.exceptions.APIConnectionError, openai.APIConnectionError, ConnectionError):
            logger.debug(f"Critic: {self.model} server not available")
            callback("[critic unavailable]", "progress", False)
        except (litellm.exceptions.Timeout, openai.APITimeoutError, subprocess.TimeoutExpired):
            logger.debug(f"Critic: {self.model} request timed out")
            callback("[critic timeout]", "progress", False)
        except Exception as e:
            logger.debug(f"Critic error ({self.model}): {e}")
            callback(f"[critic error: {type(e).__name__}]", "progress", False)

    def _summarize(
        self,
        turn: Turn,
        callback: Callable[[str, bool], None]
    ) -> None:
        """Call LLM to summarize assistant text (no tool context)."""
        try:
            content = turn.content_full
            if not content:
                callback("[empty content]", False)
                return

            # Truncate assistant response if needed
            max_content_len = 8000
            if len(content) > max_content_len:
                content = content[:max_content_len] + "..."

            # Build prompt for assistant-only summarization
            prompt = f"What is the assistant communicating in this response?\n\n{content}"

            summary = self._call_llm(prompt, SYSTEM_PROMPT)
            callback(summary, True)

        except (litellm.exceptions.APIConnectionError, openai.APIConnectionError, ConnectionError):
            logger.debug(f"Summarizer: {self.model} server not available")
            callback("[server unavailable]", False)
        except (litellm.exceptions.Timeout, openai.APITimeoutError, subprocess.TimeoutExpired):
            logger.debug(f"Summarizer: {self.model} request timed out")
            callback("[timeout]", False)
        except Exception as e:
            logger.debug(f"Summarizer error ({self.model}): {e}")
            callback(f"[error: {type(e).__name__}]", False)

    def shutdown(self) -> None:
        """Shutdown the executor, cancelling pending tasks."""
        self._executor.shutdown(wait=False, cancel_futures=True)
