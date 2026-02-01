"""LLM summarization for assistant turns using local or API providers."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable, Literal

import requests

from .models import Turn

if TYPE_CHECKING:
    from .models import Session

logger = logging.getLogger(__name__)

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


def make_tool_cache_key(tool_turns: list[Turn]) -> str:
    """Create cache key for a sequence of tool turns.

    Args:
        tool_turns: List of tool turns

    Returns:
        Cache key that uniquely identifies this tool sequence
    """
    import hashlib
    # use hash of full content to avoid collisions from truncated previews
    tool_sig = "|".join(
        f"{t.tool_name}:{hashlib.sha256(t.content_full.encode()).hexdigest()[:12]}"
        for t in tool_turns
    )
    return f"TOOLS::{tool_sig}"


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


class Summarizer:
    """Summarizes assistant turns using local server or API providers (OpenAI, OpenRouter, Anthropic)."""

    def __init__(
        self,
        provider: Literal["local", "openai", "openrouter", "anthropic"],
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_workers: int = 2,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Lazy-load API clients
        self._openai_client = None
        self._openrouter_client = None
        self._anthropic_client = None

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

            # Route to appropriate provider
            if self.provider == "local":
                summary = self._summarize_local(prompt, TOOL_SYSTEM_PROMPT)
            elif self.provider == "openai":
                summary = self._summarize_openai(prompt, TOOL_SYSTEM_PROMPT)
            elif self.provider == "openrouter":
                summary = self._summarize_openrouter(prompt, TOOL_SYSTEM_PROMPT)
            elif self.provider == "anthropic":
                summary = self._summarize_anthropic(prompt, TOOL_SYSTEM_PROMPT)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            callback(summary, True)

        except requests.exceptions.ConnectionError:
            logger.debug(f"Summarizer: {self.provider} server not available")
            callback("[server unavailable]", False)
        except requests.exceptions.Timeout:
            logger.debug(f"Summarizer: {self.provider} request timed out")
            callback("[timeout]", False)
        except Exception as e:
            logger.debug(f"Summarizer error ({self.provider}): {e}")
            callback(f"[error: {type(e).__name__}]", False)

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

            # Route to appropriate provider
            if self.provider == "local":
                summary = self._summarize_local(prompt, SYSTEM_PROMPT)
            elif self.provider == "openai":
                summary = self._summarize_openai(prompt, SYSTEM_PROMPT)
            elif self.provider == "openrouter":
                summary = self._summarize_openrouter(prompt, SYSTEM_PROMPT)
            elif self.provider == "anthropic":
                summary = self._summarize_anthropic(prompt, SYSTEM_PROMPT)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            callback(summary, True)

        except requests.exceptions.ConnectionError:
            logger.debug(f"Summarizer: {self.provider} server not available")
            callback("[server unavailable]", False)
        except requests.exceptions.Timeout:
            logger.debug(f"Summarizer: {self.provider} request timed out")
            callback("[timeout]", False)
        except Exception as e:
            logger.debug(f"Summarizer error ({self.provider}): {e}")
            callback(f"[error: {type(e).__name__}]", False)

    def _summarize_local(self, prompt: str, system_prompt: str) -> str:
        """Summarize using local OpenAI-compatible server."""
        if not self.base_url:
            raise ValueError("base_url required for local provider")

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 150,
                "temperature": 0.3,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _summarize_openai(self, prompt: str, system_prompt: str) -> str:
        """Summarize using OpenAI API."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Lazy-load openai client
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self.api_key)

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _summarize_openrouter(self, prompt: str, system_prompt: str) -> str:
        """Summarize using OpenRouter API (uses OpenAI SDK)."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        # Lazy-load openrouter client (uses OpenAI SDK with custom base_url)
        if self._openrouter_client is None:
            from openai import OpenAI
            self._openrouter_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://openrouter.ai/api/v1",
            )

        response = self._openrouter_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _summarize_anthropic(self, prompt: str, system_prompt: str) -> str:
        """Summarize using Anthropic API."""
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        # Lazy-load anthropic client
        if self._anthropic_client is None:
            from anthropic import Anthropic
            self._anthropic_client = Anthropic(api_key=self.api_key)

        response = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=150,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response.content[0].text.strip()

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
