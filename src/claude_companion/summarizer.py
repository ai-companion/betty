"""LLM summarization for assistant turns using local or API providers."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable, Literal

import requests

from .models import Turn

if TYPE_CHECKING:
    from .models import Session

logger = logging.getLogger(__name__)

# System prompt for summarization
SYSTEM_PROMPT = (
    "You are a supervisor observing an AI coding assistant at work. "
    "Summarize what the assistant accomplished in 1-2 sentences. "
    "Focus on impactful actions (edits, writes, commands) and their purpose, "
    "mentioning exploratory actions (reads, searches) only when relevant. "
    "Use third person (e.g., 'Read config.py, edited server.py to fix the bug'). "
    "Be concise and action-oriented."
)


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
        user_message: Turn | None,
        tool_context: list[Turn],
        callback: Callable[[str, bool], None]
    ) -> None:
        """Submit summarization job, calls callback with (result, success)."""
        self._executor.submit(self._summarize, turn, user_message, tool_context, callback)

    def _summarize(
        self,
        turn: Turn,
        user_message: Turn | None,
        tool_context: list[Turn],
        callback: Callable[[str, bool], None]
    ) -> None:
        """Call LLM to summarize turn content with full context."""
        try:
            content = turn.content_full
            if not content:
                callback("[empty content]", False)
                return

            # Build full context prompt
            context_parts = []

            if user_message:
                user_text = user_message.content_full
                # Truncate very long user messages
                max_user_len = 1000
                if len(user_text) > max_user_len:
                    user_text = user_text[:max_user_len] + "..."
                context_parts.append(f"User request: {user_text}")

            if tool_context:
                tool_lines = []
                for tool in tool_context:
                    tool_lines.append(f"- {tool.tool_name}: {tool.content_full}")
                tools_str = "\n".join(tool_lines)
                context_parts.append(f"Actions taken:\n{tools_str}")

            # Truncate assistant response if needed
            max_content_len = 8000
            if len(content) > max_content_len:
                content = content[:max_content_len] + "..."
            context_parts.append(f"Assistant response:\n{content}")

            # Build final prompt
            if user_message or tool_context:
                prompt = "\n\n".join(context_parts) + "\n\nSummarize what the assistant accomplished."
            else:
                # Fallback for responses with no context
                prompt = f"What is the assistant doing in this response?\n\n{content}"

            # Route to appropriate provider
            if self.provider == "local":
                summary = self._summarize_local(prompt)
            elif self.provider == "openai":
                summary = self._summarize_openai(prompt)
            elif self.provider == "openrouter":
                summary = self._summarize_openrouter(prompt)
            elif self.provider == "anthropic":
                summary = self._summarize_anthropic(prompt)
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

    def _summarize_local(self, prompt: str) -> str:
        """Summarize using local OpenAI-compatible server."""
        if not self.base_url:
            raise ValueError("base_url required for local provider")

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
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

    def _summarize_openai(self, prompt: str) -> str:
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _summarize_openrouter(self, prompt: str) -> str:
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _summarize_anthropic(self, prompt: str) -> str:
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
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response.content[0].text.strip()

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
