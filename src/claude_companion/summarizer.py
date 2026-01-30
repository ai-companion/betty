"""LLM summarization for assistant turns using local or API providers."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal

import requests

from .models import Turn

logger = logging.getLogger(__name__)

# System prompt for summarization
SYSTEM_PROMPT = (
    "You are a supervisor observing an AI assistant at work. "
    "Describe what the assistant is doing in 1-2 sentences, using third person "
    "(e.g., 'The assistant is...', 'It explains...', 'Claude is...'). "
    "Be concise and factual."
)


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

    def summarize_async(self, turn: Turn, callback: Callable[[str, bool], None]) -> None:
        """Submit summarization job, calls callback with (result, success)."""
        self._executor.submit(self._summarize, turn, callback)

    def _summarize(self, turn: Turn, callback: Callable[[str, bool], None]) -> None:
        """Call LLM to summarize turn content."""
        try:
            content = turn.content_full
            if not content:
                callback("[empty content]", False)
                return

            # Truncate very long content to avoid token limits
            max_content_len = 8000
            if len(content) > max_content_len:
                content = content[:max_content_len] + "..."

            # Route to appropriate provider
            if self.provider == "local":
                summary = self._summarize_local(content)
            elif self.provider == "openai":
                summary = self._summarize_openai(content)
            elif self.provider == "openrouter":
                summary = self._summarize_openrouter(content)
            elif self.provider == "anthropic":
                summary = self._summarize_anthropic(content)
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

    def _summarize_local(self, content: str) -> str:
        """Summarize using local OpenAI-compatible server."""
        if not self.base_url:
            raise ValueError("base_url required for local provider")

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"What is the assistant doing in this response?\n\n{content}"},
                ],
                "max_tokens": 150,
                "temperature": 0.3,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _summarize_openai(self, content: str) -> str:
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
                {"role": "user", "content": f"What is the assistant doing in this response?\n\n{content}"},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _summarize_openrouter(self, content: str) -> str:
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
                {"role": "user", "content": f"What is the assistant doing in this response?\n\n{content}"},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _summarize_anthropic(self, content: str) -> str:
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
                {"role": "user", "content": f"What is the assistant doing in this response?\n\n{content}"}
            ],
        )
        return response.content[0].text.strip()

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
