"""LLM summarization for assistant turns using vLLM server."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import requests

from .models import Turn

logger = logging.getLogger(__name__)

# Default vLLM server configuration
DEFAULT_BASE_URL = "http://localhost:8008/v1"
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class Summarizer:
    """Summarizes assistant turns using a local vLLM server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_workers: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def summarize_async(self, turn: Turn, callback: Callable[[str], None]) -> None:
        """Submit summarization job, calls callback with result."""
        self._executor.submit(self._summarize, turn, callback)

    def _summarize(self, turn: Turn, callback: Callable[[str], None]) -> None:
        """Call LLM API to summarize turn content."""
        try:
            content = turn.content_full
            if not content:
                return

            # Truncate very long content to avoid token limits
            max_content_len = 8000
            if len(content) > max_content_len:
                content = content[:max_content_len] + "..."

            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a supervisor observing an AI assistant at work. Describe what the assistant is doing in 1-2 sentences, using third person (e.g., 'The assistant is...', 'It explains...', 'Claude is...'). Be concise and factual.",
                        },
                        {
                            "role": "user",
                            "content": f"What is the assistant doing in this response?\n\n{content}",
                        }
                    ],
                    "max_tokens": 150,
                    "temperature": 0.3,
                },
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            summary = data["choices"][0]["message"]["content"].strip()
            callback(summary)

        except requests.exceptions.ConnectionError:
            logger.debug("Summarizer: vLLM server not available")
        except requests.exceptions.Timeout:
            logger.debug("Summarizer: request timed out")
        except Exception as e:
            logger.debug(f"Summarizer error: {e}")

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
