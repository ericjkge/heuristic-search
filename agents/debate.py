"""Debate agent: proposes, critiques, and revises solutions."""

from typing import Any, Callable

from utils.llm import GeminiLLM
from utils.logging import get_logger

logger = get_logger(__name__)


class DebateAgent:
    """A single debate participant. Task-agnostic via parse_fn."""

    def __init__(
        self,
        name: str,
        llm: GeminiLLM,
        parse_fn: Callable[[str], Any],
    ) -> None:
        self.name = name
        self.llm = llm
        self.parse_fn = parse_fn
        self.total_tokens = 0

    def propose(self, prompt: str) -> tuple[Any, str]:
        """Generate an initial proposal. Returns (parsed, raw)."""
        raw = self.llm.generate(prompt)
        self.total_tokens += self.llm.last_tokens
        return self.parse_fn(raw), raw

    def critique(self, prompt: str) -> tuple[Any, str]:
        """Critique and revise. Returns (parsed, raw)."""
        raw = self.llm.generate(prompt)
        self.total_tokens += self.llm.last_tokens
        return self.parse_fn(raw), raw

