"""Verifier agent: checks a candidate solution using a prompt template."""

import re
from typing import Callable

from utils.llm import GeminiLLM


class VerifierAgent:
    """Stateless verifier — one call per verification."""

    def __init__(self, name: str, llm: GeminiLLM, prompt_fn: Callable[[str], str]):
        self.name = name
        self.llm = llm
        self.prompt_fn = prompt_fn
        self.total_tokens = 0

    def verify(self, grid_str: str) -> str | None:
        """Run verification. Returns error feedback or None if no errors."""
        prompt = self.prompt_fn(grid_str)
        response = self.llm.generate(prompt)
        self.total_tokens += self.llm.last_tokens

        if not response:
            return None
        matches = re.findall(r"<OUTPUT>(.*?)</OUTPUT>", response, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        return content if content else None
