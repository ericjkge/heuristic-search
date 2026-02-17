"""Verifier agent: checks a candidate solution using a prompt template."""

import re
from typing import Callable

from utils.llm import GeminiLLM
from utils.logging import get_logger

logger = get_logger(__name__)


class VerifierAgent:
    """Stateless verifier — one call per verification."""

    def __init__(self, name: str, llm: GeminiLLM, prompt_fn: Callable[[str], str]):
        self.name = name
        self.llm = llm
        self.prompt_fn = prompt_fn
        self.total_tokens = 0

    def verify(self, grid_str: str) -> tuple[bool, str]:
        """Run verification. Returns (is_correct, feedback_text)."""
        prompt = self.prompt_fn(grid_str)
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.generate(messages)
        self.total_tokens += self.llm.last_tokens

        is_correct = self._parse_correct(response)
        logger.info(f"verifier={self.name} | correct={is_correct}")
        return is_correct, response

    def _parse_correct(self, response: str) -> bool:
        """Conservative: only return True if we find 'CORRECT: YES'."""
        match = re.search(r"CORRECT\s*:\s*(YES|NO)", response, re.IGNORECASE)
        if match:
            return match.group(1).upper() == "YES"
        return False  # Conservative default
