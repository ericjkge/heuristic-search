"""Debate agent: proposes, critiques, and votes on Sudoku solutions."""

import re
from typing import Optional

from utils.llm import GeminiLLM
from utils.logging import get_logger
from tasks.sudoku import Grid, str_to_grid

logger = get_logger(__name__)


class DebateAgent:
    """A single debate participant that can propose, critique, and vote."""

    def __init__(self, name: str, llm: GeminiLLM) -> None:
        self.name = name
        self.llm = llm
        self.total_tokens: int = 0

    def propose(self, prompt: str) -> tuple[Optional[Grid], str]:
        """Generate an initial solution proposal."""
        response = self.llm.generate(prompt)
        self.total_tokens += self.llm.last_tokens
        grid = str_to_grid(response)
        logger.info(f"[{self.name}] propose -> parsed={'ok' if grid else 'FAIL'}")
        return grid, response

    def critique(self, prompt: str) -> tuple[Optional[Grid], str]:
        """Critique others and return a (possibly revised) solution."""
        response = self.llm.generate(prompt)
        self.total_tokens += self.llm.last_tokens
        grid = str_to_grid(response)
        logger.info(f"[{self.name}] critique -> parsed={'ok' if grid else 'FAIL'}")
        return grid, response

    def vote(self, prompt: str) -> int:
        """Vote for the best solution. Returns 0-indexed choice."""
        response = self.llm.generate(prompt)
        self.total_tokens += self.llm.last_tokens
        # Extract the first digit (1, 2, or 3) from the response
        match = re.search(r"[123]", response)
        if match:
            choice = int(match.group()) - 1  # convert to 0-indexed
        else:
            choice = 0  # default to first solution
        logger.info(f"[{self.name}] vote -> choice={choice + 1}")
        return choice
