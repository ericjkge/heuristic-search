"""Generator agent: produces and revises Sudoku candidate solutions."""

from typing import Optional

from utils.llm import GeminiLLM
from utils.logging import get_logger
from tasks.sudoku import Grid, str_to_grid

logger = get_logger(__name__)


class GeneratorAgent:
    """Stateless generator: each call is a single-turn LLM request."""

    def __init__(self, llm: GeminiLLM):
        self.llm = llm
        self.total_tokens = 0

    def generate(self, prompt: str) -> tuple[Optional[Grid], str]:
        """Send a single-turn prompt, return (parsed_grid, raw_response)."""
        response = self.llm.generate(prompt)
        self.total_tokens += self.llm.last_tokens

        grid = str_to_grid(response)
        return grid, response
