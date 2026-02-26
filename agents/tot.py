"""Tree-of-Thought agent: DFS with backtracking over partial Sudoku states."""

import copy
from typing import Optional

from utils.llm import GeminiLLM
from utils.logging import get_logger
from tasks.sudoku import Grid, grid_to_str, str_to_partial_grid, check_partial_solution
from prompts.sudoku.tot import (
    initial_propose_prompt,
    continue_propose_prompt,
    backtrack_propose_prompt,
)

logger = get_logger(__name__)


class ToTAgent:
    """Proposes partial Sudoku fills via LLM."""

    def __init__(self, llm: GeminiLLM) -> None:
        self.llm = llm
        self.total_tokens: int = 0

    def propose(self, prompt: str) -> tuple[Optional[Grid], str]:
        """Call LLM and parse response as a partial grid."""
        response = self.llm.generate(prompt)
        self.total_tokens += self.llm.last_tokens
        grid = str_to_partial_grid(response)
        return grid, response


def _serialize_grid(grid: Grid) -> str:
    """Serialize grid to a hashable string for visit counting."""
    return "|".join("".join(str(c) for c in row) for row in grid)


def solve_tot(
    puzzle: Grid,
    llm: GeminiLLM,
    max_rounds: int = 30,
) -> dict:
    """DFS with backtracking over partial board states.

    Returns {"solved": bool, "rounds": int, "total_tokens": int}.
    """
    agent = ToTAgent(llm)

    # State stack: list of partial grids (0 = unfilled)
    stack: list[Grid] = [copy.deepcopy(puzzle)]
    visit_count: dict[str, int] = {}

    solved = False
    rounds_used = 0
    last_error: Optional[str] = None

    for round_num in range(1, max_rounds + 1):
        rounds_used = round_num
        current = stack[-1]
        current_str = grid_to_str(current)
        puzzle_str = grid_to_str(puzzle)

        # Choose prompt
        if last_error is not None:
            prompt = backtrack_propose_prompt(puzzle_str, current_str, last_error)
            last_error = None
        elif round_num == 1:
            prompt = initial_propose_prompt(puzzle_str)
        else:
            prompt = continue_propose_prompt(puzzle_str, current_str)

        # Propose
        candidate, raw = agent.propose(prompt)
        if candidate is None:
            logger.info(f"Round {round_num}: failed to parse LLM output")
            last_error = "Could not parse your grid. Output EXACTLY 9 lines of 9 space-separated tokens (digits or .)."
            continue

        # Check
        is_valid, is_complete, error_msg = check_partial_solution(puzzle, candidate)

        if is_complete:
            logger.info(f"Round {round_num}: SOLVED")
            solved = True
            break

        if is_valid:
            # Push valid state onto stack
            stack.append(copy.deepcopy(candidate))
            key = _serialize_grid(candidate)
            visit_count[key] = visit_count.get(key, 0) + 1
            logger.info(f"Round {round_num}: valid partial, depth={len(stack)}")
        else:
            # Invalid — rollback
            logger.info(f"Round {round_num}: invalid — {error_msg}")
            last_error = error_msg

            # Check if parent has been visited too many times → rollback 2
            if len(stack) >= 2:
                parent_key = _serialize_grid(stack[-1])
                if visit_count.get(parent_key, 0) >= 5 and len(stack) >= 3:
                    stack.pop()
                    logger.info(f"Round {round_num}: double rollback (parent visited ≥5×)")
                stack.pop()
            # If stack is empty, re-seed with puzzle
            if not stack:
                stack.append(copy.deepcopy(puzzle))

    return {
        "solved": solved,
        "rounds": rounds_used,
        "total_tokens": agent.total_tokens,
    }
