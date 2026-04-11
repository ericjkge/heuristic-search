"""Debate (MAD) prompts for the ZebraLogic task."""


def propose_prompt(puzzle_text: str) -> str:
    return f"""You are a logic puzzle solver. Solve the following puzzle.

{puzzle_text}

Output: Wrap your final answer in <OUTPUT> tags as a JSON object with "header" and "rows" keys:

<OUTPUT>
{{"header": ["House", "Attribute1", "Attribute2", ...], "rows": [["1", "value", "value", ...], ...]}}
</OUTPUT>"""


def critique_prompt(
    puzzle_text: str,
    own_solution: str,
    other_solutions: list[str],
) -> str:
    others_text = "\n\n".join(
        f"--- Solution {i+1} ---\n{s}" for i, s in enumerate(other_solutions)
    )
    return f"""You are a logic puzzle solver engaged in a debate. You have proposed a solution and have seen other agents' solutions. Revise your answer if you find errors.

{puzzle_text}

Your previous solution:
{own_solution}

Other agents' solutions:
{others_text}

Compare all solutions carefully. Check each clue against each solution. Produce your best revised solution.

Output: Wrap your final answer in <OUTPUT> tags as a JSON object with "header" and "rows" keys:

<OUTPUT>
{{"header": ["House", "Attribute1", "Attribute2", ...], "rows": [["1", "value", "value", ...], ...]}}
</OUTPUT>"""


