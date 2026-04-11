"""Self-critique prompt for the ZebraLogic self-refine baseline."""


def self_critique_prompt(puzzle_text: str, solution_str: str) -> str:
    return f"""You are a logic puzzle verifier. Carefully check whether this solution is correct.

{puzzle_text}

Candidate solution:
{solution_str}

Check the following:
1. Each attribute column has no duplicate values (bijection constraint).
2. All identity/assignment clues are satisfied (e.g. "X is Y", "X is in house N").
3. All positional/adjacency clues are satisfied (e.g. "next to", "directly left of").

Output: If you find errors, list them inside <OUTPUT> tags. If the solution is correct, output empty <OUTPUT></OUTPUT> tags.

<OUTPUT>
Clue 3 violated: "The Norwegian lives in House 1" but Norwegian is in House 3
Column "Color": "red" appears twice, "green" is missing
</OUTPUT>"""
