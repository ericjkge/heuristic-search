"""Chain-of-Thought prompt for the ZebraLogic generator agent."""


def cot_prompt(puzzle_text: str) -> str:
    return f"""You are a logic puzzle solver. Solve the following puzzle.

{puzzle_text}

Let's think step by step. Work through each clue systematically, building a constraint table and eliminating possibilities until you find the unique solution.

Output: Wrap your final answer in <OUTPUT> tags as a JSON object with "header" and "rows" keys:

<OUTPUT>
{{"header": ["House", "Attribute1", "Attribute2", ...], "rows": [["1", "value", "value", ...], ...]}}
</OUTPUT>"""
