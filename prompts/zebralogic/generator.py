"""Prompt templates for the ZebraLogic generator agent."""


def initial_prompt(puzzle_text: str) -> str:
    return f"""You are a logic puzzle solver. Solve the following puzzle.

{puzzle_text}

Output: Wrap your final answer in <OUTPUT> tags as a JSON object with "header" and "rows" keys:

<OUTPUT>
{{"header": ["House", "Attribute1", "Attribute2", ...], "rows": [["1", "value", "value", ...], ...]}}
</OUTPUT>"""


def revision_prompt(
    puzzle_text: str,
    previous_response: str,
    feedback: str,
) -> str:
    return f"""You are a logic puzzle solver. Fix your previous solution based on verifier feedback.

{puzzle_text}

Your previous response:
{previous_response}

Verifier feedback:
{feedback}

Output: Wrap your final answer in <OUTPUT> tags as a JSON object with "header" and "rows" keys:

<OUTPUT>
{{"header": ["House", "Attribute1", "Attribute2", ...], "rows": [["1", "value", "value", ...], ...]}}
</OUTPUT>"""
