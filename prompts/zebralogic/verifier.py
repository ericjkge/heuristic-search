"""Prompt templates for the ZebraLogic verifier agents (bijection, equality, positional)."""


def bijection_prompt(table_str: str, puzzle_text: str) -> str:
    return f"""You are a logic puzzle verifier. Check whether each attribute column contains a valid permutation (each value used exactly once, no duplicates, no missing values).

{puzzle_text}

Candidate solution:
{table_str}

For each attribute column (excluding House number), verify that every listed value appears exactly once.

Write down your process as you work then list any errors inside <OUTPUT> tags. If everything is correct, output empty <OUTPUT></OUTPUT> tags.

<OUTPUT>
Column "Nationality": "german" appears twice (House 1 and House 3), "brit" is missing
Column "Color": "red" appears twice, "green" is missing
</OUTPUT>"""


def equality_prompt(table_str: str, puzzle_text: str) -> str:
    return f"""You are a logic puzzle verifier. Check whether the candidate solution satisfies the identity/assignment clues (e.g. "X is Y", "X is in house N", "X is not in house N").

{puzzle_text}

Candidate solution:
{table_str}

For each clue, verify whether the candidate solution satisfies it. Check carefully.

Write down your process as you work then list any violations inside <OUTPUT> tags. If all clues are satisfied, output empty <OUTPUT></OUTPUT> tags.

<OUTPUT>
Clue 3 violated: "The Norwegian lives in House 1" but Norwegian is in House 3
Clue 7 violated: "Bob likes pizza" but Bob has stew
</OUTPUT>"""


def positional_prompt(table_str: str, puzzle_text: str) -> str:
    return f"""You are a logic puzzle verifier. Check whether the candidate solution satisfies the positional/adjacency clues (e.g. "next to", "directly left of", "somewhere to the left of").

{puzzle_text}

Candidate solution:
{table_str}

For each clue, verify the spatial relationship between houses. Remember:
- "directly left of" means exactly one position to the left (House N is directly left of House N+1)
- "somewhere to the left of" means any position with a smaller house number
- "next to" means adjacent (house numbers differ by exactly 1)
- "between" means the house number is strictly between the other two

Write down your process as you work then list any violations inside <OUTPUT> tags. If all clues are satisfied, output empty <OUTPUT></OUTPUT> tags.

<OUTPUT>
Clue 5 violated: "The dog owner is next to the blue house" but dog is in House 1 and blue is in House 4
Clue 9 violated: "Alice is directly left of Bob" but Alice is in House 3 and Bob is in House 1
</OUTPUT>"""
