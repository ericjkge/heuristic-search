PERSPECTIVES = [
    "You are a Connect 4 expert focused on OFFENSIVE play. Prioritize creating threats, building multiple winning paths, and forcing the opponent into defensive positions.",
    "You are a Connect 4 expert focused on DEFENSIVE play. Prioritize blocking opponent threats, controlling the center column, and preventing forced wins.",
    "You are a Connect 4 expert focused on POSITIONAL play. Prioritize center control, building flexible formations, and setting up double-threat situations.",
]

action_prompt = """
You are given:
- Current board: 

{board}

- You are player: {player}
- Other agents' proposals: {other_proposals}

Game rules:
- The board is a 7x6 grid (rows and columns are both 1-indexed).
- Empty cells are represented by "0".
- Player 1 is represented by "1".
- Player 2 is represented by "2".
- A move consists of choosing a column where a disc will fall to the lowest available row.
- You win by connecting four of your discs horizontally, vertically, or diagonally.
- Columns that are full are illegal moves.

Example analysis:
- Column 4 is full (7 discs), so next move CANNOT be 4
- No immediate threat for Player 2 in column 1 (only 2 vertical discs, need 4 in a row to win)
- Dropping in column 5 would create a horizontal three-in-a-row threat for Player 1

0 0 0 1 0 0 0
0 0 0 1 0 0 0
0 0 0 2 0 0 0
0 0 0 1 0 0 0
2 0 0 2 0 0 0
2 0 1 1 0 0 2


Analyze and propose the best column to play. If other agents' proposals are provided, carefully consider their reasoning and incorporate their insights before making your decision.
Format your response as:
ANALYSIS: <your reasoning in MAX 1 sentence>
MOVE: <column number 1-7>"""

policy_prompt = ""

value_prompt = ""

conclusion_prompt = """
You are a Connect 4 expert synthesizing the outputs of multiple agents to make the final decision.

You are given:
- Current board:

{board}

- The outputs of multi-agent debate: {proposals}
- You are player: {player}

Analyze the outputs and choose the single best output.
Format your response as:
ANALYSIS: <your reasoning in MAX 1 sentence>
MOVE: <column number 1-7>
"""