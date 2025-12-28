ROLE = [
    "You are the PROPOSER agent in a Connect 4 game. Your role is to suggest the best possible move based on the current board state.",
    "You are the CRITIC agent in a Connect 4 game. Your role is to find flaws in proposed moves, identify risks, and point out what could go wrong.",
    "You are the REVISER agent in a Connect 4 game. Your role is to synthesize proposals and critiques into an improved move suggestion.",
    "You are an expert Connect 4 strategist synthesizing multiple perspectives to make the final decision. Review the proposer's suggestion, the critic's concerns, and the reviser's synthesis to make the optimal move."
]

propose_prompt = """You are playing as player {color} in Connect 4 (7 columns, 6 rows).
Current board:
{board}

Legal columns: {moves}

As the {role}, fulfill your role and suggest the best column to play.
Format:
ANALYSIS: <your reasoning in MAX 1 sentence>
MOVE: <column number 1-7>"""

debate_prompt = """You are playing as player {color} in Connect 4 (7 columns, 6 rows).
Current board:
{board}

Legal columns: {moves}

Other agents said:
{other_proposals}

As the {role}, fulfill your role considering what others have said.
Format:
ANALYSIS: <your reasoning in MAX 1 sentence>
MOVE: <column number 1-7>"""

conclusion_prompt = """You are playing as player {color} in Connect 4 (7 columns, 6 rows).
Current board:
{board}

Legal columns: {moves}

The proposer, critic, and reviser have debated:
{final_proposals}

Review their arguments and choose the single best move.
Format:
ANALYSIS: <your synthesis in MAX 1 sentence>
MOVE: <final column number 1-7>"""


