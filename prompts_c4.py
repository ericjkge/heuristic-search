PERSPECTIVES = [
    "You are a Connect 4 expert focused on OFFENSIVE play. Prioritize creating threats, building multiple winning paths, and forcing the opponent into defensive positions.",
    "You are a Connect 4 expert focused on DEFENSIVE play. Prioritize blocking opponent threats, controlling the center column, and preventing forced wins.",
    "You are a Connect 4 expert focused on POSITIONAL play. Prioritize center control, building flexible formations, and setting up double-threat situations.",
]

CONCLUSION_PERSPECTIVE = "You are an expert Connect 4 strategist synthesizing multiple perspectives to make the final decision. Consider the reasons between conflicting views and choose the final move."

propose_prompt = """You are playing as {color} in Connect 4 (7 columns, 6 rows).
Current board:
{board}

Legal columns: {moves}

Analyze and propose the best column to play.
Format:
ANALYSIS: <your reasoning in MAX 1 sentence>
MOVE: <column number 1-7>"""

debate_prompt = """You are playing as {color} in Connect 4 (7 columns, 6 rows).
Current board:
{board}

Legal columns: {moves}

Other agents proposed:
{other_proposals}

Consider their perspectives. You may keep or change your move.
Format:
ANALYSIS: <your updated reasoning in MAX 1 sentence>
MOVE: <column number 1-7>"""

conclusion_prompt = """You are playing as {color} in Connect 4 (7 columns, 6 rows).
Current board:
{board}

Legal columns: {moves}

After debate, the agents propose:
{final_proposals}

Review each agent's analysis and proposed move. Choose the single best move.
Format:
ANALYSIS: <your synthesis of their arguments in MAX 1 sentence>
MOVE: <final column number 1-7>"""