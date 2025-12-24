# Perspective system prompts
PERSPECTIVES = [
    "You are a chess expert focused on TACTICAL play. Prioritize immediate threats, captures, checks, pins, forks, and short-term combinations.",
    "You are a chess expert focused on POSITIONAL play. Prioritize piece activity, pawn structure, king safety, and long-term strategic advantages.",
    "You are a chess expert focused on DEFENSIVE play. Prioritize identifying opponent threats, prophylaxis, and ensuring piece safety before attacking.",
]

propose_prompt = """You are playing as {color}.
Current board (FEN): {fen}
Legal moves: {moves}

Analyze the position from your perspective and propose the best move.
Format your response as:
ANALYSIS: <your reasoning>
MOVE: <your chosen move in UCI format, e.g., e2e4>"""

debate_prompt = """You are playing as {color}.
Current board (FEN): {fen}
Legal moves: {moves}

Other agents proposed:
{other_proposals}

Consider their perspectives and arguments. You may keep your move or change it.
Format your response as:
ANALYSIS: <your updated reasoning considering others' views>
MOVE: <your final move in UCI format>"""
