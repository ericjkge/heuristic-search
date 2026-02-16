# Single-engine-2 prompts: single-turn with previous attempts shown

initial_prompt = """You are playing {player} in a chess game.

FEN: {fen}
PGN: {pgn}
Legal moves: {legal_moves}

IMPORTANT: You must choose a move from the legal moves list above.

ANALYSIS: <1 sentence>
MOVE: <move from legal moves list>"""

retry_prompt = """You are playing {player} in a chess game.

FEN: {fen}
PGN: {pgn}
Legal moves: {legal_moves}

Previously tried moves and their engine evaluations:
{previous_attempts}

Pick a DIFFERENT move from the legal moves list. Try to find something better.

ANALYSIS: <1 sentence>
MOVE: <move from legal moves list>"""
