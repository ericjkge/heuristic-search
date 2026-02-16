# Oracle prompts for Chess - simple prompts with engine score feedback

initial_prompt = """You are playing {player} in a chess game.

FEN: {fen}
PGN: {pgn}
Legal moves: {legal_moves}

IMPORTANT: You must choose a move from the legal moves list above.

ANALYSIS: <1 sentence>
MOVE: <move from legal moves list>"""

feedback_prompt = """Your previous move was evaluated by a chess engine:

Move: {previous_move}
Engine score: {engine_score} pawns ({assessment})
Legal moves: {legal_moves}

Try to find a better move.
IMPORTANT: You must choose a move from the legal moves list above.

ANALYSIS: <1 sentence>
MOVE: <move from legal moves list>"""
