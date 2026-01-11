# === POLICY PRIOR PROMPT ===

policy_prompt = """Position (FEN): {fen}
Move history: {pgn}

Rank the top 5 next moves in UCI format (e.g. e2e4).

MOVE_1: <best move in UCI format>
MOVE_2: <second best>
MOVE_3: <third best>
MOVE_4: <fourth best>
MOVE_5: <fifth best>"""


# === VALUE ORACLE PROMPT ===

value_prompt = """Position (FEN): {fen}
Move history: {pgn}

Evaluate the current position for {player} on scale -1.0 (losing) to +1.0 (winning).

VALUE: <-1.0 to +1.0>"""

