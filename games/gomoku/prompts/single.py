action_prompt = """
You are player {player} in a 9x9 Gomoku game.

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
History: {moves}

Choose the next best move. Occupied and out-of-bounds positions are NOT allowed. Respond in this format:

ANALYSIS: <1 sentence>
MOVE: <vertex like D4, or PASS>
"""
