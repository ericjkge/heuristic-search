action_prompt = """
You are player {player} in a 9x9 Gomoku game.

Coordinates:
- Columns: A-J (no letter I)
- Rows: 1-9 (bottom to top)

Your positions: {player_positions}
Opponent positions: {opponent_positions}
Move history: {moves}

Choose the next best move. Respond in this format:

ANALYSIS: <1-2 sentences>
MOVE: <vertex like D4, or PASS>
"""
