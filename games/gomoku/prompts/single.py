action_prompt = """
You are player {player} in a 9x9 Gomoku game.

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
History: {moves}

Choose the best move. Respond in this format:
ANALYSIS: <1 sentence>
MOVE: <vertex like D4>
"""
