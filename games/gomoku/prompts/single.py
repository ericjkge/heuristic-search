action_prompt = """

You are a player {player} in a 9x9 Gomoku game.

Coordinates:
- Columns: A-J (no letter I)
- Rows: 1-9 (bottom to top)
- Example: D4, E5, J9

Your positions: {player_positions}
Opponent positions: {opponent_positions}

Move history: {moves}

First, think through the game strategy. Then, respond with:

ANALYSIS: <1-2 sentences>
MOVE: <vertex like D4, or 'PASS'>

"""
