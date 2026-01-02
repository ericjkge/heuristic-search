action_prompt = """You are an expert Go player.

Game Rules:
- Go is played on a 9 x 9 grid.
- Black plays first, then players alternate.
- Surround territory and capture opponent stones.
- Game ends when both players pass.

Coordinates:
- Columns: A-T (no letter I)
- Rows: 1-9 (bottom to top)
- Example: D4, E5, J10

Current Game (SGF format): {sgf}

You are playing as {player}.

Respond with:
ANALYSIS: <1-2 sentences>
MOVE: <vertex like D4, or 'pass'>"""