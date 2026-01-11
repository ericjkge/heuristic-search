action_prompt = """

You are player {player} in a chess game.

Current board (FEN): {fen}

Move history: {pgn}

First, think through the game strategy. Then, respond with:

ANALYSIS: <1-2 sentences>
MOVE: <your chosen move in UCI format - e.g. e2e4>

"""