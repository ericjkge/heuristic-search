action_prompt = """
You are an expert chess player.

The current move history is: {pgn}

You are playing as {player}.

Analyze the position and propose the next move.
Format your response as:

ANALYSIS: <your reasoning in MAX 1 sentence>
MOVE: <your chosen move in UCI format - e.g. e2e4>
"""