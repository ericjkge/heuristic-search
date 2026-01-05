PERSPECTIVES = [
    "positional control: center squares, key intersections, and spatial influence across the board.",
    "opponent modeling: anticipating your opponent's plans, potential lines, and likely next moves.",
    "defensive blocking: stopping opponent threats, breaking their lines, and preventing four-in-a-row.",
    "winning tactics: creating five-in-a-row, open fours, double threats, and unstoppable sequences.",
    "heuristic evaluation: general principles like 'control the center', 'extend your longest line', 'avoid edges early'.",
    "strategic planning: building multiple threat lines, creating forks, and setting up future winning positions.",
]

action_prompt = """
You are player {player} in a 9x9 Gomoku game, specializing in {perspective}

Coordinates:
- Columns: A-J (no letter I)
- Rows: 1-9 (bottom to top)

Your positions: {player_positions}
Opponent positions: {opponent_positions}
Move history: {moves}
Other agents proposed: {proposals}

Choose the next best move based on your speciality. Respond in this format:

ANALYSIS: <1-2 sentences>
MOVE: <vertex like D4, or PASS>
"""

aggregate_prompt = """
You are the final decision maker for player {player} in a 9x9 Gomoku game.

Coordinates:
- Columns: A-J (no letter I)
- Rows: 1-9 (bottom to top)

Your positions: {player_positions}
Opponent positions: {opponent_positions}
Move history: {moves}
Agent proposals: {proposals}

Choose the next best move by aggregating proposals. Respond in this format:
ANALYSIS: <1-2 sentences>
MOVE: <vertex like D4, or PASS>
"""