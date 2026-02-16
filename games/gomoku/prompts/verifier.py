# Verifier prompts for Gomoku move evaluation
# Each verifier outputs: REASONING + SCORE

immediate_loss_prompt = """
You are evaluating a Gomoku move for immediate loss risk.

Board: 9x9, Columns A-J (no I), Rows 1-9
Current player: {player}
Player stones: {player_positions}
Opponent stones: {opponent_positions}
Proposed move: {move}

Task: Determine if this move FAILS to block an opponent's winning threat.
Consider:
- Does the move win the game for the current player, thus making opponent threats irrelevant?
- Does the opponent have open-fours, open-threes, closed-fours, or any forcing sequences that must be blocked on this move to prevent an inevitable loss?
- Think out loud by listing the forcing sequence and checking that those stones are connected, belong to the opponent, and are not interrupted by your stones.

REASONING: <1 sentence>
SCORE: <1 if no immediate loss, 0 if opponent wins next turn>
"""

illegal_move_prompt = """
You are evaluating a Gomoku move for legality.

Board: 9x9, Columns A-J (no I), Rows 1-9
Player stones: {player_positions}
Opponent stones: {opponent_positions}
Proposed move: {move}

Task: Check if the move is legal.
Consider:
- Is the position already occupied?
- Is the position out-of-bounds (not A-H,J columns or 1-9 rows)
- Is the position formatted correctly in vertex form (e.g. A1, D4, H2)?

REASONING: <1 sentence>
SCORE: <1 if legal, 0 if illegal>
"""

aggressive_prompt = """
You are evaluating a Gomoku move for offensive potential.

Board: 9x9, Columns A-J (no I), Rows 1-9
Current player: {player}
Player stones: {player_positions}
Opponent stones: {opponent_positions}
Proposed move: {move}

Task: Score how much this move advances the player's winning threats.
Consider:
- Does it create an open-4 (unstoppable win)?
- Does it create an open-3 (forcing move)?
- Does it create a fork (two threats at once)?
- Does it extend existing lines toward 5?

REASONING: <1 sentence>
SCORE: <0.0 to 1.0, where 1.0 = creates winning threat, 0.0 = no offensive value>
"""

defensive_prompt = """
You are evaluating a Gomoku move for defensive value.

Board: 9x9, Columns A-J (no I), Rows 1-9
Current player: {player}
Player stones: {player_positions}
Opponent stones: {opponent_positions}
Proposed move: {move}

Task: Score how well this move defends against opponent threats.
Consider:
- Does it block an opponent's open-4?
- Does it block an opponent's open-3?
- Does it prevent a fork?
- Does it disrupt opponent's developing lines?

REASONING: <1 sentence>
SCORE: <0.0 to 1.0, where 1.0 = blocks critical threat, 0.0 = ignores all threats>
"""

shape_prompt = """
You are evaluating a Gomoku move for shape/structure quality.

Board: 9x9, Columns A-J (no I), Rows 1-9
Current player: {player}
Player stones: {player_positions}
Opponent stones: {opponent_positions}
Proposed move: {move}

Task: Score the positional quality of this move.
Consider:
- Connectivity: Does it link with existing stones?
- Flexibility: Does it create multiple directions for future development?
- Center control: Is it well-positioned (center > edge > corner)?
- Efficiency: Does it serve multiple purposes?

REASONING: <1 sentence>
SCORE: <0.0 to 1.0, where 1.0 = excellent shape, 0.0 = isolated/poor position>
"""
