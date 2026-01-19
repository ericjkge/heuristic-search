# Generator prompts for Gomoku move generation
# Initial prompts for first iteration, feedback prompts for subsequent iterations

# === INITIAL PROMPTS (Iteration 1) ===

counterfactual_initial = """
You are player {player} in a 9x9 Gomoku game.

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
History: {moves}

Instructions: Simulate each candidate move assuming optimal opponent response and choose the move with the best worst-case outcome. Respond in the following format:

ANALYSIS: <1 sentence>
MOVE: <vertex like D4>
"""

pattern_initial = """
You are player {player} in a 9x9 Gomoku game.

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
History: {moves}

Instructions: Scan the board for known Gomoku patterns (open threes, fours, forks) for each player and choose the move with the strongest pattern outcome. Respond in the following format:

ANALYSIS: <1 sentence>
MOVE: <vertex like D4>
"""

# === FEEDBACK PROMPTS (Iterations 2-3) ===

counterfactual_feedback = """
Your previous move proposal was evaluated:

Move: {previous_move}
Scores:
- Immediate Loss: {immediate_loss_score} - {immediate_loss_reasoning}
- Illegal Move: {illegal_move_score} - {illegal_move_reasoning}
- Aggressive: {aggressive_score} - {aggressive_reasoning}
- Defensive: {defensive_score} - {defensive_reasoning}
- Shape: {shape_score} - {shape_reasoning}
Final Score: {final_score}

Based on this feedback, propose a better move using counterfactual simulation (consider opponent's best response to each candidate). Respond in the following format:

ANALYSIS: <1 sentence>
MOVE: <vertex like D4>
"""

pattern_feedback = """
Your previous move proposal was evaluated:

Move: {previous_move}
Scores:
- Immediate Loss: {immediate_loss_score} - {immediate_loss_reasoning}
- Illegal Move: {illegal_move_score} - {illegal_move_reasoning}
- Aggressive: {aggressive_score} - {aggressive_reasoning}
- Defensive: {defensive_score} - {defensive_reasoning}
- Shape: {shape_score} - {shape_reasoning}
Final Score: {final_score}

Based on this feedback, propose a better move by scanning for Gomoku patterns (open threes, fours, forks). Respond in the following format:

ANALYSIS: <1 sentence>
MOVE: <vertex like D4>
"""
