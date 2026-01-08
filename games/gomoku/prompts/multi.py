FEATURES = [
  "immediate tactics: forced wins or forced defenses that resolve the position this move or next. e.g. 'H5 wins immediately: I have D5-E5-F5-G5; H5 completes five' or 'Must block D4: opponent threatens open-four at A4-B4-C4-_-E4'",
  "threat multiplicity: moves that create multiple independent winning threats the opponent cannot fully answer. e.g. 'E5 creates a fork: threatens D5-E5-F5-G5 and E3-E4-E5-E6' or 'F6 sets up two winning threats next turn'",
  "pattern strength: how a move improves or neutralizes standard Gomoku patterns (open-three, open-four, etc.). e.g. 'E5 upgrades my open-three to an open-four' or 'D4 blocks their open-three before it escalates'"
]


# === FEATURE-BASED (single move output) ===

feature_prompt = """You are player {player} in 9x9 Gomoku.
Your specialty: {feature}

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
History: {moves}

Prior proposals: {prev_proposals}

Choose the next move:
1. Are there any forced moves? (I win next, or opponent wins if I don't block)
2. What does YOUR specialty reveal that others might miss?
3. Do prior proposals have blind spots from your specialty's view?

ANALYSIS: <1 sentence>
MOVE: <vertex>"""


# === FEATURE-BASED (ranked output for Borda) ===

feature_ranked_prompt = """You are player {player} in 9x9 Gomoku.
Your specialty: {feature}

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
History: {moves}

Prior proposals: {prev_proposals}

Rank 3 choices for the next move:
1. Are there any forced moves? (I win next, or opponent wins if I don't block)
2. What does YOUR specialty reveal that others might miss?
3. Do prior proposals have blind spots from your specialty's view?

ANALYSIS: <1 sentence>
MOVE_1: <best>
MOVE_2: <second>
MOVE_3: <third>"""


# === ADVERSARIAL DEBATE (single prompt with debate history) ===

adversarial_prompt = """You are the {role} side in a debate about the best move for player {player} in 9x9 Gomoku.

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
Move history: {moves}

Debate so far:
{debate_history}

Rules:
- Propose a DIFFERENT move than your opponent when possible
- Find flaws in their reasoning and exploit them
- Only agree if you truly cannot find a better alternative

ANALYSIS: <1 sentence>
MOVE: <vertex>"""


# === CENTRALIZED DEBATE ===

centralized_propose_prompt = """You are player {player} in 9x9 Gomoku.

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
History: {moves}

{critiques_section}

Choose the best move. Consider any critiques carefully.

ANALYSIS: <1 sentence>
MOVE: <vertex>"""

centralized_critique_prompt = """You are a critic for player {player} in 9x9 Gomoku.
Your specialty: {feature}

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
History: {moves}

Current proposal:
{proposal}

Critique this proposal from your specialty's perspective. Look for blind spots.
Do NOT propose an alternative move — only critique.

CRITIQUE: <1 sentence>"""


# === JUDGE AGGREGATION ===

aggregate_prompt = """You are the judge for player {player} in 9x9 Gomoku.

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
Move history: {moves}

The debate has concluded. Final arguments:

[Affirmative]: {aff_final}

[Negative]: {neg_final}

Evaluate:
1. FORCED MOVES take priority — does either side identify an immediate win or must-block?
2. Which argument cites CONCRETE board positions and sequences?
3. Which critique of the opponent is more valid?

VERDICT: <1 sentence>
MOVE: <vertex>"""

aggregate_prompt_multi = """You are the judge for player {player} in 9x9 Gomoku.

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
Move history: {moves}

Agent proposals:
{proposals}

Evaluate:
1. FORCED MOVES take priority — does any agent identify an immediate win or must-block?
2. If agents agree on a move, that's strong evidence
3. Which argument cites CONCRETE board positions?

VERDICT: <1 sentence>
MOVE: <vertex>"""


# === SELF-CONSISTENCY (vanilla prompt) ===

action_prompt = """You are player {player} in 9x9 Gomoku.

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
History: {moves}

Choose the best move.

ANALYSIS: <1 sentence>
MOVE: <vertex>"""
