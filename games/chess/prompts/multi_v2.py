# Multi-v2 prompts: generate-verify-refine loop with multi-turn conversation

# === MOVE GENERATION ===

initial_prompt = """You are player {player} in a chess game.

FEN: {fen}
PGN: {pgn}
Legal moves: {legal_moves}

IMPORTANT: You must choose a move from the legal moves list above.

ANALYSIS: <1 sentence>
MOVE: <move from legal moves list>"""

feedback_prompt = """Your move {previous_move} was evaluated by 5 verifiers:

{verifier_feedback}

Based on this feedback, choose a better move from the legal moves list.
Legal moves: {legal_moves}

ANALYSIS: <1 sentence addressing the feedback>
MOVE: <move from legal moves list>"""

# === BINARY VERIFIER PROMPTS (each outputs SCORE: 0 or 1) ===

blunder_prompt = """You are checking a chess move for blunders.

FEN: {fen}
PGN: {pgn}
Player: {player}
Legal moves: {legal_moves}
Proposed move: {move}

Does this move hang a piece (leave it undefended and attacked), allow the opponent to win material through tactics, or allow forced checkmate?

REASONING: <1 sentence>
SCORE: <1 if safe, 0 if blunder>"""

material_prompt = """You are evaluating material for a chess move.

FEN: {fen}
PGN: {pgn}
Player: {player}
Legal moves: {legal_moves}
Proposed move: {move}

Does this move maintain or improve material balance? (Gains material, makes an equal/favorable trade, or at least doesn't lose material.)

REASONING: <1 sentence>
SCORE: <1 if OK, 0 if loses material>"""

king_safety_prompt = """You are evaluating king safety for a chess move.

FEN: {fen}
PGN: {pgn}
Player: {player}
Legal moves: {legal_moves}
Proposed move: {move}

Does this move keep the king safe? (Doesn't expose the king to attack, doesn't weaken the pawn shield, castles or prepares castling when appropriate.)

REASONING: <1 sentence>
SCORE: <1 if safe, 0 if compromised>"""

pawn_structure_prompt = """You are evaluating pawn structure for a chess move.

FEN: {fen}
PGN: {pgn}
Player: {player}
Legal moves: {legal_moves}
Proposed move: {move}

Does this move maintain or improve pawn structure? (Doesn't create doubled/isolated pawns unnecessarily, or the trade-off is worth it.)

REASONING: <1 sentence>
SCORE: <1 if OK, 0 if damaged>"""

piece_activity_prompt = """You are evaluating piece activity for a chess move.

FEN: {fen}
PGN: {pgn}
Player: {player}
Legal moves: {legal_moves}
Proposed move: {move}

Does this move maintain or improve piece activity? (Develops pieces, controls central squares, improves coordination, opens lines.)

REASONING: <1 sentence>
SCORE: <1 if active, 0 if passive>"""
