# Generate move 1
initial_prompt = """You are player {player} in a chess game.

FEN: {fen}
PGN: {pgn}
Legal moves: {legal_moves}

Choose the best move from the legal moves list. Respond in the following format (no markdown, explanation, or intro):

ANALYSIS: <1 sentence>
MOVE: <move from legal moves list>"""

# Generate move 2
feedback_prompt = """Your move {previous_move} was analyzed:

{verifier_feedback}

Based on this analysis, choose a new move from the legal moves list.
Legal moves: {legal_moves}

Respond in the following format (no markdown, explanation, or intro):

ANALYSIS: <1 sentence addressing the feedback>
MOVE: <NEW move from legal moves list>"""

# Generate verification code (before process starts)
verifier_code_prompt = """Write a Python script that extracts important information about a chess position before and after making a move. Use __FEN__ and __SAN__ as placeholders for the position in FEN format and proposed move in SAN format (they will be substituted out before execution).

# Rules:
- Only `import chess` is available (python-chess library). No other packages.
- Print all results to stdout as labeled lines (e.g. "METRIC: value").
- Do NOT score or judge the position
- Only output the Python script (no explanation, intro, etc.)

# python-chess API reference:
board = chess.Board(fen)          # create board
board.push_san("Nf3")             # make move (SAN), mutates board
board.pop()                       # undo last move
board.copy()                      # deep copy
board.turn                        # chess.WHITE or chess.BLACK
board.is_check()                  # bool
board.is_checkmate()              # bool

board.piece_at(sq)                # Piece or None
board.piece_map()                 # dict{{square: Piece}}
piece.piece_type                  # chess.PAWN / KNIGHT / BISHOP / ROOK / QUEEN / KING
piece.color                       # chess.WHITE or chess.BLACK
piece.symbol()                    # "P", "n", "B", etc.

board.legal_moves                 # iterable of Move objects
board.is_capture(move)            # bool
board.gives_check(move)           # bool
board.san(move)                   # SAN string for Move

board.attackers(color, sq)        # SquareSet of pieces attacking sq
board.is_attacked_by(color, sq)   # bool
board.king(color)                 # square of king
board.is_pinned(color, sq)        # bool

chess.square_name(sq)             # "e4"
chess.square_file(sq)             # 0-7 (a=0)
chess.square_rank(sq)             # 0-7 (rank1=0)

len(square_set)                   # count squares in a SquareSet
for sq in square_set: ...         # iterate a SquareSet
"""

# Interpret code
verifier_interpret_prompt = """Evaluate a proposed move for a chess position based on data output. Consider different evaluation heuristics and prioritize avoiding blunders.

FEN: {fen}
Player: {player}
Proposed move: {move}

Data:
{code_output}

Respond in this format:

FEEDBACK: <2 sentences>
SCORE: <0.0-1.0, where 0.0=blunder, 0.7 = good, 0.8 = 1.0=best>
"""
