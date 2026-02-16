import re
import chess


def extract_move(response, fen=None):
    """
    Extract SAN move from response (no conversion).
    Returns SAN format (e.g., e4, Nf3) or None if no valid move found.
    """
    # Match SAN format: Nf3, dxc5, O-O, e4, Qxd8+, etc.
    san_match = re.search(r"MOVE:\s*([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O-O|O-O)", response)
    if san_match:
        return san_match.group(1)
    return None


def san_to_uci(san_move, fen):
    """Convert SAN move to UCI format."""
    try:
        board = chess.Board(fen)
        move = board.parse_san(san_move)
        return move.uci()
    except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
        return None


def format_state(board, player):
    return {
        "player": "White" if player == 1 else "Black",
        "fen": board.to_positions(),
        "pgn": board.to_moves() or "(start)",
        "legal_moves": ", ".join(board.legal_moves()),
    }


config = {
    "extract_move": extract_move,
    "format_state": format_state,
    "san_to_uci": san_to_uci,
}
