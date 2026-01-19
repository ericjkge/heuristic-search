import re

def extract_move(response):
    """Extract single move from MOVE: <vertex> format."""
    match = re.search(r"MOVE:\s*([A-Oa-o][1-9][0-9]?)", response, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()

def format_state(board, player):
    pos = board.to_positions()
    p1 = "; ".join(pos["p1"]) or "None"
    p2 = "; ".join(pos["p2"]) or "None"

    return {
        "player": "Black" if player == 1 else "White",
        "player_positions": p1 if player == 1 else p2,
        "opponent_positions": p2 if player == 1 else p1,
        "moves": board.to_moves(),
    }

config = {
    "extract_move": extract_move,
    "format_state": format_state,
}

