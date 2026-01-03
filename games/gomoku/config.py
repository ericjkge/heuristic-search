import re

def extract_move(response):
    match = re.search(r"MOVE:\s*(\d+)\s*,\s*(\d+)", response) # MOVE: row,col (1-indexed)
    if match:
        row = int(match.group(1)) - 1  # Convert to 0-indexed
        col = int(match.group(2)) - 1
        return (row, col)
    return None

def format_state(board, player):
    pos = board.to_positions()
    p1 = "; ".join(f"{r},{c}" for r, c in pos["p1"]) or "None"
    p2 = "; ".join(f"{r},{c}" for r, c in pos["p2"]) or "None"

    return {
        "player": player,
        "player_positions": p1 if player == 1 else p2,
        "opponent_positions": p2 if player == 1 else p1,
    }

config = {
    "extract_move": extract_move,
    "format_state": format_state,
}

