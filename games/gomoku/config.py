import re

def extract_move(response):
    match = re.search(r"MOVE:\s*(\d+)\s*,\s*(\d+)", response) # MOVE: row,col (1-indexed)
    if match:
        row = int(match.group(1)) - 1  # Convert to 0-indexed
        col = int(match.group(2)) - 1
        return (row, col)
    return None

def format_state(board, player):
    return {
        "player": player,
        "size": board.size,
        "player_positions": board.format_positions("X" if player == 1 else "O"),
        "opponent_positions": board.format_positions("O" if player == 1 else "X")
    }

config = {
    "extract_move": extract_move,
    "format_state": format_state,
}

