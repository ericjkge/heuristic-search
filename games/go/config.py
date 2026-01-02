import re

def extract_move(response):
    match = re.search(r"MOVE:\s*([A-Ta-t][1-9][0-9]?|pass)", response, re.IGNORECASE)
    return match.group(1).upper() if match else None

def format_state(board, player):
    return {
        "player": "Black" if player == 1 else "White",
        "sgf": board.to_sgf(),
    }

config = {
    "extract_move": extract_move,
    "format_state": format_state,
}