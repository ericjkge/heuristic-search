import re

def extract_move(response):
    match = re.search(r"MOVE:\s*([1-7])", response)
    return int(match.group(1)) if match else None

def format_state(board, player):
    return {
        "board": str(board),
        "player": player
    }

config = {
    "extract_move": extract_move,
    "format_state": format_state,
}