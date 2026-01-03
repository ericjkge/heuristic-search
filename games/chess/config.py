import re

def extract_move(response):
    match = re.search(r"MOVE:\s*([a-h][1-8][a-h][1-8][qrbn]?)", response)
    return match.group(1).lower() if match else None

def format_state(board, player):
    return {
        "player": "White" if player == 1 else "Black",
        "fen": board.to_positions(),
        "pgn": board.to_moves(),
    }

config = {
    "extract_move": extract_move,
    "format_state": format_state,
}