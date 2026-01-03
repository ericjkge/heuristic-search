import re

def extract_move(response):
    COLS = "ABCDEFGHJKLMNOPQRST"
    match = re.search(r"MOVE:\s*([A-Ta-t][1-9][0-9]?|pass)", response, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid move: {response}")
    text = match.group(1).upper()
    if text == "PASS":
        return (None, None)
    col = COLS.index(text[0])
    row = int(text[1:]) - 1
    return (row, col)

def format_state(board, player):
    return {
        "player": "Black" if player == 1 else "White",
        "sgf": board.to_moves(),
    }

config = {
    "extract_move": extract_move,
    "format_state": format_state,
}