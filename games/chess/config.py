import re

def extract_move(response):
    match = re.search(r"MOVE:\s*([a-h][1-8][a-h][1-8][qrbn]?)", response)
    return match.group(1).lower() if match else None

def extract_ranked_moves(response, legal_moves):
    """Extract ranked moves from MOVE_1/2/3/4/5: <uci> format for MCTS policy."""
    moves = []
    
    # Try to find MOVE_N patterns
    pattern = r"MOVE_(\d+):\s*([a-h][1-8][a-h][1-8][qrbn]?)"
    matches = re.findall(pattern, response, re.IGNORECASE)
    
    if matches:
        # Sort by rank number and extract moves
        sorted_matches = sorted(matches, key=lambda x: int(x[0]))
        moves = [m[1].lower() for m in sorted_matches]
    else:
        # Fallback: try to find any UCI moves in the response
        uci_pattern = r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b"
        found = re.findall(uci_pattern, response, re.IGNORECASE)
        moves = [m.lower() for m in found]
    
    # Filter to legal moves
    return [m for m in moves if m in legal_moves]

def extract_value(response):
    """Extract position value from VALUE: <float> format for MCTS value oracle."""
    # Try VALUE: pattern
    pattern = r"(?:VALUE|EVALUATION):\s*([-+]?\d*\.?\d+)"
    match = re.search(pattern, response, re.IGNORECASE)
    
    if match:
        try:
            value = float(match.group(1))
            return max(-1.0, min(1.0, value))
        except ValueError:
            pass
    
    # Fallback: look for any decimal number in [-1, 1]
    decimal_pattern = r"([-+]?\d*\.?\d+)"
    matches = re.findall(decimal_pattern, response)
    for m in reversed(matches):
        try:
            value = float(m)
            if -1.0 <= value <= 1.0:
                return value
        except ValueError:
            continue
    
    return 0.0

def format_state(board, player):
    return {
        "player": "White" if player == 1 else "Black",
        "fen": board.to_positions(),
        "pgn": board.to_moves() or "(start)",
    }

config = {
    "extract_move": extract_move,
    "extract_ranked_moves": extract_ranked_moves,
    "extract_value": extract_value,
    "format_state": format_state,
}