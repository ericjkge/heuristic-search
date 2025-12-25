import re
from dotenv import load_dotenv
from models import KimiLLM
from connect4_engine import Connect4Engine
from board import Connect4Board

load_dotenv()

LOG_FILE = "logs/log_single_c4.txt"
ENGINE = Connect4Engine(depth=8)

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def get_engine_move(board):
    legal = board.legal_moves()
    
    # Convert 1-based history to 0-based for engine
    history_0based = "".join(str(int(c) - 1) for c in board.move_history)
    
    
    log(f"\n--- ENGINE DEBUG ---")
    log(f"Move history (1-based): '{board.move_history}'")
    log(f"Move history (0-based): '{history_0based}'")
    log(f"Starter: 1")
    
    try:
        result = ENGINE.suggest_move(starter=1, history_str=history_0based)
        
        if result.is_draw:
            log("Engine detected draw")
            return legal[0]
        
        if result.winner in {1, 2}:
            log(f"Engine detected Player {result.winner} won")
            return legal[0]
        
        if result.column is not None:
            move = result.column + 1
            log(f"Engine suggests column {move} (0-based: {result.column})")
            if move in legal:
                return move
        
        log(f"Invalid engine response, returning first legal move")
        return legal[0]
    except Exception as e:
        log(f"Engine error: {e}, returning first legal move")
        return legal[0]

PROMPT = """You are playing as {color} in Connect 4 (7 columns, 6 rows).
Current board:
{board}

Legal columns: {moves}

Analyze and propose the best column to play.
Format:
ANALYSIS: <your reasoning in AT MOST 2 sentences>
MOVE: <column number 1-7>"""

def extract_move(response):
    match = re.search(r"MOVE:\s*([1-7])", response)
    return int(match.group(1)) if match else None

def get_llm_move(llm, board, legal_moves):
    moves_str = " ".join(str(m) for m in legal_moves)
    prompt = PROMPT.format(color="X", board=str(board), moves=moves_str)
    
    response, _ = llm.generate(prompt)
    move = extract_move(response)
    
    log(f"{'='*60}")
    log(f"\n--- PROMPT ---\n{prompt}")
    log(f"\n--- RESPONSE ---\n{response}")
    log(f"\nEXTRACTED MOVE: {move}")
    
    if move in legal_moves:
        return move
    return legal_moves[0]

def main():
    open(LOG_FILE, "w").close()
    
    board = Connect4Board()
    llm = KimiLLM()
    
    print("Starting Connect 4 Baseline. Single LLM plays X, Engine plays O.\n")
    print(board, "\n")
    
    while True:
        winner = board.check_winner()
        if winner:
            if winner == "draw":
                print("Game over: Draw!")
            else:
                print(f"Game over: {winner} wins!")
            break
        
        if board.current_player == "X":
            legal = board.legal_moves()
            chosen = get_llm_move(llm, board, legal)
            
            log(f"\n{'#'*60}")
            log(f"CHOSEN MOVE: {chosen}")
            log(f"{'#'*60}\n")
            
            board.drop(chosen)
            print(f"LLM plays column: {chosen}")
        else:
            move = get_engine_move(board)
            board.drop(move)
            print(f"Engine plays column: {move}")
        
        print(board, "\n")

if __name__ == "__main__":
    main()

