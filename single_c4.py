import re
import time
from dotenv import load_dotenv
from models import KimiLLM
from connect4_engine import Connect4Engine
from board import Connect4Board

load_dotenv()

LOG_FILE = "logs/log_single_c4.txt"
ENGINE = Connect4Engine(depth=8)

# Stats tracking
class RoundStats:
    def __init__(self):
        self.tokens = 0
        self.time = 0.0
        self.calls = 0
    
    def add(self, tokens, elapsed):
        self.tokens += tokens
        self.time += elapsed
        self.calls += 1

def log(msg, log_file=None):
    target_file = log_file if log_file else LOG_FILE
    with open(target_file, "a") as f:
        f.write(msg + "\n")

def get_engine_move(board, log_file=None):
    legal = board.legal_moves()
    
    # Convert 1-based history to 0-based for engine
    history_0based = "".join(str(int(c) - 1) for c in board.move_history)
    
    
    log(f"\n--- ENGINE DEBUG ---", log_file)
    log(f"Move history (1-based): '{board.move_history}'", log_file)
    log(f"Move history (0-based): '{history_0based}'", log_file)
    log(f"Starter: 1", log_file)
    
    try:
        result = ENGINE.suggest_move(starter=1, history_str=history_0based)
        
        if result.is_draw:
            log("Engine detected draw", log_file)
            return legal[0]
        
        if result.winner in {1, 2}:
            log(f"Engine detected Player {result.winner} won", log_file)
            return legal[0]
        
        if result.column is not None:
            move = result.column + 1
            log(f"Engine suggests column {move} (0-based: {result.column})", log_file)
            if move in legal:
                return move
        
        log(f"Invalid engine response, returning first legal move", log_file)
        return legal[0]
    except Exception as e:
        log(f"Engine error: {e}, returning first legal move", log_file)
        return legal[0]

PROMPT = """You are playing as {color} in Connect 4 (7 columns, 6 rows).
Current board:
{board}

Legal columns: {moves}

Analyze and propose the best column to play.
Format:
ANALYSIS: <your reasoning in MAX 1 sentence>
MOVE: <column number 1-7>"""

def extract_move(response):
    match = re.search(r"MOVE:\s*([1-7])", response)
    return int(match.group(1)) if match else None

def get_llm_move(llm, board, legal_moves, color="X", log_file=None, stats=None):
    moves_str = " ".join(str(m) for m in legal_moves)
    prompt = PROMPT.format(color=color, board=str(board), moves=moves_str)
    
    start = time.time()
    response, tokens = llm.generate(prompt)
    elapsed = time.time() - start
    
    if stats:
        stats.add(tokens, elapsed)
    
    move = extract_move(response)
    
    # Compact logging
    log(f"Single Agent:", log_file)
    log(f"{response.strip()}\n", log_file)
    
    if move in legal_moves:
        return move
    return legal_moves[0]

def main():
    open(LOG_FILE, "w").close()
    
    board = Connect4Board()
    llm = KimiLLM()
    move_num = 0
    total_stats = RoundStats()
    
    print("Starting Connect 4 Baseline. Single LLM plays X, Engine plays O.\n")
    print(board, "\n")
    
    while True:
        winner = board.check_winner()
        if winner:
            if winner == "draw":
                print("Game over: Draw!")
                log("\nGame over: Draw!", LOG_FILE)
            else:
                print(f"Game over: {winner} wins!")
                log(f"\nGame over: {winner} wins!", LOG_FILE)
            log(f"\n{'='*60}", LOG_FILE)
            log(f"TOTAL STATS: {total_stats.tokens} tokens, {total_stats.time:.2f}s, {total_stats.calls} calls", LOG_FILE)
            break
        
        if board.current_player == "X":
            move_num += 1
            move_stats = RoundStats()
            legal = board.legal_moves()
            
            # Log board state once
            log(f"\n{'='*60}", LOG_FILE)
            log(f"MOVE {move_num} (X) | Legal: {' '.join(str(m) for m in legal)}", LOG_FILE)
            log(f"{board}", LOG_FILE)
            log(f"{'='*60}", LOG_FILE)
            
            chosen = get_llm_move(llm, board, legal, color="X", log_file=LOG_FILE, stats=move_stats)
            
            log(f"\n--- STATS: {move_stats.tokens} tokens, {move_stats.time:.2f}s ---", LOG_FILE)
            log(f">>> CHOSEN MOVE: {chosen} <<<\n", LOG_FILE)
            
            total_stats.tokens += move_stats.tokens
            total_stats.time += move_stats.time
            total_stats.calls += move_stats.calls
            
            board.drop(chosen)
            print(f"LLM plays column: {chosen}")
        else:
            move = get_engine_move(board, log_file=LOG_FILE)
            board.drop(move)
            print(f"Engine plays column: {move}")
        
        print(board, "\n")

if __name__ == "__main__":
    main()

