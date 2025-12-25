from dotenv import load_dotenv
from board import Connect4Board
import prompts_c4 as prompts
from multi_c4 import create_agents, get_proposals, aggregate_moves, NUM_ROUNDS
from single_c4 import get_llm_move
from models import GeminiLLM

load_dotenv()

LOG_FILE = "logs/log_vs.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def get_multi_agent_move(agents, board, legal_moves, color):
    proposals = get_proposals(agents, board, legal_moves, color, round_num=0, log_file=LOG_FILE)
    
    for r in range(NUM_ROUNDS):
        proposals = get_proposals(agents, board, legal_moves, color, proposals, round_num=r+1, log_file=LOG_FILE)
    
    chosen = aggregate_moves(board, proposals, legal_moves, color=color, log_file=LOG_FILE)
    log(f"\n{'#'*60}")
    log(f"MULTI-AGENT CHOSEN MOVE: {chosen}")
    log(f"{'#'*60}\n")
    
    return chosen

def get_single_agent_move(llm, board, legal_moves, color):
    chosen = get_llm_move(llm, board, legal_moves, color=color, log_file=LOG_FILE)
    log(f"\n{'#'*60}")
    log(f"SINGLE AGENT CHOSEN MOVE: {chosen}")
    log(f"{'#'*60}\n")
    
    return chosen

def main():
    open(LOG_FILE, "w").close()
    
    board = Connect4Board()
    multi_agents = create_agents(prompts.PERSPECTIVES)
    single_llm = GeminiLLM()
    
    # Multi-agent plays X, Single-agent plays O
    print("Starting Connect 4: Multi-Agent (X) vs Single-Agent (O)\n")
    log("Starting Connect 4: Multi-Agent (X) vs Single-Agent (O)\n")
    print(board, "\n")
    
    while True:
        winner = board.check_winner()
        if winner:
            if winner == 'draw':
                result = "Game over: Draw!"
            else:
                result = f"Game over: {winner} wins!"
            print(result)
            log(f"\n{'='*60}")
            log(result)
            log(f"{'='*60}")
            break
        
        if board.current_player == 'X':
            # Multi-agent's turn
            legal = board.legal_moves()
            chosen = get_multi_agent_move(multi_agents, board, legal, "X")
            board.drop(chosen)
            print(f"Multi-Agent plays column: {chosen}")
        else:
            # Single-agent's turn
            legal = board.legal_moves()
            chosen = get_single_agent_move(single_llm, board, legal, "O")
            board.drop(chosen)
            print(f"Single-Agent plays column: {chosen}")
        
        print(board, "\n")

if __name__ == "__main__":
    main()
