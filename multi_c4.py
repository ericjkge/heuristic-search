import re
from collections import Counter
from dotenv import load_dotenv
from models import KimiLLM
from connect4_engine import Connect4Engine
from board import Connect4Board
import prompts_c4 as prompts

load_dotenv()

NUM_ROUNDS = 1
LOG_FILE = "logs/log_multi_c4.txt"
ENGINE = Connect4Engine(depth=8)

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def get_engine_move(board):
    """Get best move from engine using Python API."""
    legal = board.legal_moves()
    
    # Convert 1-based history to 0-based for engine
    history_0based = "".join(str(int(c) - 1) for c in board.move_history)

    try:
        result = ENGINE.suggest_move(starter=1, history_str=history_0based)
        
        if result.is_draw or result.winner in {1, 2}:
            return legal[0]
        
        if result.column is not None:
            move = result.column + 1  # Convert 0-based to 1-based
            if move in legal:
                return move
        
        return legal[0]
    except Exception:
        return legal[0]

def extract_move(response):
    match = re.search(r'MOVE:\s*([1-7])', response)
    return int(match.group(1)) if match else None

def create_agents(perspectives):
    return [(KimiLLM(), p) for p in perspectives]

def get_proposals(agents, board, legal_moves, color, prev_proposals=None, round_num=0):
    proposals = []
    moves_str = " ".join(str(m) for m in legal_moves)
    board_str = str(board)
    
    for i, (llm, perspective) in enumerate(agents):
        if prev_proposals is None:
            prompt = prompts.propose_prompt.format(color=color, board=board_str, moves=moves_str)
        else:
            other = "\n".join(f"Agent {j+1}: {p}" for j, p in enumerate(prev_proposals) if j != i)
            prompt = prompts.debate_prompt.format(color=color, board=board_str, moves=moves_str, other_proposals=other)
        
        response, _ = llm.generate(prompt, system_prompt=perspective)
        move = extract_move(response)
        proposals.append((move, response))
        
        log(f"{'='*60}")
        log(f"AGENT {i+1} | ROUND {round_num}")
        log(f"PERSPECTIVE: {perspective}")
        log(f"\n--- PROMPT ---\n{prompt}")
        log(f"\n--- RESPONSE ---\n{response}")
        log(f"\nEXTRACTED MOVE: {move}")
    
    return proposals

def aggregate_moves(proposals, legal_moves):
    moves = [p[0] for p in proposals if p[0] in legal_moves]
    if not moves:
        return legal_moves[0]
    return Counter(moves).most_common(1)[0][0]

def main():
    open(LOG_FILE, "w").close()
    
    board = Connect4Board()
    agents = create_agents(prompts.PERSPECTIVES)
    
    print("Starting Connect 4. Agents play X, Engine plays O.\n")
    print(board, "\n")
    
    while True:
        winner = board.check_winner()
        if winner:
            if winner == 'draw':
                print("Game over: Draw!")
            else:
                print(f"Game over: {winner} wins!")
            break
        
        if board.current_player == 'X':
            # Agents' turn
            legal = board.legal_moves()
            proposals = get_proposals(agents, board, legal, "X", round_num=0)
            
            for r in range(NUM_ROUNDS):
                proposals = get_proposals(agents, board, legal, "X", proposals, round_num=r+1)
            
            chosen = aggregate_moves(proposals, legal)
            log(f"\n{'#'*60}")
            log(f"CHOSEN MOVE: {chosen}")
            log(f"{'#'*60}\n")
            
            board.drop(chosen)
            print(f"Agents play column: {chosen}")
        else:
            # Engine's turn
            move = get_engine_move(board)
            board.drop(move)
            print(f"Engine plays column: {move}")
        
        print(board, "\n")

if __name__ == "__main__":
    main()

# Also, switch to voting OR extra agent for conclusion
# Try using another agent as oppponent
# Run trials on Connect 4 (test baseline with single LLM)