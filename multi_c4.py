import re
import time
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

def get_engine_move(board):
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

def get_proposals(agents, board, legal_moves, color, prev_proposals=None, round_num=0, log_file=None, stats=None):
    proposals = []
    moves_str = " ".join(str(m) for m in legal_moves)
    board_str = str(board)
    
    for i, (llm, perspective) in enumerate(agents):
        if prev_proposals is None:
            prompt = prompts.propose_prompt.format(color=color, board=board_str, moves=moves_str)
        else:
            other = "\n".join(f"Agent {j+1}: {p}" for j, p in enumerate(prev_proposals) if j != i)
            prompt = prompts.debate_prompt.format(color=color, board=board_str, moves=moves_str, other_proposals=other)
        
        start = time.time()
        response, tokens = llm.generate(prompt, system_prompt=perspective)
        elapsed = time.time() - start
        
        if stats:
            stats.add(tokens, elapsed)
        
        move = extract_move(response)
        proposals.append((move, response))
        
        # Compact logging: agent name, then response on new line
        perspective_short = perspective.split("focused on ")[1].split(".")[0] if "focused on" in perspective else "UNKNOWN"
        log(f"Agent {i+1} [{perspective_short}] R{round_num}:", log_file)
        log(f"{response.strip()}", log_file)
    
    return proposals

def aggregate_moves(board, proposals, legal_moves, color="X", log_file=None, stats=None):
    # Format proposals for conclusion agent
    proposal_text = "\n".join(
        f"Agent {i+1}:\n{p}" 
        for i, p in enumerate(proposals)
    )
    
    moves_str = " ".join(str(m) for m in legal_moves)
    prompt = prompts.conclusion_prompt.format(
        color=color,
        board=str(board),
        moves=moves_str,
        final_proposals=proposal_text
    )
    
    llm = KimiLLM()
    start = time.time()
    response, tokens = llm.generate(prompt, system_prompt=prompts.CONCLUSION_PERSPECTIVE)
    elapsed = time.time() - start
    
    if stats:
        stats.add(tokens, elapsed)
    
    move = extract_move(response)
    
    log(f"\nConclusion:", log_file)
    log(f"{response.strip()}\n", log_file)
    
    if move and move in legal_moves:
        return move
    
    # Fallback to most common if conclusion agent fails
    moves = [p[0] for p in proposals if p[0] in legal_moves]
    if moves:
        return Counter(moves).most_common(1)[0][0]
    return legal_moves[0]

def main():
    open(LOG_FILE, "w").close()
    
    board = Connect4Board()
    agents = create_agents(prompts.PERSPECTIVES)
    move_num = 0
    total_stats = RoundStats()
    
    print("Starting Connect 4. Agents play X, Engine plays O.\n")
    print(board, "\n")
    
    while True:
        winner = board.check_winner()
        if winner:
            if winner == 'draw':
                print("Game over: Draw!")
                log("\nGame over: Draw!", LOG_FILE)
            else:
                print(f"Game over: {winner} wins!")
                log(f"\nGame over: {winner} wins!", LOG_FILE)
            log(f"\n{'='*60}", LOG_FILE)
            log(f"TOTAL STATS: {total_stats.tokens} tokens, {total_stats.time:.2f}s, {total_stats.calls} calls", LOG_FILE)
            break
        
        if board.current_player == 'X':
            move_num += 1
            move_stats = RoundStats()
            legal = board.legal_moves()
            
            # Log board state once at start of move
            log(f"\n{'='*60}", LOG_FILE)
            log(f"MOVE {move_num} (X) | Legal: {' '.join(str(m) for m in legal)}", LOG_FILE)
            log(f"{board}", LOG_FILE)
            log(f"{'='*60}", LOG_FILE)
            
            # Debate rounds
            proposals = get_proposals(agents, board, legal, "X", round_num=0, log_file=LOG_FILE, stats=move_stats)
            
            for r in range(NUM_ROUNDS):
                proposals = get_proposals(agents, board, legal, "X", proposals, round_num=r+1, log_file=LOG_FILE, stats=move_stats)
            
            chosen = aggregate_moves(board, proposals, legal, color="X", log_file=LOG_FILE, stats=move_stats)
            
            # Log stats for this move
            log(f"\n--- STATS: {move_stats.tokens} tokens, {move_stats.time:.2f}s, {move_stats.calls} LLM calls ---", LOG_FILE)
            log(f">>> CHOSEN MOVE: {chosen} <<<\n", LOG_FILE)
            
            total_stats.tokens += move_stats.tokens
            total_stats.time += move_stats.time
            total_stats.calls += move_stats.calls
            
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