import chess
import chess.engine
import re
from collections import Counter
from dotenv import load_dotenv
from models import KimiLLM
import prompts_chess as prompts

load_dotenv()

NUM_ROUNDS = 1 # Rounds of debate (non-inclusive of proposals)
ENGINE_PATH = "./stockfish-macos-m1-apple-silicon"
LOG_FILE = "logs/log_multi_chess.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def extract_move(response):
    match = re.search(r'MOVE:\s*([a-h][1-8][a-h][1-8][qrbn]?)', response, re.IGNORECASE)
    return match.group(1).lower() if match else ""

def create_agents(perspectives):
    return [(KimiLLM(), p) for p in perspectives]

def get_proposals(agents, fen, legal_moves, color, prev_proposals=None, round_num=0): # round_num is just for logging
    proposals = []
    moves_str = " ".join(legal_moves)
    
    for i, (llm, perspective) in enumerate(agents):
        if prev_proposals is None:
            prompt = prompts.propose_prompt.format(color=color, fen=fen, moves=moves_str)
        else:
            other = "\n".join(f"Agent {j+1}: {p}" for j, p in enumerate(prev_proposals) if j != i)
            prompt = prompts.debate_prompt.format(color=color, fen=fen, moves=moves_str, other_proposals=other)
        
        response, _ = llm.generate(prompt, system_prompt=perspective)
        move = extract_move(response)
        proposals.append((move, response))
        
        # Log the LLM call
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
        return legal_moves[0] # Fallback to first legal move if all LLMs hallucinated
    return Counter(moves).most_common(1)[0][0]

def main():
    open(LOG_FILE, "w").close()  # Clear log file
    
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    agents = create_agents(prompts.PERSPECTIVES)
    
    print("Starting game. Agents play White, Engine plays Black.\n")
    
    while not board.is_game_over():
        legal_moves = [m.uci() for m in board.legal_moves]
        proposals = get_proposals(agents, board.fen(), legal_moves, "White", round_num=0)
        
        for r in range(NUM_ROUNDS):
            proposals = get_proposals(agents, board.fen(), legal_moves, "White", proposals, round_num=r+1)
            
        chosen_move = aggregate_moves(proposals, legal_moves)
        log(f"\n{'#'*60}")
        log(f"CHOSEN MOVE: {chosen_move}")
        log(f"{'#'*60}\n")
        
        board.push(chess.Move.from_uci(chosen_move))
        print(f"Agents play: {chosen_move}")
        print(board, "\n")

        if board.is_game_over():
            break

        result = engine.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
        print(f"Engine plays: {result.move.uci()}")
        print(board, "\n")
        
    print(f"Game over: {board.result()}")
    engine.quit()

if __name__ == "__main__":
    main()
