import subprocess
import re
from collections import Counter
from dotenv import load_dotenv
from models import KimiLLM
import prompts_connect4 as prompts

load_dotenv()

NUM_ROUNDS = 1
SOLVER_PATH = "./c4solver"
LOG_FILE = "log_connect4.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

class Connect4Board:
    def __init__(self):
        self.rows = 6
        self.cols = 7 
        self.board = [[" " for _ in range(self.cols)] for _ in range(self.rows)]
        self.move_history = ""
        self.current_player = "X" # X goes first

    def legal_moves(self):
        return [c + 1 for c in range(self.cols) if self.board[0][c] == " "]
    
    def drop(self, col):
        col_index = col - 1 # 1-indexed
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col_index] == " ":
                self.board[row][col_index] = self.current_player
                self.move_history += str(col)
                self.current_player = "O" if self.current_player == "X" else "X"
                return True
        return False
    
    def check_winner(self):
        for r in range(self.rows):
            for c in range(self.cols):
                piece = self.board[r][c]
                if piece == " ":
                    continue
                # Horizontal
                if c + 3 < self.cols and all(self.board[r][c+i] == piece for i in range(4)):
                    return piece
                # Vertical
                if r + 3 < self.rows and all(self.board[r+i][c] == piece for i in range(4)):
                    return piece
                # Diagonal down-right
                if r + 3 < self.rows and c + 3 < self.cols and all(self.board[r+i][c+i] == piece for i in range(4)):
                    return piece
                # Diagonal up-right
                if r - 3 >= 0 and c + 3 < self.cols and all(self.board[r-i][c+i] == piece for i in range(4)):
                    return piece
        if not self.legal_moves():
            return 'draw'
        return None

    def __str__(self):
        lines = []
        for row in self.board:
            lines.append('|' + '|'.join(row) + '|')
        lines.append('+' + '+'.join(['-'] * self.cols) + '+')
        lines.append(' ' + ' '.join(str(i) for i in range(1, self.cols + 1)) + ' ')
        return '\n'.join(lines)

def get_solver_move(board):
    legal = board.legal_moves()
    best_col = legal[0]
    best_score = float("-inf")

    for col in legal:
        candidate = board.move_history + str(col)

        result = subprocess.run(
            [SOLVER_PATH],
            input=candidate,
            capture_output=True,
            text=True,
            timeout=2
        )

        output = result.stdout.strip()

        if "Invalid move" in output:
            continue

        score = int(output.split()[1])

        if score > best_score:
            best_score = score
            best_col = col
    
    return best_col

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
    
    print("Starting Connect 4. Agents play X, Solver plays O.\n")
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
            # Solver's turn
            move = get_solver_move(board)
            board.drop(move)
            print(f"Solver plays column: {move}")
        
        print(board, "\n")

if __name__ == "__main__":
    main()

# Also, switch to voting OR extra agent for conclusion
# Try using another agent as oppponent
# Run trials on Connect 4 (test baseline with single LLM)