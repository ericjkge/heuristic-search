import subprocess
from pathlib import Path

WIDTH = 7

class Connect4Engine:
    def __init__(self):
        self.solver_path = Path(__file__).parent / "c4solver"
        self.move_sequence = ""


    def get_move(self):
        try:
            # Use c4solver with -a flag to analyze all moves
            result = subprocess.run(
                [str(self.solver_path), "-a"],
                input=self.move_sequence + "\n",
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.solver_path.parent)
            )
            
            if result.returncode != 0:
                print(f"Solver error: {result.stderr}")
                return None
            
            # Parse output: "move_sequence score1 score2 score3 score4 score5 score6 score7"
            output = result.stdout.strip()
            parts = output.split()
            
            # Skip the first part (echoed move sequence), take the 7 scores
            scores = list(map(int, parts[1:]))
            
            # Find the best move (highest score among playable columns)
            best_col = -1
            best_score = -10000
            
            for col in range(WIDTH):
                if scores[col] > -1000:  # -1000 = INVALID_MOVE
                    if scores[col] > best_score:
                        best_score = scores[col]
                        best_col = col
            
            if best_col == -1:
                raise ValueError("No valid moves found")
            
            return best_col + 1  # Convert to 1-indexed column
            
        except subprocess.TimeoutExpired:
            print("Engine timed out!")
            return None
        except Exception as e:
            print(f"Engine error: {e}")
            return None