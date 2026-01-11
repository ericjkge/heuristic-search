import chess
import chess.engine
from pathlib import Path

class ChessEngine:
    def __init__(self, skill_level=0):
        engine_path = Path(__file__).parent / "stockfish-macos-m1-apple-silicon"
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.skill_level = skill_level
        
        if skill_level < 20:
            self.engine.configure({"UCI_LimitStrength": True, "Skill Level": skill_level})

    def get_move(self, board):
        result = self.engine.play(board._board, chess.engine.Limit(time=0.1))
        return result.move.uci()
    
    def close(self):
        self.engine.quit()