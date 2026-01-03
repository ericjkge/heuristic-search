import chess
import chess.engine
from pathlib import Path

class ChessEngine:
    def __init__(self):
        engine_path = Path(__file__).parent / "stockfish-macos-m1-apple-silicon"
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    def get_move(self, board):
        result = self.engine.play(board._board, chess.engine.Limit(time=0.1))
        return result.move.uci()
    
    def close(self):
        self.engine.quit()