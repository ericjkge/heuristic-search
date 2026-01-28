"""Deterministic chess engine wrapper."""

import chess
import chess.engine


class ChessEngine:
    """Wrapper for Stockfish with deterministic settings."""

    def __init__(self, path, depth=16, threads=1):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({'Threads': threads})
        self.depth = depth

    def analyse(self, board):
        """Analyse position with hash cleared for determinism."""
        self.engine.protocol.send_line('ucinewgame')
        return self.engine.analyse(board, chess.engine.Limit(depth=self.depth))

    def evaluate_position(self, fen):
        """Get eval in centipawns from White's perspective."""
        board = chess.Board(fen)
        result = self.analyse(board)
        score = result["score"].white()

        if score.is_mate():
            return 10000 if score.mate() > 0 else -10000
        return score.score()

    def evaluate_move(self, fen, move_uci):
        """Get eval after a move in centipawns from White's perspective."""
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        board.push(move)

        result = self.analyse(board)
        score = result["score"].white()

        if score.is_mate():
            return 10000 if score.mate() > 0 else -10000
        return score.score()

    def get_best_move(self, fen):
        """Get best move and its eval."""
        board = chess.Board(fen)
        result = self.analyse(board)
        best_move = result["pv"][0]
        return best_move.uci(), board.san(best_move)

    def evaluate_position_pawns(self, fen):
        """Get eval in pawns from White's perspective."""
        cp = self.evaluate_position(fen)
        if abs(cp) >= 10000:
            return 100.0 if cp > 0 else -100.0
        return cp / 100.0

    def evaluate_move_pawns(self, fen, move_uci):
        """Get eval after a move in pawns from White's perspective."""
        cp = self.evaluate_move(fen, move_uci)
        if abs(cp) >= 10000:
            return 100.0 if cp > 0 else -100.0
        return cp / 100.0

    def close(self):
        self.engine.quit()
