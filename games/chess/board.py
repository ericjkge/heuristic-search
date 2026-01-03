import chess
import chess.pgn

class ChessBoard:
    def __init__(self):
        self._board = chess.Board()
    
    def turn(self):
        return 1 if self._board.turn == chess.WHITE else 2
    
    def push(self, uci):
        self._board.push(chess.Move.from_uci(uci))
    
    def legal_moves(self):
        return [m.uci() for m in self._board.legal_moves]
    
    def winner(self):
        outcome = self._board.outcome()
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0  # draw
        return 1 if outcome.winner == chess.WHITE else 2
    
    def to_ascii(self):
        return str(self._board)
    
    def to_moves(self):
        game = chess.pgn.Game()
        node = game
        for move in self._board.move_stack:
            node = node.add_variation(move)
        return str(game.mainline_moves()) # PGN move list
    
    def to_positions(self):
        p1, p2 = [], []
        for sq in chess.SQUARES:
            piece = self._board.piece_at(sq)
            if piece:
                name = chess.square_name(sq)
                if piece.color == chess.WHITE:
                    p1.append(name)
                else:
                    p2.append(name)
        return {"p1": p1, "p2": p2}