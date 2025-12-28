class Connect4Board:
    def __init__(self):
        self.rows = 6
        self.cols = 7 
        self.board = [["0" for _ in range(self.cols)] for _ in range(self.rows)]
        self.move_history = ""
        self.current_player = "1"

    def legal_moves(self):
        return [c + 1 for c in range(self.cols) if self.board[0][c] == "0"]
    
    def drop(self, col):
        col_index = col - 1
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col_index] == "0":
                self.board[row][col_index] = self.current_player
                self.move_history += str(col)
                self.current_player = "2" if self.current_player == "1" else "1"
                return True
        return False
    
    def check_winner(self):
        for r in range(self.rows):
            for c in range(self.cols):
                piece = self.board[r][c]
                if piece == "0":
                    continue
                # Vertical
                if c + 3 < self.cols and all(self.board[r][c+i] == piece for i in range(4)):
                    return piece
                # Horizontal
                if r + 3 < self.rows and all(self.board[r+i][c] == piece for i in range(4)):
                    return piece
                # Diagonal
                if r + 3 < self.rows and c + 3 < self.cols and all(self.board[r+i][c+i] == piece for i in range(4)):
                    return piece
                # Anti-diagonal
                if r - 3 >= 0 and c + 3 < self.cols and all(self.board[r-i][c+i] == piece for i in range(4)):
                    return piece
        if not self.legal_moves():
            return "draw"
        return None

    def __str__(self):
        lines = []
        for row in self.board:
            lines.append(" ".join(row))
        return "\n".join(lines)