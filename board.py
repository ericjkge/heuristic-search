class Connect4Board:
    def __init__(self):
        self.rows = 6
        self.cols = 7 
        self.board = [[" " for _ in range(self.cols)] for _ in range(self.rows)]
        self.move_history = ""
        self.current_player = "X"

    def legal_moves(self):
        return [c + 1 for c in range(self.cols) if self.board[0][c] == " "]
    
    def drop(self, col):
        col_index = col - 1
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
                if c + 3 < self.cols and all(self.board[r][c+i] == piece for i in range(4)):
                    return piece
                if r + 3 < self.rows and all(self.board[r+i][c] == piece for i in range(4)):
                    return piece
                if r + 3 < self.rows and c + 3 < self.cols and all(self.board[r+i][c+i] == piece for i in range(4)):
                    return piece
                if r - 3 >= 0 and c + 3 < self.cols and all(self.board[r-i][c+i] == piece for i in range(4)):
                    return piece
        if not self.legal_moves():
            return "draw"
        return None

    def __str__(self):
        lines = []
        for row in self.board:
            lines.append("|" + "|".join(row) + "|")
        lines.append("+" + "+".join(["-"] * self.cols) + "+")
        lines.append(" " + " ".join(str(i) for i in range(1, self.cols + 1)) + " ")
        return "\n".join(lines)

