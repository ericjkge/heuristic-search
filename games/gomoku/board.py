class GomokuBoard:
    def __init__(self, size=9):
        self.size = size
        self._board = [[0 for _ in range(size)] for _ in range(size)] # 0 = empty, 1 = Player 1, 2 = Player 2
        self._turn = 1  # Start with Player 1
        self._history = []  # List of (row, col) tuples (0-indexed)
    
    def turn(self):
        return self._turn

    def legal_moves(self):
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self._board[r][c] == 0:
                    moves.append((r, c))
        return moves # (row, col) tuples (0-indexed)
    
    def push(self, row, col):
        if self._board[row][col] != 0:
            raise ValueError(f"Position ({row}, {col}) is already occupied")
        self._board[row][col] = self._turn
        self._history.append((row, col))
        self._turn = 3 - self._turn
        return True
    
    def winner(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, vertical, diagonal, anti-diagonal
        
        for r in range(self.size):
            for c in range(self.size):
                piece = self._board[r][c]
                if piece == 0:
                    continue
                for dr, dc in directions:
                    # Check if 5 in a row starting from (r, c)
                    if self._count_direction(r, c, dr, dc, piece) >= 5:
                        return piece
        
        # Check for draw (no empty spaces)
        if not self.legal_moves():
            return 0
        return None
    
    def _count_direction(self, r, c, dr, dc, player):
        count = 0
        while 0 <= r < self.size and 0 <= c < self.size and self._board[r][c] == player:
            count += 1
            r += dr
            c += dc
        return count
    
    def to_positions(self):
        p1, p2 = [], []
        for r in range(self.size):
            for c in range(self.size):
                if self._board[r][c] == 1:
                    p1.append((r + 1, c + 1))
                elif self._board[r][c] == 2:
                    p2.append((r + 1, c + 1))
        return {"p1": p1, "p2": p2} # [(row, col) tuples] (1-indexed)
    
    def to_moves(self):
        return self._history # TODO: convert to SGF?? or whatever works for Gomoku

    def to_ascii(self):
        symbols = {0: ".", 1: "X", 2: "O"} # Convert to X, O for visualization
        lines = ["   " + " ".join(f"{i+1:2}" for i in range(self.size))]
        for i, row in enumerate(self._board):
            line = f"{i+1:2} " + "  ".join(symbols[cell] for cell in row)
            lines.append(line)
        return "\n".join(lines)
