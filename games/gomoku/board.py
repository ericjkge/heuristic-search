class GomokuBoard:
    def __init__(self, size=9):
        self.size = size
        self.board = [["." for _ in range(size)] for _ in range(size)]
        self.current_player = "X"  # X goes first
        self.move_history = []  # List of (row, col) tuples (0-indexed)
    
    def legal_moves(self):
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == ".":
                    moves.append((r, c))
        return moves # (row, col) tuples (0-indexed)
    
    def place(self, row, col):
        if self.board[row][col] != ".":
            raise ValueError(f"Position ({row}, {col}) is already occupied")
        self.board[row][col] = self.current_player
        self.move_history.append((row, col))
        self.current_player = "O" if self.current_player == "X" else "X"
        return True
    
    def check_winner(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, vertical, diagonal, anti-diagonal
        
        for r in range(self.size):
            for c in range(self.size):
                piece = self.board[r][c]
                if piece == ".":
                    continue
                for dr, dc in directions:
                    # Check if 5 in a row starting from (r, c)
                    if self._count_direction(r, c, dr, dc, piece) >= 5:
                        return piece
        
        # Check for draw (no empty spaces)
        if not self.legal_moves():
            return "draw"
        return None
    
    def _count_direction(self, r, c, dr, dc, piece):
        count = 0
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == piece:
            count += 1
            r += dr
            c += dc
        return count
    
    def get_player_positions(self, player):
        positions = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == player:
                    positions.append((r + 1, c + 1))  # 1-indexed for display + prompting
        return positions
    
    def format_positions(self, player):
        positions = self.get_player_positions(player)
        if not positions:
            return "None"
        return "; ".join(f"{r},{c}" for r, c in positions) # Semicolon-separated string (1-indexed) for prompting
    
    def __str__(self):
        lines = []
        header = "   " + " ".join(f"{i+1:2}" for i in range(self.size)) # Column header
        lines.append(header)
        for i, row in enumerate(self.board):
            line = f"{i+1:2} " + "  ".join(row) # Row header + content
            lines.append(line)
        return "\n".join(lines)

