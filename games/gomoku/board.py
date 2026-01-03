class GomokuBoard:
    COLS = "ABCDEFGHJKLMNOP"  # GTP format, no 'I'
    
    def __init__(self, size=9):
        self.size = size
        self._board = [[0 for _ in range(size)] for _ in range(size)] # 0 = empty, 1 = Player 1, 2 = Player 2
        self._turn = 1  # Start with Player 1
        self._history = []  # GTP vertices: ["D4", "E5", ...]
    
    def turn(self):
        return self._turn

    def legal_moves(self):
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self._board[r][c] == 0:
                    gtp_row = self.size - r  # _board[0] = row 9 (top), _board[8] = row 1 (bottom)
                    moves.append(f"{self.COLS[c]}{gtp_row}")
        return moves  # GTP format
    
    def push(self, vertex):
        vertex = vertex.upper()
        col = self.COLS.index(vertex[0])
        gtp_row = int(vertex[1:])
        row = self.size - gtp_row
        if self._board[row][col] != 0:
            raise ValueError(f"Position {vertex} is already occupied")
        self._board[row][col] = self._turn
        self._history.append(vertex)
        self._turn = 3 - self._turn
    
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
                gtp_row = self.size - r
                if self._board[r][c] == 1:
                    p1.append(f"{self.COLS[c]}{gtp_row}")
                elif self._board[r][c] == 2:
                    p2.append(f"{self.COLS[c]}{gtp_row}")
        return {"p1": p1, "p2": p2}  # GTP format: ["D4", "E5", ...]
    
    def to_moves(self):
        moves = []
        for i, vertex in enumerate(self._history):
            color = "B" if i % 2 == 0 else "W"
            moves.append(f"{color} {vertex}")
        return ", ".join(moves)  # GTP format: "B D4, W E5, ..."

    def to_ascii(self):
        symbols = {0: ".", 1: "X", 2: "O"}
        lines = ["   " + "  ".join(self.COLS[i] for i in range(self.size))]
        for r in range(self.size):
            gtp_row = self.size - r
            line = f"{gtp_row:2} " + "  ".join(symbols[cell] for cell in self._board[r])
            lines.append(line)
        return "\n".join(lines) # GTP-style coordinates
