class GoBoard:
    """Pure game state for Go."""
    
    COLS = "abcdefghjklmnopqrst"  # SGF uses lowercase, no 'i'
    
    def __init__(self, size=9):
        self.size = size
        self.move_history = []  # (color, vertex) tuples
        self._player = 1  # 1=Black, -1=White
    
    def current_player(self) -> int:
        return self._player
    
    def make_move(self, vertex: str):
        """Apply move. vertex is 'D4' or 'PASS'."""
        color = "B" if self._player == 1 else "W"
        self.move_history.append((color, vertex.upper()))
        self._player *= -1
    
    def is_terminal(self) -> bool:
        """Two consecutive passes ends game."""
        if len(self.move_history) < 2:
            return False
        return (self.move_history[-1][1] == "PASS" and 
                self.move_history[-2][1] == "PASS")
    
    def to_sgf(self) -> str:
        """SGF format for LLM prompts."""
        sgf = f"(;GM[1]SZ[{self.size}]KM[6.5]"
        for color, vertex in self.move_history:
            if vertex == "PASS":
                sgf += f";{color}[]"
            else:
                col = self.COLS.index(vertex[0])
                row = int(vertex[1:]) - 1
                sgf += f";{color}[{self.COLS[col]}{self.COLS[row]}]"
        return sgf + ")"