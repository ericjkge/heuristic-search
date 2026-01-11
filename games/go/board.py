import pyspiel

class GoBoard:
    COLS = "ABCDEFGHJ"  # GTP format, no 'I'
    
    def __init__(self, size=9):
        self.size = size
        self._game = pyspiel.load_game("go", {"board_size": size, "komi": 6.5})
        self._state = self._game.new_initial_state()

    def turn(self):
        return self._state.current_player() + 1 # Convert to 1 or 2

    
    def push(self, vertex): # Vertex in GTP format
        vertex = vertex.upper()
        if vertex == "PASS":
            action = self.size * self.size
        else:
            col = self.COLS.index(vertex[0])
            row = int(vertex[1:]) - 1
            action = row * self.size + col
        self._state.apply_action(action)
    
    def legal_moves(self):
        moves = []
        for action in self._state.legal_actions():
            if action == self.size * self.size:  # Skip pass
                continue
            row, col = divmod(action, self.size)
            moves.append(f"{self.COLS[col]}{row + 1}")
        return moves  # GTP format: ["D4", "E5", ...]
    
    def winner(self):
        if not self._state.is_terminal():
            return None
        returns = self._state.returns()  # [black_score, white_score]
        if returns[0] > returns[1]:
            return 1
        elif returns[1] > returns[0]:
            return 2
        return 0

    def to_ascii(self):
        return str(self._state) # GTP format
    
    def to_moves(self):
        history = self._state.history()
        moves = []
        for i, action in enumerate(history):
            color = "B" if i % 2 == 0 else "W"
            if action == self.size * self.size:
                moves.append(f"{color} PASS")
            else:
                row, col = divmod(action, self.size)
                moves.append(f"{color} {self.COLS[col]}{row + 1}")
        return ", ".join(moves)  # GTP format: "B D4, W E5, ..."
    
    def to_positions(self):
        obs = self._state.observation_string()
        p1, p2 = [], []
        lines = obs.strip().split('\n')
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == 'X':
                    p1.append(f"{self.COLS[c]}{r + 1}")
                elif char == 'O':
                    p2.append(f"{self.COLS[c]}{r + 1}")
        return {"p1": p1, "p2": p2}