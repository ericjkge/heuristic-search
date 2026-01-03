import pyspiel

class GoBoard:
    def __init__(self, size=9):
        self.size = size
        self._game = pyspiel.load_game("go", {"board_size": size, "komi": 6.5})
        self._state = self._game.new_initial_state()

    def turn(self):
        return self._state.current_player() + 1 # Convert to 1 or 2

    def push(self, row, col):
        if row is None:
            action = self.size * self.size # Pass
        else:
            action = row * self.size + col
        self._state.apply_action(action)
    
    def legal_moves(self):
        moves = []
        for action in self._state.legal_actions():
            if action == self.size * self.size:  # Skip pass
                continue
            row, col = divmod(action, self.size)
            moves.append((row, col))
        return moves # [(row, col) tuples]
    
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
        return str(self._state)
    
    def to_moves(self):
        COLS = "abcdefghjklmnopqrs"  # SGF uses lowercase, no 'i'
        
        sgf = f"(;GM[1]SZ[{self.size}]KM[6.5]"
        history = self._state.history()
        
        for i, action in enumerate(history):
            color = "B" if i % 2 == 0 else "W"
            if action == self.size * self.size:  # Pass
                sgf += f";{color}[]"
            else:
                row, col = divmod(action, self.size)
                sgf += f";{color}[{COLS[col]}{COLS[row]}]"
        
        return sgf + ")" # SGF format
    
    def to_positions(self):
        obs = self._state.observation_string()
        p1, p2 = [], []
        lines = obs.strip().split('\n')
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == 'X':
                    p1.append((r, c))
                elif char == 'O':
                    p2.append((r, c))
        return {"p1": p1, "p2": p2}