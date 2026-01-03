import subprocess
from pathlib import Path

class GoEngine:    
    COLS = "ABCDEFGHJKLMNOPQRST"  # GTP uses uppercase, no 'I'

    def __init__(self, size=9):
        self.size = size
        self.p = subprocess.Popen(
            ["./katago", "gtp", "-config", "./gtp_custom.cfg",
             "-model", "./kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=Path(__file__).parent,
        )
        self._cmd(f"boardsize {size}")
        self._cmd("clear_board")
        self._cmd("komi 6.5")
    
    def _cmd(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        lines = []
        while True:
            line = self.p.stdout.readline()
            if line == "\n":
                break
            lines.append(line.rstrip())
        resp = "\n".join(lines)
        return resp[2:] if resp.startswith("= ") else resp[1:]

    def get_move(self, board):
        # Sync engine with board history
        self._cmd("clear_board")
        history = board._state.history()
        for i, action in enumerate(history):
            color = "B" if i % 2 == 0 else "W" # O-indexed players in OpenSpiel
            if action == board.size * board.size:
                self._cmd(f"play {color} pass")
            else:
                row, col = divmod(action, board.size)
                vertex = f"{self.COLS[col]}{row + 1}"
                self._cmd(f"play {color} {vertex}")
        
        # Generate move
        color = "B" if board.turn() == 1 else "W"
        result = self._cmd(f"genmove {color}").upper()
        
        if result in ("PASS", "RESIGN"):
            return (None, None) # (None, None) for pass
        
        col = self.COLS.index(result[0])
        row = int(result[1:]) - 1
        return (row, col)
    
    def close(self):
        try:
            self._cmd("quit")
        except:
            pass
        self.p.terminate()