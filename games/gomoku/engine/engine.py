import subprocess
from pathlib import Path

# Uses Piskvork protocol: (0,0) = top-left, (0,8) = bottom-left
# GTP: A1 = bottom-left, A9 = top-left
class GomokuEngine:
    COLS = "ABCDEFGHJKLMNOP"  # GTP format, no 'I'
    
    def __init__(self, size=9):
        self.size = size
        self.p = subprocess.Popen(
            ["./pbrain-rapfi-macos-apple-silicon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=Path(__file__).parent
        )
        self._cmd(f"START {size}")
        self._cmd("INFO timeout_turn 100")
        self._cmd("INFO rule 0")

    def _cmd(self, c):
        self.p.stdin.write(c + "\n")
        self.p.stdin.flush()

    def _read(self):
        while True:
            line = self.p.stdout.readline().strip()
            if not line.startswith(("OK", "MESSAGE", "DEBUG", "INFO", "ERROR")):
                return line

    def _gtp_to_xy(self, vertex):
        x = self.COLS.index(vertex[0])
        gtp_row = int(vertex[1:])
        y = self.size - gtp_row  # Flip row: GTP row 1 → y=8, row 9 → y=0
        return x, y
    
    def _xy_to_gtp(self, x, y):
        col = self.COLS[x]
        gtp_row = self.size - y  # Flip back: y=0 → row 9, y=8 → row 1
        return f"{col}{gtp_row}"

    def get_move(self, board):
        if board._history:
            last = board._history[-1]
            x, y = self._gtp_to_xy(last)
            self._cmd(f"TURN {x},{y}")
        else:
            self._cmd("BEGIN")
        r = self._read()
        x, y = map(int, r.split(","))
        return self._xy_to_gtp(x, y)  # Return GTP format

    def restart(self):
        self._cmd("RESTART")

    def close(self):
        self._cmd("END")
        self.p.terminate()