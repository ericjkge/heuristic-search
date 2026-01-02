import subprocess
from pathlib import Path

class GoEngine:
    """KataGo GTP wrapper. Syncs with GoBoard for moves."""
    
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
    
    def sync(self, board):
        """Sync engine state with board."""
        self._cmd("clear_board")
        for color, vertex in board.move_history:
            self._cmd(f"play {color} {vertex}")  # GTP engine requires move history, not just current board state
    
    def genmove(self, color) -> str:
        """Get engine's move for color. Returns vertex or PASS/RESIGN."""
        return self._cmd(f"genmove {color}").upper()
    
    def show_board(self) -> str:
        return self._cmd("showboard")
    
    def close(self):
        try:
            self._cmd("quit")
        except:
            pass
        self.p.terminate()