import subprocess
from pathlib import Path

# Uses Piskvork protocol
class GomokuEngine:
    def __init__(self, path="./rapfi", size=9):
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

    def get_move(self, board):
        if board._history:
            last = board._history[-1]
            self._cmd(f"TURN {last[0]},{last[1]}")
        else:
            self._cmd("BEGIN")
        r = self._read()
        return tuple(map(int, r.split(",")))

    def restart(self):
        self._cmd("RESTART")

    def close(self):
        self._cmd("END")
        self.p.terminate()