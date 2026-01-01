import subprocess


class Rapfi:
    def __init__(self, path="./rapfi", size=9):
        self.size = size
        self.p = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
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

    def move(self, xy=None):
        self._cmd("BEGIN" if xy is None else f"TURN {xy[0]},{xy[1]}")
        r = self._read()
        return tuple(map(int, r.split(",")))

    def restart(self):
        self._cmd("RESTART")

    def close(self):
        self._cmd("END")
        self.p.terminate()