import os
from datetime import datetime

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def default_log_path(matchup_name, root="logs"):
    path = os.path.join(root, "c4", matchup_name, f"{_timestamp()}.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

class TextLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def clear(self):
        open(self.path, "w").close()

    def log(self, msg=""):
        with open(self.path, "a") as f:
            f.write(msg + "\n")

    def header(self, title, meta=None):
        self.log("=" * 80)
        self.log(title)
        if meta:
            for k, v in meta.items():
                self.log(f"{k}: {v}")
        self.log("=" * 80)
        self.log()

    def turn_header(self, turn, player, board_render):
        self.log("=" * 80)
        self.log(f"TURN {turn} | Player {player}")
        self.log(board_render)
        self.log("=" * 80)


