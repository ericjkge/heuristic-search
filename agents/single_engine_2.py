"""Single-engine-2 agent: independent generate+score iterations, pick best."""

import chess
import time

from utils.engine import ChessEngine


class SingleEngine2Agent:
    """
    Each iteration is a fresh single-turn LLM call that sees all
    previously tried moves and their engine scores.
    Returns the move with the best engine delta across all iterations.
    """

    def __init__(self, llm, prompts, game_config, engine_path, num_iterations=3, engine_depth=12):
        self.llm = llm()
        self.prompts = prompts
        self.extract_move = game_config["extract_move"]
        self.format_state = game_config["format_state"]
        self.san_to_uci = game_config.get("san_to_uci")
        self.num_iterations = num_iterations

        self.engine = ChessEngine(engine_path, depth=engine_depth)
        self.last_stats = {}

    def evaluate_move(self, fen, move_san):
        """Evaluate a move and return delta from moving side's perspective."""
        board = chess.Board(fen)
        eval_before = self.engine.evaluate_position_pawns(fen)

        move = board.parse_san(move_san)
        side_moved = board.turn
        board.push(move)

        if board.is_checkmate():
            return 100.0

        eval_after = self.engine.evaluate_position_pawns(board.fen())

        raw_delta = eval_after - eval_before
        if side_moved == chess.WHITE:
            return raw_delta
        else:
            return -raw_delta

    def get_assessment(self, delta):
        """Get text assessment based on delta."""
        if delta < -3:
            return "blunder, loses significant material"
        elif delta < -1:
            return "mistake, loses material"
        elif delta < -0.3:
            return "inaccuracy"
        elif delta < 0.3:
            return "neutral"
        elif delta < 1:
            return "slightly good"
        else:
            return "good move"

    def format_attempts(self, attempts):
        """Format previous attempts for the prompt."""
        lines = []
        for a in attempts:
            if a["delta"] is not None:
                lines.append(f"- {a['move']}: {a['delta']:+.2f} pawns ({self.get_assessment(a['delta'])})")
            elif a.get("error"):
                lines.append(f"- {a['move']}: invalid ({a['error']})")
            else:
                lines.append(f"- (failed to parse move)")
        return "\n".join(lines)

    def choose_move(self, board, player, max_attempts=3):
        """Choose move using independent generate+score iterations."""
        start_time = time.time()

        state = self.format_state(board, player)
        fen = state.get("fen")
        legal_uci = board.legal_moves_uci()

        attempts = []
        best_move = None
        best_move_uci = None
        best_delta = float('-inf')

        for iteration in range(self.num_iterations):
            if iteration == 0:
                prompt = self.prompts.initial_prompt.format(**state)
            else:
                prompt = self.prompts.retry_prompt.format(
                    previous_attempts=self.format_attempts(attempts),
                    **state
                )

            conversation = [{"role": "user", "content": prompt}]
            response = self.llm.generate(conversation)

            move_san = self.extract_move(response, fen=fen)

            if move_san is None:
                attempts.append({"move": None, "delta": None, "iteration": iteration})
                continue

            move_uci = self.san_to_uci(move_san, fen) if self.san_to_uci else move_san

            if move_uci not in legal_uci:
                attempts.append({"move": move_san, "delta": None, "iteration": iteration, "error": "illegal"})
                continue

            try:
                delta = self.evaluate_move(fen, move_san)
            except Exception as e:
                attempts.append({"move": move_san, "delta": None, "iteration": iteration, "error": str(e)})
                continue

            attempts.append({
                "move": move_san,
                "move_uci": move_uci,
                "delta": delta,
                "iteration": iteration
            })

            if delta > best_delta:
                best_delta = delta
                best_move = move_san
                best_move_uci = move_uci

        elapsed = time.time() - start_time
        self.last_stats = {
            "elapsed": elapsed,
            "iterations": len(attempts),
            "best_delta": best_delta if best_delta != float('-inf') else None,
            "attempts": attempts
        }

        return best_move_uci

    def close(self):
        """Close the engine."""
        self.engine.close()
