"""Multi-agent: generate candidates + binary prompt-based verifiers."""

import re
import time

try:
    from games.chess.config import san_to_uci
except ImportError:
    san_to_uci = None


VERIFIER_ASPECTS = ["blunder", "material", "king_safety", "pawn_structure", "piece_activity"]


class MultiAgent:
    """
    Generate N candidate moves, score each with 5 binary verifiers,
    pick the move with the most 1s (tiebreak: first generated).
    """

    def __init__(self, llm, prompts, game_config, num_candidates=3):
        self.llm = llm()
        self.prompts = prompts
        self.extract_move = game_config["extract_move"]
        self.format_state = game_config["format_state"]
        self.san_to_uci_fn = game_config.get("san_to_uci")
        self.num_candidates = num_candidates
        self.last_stats = {}

    def _generate_candidates(self, board, player):
        """Generate N candidate moves via sequential single-turn LLM calls."""
        state = self.format_state(board, player)
        fen = state.get("fen")
        legal_uci = board.legal_moves_uci()

        candidates = []
        for i in range(self.num_candidates):
            if i == 0:
                prompt = self.prompts.initial_prompt.format(**state)
            else:
                selected = [c["move"] for c in candidates]
                prompt = self.prompts.retry_prompt.format(
                    previous_moves=", ".join(selected),
                    **state
                )

            response = self.llm.generate([{"role": "user", "content": prompt}])
            move_san = self.extract_move(response, fen=fen)

            if move_san is None:
                continue

            move_uci = self.san_to_uci_fn(move_san, fen) if self.san_to_uci_fn else move_san
            if move_uci not in legal_uci:
                continue

            candidates.append({"move": move_san, "move_uci": move_uci})

        return candidates

    def _score_candidate(self, board, player, move_san):
        """Run all binary verifiers on a move. Returns scores dict and total."""
        state = self.format_state(board, player)
        state["move"] = move_san

        scores = {}
        total = 0

        for aspect in VERIFIER_ASPECTS:
            prompt = getattr(self.prompts, f"{aspect}_prompt").format(**state)
            response = self.llm.generate([{"role": "user", "content": prompt}])

            score_match = re.search(r'SCORE:\s*([01])', response)
            score = int(score_match.group(1)) if score_match else 0
            scores[aspect] = score
            total += score

        return scores, total

    def choose_move(self, board, player):
        """Choose move: generate candidates, score with verifiers, pick best."""
        start_time = time.time()

        candidates = self._generate_candidates(board, player)

        if not candidates:
            self.last_stats = {"elapsed": time.time() - start_time, "error": "no valid candidates"}
            return None

        for candidate in candidates:
            scores, total = self._score_candidate(board, player, candidate["move"])
            candidate["scores"] = scores
            candidate["total_score"] = total

        best = max(candidates, key=lambda c: c["total_score"])

        elapsed = time.time() - start_time
        self.last_stats = {
            "elapsed": elapsed,
            "candidates": candidates,
            "best_score": best["total_score"]
        }

        return best["move_uci"]

    def close(self):
        pass
