"""Multi-v2 agent: generate-verify-refine loop with multi-turn conversation."""

import re
import time

try:
    from games.chess.config import san_to_uci
except ImportError:
    san_to_uci = None


VERIFIER_ASPECTS = ["blunder", "material", "king_safety", "pawn_structure", "piece_activity"]


class MultiV2Agent:
    """
    Multi-turn refinement with binary verifiers.

    Per turn:
        1. Generate move (fresh prompt)
        2. 5 verifiers score it (binary 0/1 with reasoning)
        3. Feed scores back to generator (multi-turn conversation)
        4. Repeat for N iterations
        5. Pick move with highest total score (tiebreak: last generated)
    """

    def __init__(self, llm, prompts, game_config, num_iterations=3):
        self.llm = llm()
        self.prompts = prompts
        self.extract_move = game_config["extract_move"]
        self.format_state = game_config["format_state"]
        self.san_to_uci_fn = game_config.get("san_to_uci")
        self.num_iterations = num_iterations
        self.last_stats = {}

    def _score_move(self, board, player, move_san):
        """Run all binary verifiers. Returns (scores_dict, total, feedback_str)."""
        state = self.format_state(board, player)
        state["move"] = move_san

        scores = {}
        total = 0
        feedback_lines = []

        for aspect in VERIFIER_ASPECTS:
            prompt = getattr(self.prompts, f"{aspect}_prompt").format(**state)
            response = self.llm.generate([{"role": "user", "content": prompt}])

            # Parse score
            score_match = re.search(r'SCORE:\s*([01])', response)
            score = int(score_match.group(1)) if score_match else 0
            scores[aspect] = score
            total += score

            # Parse reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=SCORE:|$)', response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            label = aspect.replace("_", " ").title()
            feedback_lines.append(f"- {label}: {score} - {reasoning}")

        feedback_str = "\n".join(feedback_lines)
        return scores, total, feedback_str

    def choose_move(self, board, player):
        """Choose move using generate-verify-refine loop."""
        start_time = time.time()

        state = self.format_state(board, player)
        fen = state.get("fen")
        legal_moves = board.legal_moves()
        legal_uci = board.legal_moves_uci()

        conversation = []
        attempts = []
        best_move_uci = None
        best_score = -1

        for iteration in range(self.num_iterations):
            # Generate move
            if iteration == 0 or "feedback_str" not in attempts[-1]:
                prompt = self.prompts.initial_prompt.format(**state)
            else:
                prev = attempts[-1]
                prompt = self.prompts.feedback_prompt.format(
                    previous_move=prev["move"],
                    verifier_feedback=prev["feedback_str"],
                    legal_moves=", ".join(legal_moves)
                )

            conversation.append({"role": "user", "content": prompt})
            response = self.llm.generate(conversation)
            conversation.append({"role": "model", "content": response})

            move_san = self.extract_move(response, fen=fen)

            if move_san is None:
                attempts.append({"move": None, "iteration": iteration, "total_score": 0})
                continue

            move_uci = self.san_to_uci_fn(move_san, fen) if self.san_to_uci_fn else move_san
            if move_uci not in legal_uci:
                attempts.append({"move": move_san, "iteration": iteration, "total_score": 0, "error": "illegal"})
                continue

            # Score with verifiers
            scores, total, feedback_str = self._score_move(board, player, move_san)

            attempts.append({
                "move": move_san,
                "move_uci": move_uci,
                "iteration": iteration,
                "scores": scores,
                "total_score": total,
                "feedback_str": feedback_str
            })

            # Tiebreak: last generated (use >= so later moves win ties)
            if total >= best_score:
                best_score = total
                best_move_uci = move_uci

        elapsed = time.time() - start_time
        self.last_stats = {
            "elapsed": elapsed,
            "attempts": attempts,
            "best_score": best_score if best_score >= 0 else None
        }

        return best_move_uci

    def close(self):
        pass
