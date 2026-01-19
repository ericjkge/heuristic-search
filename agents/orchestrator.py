import time


class GeneratorVerifierOrchestrator:
    """
    Orchestrates the generator-verifier loop.

    Flow:
    1. Each generator proposes a move
    2. All verifiers score each move
    3. Feedback is sent back to generators
    4. Repeat for N iterations
    5. Select highest-scoring move
    """

    def __init__(self, generators, verifiers, num_iterations=3):
        """
        Args:
            generators: List of GeneratorAgent instances
            verifiers: Dict of {name: VerifierAgent} instances
            num_iterations: Number of refinement iterations
        """
        self.generators = generators
        self.verifiers = verifiers
        self.num_iterations = num_iterations
        self.last_stats = {}  # Stats from last choose_move call

    def compute_score(self, evaluations):
        """
        Compute final score using compositional formula.

        Score = immediate_loss * illegal_move * (aggressive + defensive + shape) / 3
        """
        hard_gate = evaluations["immediate_loss"]["score"] * evaluations["illegal_move"]["score"]
        soft_score = (
            evaluations["aggressive"]["score"]
            + evaluations["defensive"]["score"]
            + evaluations["shape"]["score"]
        ) / 3
        return hard_gate * soft_score

    def evaluate_move(self, board, player, move):
        """Run all verifiers on a move and compute final score."""
        evaluations = {}
        for name, verifier in self.verifiers.items():
            evaluations[name] = verifier.evaluate(board, player, move)
        evaluations["final_score"] = self.compute_score(evaluations)
        evaluations["move"] = move
        return evaluations

    def choose_move(self, board, player):
        """
        Run the full generator-verifier loop and return the best move.

        Returns:
            tuple: (best_move, all_candidates) where all_candidates is a list of
                   dicts with generator, iteration, move, evaluations, response
        """
        start_time = time.time()

        # Reset all generators
        for gen in self.generators:
            gen.reset()

        all_candidates = []
        # Track current feedback for each generator
        generator_feedback = {id(gen): None for gen in self.generators}

        for iteration in range(self.num_iterations):
            for gen in self.generators:
                # Generate move (with feedback if not first iteration)
                feedback = generator_feedback[id(gen)]
                move, response = gen.propose_move(board, player, feedback)

                if move is None:
                    continue

                # Evaluate the move
                evaluations = self.evaluate_move(board, player, move)

                # Store candidate
                all_candidates.append({
                    "generator": gen.name,
                    "iteration": iteration + 1,
                    "move": move,
                    "evaluations": evaluations,
                    "response": response
                })

                # Update feedback for next iteration
                generator_feedback[id(gen)] = evaluations

        # Select best move
        if not all_candidates:
            self.last_stats = {"elapsed": time.time() - start_time, "candidates": 0}
            return None, []

        best = max(all_candidates, key=lambda c: c["evaluations"]["final_score"])

        # Track stats
        self.last_stats = {
            "elapsed": time.time() - start_time,
            "candidates": len(all_candidates),
            "best_score": best["evaluations"]["final_score"]
        }

        return best["move"], all_candidates
