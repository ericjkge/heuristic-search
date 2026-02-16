"""Single-code agent: Generator + LLM code execution feedback loop."""

import re
import chess


class SingleCodeAgent:
    """
    Single agent that uses LLM-generated code to evaluate moves.

    Flow:
    1. Verifier writes generic test code for position (once)
    2. Generator proposes move
    3. Run stored code with move substituted
    4. Verifier interprets output, provides feedback + score
    5. Generator proposes another move (repeat 3-4)
    """

    def __init__(self, llm, prompts, game_config, num_iterations=2):
        self.llm = llm()
        self.prompts = prompts
        self.extract_move = game_config["extract_move"]
        self.format_state = game_config["format_state"]
        self.san_to_uci = game_config.get("san_to_uci")
        self.num_iterations = num_iterations
        self.last_stats = {}
        self.last_logs = []

    def generate_test_code(self):
        """Generate reusable test code. Returns (code_template, error, logs)."""
        logs = {}

        prompt = self.prompts.verifier_code_prompt
        logs["prompt"] = prompt

        response = self.llm.generate([{"role": "user", "content": prompt}])
        logs["response"] = response

        # Extract code block
        code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)

        if not code_match:
            logs["error"] = "No code block found"
            return None, "No code block found", logs

        code = code_match.group(1)
        logs["code"] = code
        return code, None, logs

    def execute_code(self, code_template, fen, move_san):
        """Execute code with FEN and move substituted. Returns (output, error)."""
        import io
        import sys

        code = code_template.replace("__FEN__", fen).replace("__SAN__", move_san)

        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        try:
            exec(code)
            output = captured.getvalue()
            return output, None
        except Exception as e:
            return None, str(e)
        finally:
            sys.stdout = old_stdout

    def interpret_output(self, fen, player, move_san, code_output):
        """Interpret code output and return feedback + score. Returns (feedback, score, logs)."""
        logs = {}

        prompt = self.prompts.verifier_interpret_prompt.format(
            fen=fen,
            player=player,
            move=move_san,
            code_output=code_output
        )
        logs["prompt"] = prompt

        response = self.llm.generate([{"role": "user", "content": prompt}])
        logs["response"] = response

        # Extract feedback and score
        feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=SCORE:|$)', response, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else response.strip()

        score_match = re.search(r'\*?\*?SCORE:?\*?\*?:?\s*([\d.]+)', response)
        score = float(score_match.group(1)) if score_match else None

        return feedback, score, logs

    def choose_move(self, board, player):
        """Choose move using generator + code verifier feedback loop."""
        import time
        start_time = time.time()

        state = self.format_state(board, player)
        fen = state.get("fen")
        player_str = state.get("player")
        legal_moves = board.legal_moves()
        legal_uci = board.legal_moves_uci()

        attempts = []
        conversation = []
        best_move_uci = None
        best_score = -1
        self.last_logs = []

        # Step 1: Generate test code once
        code_template, code_error, code_logs = self.generate_test_code()
        self.last_logs.append({"phase": "code_generation", **code_logs})

        if code_error:
            self.last_stats = {"error": code_error, "elapsed": time.time() - start_time}
            return None

        # Step 2-N: Generator proposes moves, code evaluates
        for iteration in range(self.num_iterations):
            iter_log = {"iteration": iteration}

            # Generator prompt
            if iteration == 0:
                prompt = self.prompts.initial_prompt.format(**state)
            else:
                prev = attempts[-1]
                prompt = self.prompts.feedback_prompt.format(
                    previous_move=prev["move"],
                    verifier_feedback=prev.get("feedback", "No feedback"),
                    legal_moves=", ".join(legal_moves)
                )

            iter_log["generator_prompt"] = prompt
            conversation.append({"role": "user", "content": prompt})
            response = self.llm.generate(conversation)
            conversation.append({"role": "model", "content": response})
            iter_log["generator_response"] = response

            move_san = self.extract_move(response, fen=fen)
            iter_log["extracted_move"] = move_san

            if move_san is None:
                attempts.append({"move": None, "iteration": iteration})
                self.last_logs.append(iter_log)
                continue

            move_uci = self.san_to_uci(move_san, fen) if self.san_to_uci else move_san

            if move_uci not in legal_uci:
                attempts.append({"move": move_san, "iteration": iteration, "error": "illegal"})
                iter_log["error"] = "illegal move"
                self.last_logs.append(iter_log)
                continue

            # Run stored code with this move
            output, exec_error = self.execute_code(code_template, fen, move_san)
            if exec_error:
                code_output = f"Error: {exec_error}"
            else:
                code_output = output if output else "(no output)"
            iter_log["code_output"] = code_output

            # Interpret output
            feedback, score, interpret_logs = self.interpret_output(fen, player_str, move_san, code_output)
            iter_log["interpret"] = interpret_logs
            iter_log["feedback"] = feedback
            iter_log["score"] = score

            attempts.append({
                "move": move_san,
                "move_uci": move_uci,
                "iteration": iteration,
                "feedback": feedback,
                "score": score
            })

            # Track best move by score
            if score is not None and score > best_score:
                best_score = score
                best_move_uci = move_uci
            elif best_move_uci is None:
                best_move_uci = move_uci

            self.last_logs.append(iter_log)

        elapsed = time.time() - start_time
        self.last_stats = {
            "elapsed": elapsed,
            "iterations": len(attempts),
            "best_score": best_score if best_score >= 0 else None,
            "attempts": attempts
        }

        return best_move_uci

    def close(self):
        """No resources to clean up."""
        pass
