"""
Run Gomoku matches: Orchestrator vs Single LLM
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

from games.gomoku.board import GomokuBoard
from games.gomoku.config import config
from games.gomoku.prompts import generator as gen_prompts
from games.gomoku.prompts import verifier as ver_prompts

from agents import SingleAgent, GeneratorAgent, VerifierAgent, GeneratorVerifierOrchestrator
from utils.llm import GeminiLLM
from utils.logging import setup_logging, get_logger


def create_orchestrator(format_state, extract_move):
    """Create orchestrator with generators and verifiers."""
    llm = GeminiLLM()

    # Create generators
    generators = [
        GeneratorAgent(
            llm=llm,
            initial_prompt=gen_prompts.pattern_initial,
            feedback_prompt=gen_prompts.pattern_feedback,
            extract_move=extract_move,
            format_state=format_state,
            name="pattern"
        ),
        GeneratorAgent(
            llm=llm,
            initial_prompt=gen_prompts.counterfactual_initial,
            feedback_prompt=gen_prompts.counterfactual_feedback,
            extract_move=extract_move,
            format_state=format_state,
            name="counterfactual"
        ),
    ]

    # Create verifiers
    verifiers = {
        "immediate_loss": VerifierAgent(llm, ver_prompts.immediate_loss_prompt, "immediate_loss", format_state),
        "illegal_move": VerifierAgent(llm, ver_prompts.illegal_move_prompt, "illegal_move", format_state),
        "aggressive": VerifierAgent(llm, ver_prompts.aggressive_prompt, "aggressive", format_state),
        "defensive": VerifierAgent(llm, ver_prompts.defensive_prompt, "defensive", format_state),
        "shape": VerifierAgent(llm, ver_prompts.shape_prompt, "shape", format_state),
    }

    return GeneratorVerifierOrchestrator(generators, verifiers, num_iterations=3)


def create_single_agent(prompts, game_config):
    """Create single LLM agent (no CoT)."""
    return SingleAgent(GeminiLLM, prompts, game_config)


def play_game(player1, player2, logger):
    """Play a single game between two players."""
    board = GomokuBoard(size=9)
    players = {1: player1, 2: player2}
    move_log = []

    logger.info("=== GAME START ===")
    logger.info(f"Player 1 (Black): {type(player1).__name__}")
    logger.info(f"Player 2 (White): {type(player2).__name__}")

    while board.winner() is None:
        current_player = board.turn()
        player = players[current_player]

        logger.info(f"\n--- Move {len(board._history) + 1} (Player {current_player}) ---")
        logger.info(f"Board:\n{board.to_ascii()}")

        # Get move
        move_start = time.time()

        if isinstance(player, GeneratorVerifierOrchestrator):
            move, candidates = player.choose_move(board, current_player)
            # Log all candidates
            for c in candidates:
                logger.info(
                    f"Candidate: gen={c['generator']}, iter={c['iteration']}, "
                    f"move={c['move']}, score={c['evaluations']['final_score']:.3f}"
                )
                for name in ["immediate_loss", "illegal_move", "aggressive", "defensive", "shape"]:
                    ev = c["evaluations"][name]
                    logger.info(f"  {name}: {ev['score']} - {ev['reasoning']}")
            # Log stats
            stats = player.last_stats
            logger.info(f"Stats: elapsed={stats.get('elapsed', 0):.2f}s, candidates={stats.get('candidates', 0)}")
        else:
            move = player.choose_move(board, current_player)
            elapsed = time.time() - move_start
            logger.info(f"Stats: elapsed={elapsed:.2f}s")

        if move is None:
            logger.info(f"Player {current_player} failed to produce a valid move")
            # Other player wins by forfeit
            winner = 3 - current_player
            break

        logger.info(f"Selected move: {move}")
        move_log.append({"player": current_player, "move": move})
        board.push(move)

    winner = board.winner()
    logger.info(f"\n=== GAME END ===")
    logger.info(f"Winner: Player {winner}" if winner else "Draw")
    logger.info(f"Final board:\n{board.to_ascii()}")
    logger.info(f"Move history: {board.to_moves()}")

    return winner, move_log


def run_match(num_games=2):
    """Run a match of multiple games, alternating colors."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs/gomoku/orchestrator_vs_single") / timestamp
    setup_logging(log_dir=log_dir)
    logger = get_logger("match")

    # Simple prompt for single agent (no CoT)
    class SimplePrompts:
        action_prompt = """You are player {player} in a 9x9 Gomoku game.

Board: Columns A-J (no I), Rows 1-9
Your stones: {player_positions}
Opponent stones: {opponent_positions}
History: {moves}

Choose the best move. Respond in this format:
ANALYSIS: <1 sentence>
MOVE: <vertex like D4>
"""

    orchestrator = create_orchestrator(config["format_state"], config["extract_move"])
    single = create_single_agent(SimplePrompts(), config)

    results = {"orchestrator": 0, "single": 0, "draw": 0}

    for game_num in range(num_games):
        logger.info(f"\n{'='*50}")
        logger.info(f"GAME {game_num + 1} / {num_games}")
        logger.info(f"{'='*50}")

        # Alternate colors
        if game_num % 2 == 0:
            p1, p2 = orchestrator, single
            p1_name, p2_name = "orchestrator", "single"
        else:
            p1, p2 = single, orchestrator
            p1_name, p2_name = "single", "orchestrator"

        logger.info(f"Black: {p1_name}, White: {p2_name}")

        winner, _ = play_game(p1, p2, logger)

        if winner == 1:
            results[p1_name] += 1
        elif winner == 2:
            results[p2_name] += 1
        else:
            results["draw"] += 1

        logger.info(f"Running score: {results}")

    logger.info(f"\n{'='*50}")
    logger.info(f"FINAL RESULTS: {results}")
    logger.info(f"{'='*50}")

    print(f"Match complete. Results: {results}")
    print(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gomoku matches")
    parser.add_argument("--games", type=int, default=2, help="Number of games to play")
    args = parser.parse_args()

    run_match(num_games=args.games)
