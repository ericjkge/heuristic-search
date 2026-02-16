"""Run head-to-head chess matches between agents."""

import json
import sys
import os
from datetime import datetime
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.chess.board import ChessBoard
from games.chess.config import config as chess_config
from utils.llm import GeminiLLM

# Agent imports
from agents.single import SingleAgent
from agents.single_engine import SingleEngineAgent
from agents.single_engine_2 import SingleEngine2Agent
from agents.single_code import SingleCodeAgent
from agents.multi import MultiAgent
from agents.multi_v2 import MultiV2Agent
from games.chess.prompts import single as single_prompts
from games.chess.prompts import single_code as single_code_prompts
from games.chess.prompts import single_engine as single_engine_prompts
from games.chess.prompts import single_engine_2 as single_engine_2_prompts
from games.chess.prompts import multi as multi_prompts
from games.chess.prompts import multi_v2 as multi_v2_prompts
from utils.logging import setup_logging

STOCKFISH_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                              'games', 'chess', 'engine', 'stockfish-macos-m1-apple-silicon')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')


def create_agent(agent_type):
    """Create an agent by type name."""
    if agent_type == "single":
        return SingleAgent(GeminiLLM, single_prompts, chess_config)
    elif agent_type == "single_code":
        return SingleCodeAgent(
            GeminiLLM, single_code_prompts, chess_config,
            num_iterations=2
        )
    elif agent_type == "single_engine":
        return SingleEngineAgent(
            GeminiLLM, single_engine_prompts, chess_config,
            engine_path=STOCKFISH_PATH, engine_depth=12, num_iterations=2
        )
    elif agent_type == "single_engine_2":
        return SingleEngine2Agent(
            GeminiLLM, single_engine_2_prompts, chess_config,
            engine_path=STOCKFISH_PATH, engine_depth=12, num_iterations=3
        )
    elif agent_type == "multi":
        return MultiAgent(
            GeminiLLM, multi_prompts, chess_config,
            num_candidates=3
        )
    elif agent_type == "multi_v2":
        return MultiV2Agent(
            GeminiLLM, multi_v2_prompts, chess_config,
            num_iterations=3
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def play_game(white_agent, black_agent, max_moves=200):
    """Play a single game. Returns (result, pgn, move_count)."""
    board = ChessBoard()
    moves = []

    for move_num in range(max_moves):
        player = board.turn()
        agent = white_agent if player == 1 else black_agent

        move_uci = agent.choose_move(board, player)

        if move_uci is None:
            # Agent failed to produce move
            result = "0-1" if player == 1 else "1-0"
            return result, " ".join(moves), move_num, f"{player} failed to move"

        # Convert UCI to SAN for PGN
        temp_board = chess.Board(board._board.fen())
        move_san = temp_board.san(chess.Move.from_uci(move_uci))

        # Record move (move_num is 0-indexed total moves)
        move_pair = move_num // 2 + 1
        if player == 1:
            moves.append(f"{move_pair}. {move_san}")
        else:
            if moves:
                moves[-1] += f" {move_san}"
            else:
                moves.append(f"{move_pair}... {move_san}")

        # Apply move
        board.push(move_uci)

        # Check game over
        if board._board.is_game_over():
            if board._board.is_checkmate():
                result = "1-0" if player == 1 else "0-1"
                reason = "checkmate"
            elif board._board.is_stalemate():
                result = "1/2-1/2"
                reason = "stalemate"
            elif board._board.is_insufficient_material():
                result = "1/2-1/2"
                reason = "insufficient material"
            elif board._board.is_fifty_moves():
                result = "1/2-1/2"
                reason = "50-move rule"
            elif board._board.is_repetition():
                result = "1/2-1/2"
                reason = "repetition"
            else:
                result = "1/2-1/2"
                reason = "draw"
            return result, " ".join(moves), move_num + 1, reason

    return "1/2-1/2", " ".join(moves), max_moves, "max moves reached"


def run_match(white_type, black_type, num_games=1, enable_logging=False, game_id=None):
    """Run a match between two agent types."""
    if enable_logging:
        setup_logging(LOG_DIR, suffix=game_id)
        print(f"Logging enabled to {LOG_DIR}")

    print(f"=== {white_type} (White) vs {black_type} (Black) ===")
    print(f"Playing {num_games} game(s)\n")

    results = {"white_wins": 0, "black_wins": 0, "draws": 0}
    games = []

    for game_num in range(num_games):
        print(f"Game {game_num + 1}/{num_games}...")

        white_agent = create_agent(white_type)
        black_agent = create_agent(black_type)

        result, pgn, moves, reason = play_game(white_agent, black_agent)

        # Cleanup
        if hasattr(white_agent, 'close'):
            white_agent.close()
        if hasattr(black_agent, 'close'):
            black_agent.close()

        if result == "1-0":
            results["white_wins"] += 1
            winner = white_type
        elif result == "0-1":
            results["black_wins"] += 1
            winner = black_type
        else:
            results["draws"] += 1
            winner = "draw"

        print(f"  Result: {result} ({reason}) in {moves} moves")

        games.append({
            "game": game_num + 1,
            "white": white_type,
            "black": black_type,
            "result": result,
            "reason": reason,
            "moves": moves,
            "pgn": pgn
        })

    print(f"\n=== MATCH RESULTS ===")
    print(f"{white_type} (White) wins: {results['white_wins']}")
    print(f"{black_type} (Black) wins: {results['black_wins']}")
    print(f"Draws: {results['draws']}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{white_type}_vs_{black_type}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    output = {
        "match": {
            "white": white_type,
            "black": black_type,
            "num_games": num_games,
            "timestamp": timestamp
        },
        "results": results,
        "games": games
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {filepath}")
    return results


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    white = args[0] if len(args) > 0 else "single"
    black = args[1] if len(args) > 1 else "single"
    num = int(args[2]) if len(args) > 2 else 1
    enable_logging = "--log" in sys.argv
    game_id = next((a.split("=")[1] for a in sys.argv if a.startswith("--game-id=")), None)

    run_match(white, black, num, enable_logging, game_id)
