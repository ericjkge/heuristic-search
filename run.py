import sys
import argparse
sys.path.insert(0, '.')

from utils.logging import setup_logging

# Game registry
GAMES = {
    "chess": {
        "name": "Chess",
        "board": "games.chess.board.ChessBoard",
        "engine": "games.chess.engine.engine.ChessEngine",
        "config": "games.chess.config",
        "prompts": "games.chess.prompts.single",
        "size": None,
    },
    "go": {
        "name": "Go (9x9)",
        "board": "games.go.board.GoBoard",
        "engine": "games.go.engine.engine.GoEngine",
        "config": "games.go.config",
        "prompts": "games.go.prompts.single",
        "size": 9,
    },
    "gomoku": {
        "name": "Gomoku (9x9)",
        "board": "games.gomoku.board.GomokuBoard",
        "engine": "games.gomoku.engine.engine.GomokuEngine",
        "config": "games.gomoku.config",
        "prompts": "games.gomoku.prompts.single",
        "size": 9,
    },
}


def import_attr(path):
    module_path, attr_name = path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[attr_name])
    return getattr(module, attr_name)


def load_game(game_key):
    g = GAMES[game_key]
    Board = import_attr(g["board"])
    Engine = import_attr(g["engine"])
    config = __import__(g["config"], fromlist=["config"]).config
    prompts = __import__(g["prompts"], fromlist=["action_prompt"])
    
    size = g["size"]
    board = Board(size) if size else Board()
    engine = Engine(size) if size else Engine()
    
    return board, engine, config, prompts, g["name"]


def play_human(game_key):
    board, engine, _, _, name = load_game(game_key)
    
    print(f"=== {name}: Human vs Engine ===\n")
    
    while board.winner() is None:
        print(board.to_ascii())
        print(f"Moves: {board.to_moves()}\n")
        
        if board.turn() == 1:
            move = input("Your move: ").strip()
            if game_key != "chess":
                move = move.upper()
            if move in ('QUIT', 'quit'):
                break
            if move not in board.legal_moves() and move != 'PASS':
                print(f"Illegal. Legal: {board.legal_moves()[:10]}...")
                continue
            board.push(move)
        else:
            print("Engine thinking...")
            move = engine.get_move(board)
            print(f"Engine plays: {move}")
            board.push(move)
    
    print(f"\nGame over! Winner: {board.winner()}")
    engine.close()


def play_llm(game_key):
    from agents.single import SingleAgent
    from utils.llm import GeminiLLM
    
    setup_logging(log_dir=f"logs/{game_key}/single_engine")
    
    board, engine, config, prompts, name = load_game(game_key)
    agent = SingleAgent(GeminiLLM, prompts, config)
    
    print(f"=== {name}: Single LLM vs Engine ===\n")
    
    while board.winner() is None:
        print(board.to_ascii())
        print(f"Moves: {board.to_moves()}\n")
        
        if board.turn() == 1:
            print("LLM thinking...")
            move = agent.choose_move(board, 1, max_attempts=3)
            
            if move:
                print(f"LLM plays: {move}")
                board.push(move)
            else:
                print("LLM failed after 3 retries. Ending game.")
                break
        else:
            print("Engine thinking...")
            move = engine.get_move(board)
            print(f"Engine plays: {move}")
            board.push(move)
    
    print(f"\nGame over! Winner: {board.winner()}")
    engine.close()


def play_multi_gomoku(strategy="feature_judge"):
    """Multi-agent debate vs engine - Gomoku only."""
    from agents.multi import MultiAgent
    from utils.llm import GeminiLLM
    from games.gomoku.board import GomokuBoard
    from games.gomoku.engine.engine import GomokuEngine
    from games.gomoku.config import config
    from games.gomoku.prompts import multi as prompts
    
    setup_logging(log_dir=f"logs/gomoku/{strategy}_engine")
    
    board = GomokuBoard(9)
    engine = GomokuEngine(9)
    agent = MultiAgent(GeminiLLM, prompts, config, strategy=strategy)
    
    print(f"=== Gomoku (9x9): Multi-Agent ({strategy}) vs Engine ===\n")
    
    while board.winner() is None:
        print(board.to_ascii())
        print(f"Moves: {board.to_moves()}\n")
        
        if board.turn() == 1:
            print("Multi-agent debate...")
            move = agent.choose_move(board, 1)
            
            if move:
                print(f"Multi-agent plays: {move}")
                board.push(move)
            else:
                print("Multi-agent failed. Ending game.")
                break
        else:
            print("Engine thinking...")
            move = engine.get_move(board)
            print(f"Engine plays: {move}")
            board.push(move)
    
    print(f"\nGame over! Winner: {board.winner()}")
    engine.close()


def play_single_vs_multi_gomoku(game_id=None, strategy="feature_judge"):
    """Single LLM vs Multi-agent debate - Gomoku only."""
    from agents.single import SingleAgent
    from agents.multi import MultiAgent
    from utils.llm import GeminiLLM
    from games.gomoku.board import GomokuBoard
    from games.gomoku.config import config
    from games.gomoku.prompts import single as single_prompts
    from games.gomoku.prompts import multi as multi_prompts
    
    log_dir = f"logs/gomoku/single_{strategy}/game_{game_id}" if game_id else f"logs/gomoku/single_{strategy}"
    setup_logging(log_dir=log_dir)
    
    board = GomokuBoard(9)
    single_agent = SingleAgent(GeminiLLM, single_prompts, config)
    multi_agent = MultiAgent(GeminiLLM, multi_prompts, config, strategy=strategy)
    
    print(f"=== Gomoku (9x9): Single LLM (Black) vs Multi-Agent/{strategy} (White) [Game {game_id or 'N/A'}] ===\n")
    
    while board.winner() is None:
        print(board.to_ascii())
        print(f"Moves: {board.to_moves()}\n")
        
        if board.turn() == 1:
            print("Single LLM thinking...")
            move = single_agent.choose_move(board, 1, max_attempts=3)
            
            if move:
                print(f"Single LLM plays: {move}")
                board.push(move)
            else:
                print("Single LLM failed. Ending game.")
                break
        else:
            print("Multi-agent debate...")
            move = multi_agent.choose_move(board, 2)
            
            if move:
                print(f"Multi-agent plays: {move}")
                board.push(move)
            else:
                print("Multi-agent failed. Ending game.")
                break
    
    winner = board.winner()
    if winner == 1:
        print(f"\nGame {game_id or 'N/A'} over! Winner: Single LLM (Black)")
    elif winner == 2:
        print(f"\nGame {game_id or 'N/A'} over! Winner: Multi-Agent (White)")
    else:
        print(f"\nGame {game_id or 'N/A'} over! Result: {'Draw' if winner == 0 else 'Incomplete'}")


def play_mcts_chess(game_id=None):
    """MCTS LLM vs Stockfish engine - Chess only."""
    from agents.mcts import MCTSAgent
    from search.mcts import NUM_SIMULATIONS
    from utils.llm import GeminiLLM
    from games.chess.board import ChessBoard
    from games.chess.engine.engine import ChessEngine
    
    log_dir = f"logs/chess/mcts_engine/game_{game_id}" if game_id else "logs/chess/mcts_engine"
    setup_logging(log_dir=log_dir)
    
    board = ChessBoard()
    engine = ChessEngine()
    agent = MCTSAgent(GeminiLLM)
    
    print(f"=== Chess: MCTS LLM (White) vs Stockfish (Black) ===")
    print(f"    Simulations per move: {NUM_SIMULATIONS}")
    print()
    
    move_count = 0
    while board.winner() is None:
        print(board.to_ascii())
        print(f"Moves: {board.to_moves()}\n")
        
        if board.turn() == 1:  # White (MCTS)
            print(f"MCTS thinking ({NUM_SIMULATIONS} simulations)...")
            move = agent.choose_move(board, 1)
            
            if move:
                print(f"MCTS plays: {move}")
                board.push(move)
                move_count += 1
            else:
                print("MCTS failed. Ending game.")
                break
        else:  # Black (Stockfish)
            print("Stockfish thinking...")
            move = engine.get_move(board)
            print(f"Stockfish plays: {move}")
            board.push(move)
            move_count += 1
        
        if move_count > 200:
            print("\nMove limit reached (200). Ending game.")
            break
    
    winner = board.winner()
    
    print(f"\nGame over after {move_count} moves!")
    if winner == 1:
        print("Winner: MCTS LLM (White)")
    elif winner == 2:
        print("Winner: Stockfish (Black)")
    elif winner == 0:
        print("Result: Draw")
    else:
        print("Result: Incomplete")
    
    print(f"Policy cache size: {len(agent.mcts._policy_cache)} positions")
    engine.close()


def main():
    from agents.multi import STRATEGIES
    
    parser = argparse.ArgumentParser(
        description="Run game experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode single_vs_multi --strategy feature_judge
  python run.py --mode single_vs_multi --strategy adversarial_judge
  python run.py --mode mcts_vs_engine
  python run.py --list-strategies
        """
    )
    parser.add_argument("--mode", type=str, help="Mode: single_vs_multi, multi_vs_engine, llm_vs_engine, mcts_vs_engine")
    parser.add_argument("--game", type=str, help="Game: chess, go, gomoku")
    parser.add_argument("--game-id", type=str, help="Game ID for logging (used in parallel runs)")
    parser.add_argument("--strategy", type=str, default="feature_judge", 
                        choices=STRATEGIES, help="Multi-agent strategy")
    parser.add_argument("--list-strategies", action="store_true", help="List all available strategies")
    args = parser.parse_args()
    
    if args.list_strategies:
        print("Available strategies:")
        for s in STRATEGIES:
            print(f"  - {s}")
        return
    
    # CLI mode
    if args.mode:
        if args.mode == "single_vs_multi":
            play_single_vs_multi_gomoku(game_id=args.game_id, strategy=args.strategy)
        elif args.mode == "multi_vs_engine":
            play_multi_gomoku(strategy=args.strategy)
        elif args.mode == "llm_vs_engine" and args.game:
            play_llm(args.game)
        elif args.mode == "mcts_vs_engine":
            play_mcts_chess(game_id=args.game_id)
        else:
            print(f"Invalid mode: {args.mode}")
        return
    
    # Interactive mode
    print("Select mode:")
    print("1. Human vs Engine")
    print("2. Single LLM vs Engine")
    print("3. Multi-Agent vs Engine (Gomoku only)")
    print("4. Single LLM vs Multi-Agent (Gomoku only)")
    print("5. MCTS LLM vs Engine (Chess only)")
    mode = input("\nMode (1/2/3/4/5): ").strip()
    
    if mode == "3":
        play_multi_gomoku()
        return
    
    if mode == "4":
        play_single_vs_multi_gomoku()
        return
    
    if mode == "5":
        play_mcts_chess()
        return
    
    print("\nSelect game:")
    print("1. Chess")
    print("2. Go")
    print("3. Gomoku")
    game_choice = input("\nGame (1/2/3): ").strip()
    
    game_map = {"1": "chess", "2": "go", "3": "gomoku"}
    game_key = game_map.get(game_choice)
    
    if not game_key:
        print("Invalid game choice")
        return
    
    if mode == "1":
        play_human(game_key)
    elif mode == "2":
        play_llm(game_key)
    else:
        print("Invalid mode choice")


if __name__ == "__main__":
    main()
