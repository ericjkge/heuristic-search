import sys
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
    from utils.llm import QwenLLM
    
    setup_logging(log_dir=f"logs/{game_key}")
    
    board, engine, config, prompts, name = load_game(game_key)
    agent = SingleAgent(QwenLLM, prompts, config)
    
    print(f"=== {name}: LLM vs Engine ===\n")
    
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


def play_multi_gomoku():
    """Multi-agent debate vs engine - Gomoku only."""
    from agents.multi import MultiAgent
    from utils.llm import QwenLLM
    from games.gomoku.board import GomokuBoard
    from games.gomoku.engine.engine import GomokuEngine
    from games.gomoku.config import config
    from games.gomoku.prompts import multi as prompts
    
    setup_logging(log_dir="logs/gomoku_multi")
    
    board = GomokuBoard(9)
    engine = GomokuEngine(9)
    agent = MultiAgent(QwenLLM, prompts, prompts.PERSPECTIVES, config)
    
    print("=== Gomoku (9x9): Multi-Agent vs Engine ===\n")
    
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


def main():
    print("Select mode:")
    print("1. Human vs Engine")
    print("2. Single LLM vs Engine")
    print("3. Multi-Agent vs Engine (Gomoku only)")
    mode = input("\nMode (1/2/3): ").strip()
    
    if mode == "3":
        play_multi_gomoku()
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
