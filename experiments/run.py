import sys
sys.path.insert(0, '.')

def play_chess():
    from games.chess.board import ChessBoard
    from games.chess.engine.engine import ChessEngine
    
    board = ChessBoard()
    engine = ChessEngine()
    
    print("=== CHESS ===")
    print("You are White. Enter moves in UCI format (e.g., e2e4)")
    print("Type 'quit' to exit\n")
    
    while board.winner() is None:
        print(board.to_ascii())
        print(f"Moves: {board.to_moves()}\n")
        
        if board.turn() == 1:  # Human (White)
            move = input("Your move: ").strip()
            if move == 'quit':
                break
            if move not in board.legal_moves():
                print(f"Illegal move. Legal: {board.legal_moves()[:10]}...")
                continue
            board.push(move)
        else:  # Engine (Black)
            print("Engine thinking...")
            move = engine.get_move(board)
            print(f"Engine plays: {move}")
            board.push(move)
    
    print(f"\nGame over! Winner: {board.winner()}")
    engine.close()


def play_go():
    from games.go.board import GoBoard
    from games.go.engine.engine import GoEngine
    
    board = GoBoard(9)
    engine = GoEngine(9)
    
    print("=== GO (9x9) ===")
    print("You are Black. Enter moves in GTP format (e.g., D4, E5) or 'pass'")
    print("Type 'quit' to exit\n")
    
    while board.winner() is None:
        print(board.to_ascii())
        print(f"Moves: {board.to_moves()}\n")
        
        if board.turn() == 1:  # Human (Black)
            move = input("Your move: ").strip().upper()
            if move == 'QUIT':
                break
            if move != 'PASS' and move not in board.legal_moves():
                print(f"Illegal. Legal moves: {board.legal_moves()[:10]}...")
                continue
            board.push(move)
        else:  # Engine (White)
            print("Engine thinking...")
            move = engine.get_move(board)
            print(f"Engine plays: {move}")
            board.push(move)
    
    print(f"\nGame over! Winner: {board.winner()}")
    engine.close()


def play_gomoku():
    from games.gomoku.board import GomokuBoard
    from games.gomoku.engine.engine import GomokuEngine
    
    board = GomokuBoard(9)
    engine = GomokuEngine(size=9)
    
    print("=== GOMOKU (9x9) ===")
    print("You are Black (X). Enter moves in GTP format (e.g., D4, E5)")
    print("Type 'quit' to exit\n")
    
    while board.winner() is None:
        print(board.to_ascii())
        print(f"Moves: {board.to_moves()}\n")
        
        if board.turn() == 1:  # Human (Black/X)
            move = input("Your move: ").strip().upper()
            if move == 'QUIT':
                break
            if move not in board.legal_moves():
                print(f"Illegal. Legal moves: {board.legal_moves()[:10]}...")
                continue
            board.push(move)
        else:  # Engine (White/O)
            print("Engine thinking...")
            move = engine.get_move(board)
            print(f"Engine plays: {move}")
            board.push(move)
    
    print(f"\nGame over! Winner: {board.winner()}")
    engine.close()


def main():
    print("Select game:")
    print("1. Chess")
    print("2. Go")
    print("3. Gomoku")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == '1':
        play_chess()
    elif choice == '2':
        play_go()
    elif choice == '3':
        play_gomoku()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()