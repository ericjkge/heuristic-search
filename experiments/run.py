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
    print("You are Black. Enter moves as 'row,col' (0-indexed) or 'pass'")
    print("Type 'quit' to exit\n")
    
    while board.winner() is None:
        print(board.to_ascii())
        print(f"SGF: {board.to_moves()}\n")
        
        if board.turn() == 1:  # Human (Black)
            move = input("Your move (row,col or pass): ").strip()
            if move == 'quit':
                break
            if move == 'pass':
                board.push(None, None)
            else:
                try:
                    row, col = map(int, move.split(','))
                    if (row, col) not in board.legal_moves():
                        print(f"Illegal. Legal moves: {board.legal_moves()[:10]}...")
                        continue
                    board.push(row, col)
                except:
                    print("Invalid format. Use 'row,col' e.g., '4,4'")
                    continue
        else:  # Engine (White)
            print("Engine thinking...")
            move = engine.get_move(board)
            if move == (None, None):
                print("Engine passes")
                board.push(None, None)
            else:
                print(f"Engine plays: {move}")
                board.push(*move)
    
    print(f"\nGame over! Winner: {board.winner()}")
    engine.close()


def play_gomoku():
    from games.gomoku.board import GomokuBoard
    from games.gomoku.engine.engine import GomokuEngine
    
    board = GomokuBoard(9)
    engine = GomokuEngine()
    
    print("=== GOMOKU (15x15) ===")
    print("You are X (Player 1). Enter moves as 'row,col' (0-indexed)")
    print("Type 'quit' to exit\n")
    
    while board.winner() is None:
        print(board.to_ascii())
        print()
        
        if board.turn() == 1:  # Human (X)
            move = input("Your move (row,col): ").strip()
            if move == 'quit':
                break
            try:
                row, col = map(int, move.split(','))
                if (row, col) not in board.legal_moves():
                    print("Illegal move!")
                    continue
                board.push(row, col)
            except:
                print("Invalid format. Use 'row,col' e.g., '7,7'")
                continue
        else:  # Engine (O)
            print("Engine thinking...")
            move = engine.get_move(board)
            print(f"Engine plays: {move}")
            board.push(*move)
    
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