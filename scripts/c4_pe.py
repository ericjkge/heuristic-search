from dotenv import load_dotenv

from connect4_engine import Connect4Engine
from c4_board import Connect4Board
from log_utils import TextLogger, default_log_path
from models import KimiLLM
from prompts import c4_perspective
from agents.multi_agent import MultiAgent

load_dotenv()


def get_engine_move(engine, board):
    legal = board.legal_moves()
    history_0based = "".join(str(int(c) - 1) for c in board.move_history)

    try:
        result = engine.suggest_move(starter=1, history_str=history_0based)
        if result.is_draw or result.winner in {1, 2}:
            return legal[0]
        if result.column is not None:
            move = result.column + 1
            if move in legal:
                return move
        return legal[0]
    except Exception:
        return legal[0]


def main():
    log_path = default_log_path("pe")
    logger = TextLogger(log_path)
    logger.clear()

    board = Connect4Board()
    engine = Connect4Engine(depth=8)

    system_prompts = c4_perspective.PERSPECTIVES
    roles = ["OFFENSIVE", "DEFENSIVE", "POSITIONAL", "CONCLUSION"]

    agent = MultiAgent(
        KimiLLM,
        system_prompts,
        c4_perspective.propose_prompt,
        c4_perspective.debate_prompt,
        c4_perspective.conclusion_prompt,
        num_rounds=1,
        roles=roles,
    )

    print("Starting Connect 4. Multi (Perspective) (1) vs Engine (2)\n")
    print(board, "\n")
    logger.header("Connect4: Multi (Perspective) (1) vs Engine (2)", {"log": log_path})

    turn = 0
    while True:
        winner = board.check_winner()
        if winner:
            print(f"Game over: {winner}")
            logger.log()
            logger.log(f"Game over: {winner}")
            break

        if board.current_player == "1":
            turn += 1
            legal = board.legal_moves()
            logger.turn_header(turn, "1", str(board))
            logger.log(f"Legal: {' '.join(str(m) for m in legal)}")
            move = agent.choose_move(board, legal, "1", logger=logger)
            board.drop(move)
            print(f"Multi(P) plays column: {move}")
        else:
            move = get_engine_move(engine, board)
            board.drop(move)
            print(f"Engine plays column: {move}")

        print(board, "\n")


if __name__ == "__main__":
    main()
