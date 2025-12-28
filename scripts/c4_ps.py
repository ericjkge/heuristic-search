from dotenv import load_dotenv

from c4_board import Connect4Board
from log_utils import TextLogger, default_log_path
from models import KimiLLM
from prompts import c4_perspective, c4_single
from agents.multi_agent import MultiAgent
from agents.single_agent import SingleAgent

load_dotenv()


def main():
    log_path = default_log_path("ps")
    logger = TextLogger(log_path)
    logger.clear()

    board = Connect4Board()

    # Multi-agent setup (Perspective-based)
    system_prompts = c4_perspective.PERSPECTIVES
    roles = ["OFFENSIVE", "DEFENSIVE", "POSITIONAL", "CONCLUSION"]
    multi_agent = MultiAgent(
        KimiLLM,
        system_prompts,
        c4_perspective.propose_prompt,
        c4_perspective.debate_prompt,
        c4_perspective.conclusion_prompt,
        num_rounds=1,
        roles=roles,
    )

    # Single agent setup
    single_llm = KimiLLM()
    single_agent = SingleAgent(single_llm, c4_single)

    print("Starting Connect 4. Multi (Perspective) (1) vs Single (2)\n")
    print(board, "\n")
    logger.header("Connect4: Multi (Perspective) (1) vs Single (2)", {"log": log_path})

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
            move = multi_agent.choose_move(board, legal, "1", logger=logger)
            board.drop(move)
            print(f"Multi(P) plays column: {move}")
        else:
            legal = board.legal_moves()
            logger.log()
            logger.log(f"Player 2 | Legal: {' '.join(str(m) for m in legal)}")
            move = single_agent.choose_move(board, legal, "2", logger=logger)
            board.drop(move)
            print(f"Single plays column: {move}")

        print(board, "\n")


if __name__ == "__main__":
    main()
