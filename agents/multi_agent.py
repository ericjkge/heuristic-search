import re
from collections import Counter
import time


class MultiAgent:
    def __init__(
        self,
        llm,
        system_prompts,
        propose_prompt,
        debate_prompt,
        conclusion_prompt,
        num_rounds=1,
        roles=None,
    ):
        self.agents = []
        for sys_prompt, role in zip(system_prompts, roles):
            self.agents.append((llm(), sys_prompt, role))
        self.propose_prompt = propose_prompt
        self.debate_prompt = debate_prompt
        self.conclusion_prompt = conclusion_prompt
        self.num_rounds = num_rounds

    def extract_move(self, response):
        match = re.search(r"MOVE:\s*([1-7])", response)
        return int(match.group(1)) if match else None

    def get_proposals(self, agents, board, legal_moves, color, prev_proposals=None, round_num=0, logger=None):
        proposals = []
        moves_str = " ".join(str(m) for m in legal_moves)
        board_str = str(board)

        total_tokens = 0
        total_time = 0.0
        total_calls = 0

        for i, (llm, sys_prompt, role) in enumerate(agents[:-1]):
            if prev_proposals is None:
                prompt = self.propose_prompt.format(color=color, board=board_str, moves=moves_str, role=role)
            else:
                other = "\n".join(f"Agent {j+1}: {p}" for j, p in enumerate(prev_proposals) if j != i)
                prompt = self.debate_prompt.format(color=color, board=board_str, moves=moves_str, other_proposals=other, role=role)

            start = time.time()
            response, tokens = llm.generate(prompt, system_prompt=sys_prompt)
            elapsed = time.time() - start
            total_tokens += tokens
            total_time += elapsed
            total_calls += 1
            move = self.extract_move(response)
            proposals.append((move, response))

            if logger:
                logger.log(f"{role} R{round_num}:")
                logger.log(response.strip())

        return proposals, (total_tokens, total_time, total_calls)

    def aggregate_moves(self, board, proposals, legal_moves, color, logger=None):
        proposal_text = "\n".join(f"Agent {i+1}:\n{p}" for i, p in enumerate(proposals))

        moves_str = " ".join(str(m) for m in legal_moves)
        prompt = self.conclusion_prompt.format(color=color, board=str(board), moves=moves_str, final_proposals=proposal_text)

        llm = self.agents[-1][0]
        start = time.time()
        response, tokens = llm.generate(prompt, system_prompt=self.conclusion_prompt)
        elapsed = time.time() - start
        move = self.extract_move(response)

        if logger:
            logger.log()
            logger.log("Conclusion:")
            logger.log(response.strip())
            logger.log()

        if move in legal_moves:
            return move, (tokens, elapsed, 1)
        else:
            raise ValueError(f"Conclusion agent chose invalid move: {move}. Legal moves: {legal_moves}")

    def choose_move(self, board, legal_moves, color, logger=None):
        total_tokens = 0
        total_time = 0.0
        total_calls = 0

        proposals, (toks, tsec, tcalls) = self.get_proposals(
            self.agents, board, legal_moves, color, prev_proposals=None, round_num=0, logger=logger
        )
        total_tokens += toks
        total_time += tsec
        total_calls += tcalls

        for r in range(self.num_rounds):
            proposals, (toks, tsec, tcalls) = self.get_proposals(
                self.agents,
                board,
                legal_moves,
                color,
                prev_proposals=proposals,
                round_num=r + 1,
                logger=logger,
            )
            total_tokens += toks
            total_time += tsec
            total_calls += tcalls

        move, (ctoks, csec, ccalls) = self.aggregate_moves(board, proposals, legal_moves, color, logger=logger)
        total_tokens += ctoks
        total_time += csec
        total_calls += ccalls

        if logger:
            logger.log(f"--- STATS: {total_tokens} tokens, {total_time:.2f}s, {total_calls} LLM calls ---")
            logger.log(f">>> MULTI CHOSEN: {move} <<<")
            logger.log()

        return move


