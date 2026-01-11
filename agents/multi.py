from collections import Counter
from utils.logging import get_logger

logger = get_logger(__name__)

STRATEGIES = [
    "feature_judge", "feature_majority", "feature_borda",
    "adversarial_judge",
    "halfhalf_judge",
    "centralized_judge",
    "self_consistency"
]


class MultiAgent:
    def __init__(self, llm, prompts, game_config, strategy="feature_judge", n_rounds=2):
        self.llm = llm()
        self.prompts = prompts
        self.extract_move = game_config["extract_move"]
        self.extract_ranked_moves = game_config["extract_ranked_moves"]
        self.format_state = game_config["format_state"]
        self.strategy = strategy
        self.n_rounds = n_rounds
        
        parts = strategy.split("_")
        self.diversity = parts[0]
        self.aggregation = parts[1] if len(parts) > 1 else "majority"
        
        # Borda needs ranked output
        self.use_ranked = (self.aggregation == "borda")

    def _query(self, board, prompt, max_attempts=3):
        """Query LLM, return (response, move) or (response, [moves]) for ranked."""
        messages = [{"role": "user", "content": prompt}]
        for _ in range(max_attempts):
            response = self.llm.generate(messages)
            
            if self.use_ranked:
                moves = self.extract_ranked_moves(response)
                valid = [m for m in moves if m in board.legal_moves()]
                if valid:
                    return response, valid
            else:
                move = self.extract_move(response)
                if move and (move in board.legal_moves() or move == 'PASS'):
                    return response, move
            
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Invalid move(s). Try again."})
        return None, [] if self.use_ranked else None

    def _query_no_move(self, prompt):
        """Query LLM for critique only (no move extraction)."""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.generate(messages)

    # === DIVERSITY STRATEGIES ===

    def _feature_debate(self, board, player):
        """Multi-round parallel debate. All agents see all proposals from t-1."""
        state = self.format_state(board, player)
        prev_proposals = []
        all_moves = []
        
        prompt_key = "feature_ranked_prompt" if self.use_ranked else "feature_prompt"
        template = getattr(self.prompts, prompt_key)
        
        for round_num in range(self.n_rounds):
            current_proposals = []
            current_moves = []
            
            for feature in self.prompts.FEATURES:
                prompt = template.format(
                    **state, feature=feature,
                    prev_proposals=prev_proposals or "None yet"
                )
                response, moves = self._query(board, prompt)
                if response:
                    current_proposals.append(response)
                    current_moves.append(moves)
                logger.info(f"Feature R{round_num+1} | {feature[:25]}... | Moves={moves}")
            
            prev_proposals = current_proposals
            all_moves = current_moves
        
        return prev_proposals, all_moves

    def _adversarial_debate(self, board, player):
        """Adversarial debate: single prompt with formatted debate history."""
        state = self.format_state(board, player)
        template = self.prompts.adversarial_prompt
        
        # Debate history as formatted text
        debate_history = []
        aff_move, neg_move = None, None
        aff_resp, neg_resp = "", ""
        
        for round_num in range(self.n_rounds):
            # === AFFIRMATIVE TURN ===
            history_text = "\n".join(debate_history) if debate_history else "(You speak first)"
            aff_prompt = template.format(
                **state, role="Affirmative", debate_history=history_text
            )
            aff_resp, aff_move = self._query(board, aff_prompt)
            debate_history.append(f"Affirmative: {aff_resp}")
            logger.info(f"Adversarial R{round_num+1} | Affirmative | Move={aff_move}")
            
            # === NEGATIVE TURN ===
            history_text = "\n".join(debate_history)
            neg_prompt = template.format(
                **state, role="Negative", debate_history=history_text
            )
            neg_resp, neg_move = self._query(board, neg_prompt)
            debate_history.append(f"Negative: {neg_resp}")
            logger.info(f"Adversarial R{round_num+1} | Negative | Move={neg_move}")
        
        # Return final proposals and moves
        proposals = [aff_resp or "", neg_resp or ""]
        moves = [aff_move, neg_move]
        return proposals, moves

    def _halfhalf_debate(self, board, player):
        """Half-half debate: 3 feature + 3 vanilla agents, parallel structure."""
        state = self.format_state(board, player)
        prev_proposals = []
        all_moves = []
        
        for round_num in range(self.n_rounds):
            current_proposals = []
            current_moves = []
            
            # 3 feature agents
            for feature in self.prompts.FEATURES:
                prompt = self.prompts.feature_prompt.format(
                    **state, feature=feature,
                    prev_proposals=prev_proposals or "None yet"
                )
                response, move = self._query(board, prompt)
                if response:
                    current_proposals.append(response)
                    current_moves.append(move)
                logger.info(f"HalfHalf R{round_num+1} | {feature[:20]}... | Move={move}")
            
            # 3 vanilla agents
            for i in range(3):
                prompt = self.prompts.action_prompt.format(**state)
                response, move = self._query(board, prompt)
                if response:
                    current_proposals.append(response)
                    current_moves.append(move)
                logger.info(f"HalfHalf R{round_num+1} | vanilla-{i+1} | Move={move}")
            
            prev_proposals = current_proposals
            all_moves = current_moves
        
        return prev_proposals, all_moves

    def _centralized_debate(self, board, player):
        """Centralized debate: central agent proposes, feature agents critique."""
        state = self.format_state(board, player)
        
        proposal = ""
        final_move = None
        
        for round_num in range(self.n_rounds):
            # === CENTRAL AGENT PROPOSES ===
            if round_num == 0:
                critiques_section = "This is your first proposal."
            else:
                critiques_section = f"Critiques from previous round:\n{critiques_text}"
            
            propose_prompt = self.prompts.centralized_propose_prompt.format(
                **state, critiques_section=critiques_section
            )
            proposal, final_move = self._query(board, propose_prompt)
            logger.info(f"Centralized R{round_num+1} | Central | Move={final_move}")
            
            # === FEATURE AGENTS CRITIQUE ===
            critiques = []
            for feature in self.prompts.FEATURES:
                critique_prompt = self.prompts.centralized_critique_prompt.format(
                    **state, feature=feature, proposal=proposal
                )
                critique = self._query_no_move(critique_prompt)
                critiques.append(f"[{feature[:20]}]: {critique}")
                logger.info(f"Centralized R{round_num+1} | Critique | {feature[:20]}...")
            
            critiques_text = "\n".join(critiques)
        
        # Final round: central sees last critiques and makes final decision
        critiques_section = f"Critiques from previous round:\n{critiques_text}"
        propose_prompt = self.prompts.centralized_propose_prompt.format(
            **state, critiques_section=critiques_section
        )
        proposal, final_move = self._query(board, propose_prompt)
        logger.info(f"Centralized Final | Central | Move={final_move}")
        
        return [proposal], [final_move]

    def _self_consistency(self, board, player):
        """Same prompt 5 times, majority vote."""
        state = self.format_state(board, player)
        prompt = self.prompts.action_prompt.format(**state)
        
        proposals, moves = [], []
        for i in range(5):
            response, move = self._query(board, prompt)
            if response:
                proposals.append(response)
                moves.append(move)
            logger.info(f"SelfConsistency | Sample {i+1} | Move={move}")
        
        return proposals, moves

    # === AGGREGATION METHODS ===

    def _majority_vote(self, moves):
        """Simple majority vote on single moves."""
        valid = [m for m in moves if m]
        return Counter(valid).most_common(1)[0][0] if valid else None

    def _borda_count(self, ranked_moves):
        """Borda count: 1st=3pts, 2nd=2pts, 3rd=1pt."""
        scores = Counter()
        for ranking in ranked_moves:
            if isinstance(ranking, list):
                for i, move in enumerate(ranking[:3]):
                    scores[move] += 3 - i
            elif ranking:  # Single move fallback
                scores[ranking] += 3
        return scores.most_common(1)[0][0] if scores else None

    def _llm_judge(self, board, player, proposals):
        """LLM judge picks best move from proposals."""
        state = self.format_state(board, player)
        
        # Use debate-style prompt for adversarial (2 proposals: Aff, Neg)
        if self.diversity == "adversarial" and len(proposals) == 2:
            prompt = self.prompts.aggregate_prompt.format(
                **state,
                aff_final=proposals[0],
                neg_final=proposals[1]
            )
        else:
            # Multi-agent format for feature/halfhalf
            proposals_text = "\n\n".join(proposals)
            prompt = self.prompts.aggregate_prompt_multi.format(
                **state, proposals=proposals_text
            )
        
        messages = [{"role": "user", "content": prompt}]
        
        for _ in range(3):
            response = self.llm.generate(messages)
            move = self.extract_move(response)
            if move and (move in board.legal_moves() or move == 'PASS'):
                return move
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Invalid move '{move}'. Try again."})
        return None

    # === MAIN ENTRY ===

    def choose_move(self, board, player):
        # Get proposals
        if self.diversity == "feature":
            proposals, moves = self._feature_debate(board, player)
        elif self.diversity == "adversarial":
            proposals, moves = self._adversarial_debate(board, player)
        elif self.diversity == "halfhalf":
            proposals, moves = self._halfhalf_debate(board, player)
        elif self.diversity == "centralized":
            proposals, moves = self._centralized_debate(board, player)
        else:  # self_consistency
            proposals, moves = self._self_consistency(board, player)
        
        logger.info(f"Strategy={self.strategy} | AllMoves={moves}")
        
        # Aggregate (centralized returns final move directly)
        if self.diversity == "centralized":
            return moves[0] if moves else None
        elif self.aggregation == "judge":
            return self._llm_judge(board, player, proposals)
        elif self.aggregation == "borda":
            return self._borda_count(moves)
        else:  # majority
            return self._majority_vote(moves)
