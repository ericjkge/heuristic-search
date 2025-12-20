from dotenv import load_dotenv
import re
import prompts
from models import KimiLLM, BaseLLM

# Load environment variables
load_dotenv()


class TreeNode:
    """A node in the reasoning tree representing a single reasoning step."""
    
    def __init__(self, content: str, parent: 'TreeNode' = None, score: float = 0.0, agent_id: int = None
    ):
        self.content = content          # Step string (e.g. "2 + 8 = 10 (left: 8 10 14)")
        self.parent = parent
        self.children: list['TreeNode'] = []
        self.score = score              # Evaluation score
        self.visits = 0                 # Visit count (for future MCTS/UCB/etc.)
        self.agent_id = agent_id        # Which agent created this node
        
        # Link to parent
        if parent:
            parent.children.append(self)

    def is_leaf(self) -> bool:
        """Check if this node has no children."""
        return len(self.children) == 0

    def get_history(self) -> str:
        """Walk from this node to root and return the reasoning path."""
        history = []
        curr = self
        while curr and curr.content:  # Stop if content is empty (root dummy)
            history.append(curr.content)
            curr = curr.parent
        return "\n".join(reversed(history))
    
    def get_depth(self) -> int:
        """Return the depth of this node (root = 0)."""
        depth = 0
        curr = self.parent
        while curr:
            depth += 1
            curr = curr.parent
        return depth

    def __repr__(self) -> str:
        return f"TreeNode(score={self.score:.2f}, agent={self.agent_id}, content='{self.content[:30]}...')"


class Tree:
    def __init__(self, problem: str):
        self.problem = problem
        self.root = TreeNode(content="", parent=None)  # Dummy root node
    
    def get_all_leaves(self) -> list[TreeNode]:
        """Return all leaf nodes in the tree using BFS."""
        leaves = []
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            if node.is_leaf():
                # Don't include empty root as a leaf unless it's the only node
                if node.content or node == self.root:
                    leaves.append(node)
            else:
                queue.extend(node.children)
        
        return leaves
    
    def get_best_leaf(self) -> TreeNode:
        """Return the leaf node with the highest score (greedy selection)."""
        leaves = self.get_all_leaves()
        if not leaves:
            return self.root
        return max(leaves, key=lambda n: n.score)
    
    def add_node(
        self,
        parent: TreeNode,
        content: str,
        score: float = 0.0,
        agent_id: int = None
    ) -> TreeNode:
        """Create a new node and attach it to the parent."""
        node = TreeNode(
            content=content,
            parent=parent,
            score=score,
            agent_id=agent_id
        )
        return node
    
    def get_node_count(self) -> int:
        """Return total number of nodes in the tree."""
        count = 0
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            count += 1
            queue.extend(node.children)
        return count

    def print_tree(self, node: TreeNode = None, indent: int = 0):
        """Pretty print the tree structure."""
        if node is None:
            node = self.root
        
        prefix = "  " * indent
        if node.content:
            print(f"{prefix}├─ [A{node.agent_id}|{node.score:.1f}] {node.content[:50]}")
        else:
            print(f"{prefix}[ROOT]")
        
        for child in node.children:
            self.print_tree(child, indent + 1)


class Agent:    
    def __init__(self, agent_id: int, llm: BaseLLM):
        self.agent_id = agent_id
        self.llm = llm
        self.current_node: TreeNode = None      # Current position in tree
        self.memory: list[str] = []             # Exploration history
    
    def propose(self, tree: Tree) -> str:
        history = self.current_node.get_history() if self.current_node.content else ""
        
        prompt = prompts.propose_prompt.format(
            input=tree.problem,
            history=history if history else "(No steps yet)",
            k=1  # Generate 1 proposal
        )
        
        response, _ = self.llm.generate(prompt, prompts.system_prompt)
        # Take first non-empty line as the proposal
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        return lines[0] if lines else response.strip()
    
    def evaluate(self, tree: Tree, step: str) -> float:
        history = self.current_node.get_history() if self.current_node.content else ""
        
        prompt = prompts.value_prompt.format(
            input=tree.problem,
            history=history if history else "(No steps yet)",
            candidate=step
        )
        
        response, _ = self.llm.generate(prompt, prompts.system_prompt)
        
        # Parse score from response (expect "Score: X.X")
        score_match = re.search(r'Score:\s*([0-9.]+)', response, re.IGNORECASE)
        if score_match:
            try:
                return float(score_match.group(1))
            except ValueError:
                pass
        
        # Fallback: check for sure/likely/impossible (ToT style)
        response_lower = response.lower()
        if 'sure' in response_lower:
            return 20.0
        elif 'likely' in response_lower:
            return 1.0
        else:
            return 0.001
    
    def propose_and_evaluate(self, tree: Tree) -> TreeNode:
        """Generate a new step, evaluate it, and add to tree."""
        # Propose next step
        step = self.propose(tree)
        
        # Evaluate the step
        score = self.evaluate(tree, step)
        
        # Add to tree
        new_node = tree.add_node(
            parent=self.current_node,
            content=step,
            score=score,
            agent_id=self.agent_id
        )
        
        # Update agent state
        self.current_node = new_node
        self.memory.append(f"Round {len(self.memory)+1}: {step} (score: {score:.2f})")
        
        return new_node
    
    def decide_next_position(self, tree: Tree, threshold: float = 2.0):
        best_leaf = tree.get_best_leaf()
        best_score = best_leaf.score
        own_score = self.current_node.score
        
        # Prefer own branch if within threshold of best
        if best_score - own_score <= threshold:
            # Stay at current node
            self.memory.append(f"  -> Staying (own: {own_score:.2f}, best: {best_score:.2f})")
        else:
            # Jump to best leaf
            self.memory.append(
                f"  -> Jumping to best leaf (own: {own_score:.2f}, best: {best_score:.2f})"
            )
            self.current_node = best_leaf
    
    def get_memory_str(self) -> str:
        """Return formatted memory history."""
        return "\n".join(self.memory)

def solve(
    problem: str,
    agents: list[Agent],
    max_rounds: int = 5,
    threshold: float = 2.0
) -> str:
    """
    Run the multi-agent reasoning process.
    
    Args:
        problem: The problem to solve
        agents: List of agents to use
        max_rounds: Maximum number of reasoning rounds
        threshold: Score difference threshold for branch jumping
    
    Returns:
        The best reasoning path found
    """
    tree = Tree(problem)
    
    # Initialize all agents at root
    for agent in agents:
        agent.current_node = tree.root
        agent.memory = []
    
    for round_num in range(max_rounds):
        print(f"\n=== Round {round_num + 1}/{max_rounds} ===")
        
        # Phase 1: All agents propose and evaluate
        for agent in agents:
            node = agent.propose_and_evaluate(tree)
            print(f"Agent {agent.agent_id}: '{node.content[:50]}...' (score: {node.score:.2f})")
        
        # Phase 2: All agents decide next position
        for agent in agents:
            agent.decide_next_position(tree, threshold)
        
        # Show tree state
        print(f"\nTree has {tree.get_node_count()} nodes")
    
    # Return best path
    best_leaf = tree.get_best_leaf()
    return best_leaf.get_history()


def main():
    # Example usage
    llm = KimiLLM()
    
    agents = [
        Agent(agent_id=0, llm=llm),
        Agent(agent_id=1, llm=llm),
        Agent(agent_id=2, llm=llm),
    ]
    
    problem = "2 8 8 14"  # Game of 24
    result = solve(problem, agents, max_rounds=3, threshold=2.0)
    
    print("\n=== BEST REASONING PATH ===")
    print(result)


if __name__ == "__main__":
    main()
