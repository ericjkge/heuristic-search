from dotenv import load_dotenv
import prompts
from models import GeminiLLM, BaseLLM

# Load environment variables
load_dotenv()


class TreeNode:
    def __init__(self, content: str, parent: 'TreeNode' = None, score: float = 0.0):
        self.content = content      # Step string (e.g. "2+2=4")
        self.parent = parent
        self.children = []
        self.score = score          # Evaluation score
        self.visits = 0             # Visit count (for future MCTS/UCB/etc.)

    # Convert history to string (walk from node to root)        
    def get_history(self) -> str:
        history = []
        curr = self
        while curr and curr.content: # Stop if content is empty (root dummy)
            history.append(curr.content)
            curr = curr.parent
        return "\n".join(reversed(history))

# Global blackboard with ToT, work queue, and completed solutions
class Blackboard:
    def __init__(self, problem: str):
        self.problem = problem              # Original problem
        self.root = TreeNode(content="")    # Tree root
        self.work_queue = []                # List of nodes to expand (NOTE: rename to blackboard, rename blackboard to global memory)
        self.solutions = []                 # Completed solution paths
        self.total_tokens = 0
    
    def add_work(self, node: TreeNode):
        """Add a node to the work queue."""
        self.work_queue.append(node)
    
    def get_work(self) -> TreeNode | None:
        """Get next node to expand (returns None if queue empty)."""
        if self.work_queue:
            return self.work_queue.pop(0)
        return None
    
    def add_node(self, parent: TreeNode, content: str, score: float) -> TreeNode:
        """Create a new node and attach to tree."""
        new_node = TreeNode(content=content, parent=parent, score=score)
        parent.children.append(new_node)
        return new_node
    
    def add_solution(self, node: TreeNode):
        """Record a completed solution."""
        self.solutions.append(node)
    
    def add_tokens(self, count: int):
        """Track token usage."""
        self.total_tokens += count


class Agent:
    def __init__(self, agent_id: int, llm: BaseLLM):
        self.agent_id = agent_id
        self.llm = llm
    
    def run(self, blackboard: Blackboard, max_iterations: int = 10):
        pass


def main():
    pass

if __name__ == "__main__":
    main()
