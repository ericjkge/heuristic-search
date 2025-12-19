from dotenv import load_dotenv
import prompts
from tasks import Game24Task, BaseTask
from models import GeminiLLM, BaseLLM
import re

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
    
    # Main agent loop (get node, propose steps, evaluate steps, push results)
    def run(self, blackboard: Blackboard, max_iterations: int = 10):
        """Main agent loop: pull work, expand, evaluate, push results."""
        for _ in range(max_iterations):
            # Get next node to work on
            node = blackboard.get_work()
            if node is None:
                continue
            
            # Propose next steps
            proposals, tokens = self._propose(blackboard.problem, node.content)
            blackboard.add_tokens(tokens)
            
            # Evaluate and add to tree
            for content in proposals:
                score, tokens = self._evaluate(content)
                blackboard.add_tokens(tokens)
                new_node = blackboard.add_node(node, content, score)
                
                # Check if solution (only "24" left)
                remaining = self._extract_remaining(content)
                if remaining.strip() == "24":
                    blackboard.add_solution(new_node)
                else:
                    # Add to work queue for further expansion
                    blackboard.add_work(new_node)
    
    def _extract_remaining(self, step: str) -> str:
        """Extract remaining numbers from step."""
        match = re.search(r'\(left:\s*([^\)]+)\)', step)
        return match.group(1).strip() if match else ""
    
    def _propose(self, problem: str, parent_content: str) -> tuple[list[str], int]:
        """Generate next step proposals."""
        if parent_content:
            remaining = self._extract_remaining(parent_content)
        else:
            remaining = problem
        
        prompt = prompts.propose_prompt.format(input=remaining)
        response_text, token_count = self.llm.generate(prompt, system_prompt=prompts.system_prompt)
        
        proposals = [line.strip() for line in response_text.split('\n') if line.strip()]
        return proposals, token_count
    
    def _evaluate(self, candidate: str) -> tuple[float, int]:
        """Evaluate a candidate step."""
        remaining = self._extract_remaining(candidate)
        if not remaining:
            return 0.001, 0
        
        prompt = prompts.value_prompt.format(input=remaining)
        response_text, token_count = self.llm.generate(prompt, system_prompt=prompts.system_prompt)
        
        response_lower = response_text.lower()
        if 'sure' in response_lower:
            return 20, token_count
        elif 'likely' in response_lower:
            return 1, token_count
        else:
            return 0.001, token_count

class TreeOfThoughts:
    def __init__(self, task: BaseTask, llm: BaseLLM):
        self.task = task
        self.llm = llm

    def solve(self, initial_problem: str, k: int = 3, b: int = 5, d: int = 3, log_file=None):
        """
        Solves the problem using BFS.

        k: Number of proposed thoughts per state
        b: Branching factor (top states to keep)
        d: Max depth (steps)
        log_file: Open file handle for logging (optional)
        """
        # Validate input
        if not self.task.validate_input(initial_problem):
            return "Invalid input for task."

        # Root node (empty content)
        current_nodes = [TreeNode(content="")] 
        total_tokens = 0
        
        # Loop for "d" steps
        for step in range(d):
            print(f"Step {step+1}/{d}, current nodes: {len(current_nodes)}")
            
            # 1. Generate k new thoughts per path (NOTE: * unpacks list created by list comprehension)
            candidates = []
            for node in current_nodes:
                proposals, tokens = self._propose(initial_problem, node.content, k)
                total_tokens += tokens
                for content in proposals:
                    candidates.append((node, content))

            # 2. Evaluate each next step
            scored_candidates = []
            for parent_node, content in candidates:
                score, tokens = self._evaluate(content)
                total_tokens += tokens
                scored_candidates.append((score, parent_node, content))

            # 3. Select top "b" next steps and create TreeNodes
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            selected = scored_candidates[:b]
            
            current_nodes = []
            for score, parent_node, content in selected:
                new_node = TreeNode(content=content, parent=parent_node, score=score)
                parent_node.children.append(new_node)
                current_nodes.append(new_node)
            
            # Log top thoughts
            if log_file:
                log_file.write(f"\n--- Step {step+1} Top Thoughts ---\n")
                for i, node in enumerate(current_nodes):
                    log_file.write(f"Rank {i+1} (Score: {node.score}):\n{node.content}\n{'-'*20}\n")
                    print(f"Rank {i+1} (Score: {node.score}):\n{node.content}\n{'-'*20}\n")

            if current_nodes:
                print(f"Top score: {current_nodes[0].score}")

        print(f"Total tokens used: {total_tokens}")
        # Best solution has highest-scoring newest thought
        return current_nodes[0].get_history() if current_nodes else "No solution found"

    # Internal helper for parsing "left" numbers (e.g. "2 + 8 = 10 (left: 8 10 14)"" -> "8 10 14")
    def _extract_remaining(self, step: str) -> str:
        match = re.search(r'\(left:\s*([^\)]+)\)', step)
        return match.group(1).strip() if match else ""

    # Internal function for proposing new thoughts
    def _propose(self, problem: str, parent_content: str, k: int) -> tuple[list[str], int]:
        # Extract state from parent's "left" numbers, or use original problem
        if parent_content:
            remaining = self._extract_remaining(parent_content)
        else:
            remaining = problem
        
        prompt = prompts.propose_prompt.format(input=remaining)
        response_text, token_count = self.llm.generate(prompt, system_prompt=prompts.system_prompt)
        
        # Split by newline and filter empty lines
        proposals = [line.strip() for line in response_text.split('\n') if line.strip()]
        return proposals, token_count

    # Internal function for evaluating new thoughts
    def _evaluate(self, candidate: str) -> tuple[float, int]:
        # Extract "left" numbers from candidate step
        remaining = self._extract_remaining(candidate)
        if not remaining:
            return 0.001, 0  # Invalid format
        
        prompt = prompts.value_prompt.format(input=remaining)
        response_text, token_count = self.llm.generate(prompt, system_prompt=prompts.system_prompt)

        # Map sure/likely/impossible to scores (NOTE: values copied from Princeton ToT)
        response_lower = response_text.lower()
        if 'sure' in response_lower:
            return 20, token_count
        elif 'likely' in response_lower:
            return 1, token_count
        else:  # impossible or unrecognized
            return 0.001, token_count

def main():
    # Initialize task and solver
    task = Game24Task()
    llm = GeminiLLM()
    tot = TreeOfThoughts(task, llm)
    
    # Example problem
    problem = "2 2 6 8"
    print(f"Solving: {problem}")
    
    # Run Tree of Thoughts
    with open("output.txt", "w") as f:
        f.write(f"Problem: {problem}\n")
        best_path = tot.solve(problem, k=3, b=5, d=3, log_file=f)
        
        f.write("\n=== BEST PATH ===\n")
        f.write(best_path)
    
    # Output results
    print("\n=== BEST PATH ===")
    print(best_path)

if __name__ == "__main__":
    main()
