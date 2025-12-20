# System prompt for reasoning agents
system_prompt = """You are a precise reasoning agent. Follow instructions exactly.
- NO markdown formatting
- Be concise and direct
- Follow the exact output format requested
"""

# Generate the next reasoning step given current history
propose_prompt = """You are solving a step-by-step reasoning problem.

Problem: {input}

Current Progress:
{history}

Task: Propose {k} distinct, valid next step(s) to move towards the solution.
- Do not try to solve the entire problem, just the immediate next step
- Each step should be a single logical operation or deduction
- Be specific and show your work

Example format for math problems:
2 + 8 = 10 (left: 10 14 8)

Example format for logic problems:
From premise A and B, we can deduce C

Your next step:
"""

# Evaluate a specific reasoning step
value_prompt = """You are evaluating a reasoning step.

Problem: {input}

Current Progress:
{history}

Proposed Next Step: {candidate}

Task: Evaluate whether this step is valid and likely to lead to a correct solution.

Rate using these categories:
- "sure" (score: 20) = Step is clearly correct and makes good progress
- "likely" (score: 1) = Step is valid but uncertain if it leads to solution  
- "impossible" (score: 0.001) = Step is invalid, wrong, or leads to dead end

Examples:
- "2 + 2 = 4" for adding numbers → sure (correct arithmetic)
- "try a different approach" → likely (valid but vague)
- "2 + 2 = 5" → impossible (wrong arithmetic)

Evaluate the proposed step and end with exactly one word: sure, likely, or impossible
"""