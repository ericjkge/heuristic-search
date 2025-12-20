# # Generate possible next steps given the current history
# propose_prompt = """
# Problem: {input}

# Current Progress: {history}

# Task: Propose {k} distinct, valid next steps to move towards the solution.
# Do not try to solve the entire problem, just the immediate next step.
# Separate each proposal using newlines.
# """

# # Evaluate a specific state/step (NOTE: reason is currently unused)
# value_prompt = """
# Problem: {input}

# Current Progress: {history}

# Proposed Next Step: {candidate}

# Task: Evaluate the likelihood that this step leads to a correct solution to the problem.
# Rate the quality of this step on a scale from 0.1 (impossible/invalid) to 1.0 (sure/optimal).
# Provide the score and a brief reason.

# Format:
# Score: [0.1-1.0]
# Reason: [Short explanation]
# """