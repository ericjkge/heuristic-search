"""Core result dataclasses for math reasoning search."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VerifierResult:
    name: str
    score: float          # [0.0, 1.0]
    passed: bool | None   # None for soft LLM judges (continuous score)
    feedback: str
    is_hard: bool         # True = code-based check, no LLM


@dataclass
class MathCandidate:
    candidate_id: str
    problem_id: str
    solution_text: str
    extracted_answer: str | None
    verifier_scores: dict[str, VerifierResult]  # verifier name -> result
    parent_ids: list[str]    # [] for seeds; [p.id] for revise; [a.id, b.id] for combine
    search_depth: int        # 0 for seeds
    operation: str           # "seed" | "revise" | "combine"
    generation_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "problem_id": self.problem_id,
            "solution_text": self.solution_text,
            "extracted_answer": self.extracted_answer,
            "verifier_scores": {
                k: {
                    "name": v.name,
                    "score": v.score,
                    "passed": v.passed,
                    "feedback": v.feedback,
                    "is_hard": v.is_hard,
                }
                for k, v in self.verifier_scores.items()
            },
            "parent_ids": self.parent_ids,
            "search_depth": self.search_depth,
            "operation": self.operation,
            "generation_metadata": self.generation_metadata,
        }


@dataclass
class MathResult:
    problem_id: str
    source: str
    mode: str                              # "verifier_guided_search" | "bon" | "single" | "verifier_rerank"
    solved: bool
    ground_truth_answer: str
    predicted_answer: str | None
    selected_candidate: MathCandidate | None
    all_candidates: list[MathCandidate]
    model_calls: int
    total_tokens: int
    best_verifier_score: float
    trajectory: list[float]                # best aggregate score after each step
    early_stop: bool
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "mode": self.mode,
            "solved": self.solved,
            "ground_truth_answer": self.ground_truth_answer,
            "predicted_answer": self.predicted_answer,
            "selected_candidate": self.selected_candidate.to_dict() if self.selected_candidate else None,
            "all_candidates": [c.to_dict() for c in self.all_candidates],
            "model_calls": self.model_calls,
            "total_tokens": self.total_tokens,
            "best_verifier_score": self.best_verifier_score,
            "trajectory": self.trajectory,
            "early_stop": self.early_stop,
            "extra": self.extra,
        }
