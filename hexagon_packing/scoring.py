"""combined = w_raw * raw01(s) + w_soft * mean(soft). Only feasible candidates scored.

raw01(s) = (s_ref - s) / (s_ref - s_target): 0 at the baseline seed, 1 at SOTA,
>1 beyond (rewarded, not clipped) -- keeps the raw term commensurate with the
[0,1] soft scores. w_soft is THE knob: how much verifiers steer selection vs the
true objective; w_soft=0 is the raw-only ablation. mean (not sum) keeps the soft
term invariant to the number of verifiers.
"""

W_RAW = 1.0
W_SOFT = 0.3


def raw01(s, s_ref, s_target):
    span = s_ref - s_target
    if span <= 0:
        raise ValueError(f"need s_ref ({s_ref}) > s_target ({s_target})")
    return (s_ref - s) / span


def combined_score(s, soft_scores, s_ref, s_target, w_raw=W_RAW, w_soft=W_SOFT):
    soft_mean = sum(soft_scores) / len(soft_scores) if soft_scores else 0.0
    return w_raw * raw01(s, s_ref, s_target) + w_soft * soft_mean


if __name__ == "__main__":
    s_ref, s_target = 4.619, 3.931  # n=11 baseline / SOTA
    for name, s, soft in [
        ("baseline", 4.619, [0.3, 0.2, 0.4]),
        ("modest improve", 4.30, [0.2, 0.1, 0.3]),
        ("worse s, high soft", 4.35, [0.9, 0.8, 0.9]),  # additive lets this outrank ^
        ("beats SOTA", 3.90, [0.5, 0.6, 0.5]),
    ]:
        print(f"{name:20s} s={s:.3f}  combined={combined_score(s, soft, s_ref, s_target):+.3f}"
              f"  raw-only={combined_score(s, soft, s_ref, s_target, w_soft=0.0):+.3f}")
