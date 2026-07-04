"""combined = w_raw * raw01(s) + w_soft * weighted_mean(quality). Feasible only.

raw01(s) = (s_ref - s) / (s_ref - s_target): 0 at the baseline seed, 1 at the
best-known record, >1 beyond (rewarded, not clipped). w_soft is THE knob: how
much quality verifiers steer selection vs the true objective; w_soft=0 is the
raw-only ablation. The quality mean is weight-tiered (verifiers.QUALITY_WEIGHTS).

Diversity is deliberately NOT part of this score -- it enters only through
population truncation and parent selection (search.py), mirroring the AIME v4
disjoint quality/diversity design.
"""

from verifiers import weighted_soft_mean

W_RAW = 1.0
W_SOFT = 0.3


def raw01(s, s_ref, s_target):
    span = s_ref - s_target
    if span <= 0:
        raise ValueError(f"need s_ref ({s_ref}) > s_target ({s_target})")
    return (s_ref - s) / span


def combined_score(s, quality, s_ref, s_target, w_raw=W_RAW, w_soft=W_SOFT):
    return w_raw * raw01(s, s_ref, s_target) + w_soft * weighted_soft_mean(quality)


if __name__ == "__main__":
    s_ref, s_target = 6.6, 4.906  # n=10 grid baseline / best known (July 2026)
    for name, s, q in [
        ("baseline", 6.6, {"contact_tightness": 0.4, "evenness": 0.8}),
        ("modest improve", 5.8, {"contact_tightness": 0.5, "evenness": 0.7}),
        ("beats record", 4.85, {"contact_tightness": 0.9, "evenness": 0.9}),
    ]:
        print(f"{name:16} s={s:.3f}  combined={combined_score(s, q, s_ref, s_target):+.3f}"
              f"  raw-only={combined_score(s, q, s_ref, s_target, w_soft=0.0):+.3f}")
