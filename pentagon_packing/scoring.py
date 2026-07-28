"""Effective scores: adaptive bucket interpolation (BES) over raw s, with the
soft-verifier mean as the within-bucket tiebreak.

Raw s is the dominant key: candidates are bucketed at BUCKET_PRECISION (raw
differences below it are treated as ties, decided by verifiers; differences
above it are decided by raw alone). Within a bucket, the plain mean of the
[0,1] verifier scores interpolates toward the next-BETTER (lower) bucket:

    effective = bucket - soft_mean * gap * (1 - SOFT_SAFETY)     (lower = better)

where gap is the distance to the next-lower bucket in the set being ranked
(the best bucket reuses the runner-up's gap). Scaling by gap means even a
perfect verifier score moves a candidate at most to the doorstep of the next
raw bucket -- verifiers can order within a raw plateau but never override a
real raw difference. SOFT_SAFETY keeps a perfect score strictly short of the
boundary (no ties with, or float-crossing into, the better bucket).

Scores are SET-RELATIVE (the gap landscape depends on which candidates are
ranked together): always pass the specific set being compared.
"""

BUCKET_PRECISION = 0.01  # bucket width in s units
SOFT_SAFETY = 0.01       # fraction of the gap the tiebreak may never enter


def soft_mean(quality):
    """Plain mean of the verifier scores dict, in [0, 1]; 0.0 when empty."""
    if not quality:
        return 0.0
    return sum(quality.values()) / len(quality)


def effective_scores(candidates):
    """One effective score per feasible candidate (same order); lower is better.

    Candidates need .raw_s (float) and .quality (dict name -> [0,1] score).
    """
    if not candidates:
        return []
    buckets = [round(c.raw_s / BUCKET_PRECISION) * BUCKET_PRECISION
               for c in candidates]
    ordered = sorted(set(buckets))  # best (lowest) first
    gaps = {}
    if len(ordered) >= 2:
        gaps[ordered[0]] = ordered[1] - ordered[0]  # best reuses runner-up's gap
        for k in range(1, len(ordered)):
            gaps[ordered[k]] = ordered[k] - ordered[k - 1]
    else:
        # Single bucket: BES zeroes the gap (verifiers dead); we use the bucket
        # width instead so verifiers still order the set -- there is no better
        # bucket to cross, so the guarantee is vacuous here.
        gaps[ordered[0]] = BUCKET_PRECISION
    return [b - soft_mean(c.quality) * gaps[b] * (1.0 - SOFT_SAFETY)
            for c, b in zip(candidates, buckets)]


def rank_by_effective(candidates):
    """Candidates sorted best-first by effective score (stable on exact ties)."""
    scores = effective_scores(candidates)
    order = sorted(range(len(candidates)), key=lambda i: scores[i])
    return [candidates[i] for i in order]
