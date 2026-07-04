"""Quality verifiers: hand-coded geometric heuristics, dense in [0,1].

Unlike the AIME harness (LLM judges, because correctness there isn't computable)
every property of a packing IS computable, so quality verifiers are deterministic
numpy functions: free, reproducible, and immune to judge noise.

Design rules (same as hexagon_packing/decompose):
- never a function of s alone (must distinguish packings with equal s);
- dense, not saturated at the seed (headroom as packings improve);
- constraint-only: a verifier measures a property, it never suggests a fix.

Weights follow the AIME convention: tiers meaningfully apart (1 / 2 / 4).
"""

import math

import numpy as np

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
_PENT_ANGLES = np.arange(5) * (2.0 * math.pi / 5.0)
_SYM = 2.0 * math.pi / 5.0  # pentagon rotational symmetry: 72 degrees

QUALITY_WEIGHTS = {
    "contact_tightness": 4.0,
    "contact_count": 4.0,
    "boundary_utilization": 2.0,
    "double_lattice_structure": 2.0,
    "evenness": 1.0,
}


def geometry_vars(centers, angles, s):
    """Build shared geometry: centers (n,2), angles (n,), vertices (n,5,2), s, n."""
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    angles = np.asarray(angles, dtype=float).reshape(-1)
    ang = angles[:, None] + _PENT_ANGLES[None, :]
    vertices = np.stack(
        [centers[:, 0, None] + R * np.cos(ang), centers[:, 1, None] + R * np.sin(ang)],
        axis=-1,
    )
    return {
        "centers": centers,
        "angles": angles,
        "vertices": vertices,
        "s": float(s),
        "n": int(centers.shape[0]),
    }


def _nn_center_gaps(centers):
    """Distance from each center to its nearest neighbour center (n>=2)."""
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def contact_tightness(ev):
    """Mean exp(-gap/sigma) over per-pentagon nearest gaps (neighbour or wall).
    Gap proxy: nearest center distance minus one pentagon width-at-contact (2*apothem
    lower bound); wall gap from vertex support. Dense: 1 when everything touches."""
    n, s = ev["n"], ev["s"]
    if n < 2:
        return 1.0
    sigma = 0.25
    apothem2 = 2.0 * (SIDE / (2.0 * math.tan(math.pi / 5.0)))  # 2*apothem ~ 1.376
    nn = _nn_center_gaps(ev["centers"]) - apothem2
    wall = (s / 2.0) - np.abs(ev["vertices"]).reshape(n, -1).max(axis=1)
    gaps = np.minimum(np.maximum(nn, 0.0), np.maximum(wall, 0.0))
    return float(np.mean(np.exp(-gaps / sigma)))


def contact_count(ev):
    """Mean min(1, touching neighbours / 3). Touching = center distance within
    tol of the two-pentagon contact range [2*apothem, 2*R]."""
    n = ev["n"]
    if n < 2:
        return 1.0
    d = np.linalg.norm(ev["centers"][:, None, :] - ev["centers"][None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    touching = (d < 2.0 * R + 0.05).sum(axis=1)
    return float(np.mean(np.minimum(1.0, touching / 3.0)))


def boundary_utilization(ev):
    """Mean over the 4 square edges of the fraction of edge length covered by the
    projection of pentagons whose vertices come within eps of that edge."""
    s, verts, n = ev["s"], ev["vertices"], ev["n"]
    half, eps = s / 2.0, 0.05
    total = 0.0
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        other = 1 - axis
        covered = 0.0
        for i in range(n):
            v = verts[i]
            near = np.abs(sign * half - v[:, axis]) < eps
            if near.any():
                lo, hi = v[near, other].min(), v[near, other].max()
                covered += max(0.0, min(hi, half) - max(lo, -half))
        total += min(1.0, covered / s)
    return total / 4.0


def double_lattice_structure(ev):
    """Kuperberg prior: optimal plane packings pair pentagons in opposite
    orientations. Score = mean closeness of nearest-neighbour orientation gap to
    36 degrees (half the 72-degree symmetry angle)."""
    n = ev["n"]
    if n < 2:
        return 1.0
    centers, angles = ev["centers"], ev["angles"]
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.argmin(axis=1)
    delta = np.abs(angles - angles[nn]) % _SYM        # in [0, 72) degrees
    off = np.abs(delta - _SYM / 2.0) / (_SYM / 2.0)   # 0 at 36 deg, 1 at 0/72 deg
    return float(np.mean(1.0 - off))


def evenness(ev):
    """1/(1+CV) of nearest-neighbour center distances: penalises one big hole."""
    if ev["n"] < 3:
        return 1.0
    nn = _nn_center_gaps(ev["centers"])
    mean = nn.mean()
    if mean <= 0:
        return 0.0
    return float(1.0 / (1.0 + nn.std() / mean))


_QUALITY_FNS = {
    "contact_tightness": contact_tightness,
    "contact_count": contact_count,
    "boundary_utilization": boundary_utilization,
    "double_lattice_structure": double_lattice_structure,
    "evenness": evenness,
}


def quality_scores(ev):
    """name -> score in [0,1]; a crashing verifier scores 0.0, never raises."""
    out = {}
    for name, fn in _QUALITY_FNS.items():
        try:
            v = float(fn(ev))
            out[name] = 0.0 if not math.isfinite(v) else max(0.0, min(1.0, v))
        except Exception:
            out[name] = 0.0
    return out


def weighted_soft_mean(scores):
    """Weighted mean of quality scores using QUALITY_WEIGHTS (1/2/4 tiers)."""
    if not scores:
        return 0.0
    tw = sum(QUALITY_WEIGHTS.get(k, 1.0) for k in scores)
    return sum(v * QUALITY_WEIGHTS.get(k, 1.0) for k, v in scores.items()) / tw


if __name__ == "__main__":
    import initial

    centers, angles, s = initial.pack(10)
    ev = geometry_vars(centers, angles, s)
    scores = quality_scores(ev)
    for k, v in scores.items():
        print(f"  {k:26} {v:.3f}  (w={QUALITY_WEIGHTS[k]})")
    print(f"  weighted mean: {weighted_soft_mean(scores):.3f}  (seed s={s:.3f})")
