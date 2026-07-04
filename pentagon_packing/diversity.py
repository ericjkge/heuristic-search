"""Diversity descriptors: how DIFFERENT two packings are in structure, never how
good. Disjoint from quality (verifiers.py) by construction -- the same split as
the AIME technique taxonomy, but code-computed and free because geometry is
measurable.

Descriptor vector per candidate (each normalised to [0,1]):
  orientation_clusters  distinct angle clusters mod 72deg (1=lattice, 2=double
                        lattice, more=mixed), /4 capped
  boundary_fraction     fraction of pentagons touching the container wall
  row_count             y-clusters of centers (layering), /6 capped
  angle_entropy         circular entropy of angles mod 72deg
  uses_scipy            program-level bit: candidate source imports scipy
  uses_stochastic       program-level bit: candidate source uses randomness

diversity_distance = L2 / sqrt(dim) between vectors, in [0,1].
"""

import math

import numpy as np

_SYM = 2.0 * math.pi / 5.0
DESCRIPTOR_NAMES = [
    "orientation_clusters", "boundary_fraction", "row_count",
    "angle_entropy", "uses_scipy", "uses_stochastic",
]


def _cluster_count(values, period, tol):
    """Distinct clusters of circular values (radians, given period)."""
    if len(values) == 0:
        return 0
    vals = np.sort(np.asarray(values) % period)
    gaps = np.diff(np.concatenate([vals, [vals[0] + period]]))
    return max(1, int((gaps > tol).sum()))


def descriptor_vector(ev, source=""):
    """Compute the 6-dim descriptor vector from geometry_vars + candidate source."""
    n, s = ev["n"], ev["s"]
    angles, centers, verts = ev["angles"], ev["centers"], ev["vertices"]

    clusters = _cluster_count(angles, _SYM, tol=math.radians(5.0))
    orientation_clusters = min(1.0, clusters / 4.0)

    wall = (s / 2.0) - np.abs(verts).reshape(n, -1).max(axis=1)
    boundary_fraction = float((wall < 0.05).mean())

    rows = _cluster_count(centers[:, 1], period=max(s, 1e-9), tol=0.5)
    row_count = min(1.0, rows / 6.0)

    hist, _ = np.histogram(angles % _SYM, bins=12, range=(0.0, _SYM))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    angle_entropy = float(-(p * np.log(p)).sum() / math.log(12))

    uses_scipy = 1.0 if "scipy" in source else 0.0
    uses_stochastic = 1.0 if ("random" in source or "np.random" in source) else 0.0

    return [orientation_clusters, boundary_fraction, row_count,
            angle_entropy, uses_scipy, uses_stochastic]


def diversity_distance(vec_a, vec_b):
    """Normalised L2 in [0,1]. Empty vector (failed candidate) -> distance 0."""
    if not vec_a or not vec_b:
        return 0.0
    a, b = np.asarray(vec_a), np.asarray(vec_b)
    return float(np.linalg.norm(a - b) / math.sqrt(len(vec_a)))


def min_distance_to(vec, others):
    """Min diversity distance from vec to a list of vectors (1.0 if none)."""
    dists = [diversity_distance(vec, o) for o in others if o]
    return min(dists) if dists else 1.0


if __name__ == "__main__":
    import initial
    import verifiers

    centers, angles, s = initial.pack(10)
    ev = verifiers.geometry_vars(centers, angles, s)
    vec = descriptor_vector(ev, source="import scipy.optimize")
    for name, v in zip(DESCRIPTOR_NAMES, vec):
        print(f"  {name:22} {v:.3f}")
    rotated = verifiers.geometry_vars(centers, [a + 0.3 * i for i, a in enumerate(angles)], s)
    vec2 = descriptor_vector(rotated, source="")
    print(f"  distance(seed, mixed-rotations): {diversity_distance(vec, vec2):.3f}")
