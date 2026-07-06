"""Improved packer for unit regular pentagons into the smallest origin-centered
axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles: list[float] rotation in radians
- s: side length of the smallest origin-centered axis-aligned square enclosing all
     pentagons.

This version uses a mixed strategy:
1) A hand-built double-lattice / staggered arrangement based on the known tendency
   of regular pentagons to pack efficiently in opposite orientations.
2) Local randomized improvement over a small parameter space.
3) Exact geometric validation and conservative binary-search shrinking when possible.

The implementation is self-contained and does not require SciPy.
"""

import math
import random
from typing import List, Tuple

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

TAU = 2.0 * math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0

# A small epsilon for robust geometric comparisons
EPS = 1e-10


def pentagon_vertices(cx: float, cy: float, angle: float):
    return [
        (cx + R * math.cos(angle + k * TAU / 5.0),
         cy + R * math.sin(angle + k * TAU / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, ang)
    )


def polygon_axes(poly):
    axes = []
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        ux, uy = -(y2 - y1), (x2 - x1)
        norm = math.hypot(ux, uy)
        if norm > 0:
            axes.append((ux / norm, uy / norm))
    return axes


def project(poly, ax, ay):
    vals = [x * ax + y * ay for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb):
    # Separating axis test
    for ax, ay in polygon_axes(pa) + polygon_axes(pb):
        amin, amax = project(pa, ax, ay)
        bmin, bmax = project(pb, ax, ay)
        if amax < bmin - EPS or bmax < amin - EPS:
            return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def valid_pack(centers, angles):
    if has_overlap(centers, angles):
        return False
    s = enclosing_side(centers, angles)
    half = s / 2.0 + 1e-9
    for (cx, cy), ang in zip(centers, angles):
        for x, y in pentagon_vertices(cx, cy, ang):
            if abs(x) > half or abs(y) > half:
                return False
    return True


def bounding_side_for_shift(centers, angles, dx=0.0, dy=0.0):
    shifted = [(x + dx, y + dy) for x, y in centers]
    return enclosing_side(shifted, angles)


def center_square(centers, angles):
    # Keep origin-centered convention; here the input layout is already centered.
    # If a generated layout is slightly biased, do not translate: container must
    # stay origin-centered. Instead, use symmetric constructions.
    return centers


def shift_pattern(points, x0, y0):
    return [(x + x0, y + y0) for x, y in points]


def build_double_lattice(n, a, b, rx, ry, jitter=0.0, phase=0.0):
    """
    Build a staggered arrangement in rows:
    - alternating orientations
    - row shifts to encourage opposite-orientation interlocking
    - optional tiny jitter to diversify local search
    """
    centers = []
    angles = []

    # Two pointy orientations: one flipped by pi
    ang0 = math.pi / 2.0 + phase
    ang1 = -math.pi / 2.0 + phase

    # Grid dimensions
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # Stagger rows to interlock
    for j in range(rows):
        row_shift = (0.5 * a if (j % 2 == 1) else 0.0)
        # slight centering correction for odd/even row lengths
        for i in range(cols):
            k = j * cols + i
            if k >= n:
                break
            x = (i - (cols - 1) / 2.0) * a + row_shift
            y = (j - (rows - 1) / 2.0) * b
            if jitter:
                x += jitter * math.sin(1.7 * k + 0.3)
                y += jitter * math.cos(2.3 * k + 0.8)
            centers.append((x * rx, y * ry))
            angles.append(ang0 if ((i + j) % 2 == 0) else ang1)
    return centers, angles


def build_spiral_shell(n, scale=1.0, phase=0.0):
    """
    A boundary-friendly fallback: points placed on a gentle spiral with alternating
    orientations. Useful for small n or as a diversification seed.
    """
    centers = []
    angles = []
    base_r = 0.55 * scale
    dr = 0.32 * scale
    for k in range(n):
        t = 2.0 * math.pi * (k / max(1, n)) * (1.0 + 0.18 * n)
        r = base_r + dr * math.sqrt(k)
        x = r * math.cos(t)
        y = r * math.sin(t)
        centers.append((x, y))
        angles.append((math.pi / 2.0 if k % 2 == 0 else -math.pi / 2.0) + phase)
    return centers, angles


def try_shrink_about_origin(centers, angles):
    """
    Uniformly scale centers toward the origin until just before overlap.
    This preserves orientations.
    """
    if has_overlap(centers, angles):
        return centers

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 0.0, 1.0
    # first ensure hi feasible
    if has_overlap(scaled(hi), angles):
        return centers
    # search lower bound by shrinking until overlap appears or too small
    for _ in range(35):
        mid = (lo + hi) / 2.0
        if mid <= 0:
            break
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi)


def local_search(centers, angles, iters=1200, step0=0.06, rng=None):
    """
    Randomized hill-climbing on a small parameterization:
    each move perturbs a center and optionally flips/rotates a pentagon.
    The objective is the enclosing square side; infeasible states are rejected.
    """
    if rng is None:
        rng = random.Random()

    best_centers = [tuple(p) for p in centers]
    best_angles = list(angles)
    best_s = enclosing_side(best_centers, best_angles)

    n = len(best_centers)

    for t in range(iters):
        idx = rng.randrange(n)
        step = step0 * (1.0 - t / max(1, iters)) ** 1.5

        cand_centers = [tuple(p) for p in best_centers]
        cand_angles = list(best_angles)

        # perturb one pentagon
        dx = rng.uniform(-step, step)
        dy = rng.uniform(-step, step)
        cand_centers[idx] = (cand_centers[idx][0] + dx, cand_centers[idx][1] + dy)

        # sometimes flip orientation or apply a small rotation
        r = rng.random()
        if r < 0.18:
            cand_angles[idx] += math.pi
        elif r < 0.38:
            cand_angles[idx] += rng.uniform(-0.15, 0.15)

        # occasional global gentle shrink/expand around origin
        if rng.random() < 0.08:
            lam = rng.uniform(0.985, 1.01)
            cand_centers = [(x * lam, y * lam) for x, y in cand_centers]

        if has_overlap(cand_centers, cand_angles):
            continue

        s = enclosing_side(cand_centers, cand_angles)
        if s + 1e-12 < best_s:
            best_s = s
            best_centers = cand_centers
            best_angles = cand_angles

    return best_centers, best_angles


def build_initial_candidates(n):
    """
    Generate several distinct initial layouts, then choose the best valid one.
    """
    candidates = []

    # Base geometry constants tuned for pentagon packing
    w = WIDTH * 0.94
    h = HEIGHT * 0.90

    # Candidate 1: staggered double-lattice grid
    for rx in (0.88, 0.92, 0.96, 1.0):
        for ry in (0.86, 0.90, 0.94, 0.98):
            for phase in (0.0, math.pi / 10.0, -math.pi / 10.0):
                c, a = build_double_lattice(n, w, h, rx, ry, jitter=0.0, phase=phase)
                candidates.append((c, a))

    # Candidate 2: slightly irregular staggered grid
    for rx in (0.90, 0.95, 1.00):
        for ry in (0.88, 0.93, 0.98):
            c, a = build_double_lattice(n, w, h, rx, ry, jitter=0.02, phase=math.pi / 20.0)
            candidates.append((c, a))

    # Candidate 3: spiral shell, often helpful for small n
    for phase in (0.0, math.pi / 15.0, -math.pi / 15.0):
        c, a = build_spiral_shell(n, scale=max(w, h) * 0.55, phase=phase)
        candidates.append((c, a))

    # Candidate 4: compact hex-ish staggered rows with alternate flip
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    dx = w * 0.98
    dy = h * 0.96
    c = []
    a = []
    for j in range(rows):
        offset = 0.5 * dx if j % 2 else 0.0
        for i in range(cols):
            k = j * cols + i
            if k >= n:
                break
            x = (i - (cols - 1) / 2.0) * dx + offset
            y = (j - (rows - 1) / 2.0) * dy
            c.append((x, y))
            a.append(math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0)
    candidates.append((c, a))

    return candidates


def optimize_layout(n, seed=0):
    rng = random.Random(seed)
    candidates = build_initial_candidates(n)

    # Validate and score candidates
    scored = []
    for c, a in candidates:
        if not has_overlap(c, a):
            scored.append((enclosing_side(c, a), c, a))

    if not scored:
        # Fallback: very conservative grid
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        dx = WIDTH + 0.25
        dy = HEIGHT + 0.25
        c = []
        a = []
        for j in range(rows):
            for i in range(cols):
                k = j * cols + i
                if k >= n:
                    break
                c.append(((i - (cols - 1) / 2.0) * dx, (j - (rows - 1) / 2.0) * dy))
                a.append(math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0)
        return c, a

    scored.sort(key=lambda t: t[0])
    best_s, best_c, best_a = scored[0]

    # Local search from several top seeds
    topk = min(6, len(scored))
    for idx in range(topk):
        s0, c0, a0 = scored[idx]
        c1, a1 = local_search(c0, a0, iters=1800, step0=0.05 + 0.01 * idx, rng=rng)
        if not has_overlap(c1, a1):
            s1 = enclosing_side(c1, a1)
            if s1 < best_s:
                best_s, best_c, best_a = s1, c1, a1

        # Try a symmetric shrink if the configuration is feasible
        c2 = try_shrink_about_origin(c1, a1)
        if not has_overlap(c2, a1):
            s2 = enclosing_side(c2, a1)
            if s2 < best_s:
                best_s, best_c, best_a = s2, c2, a1

    # Final polish: small random perturbation hill-climbing
    for _ in range(4):
        c_try, a_try = local_search(best_c, best_a, iters=1200, step0=0.03, rng=rng)
        if not has_overlap(c_try, a_try):
            s_try = enclosing_side(c_try, a_try)
            if s_try < best_s:
                best_s, best_c, best_a = s_try, c_try, a_try

    return best_c, best_a


def pack(n):
    """
    Return a valid packing of n unit regular pentagons.
    """
    if n <= 0:
        return [], [], 0.0

    centers, angles = optimize_layout(n, seed=1234567 + 7919 * n)

    # Ensure validity; if something went wrong, fall back to a safe grid.
    if not has_overlap(centers, angles):
        s = enclosing_side(centers, angles)
        return centers, angles, s

    # Conservative fallback
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pitch_x = WIDTH + 0.30
    pitch_y = HEIGHT + 0.30
    centers, angles = [], []
    for k in range(n):
        i, j = k % cols, k // cols
        x = (i - (cols - 1) / 2.0) * pitch_x
        y = (j - (rows - 1) / 2.0) * pitch_y
        centers.append((x, y))
        angles.append(math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0)

    return centers, angles, enclosing_side(centers, angles)
