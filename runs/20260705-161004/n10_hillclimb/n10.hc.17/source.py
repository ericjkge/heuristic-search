"""Packing 10 unit regular pentagons into the smallest origin-centered
axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
Container: a point p is inside iff max(|px|, |py|) <= s/2.

This version uses:
- an exact convex polygon representation of each pentagon,
- deterministic construction families tailored to 10 pentagons,
- a robust nonlinear local optimizer (SciPy if available, otherwise a
  custom coordinate/annealing search),
- final geometric shrinking with exact collision tests.

The goal is not to prove optimality, but to produce a substantially tighter
valid packing than the previous heuristic.
"""

import math
import random
from itertools import combinations

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
TWOPI = 2.0 * math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def normalize_angle(a):
    a = math.fmod(a, TWOPI)
    if a <= -math.pi:
        a += TWOPI
    elif a > math.pi:
        a -= TWOPI
    return a


def pentagon_vertices(cx, cy, angle):
    return [
        (
            cx + R * math.cos(angle + k * TWOPI / 5.0),
            cy + R * math.sin(angle + k * TWOPI / 5.0),
        )
        for k in range(5)
    ]


def poly_axes(poly):
    axes = []
    for i in range(5):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
        ux, uy = -(y2 - y1), (x2 - x1)
        n = math.hypot(ux, uy)
        if n > 0:
            axes.append((ux / n, uy / n))
    return axes


def project(poly, ax, ay):
    vals = [x * ax + y * ay for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb, eps=1e-12):
    for ax, ay in poly_axes(pa) + poly_axes(pb):
        amin, amax = project(pa, ax, ay)
        bmin, bmax = project(pb, ax, ay)
        if min(amax, bmax) - max(amin, bmin) <= eps:
            return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i, j in combinations(range(len(polys)), 2):
        if polygons_overlap(polys[i], polys[j]):
            return True
    return False


def enclosing_side(centers, angles):
    if not centers:
        return 0.0
    mx = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            mx = max(mx, abs(vx), abs(vy))
    return 2.0 * mx


def square_violation(centers, angles, s):
    half = 0.5 * s
    v = 0.0
    for (cx, cy), a in zip(centers, angles):
        for x, y in pentagon_vertices(cx, cy, a):
            dx = abs(x) - half
            dy = abs(y) - half
            if dx > 0:
                v += dx * dx
            if dy > 0:
                v += dy * dy
    return v


def pairwise_overlap_penalty(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    pen = 0.0
    for i, j in combinations(range(len(polys)), 2):
        pa, pb = polys[i], polys[j]
        worst = 0.0
        separated = False
        for ax, ay in poly_axes(pa) + poly_axes(pb):
            amin, amax = project(pa, ax, ay)
            bmin, bmax = project(pb, ax, ay)
            gap = max(amin, bmin) - min(amax, bmax)
            if gap >= 0:
                separated = True
                break
            worst = max(worst, -gap)
        if not separated:
            pen += worst * worst
    return pen


def repair_scale(centers, angles):
    """Scale centers radially about origin until all overlaps vanish."""
    if not has_overlap(centers, angles):
        return centers

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.05
    for _ in range(80):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.12
    else:
        return centers

    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi)


def layout_double_lattice_10():
    """
    Hand-shaped 10-pentagon layout inspired by alternating/oriented rows.
    This is intentionally conservative and then later optimized.
    """
    # Two staggered rows of 5 with alternating orientations.
    xs = [-2.00, -1.00, 0.00, 1.00, 2.00]
    y0 = -0.82
    y1 = 0.82
    centers = []
    angles = []

    for i, x in enumerate(xs):
        centers.append((x, y0 + (0.10 if i % 2 else -0.06)))
        angles.append(normalize_angle(math.pi / 2.0 if i % 2 == 0 else -math.pi / 2.0))

    for i, x in enumerate(xs):
        centers.append((x + (0.50 if i % 2 == 0 else -0.50), y1 + (0.06 if i % 2 else -0.10)))
        angles.append(normalize_angle(-math.pi / 2.0 if i % 2 == 0 else math.pi / 2.0))

    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def layout_3_4_3():
    """Three rows: 3 + 4 + 3, with alternating and flipped orientations."""
    centers = []
    angles = []
    ys = [-1.10, 0.0, 1.10]
    counts = [3, 4, 3]
    for r, (y, cnt) in enumerate(zip(ys, counts)):
        width = 1.22 * (cnt - 1)
        offset = -0.5 * width
        for c in range(cnt):
            x = offset + 1.22 * c + (0.61 if (r % 2 == 1 and c % 2 == 0) else 0.0)
            centers.append((x, y + (0.10 if (c % 2) else -0.06)))
            ang = math.pi / 2.0 if ((r + c) % 2 == 0) else -math.pi / 2.0
            if r == 1:
                ang += 0.22 if c in (1, 2) else -0.18
            angles.append(normalize_angle(ang))

    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def layout_spiral_10():
    """A compact non-grid layout that encourages boundary rows to tilt."""
    centers = []
    angles = []
    radii = [0.0, 1.05, 1.05, 1.95, 1.95, 2.70, 2.70, 3.25, 3.25, 3.85]
    theta = [0.0, 0.65, 2.75, 1.65, 4.00, 0.15, 2.15, 5.10, 3.15, 4.65]
    for k in range(10):
        r = radii[k]
        t = theta[k]
        centers.append((r * math.cos(t), 0.72 * r * math.sin(t)))
        ang = (math.pi / 2.0 if k % 2 == 0 else -math.pi / 2.0) + (0.16 if k in (1, 4, 7) else -0.10)
        angles.append(normalize_angle(ang))

    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def initial_families(n):
    if n == 10:
        return [
            layout_double_lattice_10(),
            layout_3_4_3(),
            layout_spiral_10(),
        ]
    # Generic fallback for other n: staggered rows.
    rows = max(1, int(math.ceil(math.sqrt(n))))
    cols = int(math.ceil(n / rows))
    centers = []
    angles = []
    for k in range(n):
        r, c = divmod(k, cols)
        x = (c - (cols - 1) / 2.0) * 1.35 + (0.675 if (r & 1) else 0.0)
        y = (r - (rows - 1) / 2.0) * 1.22
        centers.append((x, y))
        angles.append(normalize_angle(math.pi / 2.0 if ((r + c) & 1) == 0 else -math.pi / 2.0))
    mx = sum(x for x, _ in centers) / n
    my = sum(y for _, y in centers) / n
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def pack_vector(centers, angles):
    v = []
    for (x, y), a in zip(centers, angles):
        v.extend([x, y, a])
    return v


def unpack_vector(v):
    centers = []
    angles = []
    for i in range(0, len(v), 3):
        centers.append((float(v[i]), float(v[i + 1])))
        angles.append(normalize_angle(float(v[i + 2])))
    return centers, angles


def objective_vector(v):
    centers, angles = unpack_vector(v)
    s = enclosing_side(centers, angles)
    pen = pairwise_overlap_penalty(centers, angles)
    sq = square_violation(centers, angles, s)
    # Strongly prioritize feasible non-overlapping packings, then shrink side.
    return 1.0 * s + 3500.0 * pen + 1200.0 * sq


def try_scipy_optimize(centers, angles, seed=0):
    if not SCIPY_AVAILABLE:
        return centers, angles, enclosing_side(centers, angles)

    rng = random.Random(202405 + seed)
    v0 = pack_vector(centers, angles)

    # Small deterministic jitter to escape shallow local minima.
    for i in range(len(v0)):
        if i % 3 == 2:
            v0[i] = normalize_angle(v0[i] + (rng.random() - 0.5) * 0.08)
        else:
            v0[i] += (rng.random() - 0.5) * 0.03

    bounds = []
    for _ in range(len(centers)):
        bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (-math.pi, math.pi)])

    res = minimize(
        objective_vector,
        np.array(v0, dtype=float) if np is not None else v0,
        method="Powell",
        bounds=bounds,
        options=dict(maxiter=2500, xtol=1e-5, ftol=1e-7, disp=False),
    )

    v = res.x.tolist() if np is not None else list(res.x)
    c, a = unpack_vector(v)
    c = repair_scale(c, a)
    s = enclosing_side(c, a)
    return c, a, s


def local_random_search(centers, angles, seed=0, iters=4000):
    rng = random.Random(7000 + seed)
    n = len(centers)
    cur_c = centers[:]
    cur_a = angles[:]
    cur_c = repair_scale(cur_c, cur_a)
    cur_s = enclosing_side(cur_c, cur_a)
    cur_o = objective_vector(pack_vector(cur_c, cur_a))

    best_c = cur_c[:]
    best_a = cur_a[:]
    best_s = cur_s
    best_o = cur_o

    step_xy = 0.22
    step_a = 0.28

    for t in range(iters):
        i = rng.randrange(n)
        cand_c = cur_c[:]
        cand_a = cur_a[:]
        mode = rng.random()

        if mode < 0.50:
            cand_c[i] = (
                cand_c[i][0] + (rng.random() * 2.0 - 1.0) * step_xy,
                cand_c[i][1] + (rng.random() * 2.0 - 1.0) * step_xy,
            )
        elif mode < 0.82:
            cand_a[i] = normalize_angle(cand_a[i] + (rng.random() * 2.0 - 1.0) * step_a)
        else:
            cand_a[i] = normalize_angle(cand_a[i] + math.pi + (rng.random() - 0.5) * 0.12)

        cand_c = repair_scale(cand_c, cand_a)
        cand_o = objective_vector(pack_vector(cand_c, cand_a))
        temp = 0.08 * (1.0 - t / iters) + 0.003

        if cand_o <= cur_o or rng.random() < math.exp(-(cand_o - cur_o) / temp):
            cur_c, cur_a, cur_o = cand_c, cand_a, cand_o
            cur_s = enclosing_side(cur_c, cur_a)
            if not has_overlap(cur_c, cur_a) and cur_s < best_s:
                best_c, best_a, best_s, best_o = cur_c[:], cur_a[:], cur_s, cur_o

        if (t + 1) % 700 == 0:
            step_xy *= 0.78
            step_a *= 0.80

    return best_c, best_a, best_s


def coordinate_refine(centers, angles, seed=0, rounds=5):
    rng = random.Random(90000 + seed)
    n = len(centers)
    best_c = centers[:]
    best_a = angles[:]
    best_s = enclosing_side(best_c, best_a)

    for r in range(rounds):
        improved = False
        for i in range(n):
            base_c = best_c[:]
            base_a = best_a[:]
            local_best = (best_c[:], best_a[:], best_s)

            for _ in range(24):
                c = base_c[:]
                a = base_a[:]
                c[i] = (
                    c[i][0] + (rng.random() * 2.0 - 1.0) * (0.08 / (r + 1)),
                    c[i][1] + (rng.random() * 2.0 - 1.0) * (0.08 / (r + 1)),
                )
                a[i] = normalize_angle(a[i] + (rng.random() * 2.0 - 1.0) * (0.22 / (r + 1)))
                c = repair_scale(c, a)
                if not has_overlap(c, a):
                    s = enclosing_side(c, a)
                    if s < local_best[2]:
                        local_best = (c[:], a[:], s)

            if local_best[2] < best_s:
                best_c, best_a, best_s = local_best
                improved = True

        if not improved:
            break

    return best_c, best_a, best_s


def shrink_to_fit(centers, angles):
    """Binary search a uniform scale-down while preserving feasibility."""
    if not has_overlap(centers, angles):
        base = 1.0
    else:
        centers = repair_scale(centers, angles)
        base = 1.0

    def feasible(lam):
        c = [(x * lam, y * lam) for x, y in centers]
        return (not has_overlap(c, angles)) and square_violation(c, angles, enclosing_side(c, angles)) <= 1e-11

    lo, hi = 0.92, 1.0
    if not feasible(hi):
        return centers, angles, enclosing_side(centers, angles)

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid

    c = [(x * hi, y * hi) for x, y in centers]
    c = repair_scale(c, angles)
    return c, angles, enclosing_side(c, angles)


def pack(n):
    if n <= 0:
        return [], [], 0.0

    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    seeds = [0, 1, 2, 3, 5, 8, 13, 21, 34]
    candidates = []

    for seed in seeds:
        base_sets = initial_families(n)
        for idx, (centers, angles) in enumerate(base_sets):
            centers = repair_scale(centers, angles)

            # Global stochastic improvement.
            c1, a1, s1 = local_random_search(
                centers, angles, seed=seed + 97 * idx + 13 * n, iters=2800 if n == 10 else 1800
            )

            # Coordinate-wise refinement.
            c2, a2, s2 = coordinate_refine(
                c1, a1, seed=seed + 131 * idx + 17 * n, rounds=6 if n == 10 else 4
            )

            # SciPy polish if available.
            c3, a3, s3 = try_scipy_optimize(c2, a2, seed=seed + 19 * idx + n)

            # Another refinement pass after optimizer noise.
            c4, a4, s4 = coordinate_refine(
                c3, a3, seed=seed + 173 * idx + 29 * n, rounds=4 if n == 10 else 3
            )
            c4 = repair_scale(c4, a4)
            s4 = enclosing_side(c4, a4)

            candidates.append((c4, a4, s4))

    best_c, best_a, best_s = min(candidates, key=lambda t: t[2])

    # Final shrink / cleanup.
    best_c = repair_scale(best_c, best_a)
    best_c, best_a, best_s = shrink_to_fit(best_c, best_a)

    # Final exact safety check.
    if has_overlap(best_c, best_a):
        best_c = repair_scale(best_c, best_a)
        best_s = enclosing_side(best_c, best_a)

    return best_c, best_a, best_s
