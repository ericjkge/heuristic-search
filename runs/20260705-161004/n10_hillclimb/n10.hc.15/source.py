"""Packing regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
Container: point p is inside iff max(|px|, |py|) <= s/2.

This version replaces the previous heuristic with a more geometry-driven optimizer:

- exact convex-polygon overlap checks via SAT,
- a robust penalty function with soft square bounds,
- deterministic construction from several hand-tuned motif families,
- multi-start local optimization using scipy if available, otherwise a pure-Python fallback,
- final coordinate refinement and shrinking.

The program keeps the required signature and returns valid packings.
"""

import math
import random

SIDE = 1.0
TWOPI = 2.0 * math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


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
    m = len(polys)
    for i in range(m):
        for j in range(i + 1, m):
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


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def shrink_to_fit(centers, angles):
    """Scale centers toward origin until all pentagons fit and do not overlap."""
    if not centers:
        return centers, angles
    if not has_overlap(centers, angles):
        return centers, angles

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.05
    for _ in range(90):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.12
    else:
        return centers, angles

    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi), angles


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


def overlap_penalty(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    n = len(polys)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pa, pb = polys[i], polys[j]
            sep = False
            worst = 0.0
            for ax, ay in poly_axes(pa) + poly_axes(pb):
                amin, amax = project(pa, ax, ay)
                bmin, bmax = project(pb, ax, ay)
                gap = max(amin, bmin) - min(amax, bmax)
                if gap >= 0:
                    sep = True
                    break
                worst = max(worst, -gap)
            if not sep:
                pen += worst * worst
    return pen


def center_penalty(centers):
    return sum(x * x + y * y for x, y in centers)


def objective_vec(x, n, w_overlap=4000.0, w_square=3000.0, w_center=0.02):
    centers = [(x[3 * i], x[3 * i + 1]) for i in range(n)]
    angles = [normalize_angle(x[3 * i + 2]) for i in range(n)]
    s = enclosing_side(centers, angles)
    return s + w_overlap * overlap_penalty(centers, angles) + w_square * square_violation(centers, angles, s) + w_center * center_penalty(centers)


def pack_from_vec(x, n):
    centers = [(float(x[3 * i]), float(x[3 * i + 1])) for i in range(n)]
    angles = [normalize_angle(float(x[3 * i + 2])) for i in range(n)]
    centers, angles = shrink_to_fit(centers, angles)
    s = enclosing_side(centers, angles)
    return centers, angles, s


def motif_layout(n, kind, rng):
    """Construct several deterministic seed patterns with opposite orientations."""
    centers = []
    angles = []

    if n == 1:
        return [(0.0, 0.0)], [0.0]

    # Base scale chosen to start feasible after one shrink step.
    if kind == 0:
        # Two-row stagger with alternating opposite angles.
        cols = int(math.ceil(n / 2.0))
        dx = 1.18
        dy = 1.05
        for k in range(n):
            r, c = divmod(k, cols)
            x = (c - (cols - 1) / 2.0) * dx + (0.5 * dx if r == 1 else 0.0)
            y = (r - 0.5) * dy
            a = (math.pi / 5.0 if (k % 2 == 0) else -math.pi / 5.0) + (rng.random() - 0.5) * 0.07
            centers.append((x, y))
            angles.append(normalize_angle(a))

    elif kind == 1:
        # Three-row diagonal lattice.
        rows = 3 if n >= 7 else 2
        cols = int(math.ceil(n / rows))
        dx = 1.10
        dy = 0.96
        k = 0
        for r in range(rows):
            for c in range(cols):
                if k >= n:
                    break
                x = (c - (cols - 1) / 2.0) * dx + ((r % 2) * 0.5 * dx)
                y = (r - (rows - 1) / 2.0) * dy
                a = (math.pi if (r + c) % 2 else 0.0) + (rng.random() - 0.5) * 0.06
                centers.append((x, y))
                angles.append(normalize_angle(a))
                k += 1

    elif kind == 2:
        # Fivefold angular motif around origin.
        rad = 1.05
        for k in range(n):
            t = TWOPI * k / max(5, n)
            if n > 5:
                ring = 0.0 if k < 5 else 0.55
            else:
                ring = 0.0
            x = (rad + ring) * math.cos(t)
            y = (rad + ring) * math.sin(t)
            a = normalize_angle(t + math.pi / 2.0 + (math.pi if k % 2 else 0.0))
            centers.append((x, y))
            angles.append(a)

    else:
        # Compact rectangular seed with opposite orientations.
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        dx = 1.06
        dy = 1.00
        k = 0
        for r in range(rows):
            for c in range(cols):
                if k >= n:
                    break
                x = (c - (cols - 1) / 2.0) * dx
                y = (r - (rows - 1) / 2.0) * dy
                if r & 1:
                    x += 0.5 * dx
                a = (0.0 if (k & 1) == 0 else math.pi) + (rng.random() - 0.5) * 0.05
                centers.append((x, y))
                angles.append(normalize_angle(a))
                k += 1

    mx = sum(x for x, _ in centers) / n
    my = sum(y for _, y in centers) / n
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def eval_solution(centers, angles):
    centers, angles = shrink_to_fit(centers, angles)
    s = enclosing_side(centers, angles)
    return centers, angles, s


def local_refine(centers, angles, seed=0, iters=1200):
    rng = random.Random(202501 + seed)
    n = len(centers)
    c = centers[:]
    a = angles[:]
    best_c, best_a, best_s = eval_solution(c, a)
    cur = best_s

    step_xy = max(0.015, best_s * 0.012)
    step_a = 0.18

    for t in range(iters):
        i = rng.randrange(n)
        oldc = c[i]
        olda = a[i]

        move = rng.random()
        if move < 0.45:
            dx = (rng.random() * 2 - 1) * step_xy
            dy = (rng.random() * 2 - 1) * step_xy
            c[i] = (oldc[0] + dx, oldc[1] + dy)
        elif move < 0.85:
            da = (rng.random() * 2 - 1) * step_a
            a[i] = normalize_angle(olda + da)
        else:
            a[i] = normalize_angle(olda + math.pi + (rng.random() - 0.5) * 0.1)

        tc, ta = shrink_to_fit(c, a)
        s = enclosing_side(tc, ta)
        pen = overlap_penalty(tc, ta) + square_violation(tc, ta, s)
        score = s + 1500.0 * pen + 0.02 * center_penalty(tc)

        temp = 0.03 * (1.0 - t / iters) + 0.002
        accept = score <= cur or rng.random() < math.exp(-(score - cur) / temp)
        if accept:
            c, a = tc, ta
            cur = score
            if has_overlap(c, a):
                continue
            if s < best_s:
                best_c, best_a, best_s = c[:], a[:], s
        else:
            c[i] = oldc
            a[i] = olda

        if (t + 1) % 300 == 0:
            step_xy *= 0.82
            step_a *= 0.88

    return best_c, best_a, best_s


def scipy_refine(centers, angles, n, seed=0):
    if minimize is None:
        return centers, angles, enclosing_side(centers, angles)

    x0 = []
    for (x, y), ang in zip(centers, angles):
        x0.extend([x, y, ang])
    x0 = np.array(x0, dtype=float)

    def f(x):
        return objective_vec(x, n)

    options = dict(maxiter=1200, fatol=1e-10, xatol=1e-10, adaptive=True)

    res = minimize(f, x0, method="Nelder-Mead", options=options)
    x = res.x if res.success or res.x is not None else x0
    return pack_from_vec(x, n)


def coordinate_sweep(centers, angles, rounds=4):
    n = len(centers)
    best_c, best_a, best_s = eval_solution(centers, angles)
    rng = random.Random(99173 + n)
    for r in range(rounds):
        improved = False
        for i in range(n):
            base_c = best_c[:]
            base_a = best_a[:]
            local_best = (best_c, best_a, best_s)
            for _ in range(12):
                c = base_c[:]
                a = base_a[:]
                c[i] = (c[i][0] + (rng.random() * 2 - 1) * 0.04 / (r + 1),
                        c[i][1] + (rng.random() * 2 - 1) * 0.04 / (r + 1))
                a[i] = normalize_angle(a[i] + (rng.random() * 2 - 1) * 0.14 / (r + 1))
                c, a = shrink_to_fit(c, a)
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


def pack(n):
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [0.0]
        return centers, angles, enclosing_side(centers, angles)

    seeds = [0, 1, 2, 3, 5, 8, 13, 21, 34]
    kinds = [0, 1, 2, 3]

    best = None

    for seed in seeds:
        rng = random.Random(1000003 + 97 * n + seed)
        for kind in kinds:
            centers, angles = motif_layout(n, kind, rng)
            centers, angles = shrink_to_fit(centers, angles)
            centers, angles, _ = local_refine(
                centers, angles, seed=seed + 17 * kind,
                iters=900 if n <= 12 else 650
            )
            centers, angles, s = coordinate_sweep(centers, angles, rounds=3)
            centers, angles = shrink_to_fit(centers, angles)

            if np is not None and minimize is not None and n <= 18:
                # Use scipy only on moderate sizes; it often helps tighten the boundary.
                sc, sa, ss = scipy_refine(centers, angles, n, seed=seed)
                if ss < s and not has_overlap(sc, sa):
                    centers, angles, s = sc, sa, ss

            if best is None or s < best[2]:
                best = (centers[:], angles[:], s)

    centers, angles, s = best

    # Final deterministic tightening.
    centers, angles, s = coordinate_sweep(centers, angles, rounds=5)
    centers, angles = shrink_to_fit(centers, angles)
    s = enclosing_side(centers, angles)

    # Last validation and repair if needed.
    if has_overlap(centers, angles):
        centers, angles = shrink_to_fit(centers, angles)
        s = enclosing_side(centers, angles)

    return centers, angles, s
