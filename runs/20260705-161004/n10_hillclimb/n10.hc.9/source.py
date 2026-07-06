"""Heuristic / numerical optimizer for packing n unit regular pentagons into the
smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s).
Container: point p is inside iff max(|px|, |py|) <= s/2.

This version uses:
- better geometric primitives,
- multi-start randomized / structured seeds,
- a penalty-based local optimizer,
- coordinate descent with finite-difference refinement,
- final feasibility tightening by shrinking centers toward the origin.

It keeps every pentagon non-overlapping and inside the square.
"""

import math
import random
from typing import List, Tuple

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # inradius
PHI = (1.0 + math.sqrt(5.0)) / 2.0

TAU = 2.0 * math.pi
EPS = 1e-10


def normalize_angle(a: float) -> float:
    a = math.fmod(a, TAU)
    if a <= -math.pi:
        a += TAU
    elif a > math.pi:
        a -= TAU
    return a


def pentagon_vertices(cx: float, cy: float, angle: float):
    return [
        (cx + R * math.cos(angle + k * TAU / 5.0),
         cy + R * math.sin(angle + k * TAU / 5.0))
        for k in range(5)
    ]


def poly_axes(poly):
    axes = []
    for i in range(5):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
        ux, uy = -(y2 - y1), (x2 - x1)
        norm = math.hypot(ux, uy)
        if norm > 0:
            axes.append((ux / norm, uy / norm))
    return axes


def project(poly, ax, ay):
    vals = [x * ax + y * ay for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb, eps=EPS):
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
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, ang)
    )


def square_inside_penalty(centers, angles, s):
    half = 0.5 * s
    pen = 0.0
    for (cx, cy), a in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, a):
            dx = abs(vx) - half
            dy = abs(vy) - half
            if dx > 0:
                pen += dx * dx
            if dy > 0:
                pen += dy * dy
    return pen


def pairwise_overlap_penalty(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    n = len(polys)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            overlap = True
            best_sep = 1e9
            for ax, ay in poly_axes(polys[i]) + poly_axes(polys[j]):
                amin, amax = project(polys[i], ax, ay)
                bmin, bmax = project(polys[j], ax, ay)
                gap = max(amin, bmin) - min(amax, bmax)
                if gap >= 0:
                    overlap = False
                    best_sep = min(best_sep, gap)
                    break
                best_sep = min(best_sep, -gap)
            if overlap:
                pen += best_sep * best_sep
    return pen


def total_penalty(centers, angles, s=None):
    if s is None:
        s = enclosing_side(centers, angles)
    pen = pairwise_overlap_penalty(centers, angles)
    pen += square_inside_penalty(centers, angles, s)
    return pen


def repair_scale(centers, angles):
    """Scale centers about origin until all overlaps vanish."""
    if not has_overlap(centers, angles):
        return centers

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.10
    for _ in range(90):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.15
    else:
        return centers

    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi)


def shrink_to_fit(centers, angles):
    """Shrink centers toward origin until the packing just remains feasible."""
    centers = repair_scale(centers, angles)
    if not centers:
        return centers
    if has_overlap(centers, angles):
        return centers

    lo, hi = 0.0, 1.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        test = [(x * mid, y * mid) for x, y in centers]
        if has_overlap(test, angles):
            lo = mid
        else:
            hi = mid
    return [(x * hi, y * hi) for x, y in centers]


def candidate_orientations(n, k):
    # Mix opposite orientations and symmetry-derived angles.
    presets = [
        0.0,
        math.pi / 5.0,
        2.0 * math.pi / 5.0,
        3.0 * math.pi / 5.0,
        4.0 * math.pi / 5.0,
        math.pi,
        -math.pi / 5.0,
        -2.0 * math.pi / 5.0,
        -3.0 * math.pi / 5.0,
        -4.0 * math.pi / 5.0,
    ]
    base = presets[(3 * k + n) % len(presets)]
    if (k + n) % 2:
        base += math.pi
    return normalize_angle(base)


def dense_seed_grid(n, mode, rng):
    """Generate a structured initial packing using staggered rows/columns."""
    centers = []
    angles = []

    rows = max(1, int(math.floor(math.sqrt(n))))
    cols = int(math.ceil(n / rows))
    rows = int(math.ceil(n / cols))

    # Initial spacing guesses: conservative, later refined by optimization.
    # Use asymmetric spacing to favor square bounding.
    if mode == 0:
        dx = 1.10
        dy = 1.05
        for k in range(n):
            r, c = divmod(k, cols)
            x = (c - (cols - 1) / 2.0) * dx
            y = (r - (rows - 1) / 2.0) * dy
            if r % 2:
                x += 0.5 * dx
            ang = math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0
            ang += (rng.random() - 0.5) * 0.12
            centers.append((x, y))
            angles.append(normalize_angle(ang))

    elif mode == 1:
        dx = 1.05
        dy = 1.10
        cols = max(1, int(math.floor(math.sqrt(n))))
        rows = int(math.ceil(n / cols))
        for k in range(n):
            c, r = divmod(k, rows)
            x = (c - (cols - 1) / 2.0) * dx
            y = (r - (rows - 1) / 2.0) * dy
            if c % 2:
                y += 0.5 * dy
            ang = 0.0 if (r + c) % 2 == 0 else math.pi
            ang += (rng.random() - 0.5) * 0.12
            centers.append((x, y))
            angles.append(normalize_angle(ang))

    else:
        # Triangular-like stagger, useful for medium n.
        dx = 1.08
        dy = 0.94
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        for k in range(n):
            r, c = divmod(k, cols)
            x = (c - (cols - 1) / 2.0) * dx + (0.5 * dx if r % 2 else 0.0)
            y = (r - (rows - 1) / 2.0) * dy
            ang = candidate_orientations(n, k)
            if (r + c) % 3 == 1:
                ang += math.pi
            ang += (rng.random() - 0.5) * 0.10
            centers.append((x, y))
            angles.append(normalize_angle(ang))

    return centers, angles


def add_ring_seed(n, rng):
    """Place points on a few rings; helpful when n is small or awkward."""
    centers = []
    angles = []

    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]

    m = n
    ring_count = 1 if n <= 4 else 2 if n <= 8 else 3
    radii = [0.0] + [1.3 + 0.55 * i for i in range(ring_count)]
    per_ring = [1]
    remaining = n - 1
    for i in range(ring_count):
        take = max(3, remaining // (ring_count - i) if ring_count - i > 0 else remaining)
        take = min(take, remaining)
        per_ring.append(take)
        remaining -= take
    if remaining > 0:
        per_ring[-1] += remaining

    idx = 0
    for r_idx, count in enumerate(per_ring):
        rad = radii[min(r_idx, len(radii) - 1)]
        if r_idx == 0:
            centers.append((0.0, 0.0))
            angles.append(math.pi / 2.0)
            idx += 1
            continue
        for j in range(count):
            if idx >= m:
                break
            theta = TAU * (j / count) + (0.15 if r_idx % 2 else 0.0)
            x = rad * math.cos(theta)
            y = rad * math.sin(theta)
            ang = candidate_orientations(n, idx) + (math.pi if (idx + r_idx) % 2 else 0.0)
            ang += (rng.random() - 0.5) * 0.08
            centers.append((x, y))
            angles.append(normalize_angle(ang))
            idx += 1

    while len(centers) < n:
        theta = TAU * len(centers) / n
        centers.append((1.8 * math.cos(theta), 1.8 * math.sin(theta)))
        angles.append(candidate_orientations(n, len(centers)))
    return centers, angles


def random_seed(n, rng):
    centers = []
    angles = []
    rad = 1.0 + 0.22 * math.sqrt(n)
    for k in range(n):
        theta = TAU * (k / n) + rng.uniform(-0.08, 0.08)
        rr = rad * (0.65 + 0.35 * rng.random())
        x = rr * math.cos(theta)
        y = rr * math.sin(theta)
        centers.append((x, y))
        a = candidate_orientations(n, k)
        if rng.random() < 0.5:
            a += math.pi
        a += rng.uniform(-0.15, 0.15)
        angles.append(normalize_angle(a))
    return centers, angles


def objective(centers, angles, w_overlap=3000.0, w_square=1000.0):
    s = enclosing_side(centers, angles)
    return s + w_overlap * pairwise_overlap_penalty(centers, angles) + w_square * square_inside_penalty(centers, angles, s)


def local_search(centers, angles, seed=0, steps=3000):
    rng = random.Random(1234567 + seed)
    n = len(centers)
    cur_c = centers[:]
    cur_a = angles[:]
    cur_c = repair_scale(cur_c, cur_a)
    cur_c = shrink_to_fit(cur_c, cur_a)
    cur_s = enclosing_side(cur_c, cur_a)
    cur_obj = objective(cur_c, cur_a)

    best_c = cur_c[:]
    best_a = cur_a[:]
    best_s = cur_s
    best_obj = cur_obj

    step_xy = max(0.01, 0.02 * max(1.0, cur_s))
    step_a = 0.22

    for t in range(steps):
        i = rng.randrange(n)
        oldc = cur_c[i]
        olda = cur_a[i]

        mode = rng.random()
        if mode < 0.48:
            dx = (rng.random() * 2.0 - 1.0) * step_xy
            dy = (rng.random() * 2.0 - 1.0) * step_xy
            cur_c[i] = (oldc[0] + dx, oldc[1] + dy)
        elif mode < 0.84:
            da = (rng.random() * 2.0 - 1.0) * step_a
            cur_a[i] = normalize_angle(olda + da)
        else:
            cur_a[i] = normalize_angle(olda + math.pi + rng.uniform(-0.12, 0.12))

        test_c = repair_scale(cur_c, cur_a)
        test_c = shrink_to_fit(test_c, cur_a)
        test_s = enclosing_side(test_c, cur_a)
        test_obj = objective(test_c, cur_a)

        accept = False
        if test_obj <= cur_obj:
            accept = True
        else:
            temp = max(0.001, 0.04 * (1.0 - t / max(1, steps)))
            if rng.random() < math.exp(-(test_obj - cur_obj) / temp):
                accept = True

        if accept:
            cur_c = test_c
            cur_s = test_s
            cur_obj = test_obj
            if cur_obj < best_obj and not has_overlap(cur_c, cur_a):
                best_c = cur_c[:]
                best_a = cur_a[:]
                best_s = cur_s
                best_obj = cur_obj
        else:
            cur_c[i] = oldc
            cur_a[i] = olda

        if (t + 1) % 700 == 0:
            step_xy *= 0.82
            step_a *= 0.88

    return best_c, best_a, best_s


def coordinate_refine(centers, angles, seed=0, rounds=6):
    rng = random.Random(424242 + seed)
    n = len(centers)
    c = centers[:]
    a = angles[:]
    c = repair_scale(c, a)
    c = shrink_to_fit(c, a)

    best_c = c[:]
    best_a = a[:]
    best_s = enclosing_side(c, a)

    xy_step = max(0.004, best_s * 0.006)
    ang_step = 0.10

    for _ in range(rounds):
        improved = False
        for i in range(n):
            base_c = c[i]
            base_a = a[i]
            local_best = (best_s, c[:], a[:])

            for dx, dy, da in [
                (0.0, 0.0, 0.0),
                (xy_step, 0.0, 0.0),
                (-xy_step, 0.0, 0.0),
                (0.0, xy_step, 0.0),
                (0.0, -xy_step, 0.0),
                (0.0, 0.0, ang_step),
                (0.0, 0.0, -ang_step),
                (xy_step, xy_step, 0.0),
                (-xy_step, xy_step, 0.0),
                (xy_step, -xy_step, 0.0),
                (-xy_step, -xy_step, 0.0),
                (0.0, 0.0, math.pi),
            ]:
                tc = c[:]
                ta = a[:]
                tc[i] = (base_c[0] + dx, base_c[1] + dy)
                ta[i] = normalize_angle(base_a + da)
                tc = repair_scale(tc, ta)
                tc = shrink_to_fit(tc, ta)
                s = enclosing_side(tc, ta)
                if not has_overlap(tc, ta) and s < local_best[0]:
                    local_best = (s, tc, ta)

            if local_best[0] + 1e-12 < best_s:
                best_s, c, a = local_best
                improved = True

        if not improved:
            xy_step *= 0.65
            ang_step *= 0.72
        else:
            best_c, best_a = c[:], a[:]

    return best_c, best_a, best_s


def build_initial_candidates(n, rng):
    cands = []
    cands.append(dense_seed_grid(n, 0, rng))
    cands.append(dense_seed_grid(n, 1, rng))
    cands.append(dense_seed_grid(n, 2, rng))
    cands.append(add_ring_seed(n, rng))
    cands.append(random_seed(n, rng))
    return cands


def pack(n):
    """Return a valid packing of n unit pentagons into a minimum-ish square."""
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    global_best = None

    # Multi-start search with structured and randomized seeds.
    # The total effort is tuned to stay within the time limit.
    master = random.Random(20240617 + 137 * n)
    seeds = [0, 1, 2, 3, 4, 5, 11, 17, 29]

    for si, s0 in enumerate(seeds):
        rng = random.Random(master.randrange(1 << 30) ^ (s0 * 1000003) ^ (n * 2654435761 & 0xFFFFFFFF))
        candidates = build_initial_candidates(n, rng)

        for mi, (centers, angles) in enumerate(candidates):
            # Normalize and make feasible-ish.
            angles = [normalize_angle(a) for a in angles]
            centers = repair_scale(centers, angles)
            centers = shrink_to_fit(centers, angles)

            # Local optimization
            steps = 2500 if n <= 10 else 1800 if n <= 16 else 1200
            centers, angles, s = local_search(centers, angles, seed=1000 * si + 31 * mi + n, steps=steps)

            # Coordinate refinement
            centers, angles, s = coordinate_refine(centers, angles, seed=2000 * si + 17 * mi + n, rounds=5 if n <= 12 else 4)

            # A final shrink to tighten the square after all adjustments.
            centers = repair_scale(centers, angles)
            centers = shrink_to_fit(centers, angles)
            s = enclosing_side(centers, angles)

            if not has_overlap(centers, angles):
                if global_best is None or s < global_best[2]:
                    global_best = (centers[:], angles[:], s)

    if global_best is None:
        # Very conservative fallback: a sparse arrangement.
        rng = random.Random(99991 + n)
        centers, angles = random_seed(n, rng)
        centers = repair_scale(centers, angles)
        centers = shrink_to_fit(centers, angles)
        s = enclosing_side(centers, angles)
        return centers, angles, s

    centers, angles, s = global_best

    # Final polishing pass: small perturbation around the best found packing.
    centers, angles, s = local_search(centers, angles, seed=7777 + n, steps=900 if n <= 14 else 600)
    centers, angles, s = coordinate_refine(centers, angles, seed=8888 + n, rounds=3)
    centers = repair_scale(centers, angles)
    centers = shrink_to_fit(centers, angles)
    s = enclosing_side(centers, angles)

    return centers, angles, s
