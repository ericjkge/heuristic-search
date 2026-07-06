"""Seed program: pack n unit regular pentagons into a square, minimizing side s.

Contract: pack(n) -> (centers, angles, s). Container = origin-centered
axis-aligned square, side s: point p inside iff max(|px|, |py|) <= s/2.

This rewrite uses a hand-tuned constructive search plus local numerical
improvement. It focuses on the n=10 case, but works for general n.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius ~ 0.8507
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # ~ 0.6882
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # widest extent (diagonal) ~ 1.618
HEIGHT = R + APOTHEM                              # point-up bounding height ~ 1.539

OVERLAP_EPS = 1e-9
BOUND_EPS = 1e-10


def pentagon_vertices(cx, cy, angle):
    return [
        (cx + R * math.cos(angle + k * 2.0 * math.pi / 5.0),
         cy + R * math.sin(angle + k * 2.0 * math.pi / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, ang)
    )


def poly_edges(poly):
    return list(zip(poly, poly[1:] + poly[:1]))


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def point_in_convex(poly, p):
    # inclusive point-in-convex-polygon
    sign = None
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        c = cross(sub(b, a), sub(p, a))
        if abs(c) <= 1e-12:
            continue
        s = c > 0
        if sign is None:
            sign = s
        elif sign != s:
            return False
    return True


def segment_intersect(a, b, c, d):
    def orient(p, q, r):
        return cross(sub(q, p), sub(r, p))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if (abs(o1) <= 1e-12 and min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and
            min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12):
        return True
    if (abs(o2) <= 1e-12 and min(a[0], b[0]) - 1e-12 <= d[0] <= max(a[0], b[0]) + 1e-12 and
            min(a[1], b[1]) - 1e-12 <= d[1] <= max(a[1], b[1]) + 1e-12):
        return True
    if (abs(o3) <= 1e-12 and min(c[0], d[0]) - 1e-12 <= a[0] <= max(c[0], d[0]) + 1e-12 and
            min(c[1], d[1]) - 1e-12 <= a[1] <= max(c[1], d[1]) + 1e-12):
        return True
    if (abs(o4) <= 1e-12 and min(c[0], d[0]) - 1e-12 <= b[0] <= max(c[0], d[0]) + 1e-12 and
            min(c[1], d[1]) - 1e-12 <= b[1] <= max(c[1], d[1]) + 1e-12):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def pentagons_overlap(pa, pb):
    # Proper overlap or touching counts as overlap (we need strict separation).
    for poly in (pa, pb):
        for i in range(5):
            a1, a2 = poly[i], poly[(i + 1) % 5]
            for j in range(5):
                b1, b2 = (pb if poly is pa else pa)[j], (pb if poly is pa else pa)[(j + 1) % 5]
                if segment_intersect(a1, a2, b1, b2):
                    return True
    if point_in_convex(pa, pb[0]) or point_in_convex(pb, pa[0]):
        return True
    return False


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if pentagons_overlap(polys[i], polys[j]):
                return True
    return False


def inside_square(centers, angles, s):
    half = s / 2.0 + 1e-12
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            if abs(vx) > half or abs(vy) > half:
                return False
    return True


def repair_dilate(centers, angles):
    # Dilate about origin until all overlaps vanish.
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.1
    for _ in range(60):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.2
    else:
        return centers

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def neighbor_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def objective(centers, angles):
    return enclosing_side(centers, angles)


def constraints_score(centers, angles):
    # Nonnegative penalty: overlaps + boundary violations
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    pen = 0.0

    # Pairwise separation by SAT
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            pa, pb = polys[i], polys[j]
            min_sep = float("inf")
            for poly in (pa, pb):
                for k in range(5):
                    (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % 5]
                    ux, uy = -(y2 - y1), (x2 - x1)
                    amin = min(x * ux + y * uy for x, y in pa)
                    amax = max(x * ux + y * uy for x, y in pa)
                    bmin = min(x * ux + y * uy for x, y in pb)
                    bmax = max(x * ux + y * uy for x, y in pb)
                    norm = math.hypot(ux, uy)
                    sep = min(amax, bmax) - max(amin, bmin)
                    min_sep = min(min_sep, sep / norm)
            if min_sep > OVERLAP_EPS:
                pen += min_sep * min_sep * 1e3
            elif min_sep > -OVERLAP_EPS:
                pen += (min_sep + OVERLAP_EPS) ** 2 * 1e2

    half = objective(centers, angles) / 2.0
    for poly in polys:
        for x, y in poly:
            dx = max(0.0, abs(x) - half)
            dy = max(0.0, abs(y) - half)
            pen += (dx * dx + dy * dy) * 1e4
    return pen


def random_initial_10():
    # Two opposite-orientation motifs with slight staggering.
    # Geometry inspired by dense finite packings: alternating flips and boundary tilt.
    angles = []
    centers = []

    # base motifs
    motif = [
        (-1.65, -0.95, math.pi / 2.0 + 0.08),
        (-0.35, -1.05, -math.pi / 2.0 - 0.06),
        (1.05, -0.95, math.pi / 2.0 + 0.03),
        (-1.10, 0.15, -math.pi / 2.0 + 0.08),
        (0.35, 0.10, math.pi / 2.0 - 0.02),
        (1.65, 0.05, -math.pi / 2.0 + 0.06),
        (-1.45, 1.15, math.pi / 2.0 - 0.10),
        (-0.05, 1.05, -math.pi / 2.0 + 0.02),
        (1.25, 1.10, math.pi / 2.0 + 0.04),
        (0.00, -2.00, -math.pi / 2.0 - 0.05),
    ]
    for x, y, a in motif:
        centers.append((x, y))
        angles.append(a)

    # Center and scale into a reasonable region
    mx = sum(x for x, y in centers) / len(centers)
    my = sum(y for x, y in centers) / len(centers)
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def build_grid(n):
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pitch_x = WIDTH * 0.84
    pitch_y = HEIGHT * 0.84
    centers, angles = [], []
    for k in range(n):
        i, j = k % cols, k // cols
        x = (i - (cols - 1) / 2.0) * pitch_x
        y = (j - (rows - 1) / 2.0) * pitch_y
        centers.append((x, y))
        angles.append(math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0)
    return centers, angles


def local_search(centers, angles, steps=5000, seed=0):
    rng = random.Random(seed)
    best_c = [list(p) for p in centers]
    best_a = list(angles)

    def pack_score(c, a):
        return objective(c, a) + 10.0 * constraints_score(c, a)

    best = pack_score(best_c, best_a)
    temp = 0.3

    for t in range(steps):
        c = [p[:] for p in best_c]
        a = best_a[:]

        i = rng.randrange(len(c))
        move = temp * (0.5 + rng.random())
        if rng.random() < 0.55:
            c[i][0] += rng.uniform(-move, move)
            c[i][1] += rng.uniform(-move, move)
        else:
            a[i] += rng.uniform(-0.35, 0.35) * move

        # occasional global perturbation
        if rng.random() < 0.08:
            for j in range(len(c)):
                if rng.random() < 0.25:
                    c[j][0] += rng.uniform(-0.05, 0.05)
                    c[j][1] += rng.uniform(-0.05, 0.05)
                    a[j] += rng.uniform(-0.03, 0.03)

        # recentre
        mx = sum(p[0] for p in c) / len(c)
        my = sum(p[1] for p in c) / len(c)
        c = [(x - mx, y - my) for x, y in c]

        sc = pack_score(c, a)
        if sc < best or rng.random() < math.exp((best - sc) / max(1e-6, temp)):
            best = sc
            best_c, best_a = c, a

        temp *= 0.9994

    best_c = repair_dilate(best_c, best_a)
    return best_c, best_a


def refine_boundary(centers, angles, iters=2500):
    # Deterministic coordinate descent on bounding side with overlap penalties.
    c = [list(p) for p in centers]
    a = list(angles)

    def score():
        return objective(c, a) + 50.0 * constraints_score(c, a)

    best = score()
    step_xy = 0.08
    step_a = 0.04

    for _ in range(iters):
        improved = False
        for i in range(len(c)):
            for dx, dy, da in (
                (step_xy, 0.0, 0.0), (-step_xy, 0.0, 0.0),
                (0.0, step_xy, 0.0), (0.0, -step_xy, 0.0),
                (0.0, 0.0, step_a), (0.0, 0.0, -step_a),
            ):
                old = (c[i][0], c[i][1], a[i])
                c[i][0] += dx
                c[i][1] += dy
                a[i] += da

                mx = sum(p[0] for p in c) / len(c)
                my = sum(p[1] for p in c) / len(c)
                c2 = [(x - mx, y - my) for x, y in c]
                c_old = c
                c = [list(p) for p in c2]

                sc = score()
                if sc + 1e-12 < best:
                    best = sc
                    improved = True
                    break

                c = c_old
                c[i][0], c[i][1], a[i] = old
            if improved:
                break

        if not improved:
            step_xy *= 0.95
            step_a *= 0.95
            if step_xy < 1e-4:
                break

    c = repair_dilate([(x, y) for x, y in c], a)
    return c, a


def pack(n):
    """Return a valid packing of n unit regular pentagons."""
    if n <= 0:
        return [], [], 0.0

    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0], HEIGHT

    if n == 10:
        # Multi-start over a bespoke 10-pentagon motif and a staggered grid.
        candidates = []

        c0, a0 = random_initial_10()
        candidates.append((c0, a0))

        c1, a1 = build_grid(10)
        candidates.append((c1, a1))

        # A looser hand-built arrangement with opposite orientations.
        c2 = [
            (-1.55, -1.10), (-0.40, -1.25), (0.95, -1.15), (1.55, -0.15), (0.10, -0.10),
            (-1.35, 0.95), (-0.10, 1.10), (1.20, 0.95), (1.55, 1.45), (-0.85, 1.65),
        ]
        a2 = [
            math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0,
            -math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0,
        ]
        candidates.append((c2, a2))

        best = None
        best_s = float("inf")

        for idx, (cc, aa) in enumerate(candidates):
            cc = repair_dilate(cc, aa)
            cc, aa = local_search(cc, aa, steps=7000, seed=12345 + idx)
            cc, aa = refine_boundary(cc, aa, iters=1800)
            s = enclosing_side(cc, aa)
            if inside_square(cc, aa, s) and not has_overlap(cc, aa) and s < best_s:
                best = (cc, aa, s)
                best_s = s

        if best is None:
            # Fallback to a safe construction.
            cc, aa = build_grid(10)
            cc = repair_dilate(cc, aa)
            return cc, aa, enclosing_side(cc, aa)

        return best

    # General fallback: staggered grid with alternating orientations, then local improvement.
    c, a = build_grid(n)
    c = repair_dilate(c, a)
    c, a = local_search(c, a, steps=max(2000, 400 * n), seed=999 + n)
    c, a = refine_boundary(c, a, iters=max(500, 120 * n))
    s = enclosing_side(c, a)
    return c, a, s
