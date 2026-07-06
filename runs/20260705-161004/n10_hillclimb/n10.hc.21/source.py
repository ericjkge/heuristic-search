"""Packing regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
Container: point p is inside iff max(|px|, |py|) <= s/2.

This rewrite uses a more structure-driven search for n=10:
- exact regular pentagon geometry,
- robust collision and enclosure checks,
- several hand-crafted motifs motivated by opposite-orientation / double-lattice behavior,
- local optimization in a reduced parameterization for the 10-gon case,
- deterministic multi-start refinement.

For n=10 the code focuses on a few strong symmetric/asymmetric templates and then
optimizes them directly under exact geometry constraints.
"""

import math
import random

SIDE = 1.0
TWOPI = 2.0 * math.pi
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
PHI = (1.0 + math.sqrt(5.0)) / 2.0


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
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def enclosing_side(centers, angles):
    if not centers:
        return 0.0
    mx = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            ax = abs(vx)
            ay = abs(vy)
            if ax > mx:
                mx = ax
            if ay > mx:
                mx = ay
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
    for i in range(len(polys)):
        pa = polys[i]
        for j in range(i + 1, len(polys)):
            pb = polys[j]
            worst = 0.0
            separated = False
            for ax, ay in poly_axes(pa) + poly_axes(pb):
                amin, amax = project(pa, ax, ay)
                bmin, bmax = project(pb, ax, ay)
                gap = max(amin, bmin) - min(amax, bmax)
                if gap >= 0:
                    separated = True
                    break
                if -gap > worst:
                    worst = -gap
            if not separated:
                pen += worst * worst
    return pen


def repair_scale(centers, angles):
    """Scale centers about origin until all overlaps vanish."""
    if not has_overlap(centers, angles):
        return centers

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.05
    for _ in range(80):
        if not has_overlap(scaled(hi), angles):
            break
        lo = hi
        hi *= 1.12
    else:
        return centers

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi)


def center_layout(centers):
    if not centers:
        return centers
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    return [(x - mx, y - my) for x, y in centers]


def objective(centers, angles, w_s=3.0, w_ov=5000.0, w_sq=800.0):
    s = enclosing_side(centers, angles)
    return w_s * s + w_ov * pairwise_overlap_penalty(centers, angles) + w_sq * square_violation(centers, angles, s)


def random_perturb(x, scale, rng):
    return x + (rng.random() * 2.0 - 1.0) * scale


def build_template_10(mode, rng):
    """Return a 10-gon starting point with structured motifs."""
    c = []
    a = []

    if mode == 0:
        # 2x5 zig-zag, alternating opposite orientations
        dx, dy = 1.40, 1.15
        for r in range(2):
            for col in range(5):
                x = (col - 2.0) * dx + (0.5 * dx if r == 1 else 0.0)
                y = (r - 0.5) * dy
                ang = (math.pi / 2.0 if ((r + col) & 1) == 0 else -math.pi / 2.0)
                ang += (rng.random() - 0.5) * 0.08
                c.append((x, y))
                a.append(normalize_angle(ang))

    elif mode == 1:
        # 5 mirrored pairs, slightly offset in x and y; encourages opposite-orientation motifs.
        dx = 1.25
        dy = 0.90
        xs = [-2, -1, 0, 1, 2]
        for i, x0 in enumerate(xs):
            y0 = (i - 2) * dy
            c.append((x0 * dx - 0.35, y0 + 0.15))
            a.append(normalize_angle(0.15 + (rng.random() - 0.5) * 0.10))
            c.append((x0 * dx + 0.35, -y0 - 0.15))
            a.append(normalize_angle(math.pi + 0.15 + (rng.random() - 0.5) * 0.10))

    elif mode == 2:
        # Staggered "two chains" with a central bridge.
        xs1 = [-2.4, -1.35, -0.25, 0.85, 1.95]
        xs2 = [-1.95, -0.85, 0.25, 1.35, 2.45]
        for i in range(5):
            c.append((xs1[i], -0.62 + 0.42 * i))
            a.append(normalize_angle((0.0 if i % 2 == 0 else math.pi) + (rng.random() - 0.5) * 0.10))
            c.append((xs2[i], 0.62 - 0.42 * i))
            a.append(normalize_angle((math.pi if i % 2 == 0 else 0.0) + (rng.random() - 0.5) * 0.10))

    elif mode == 3:
        # Near-hexagonal cloud, then alternating half-turns.
        pts = [
            (-1.95, -0.55), (-1.00, -1.15), (0.00, -1.35), (1.00, -1.15), (1.95, -0.55),
            (-1.70,  0.55), (-0.70,  0.95), (0.30,  1.15), (1.30,  0.95), (2.10,  0.55),
        ]
        for i, (x, y) in enumerate(pts):
            c.append((x, y))
            a.append(normalize_angle((0.0 if i % 2 == 0 else math.pi) + (rng.random() - 0.5) * 0.12))

    else:
        # Compact spiral-ish structure.
        pts = [
            (0.0, 0.0),
            (1.05, 0.20), (1.95, 0.95), (1.55, 1.95), (0.35, 2.35),
            (-0.80, 1.75), (-1.55, 0.80), (-1.70, -0.45), (-0.75, -1.35), (0.55, -1.55),
        ]
        for i, (x, y) in enumerate(pts):
            c.append((x, y))
            a.append(normalize_angle((i * TWOPI / 5.0) + (math.pi if i in (1, 3, 5, 7, 9) else 0.0) +
                                     (rng.random() - 0.5) * 0.10))

    return center_layout(c), a


def mutate_solution(centers, angles, rng, scale_xy=0.05, scale_a=0.10, p_flip=0.22):
    c = centers[:]
    a = angles[:]
    i = rng.randrange(len(c))
    if rng.random() < p_flip:
        a[i] = normalize_angle(a[i] + math.pi + (rng.random() - 0.5) * 0.15)
    else:
        c[i] = (
            c[i][0] + (rng.random() * 2.0 - 1.0) * scale_xy,
            c[i][1] + (rng.random() * 2.0 - 1.0) * scale_xy,
        )
        a[i] = normalize_angle(a[i] + (rng.random() * 2.0 - 1.0) * scale_a)
    if rng.random() < 0.35:
        j = rng.randrange(len(c))
        c[i], c[j] = c[j], c[i]
        a[i], a[j] = a[j], a[i]
    return c, a


def stochastic_refine(centers, angles, seed=0, steps=3500):
    rng = random.Random(1000003 + seed)
    cur_c = centers[:]
    cur_a = angles[:]
    cur_c = repair_scale(cur_c, cur_a)
    cur_o = objective(cur_c, cur_a)

    best_c = cur_c[:]
    best_a = cur_a[:]
    best_s = enclosing_side(best_c, best_a)

    xy = 0.12
    aa = 0.18
    for t in range(steps):
        cand_c, cand_a = mutate_solution(cur_c, cur_a, rng, xy, aa)
        cand_c = repair_scale(cand_c, cand_a)
        cand_o = objective(cand_c, cand_a)

        temp = 0.06 * (1.0 - t / steps) + 0.001
        accept = cand_o <= cur_o or rng.random() < math.exp(-(cand_o - cur_o) / temp)

        if accept:
            cur_c, cur_a, cur_o = cand_c, cand_a, cand_o
            if not has_overlap(cur_c, cur_a):
                s = enclosing_side(cur_c, cur_a)
                if s < best_s:
                    best_c, best_a, best_s = cur_c[:], cur_a[:], s

        if (t + 1) % 500 == 0:
            xy *= 0.78
            aa *= 0.82

    return best_c, best_a, best_s


def coordinate_descent(centers, angles, seed=0, rounds=6):
    rng = random.Random(700003 + seed)
    best_c = centers[:]
    best_a = angles[:]
    best_s = enclosing_side(best_c, best_a)
    n = len(best_c)

    for r in range(rounds):
        improved = False
        for i in range(n):
            base_c = best_c[:]
            base_a = best_a[:]
            local_best = (best_c[:], best_a[:], best_s)
            for _ in range(30):
                c = base_c[:]
                a = base_a[:]
                c[i] = (c[i][0] + (rng.random() * 2.0 - 1.0) * (0.05 / (r + 1)),
                        c[i][1] + (rng.random() * 2.0 - 1.0) * (0.05 / (r + 1)))
                a[i] = normalize_angle(a[i] + (rng.random() * 2.0 - 1.0) * (0.12 / (r + 1)))
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


def final_cleanup(centers, angles, seed=0):
    c, a, s = stochastic_refine(centers, angles, seed=9000 + seed, steps=1800)
    c, a, s = coordinate_descent(c, a, seed=12000 + seed, rounds=4)
    c = repair_scale(c, a)
    s = enclosing_side(c, a)
    return c, a, s


def pack(n):
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    # Generic fallback for other n: a few deterministic motifs plus local search.
    seeds = [0, 1, 2, 3, 5, 8, 13, 21]
    best = None

    if n == 10:
        modes = [0, 1, 2, 3, 4]
    else:
        modes = [0, 1, 2]

    for sd in seeds:
        rng = random.Random(20240000 + 97 * n + sd)
        for mode in modes:
            centers, angles = build_template_10(mode % 5, rng) if n == 10 else build_template_generic(n, mode, rng)
            if len(centers) != n:
                centers, angles = build_template_generic(n, mode, rng)
            centers = repair_scale(centers, angles)

            c1, a1, s1 = stochastic_refine(centers, angles, seed=sd + 31 * mode, steps=3200 if n == 10 else 2000)
            c2, a2, s2 = coordinate_descent(c1, a1, seed=sd + 101 * mode, rounds=7 if n == 10 else 5)
            c2 = repair_scale(c2, a2)
            s2 = enclosing_side(c2, a2)

            if best is None or s2 < best[2]:
                best = (c2[:], a2[:], s2)

    centers, angles, s = best
    centers, angles, s = final_cleanup(centers, angles, seed=n)

    if has_overlap(centers, angles):
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)

    return centers, angles, s


def build_template_generic(n, mode, rng):
    c = []
    a = []
    rows = max(1, int(math.ceil(math.sqrt(n))))
    cols = int(math.ceil(n / rows))

    if mode == 0:
        dx, dy = 1.45, 1.22
        k = 0
        for r in range(rows):
            for col in range(cols):
                if k >= n:
                    break
                x = (col - (cols - 1) / 2.0) * dx + (0.5 * dx if (r & 1) else 0.0)
                y = (r - (rows - 1) / 2.0) * dy
                ang = 0.0 if ((r + col) & 1) == 0 else math.pi
                ang += (rng.random() - 0.5) * 0.10
                c.append((x, y))
                a.append(normalize_angle(ang))
                k += 1
    elif mode == 1:
        dx, dy = 1.25, 1.45
        k = 0
        for col in range(cols):
            for r in range(rows):
                if k >= n:
                    break
                x = (col - (cols - 1) / 2.0) * dx
                y = (r - (rows - 1) / 2.0) * dy + (0.5 * dy if (col & 1) else 0.0)
                ang = math.pi / 2.0 if (k % 2 == 0) else -math.pi / 2.0
                ang += (rng.random() - 0.5) * 0.10
                c.append((x, y))
                a.append(normalize_angle(ang))
                k += 1
    else:
        angs = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
        for k in range(n):
            t = k / max(1, n - 1)
            x = (t - 0.5) * 3.0 + (rng.random() - 0.5) * 0.15
            y = math.sin(t * math.pi * 2.0) * 0.8 + (rng.random() - 0.5) * 0.15
            c.append((x, y))
            a.append(normalize_angle(angs[k % 4] + (rng.random() - 0.5) * 0.12))
    return center_layout(c), a
