"""Packing regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
Container: point p is inside iff max(|px|, |py|) <= s/2.

This version uses a geometry-first optimization strategy:
- exact polygon construction for unit regular pentagons,
- robust pairwise collision detection via SAT,
- boundary-aware objective,
- multiple deterministic seed patterns inspired by staggered / opposite-orientation motifs,
- coordinate refinement with adaptive random search and local shape-aware moves,
- final exact validation and container tightening.

The goal is to minimize the outer square side s while keeping all pentagons
non-overlapping and inside the square.
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


def initial_layout(n, mode, rng):
    centers = []
    angles = []

    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]

    # Layouts are intentionally over-dispersed; optimization will tighten them.
    rows = max(1, int(math.ceil(math.sqrt(n))))
    cols = int(math.ceil(n / rows))

    if mode == 0:
        # Staggered rows with alternating opposite orientations.
        dx = 1.58
        dy = 1.36
        for k in range(n):
            r, c = divmod(k, cols)
            x = (c - (cols - 1) / 2.0) * dx + (0.5 * dx if (r & 1) else 0.0)
            y = (r - (rows - 1) / 2.0) * dy
            a = (math.pi / 2.0 if ((r + c) & 1) == 0 else -math.pi / 2.0)
            a += (rng.random() - 0.5) * 0.12
            centers.append((x, y))
            angles.append(normalize_angle(a))

    elif mode == 1:
        # Column-staggered arrangement, often useful for odd n.
        dx = 1.34
        dy = 1.58
        cols2 = max(1, int(math.ceil(math.sqrt(n))))
        rows2 = int(math.ceil(n / cols2))
        for k in range(n):
            c, r = divmod(k, rows2)
            x = (c - (cols2 - 1) / 2.0) * dx
            y = (r - (rows2 - 1) / 2.0) * dy + (0.5 * dy if (c & 1) else 0.0)
            a = 0.0 if ((r + c) & 1) == 0 else math.pi
            a += (rng.random() - 0.5) * 0.12
            centers.append((x, y))
            angles.append(normalize_angle(a))

    elif mode == 2:
        # Triangular/hex-like lattice with orientation cycling.
        dx = 1.40
        dy = 1.22
        k = 0
        for r in range(rows):
            for c in range(cols):
                if k >= n:
                    break
                x = (c - (cols - 1) / 2.0) * dx + (0.5 * dx if (r & 1) else 0.0)
                y = (r - (rows - 1) / 2.0) * dy
                a = (k % 5) * TWOPI / 5.0
                if (r + c) % 3 == 1:
                    a += math.pi
                a += (rng.random() - 0.5) * 0.10
                centers.append((x, y))
                angles.append(normalize_angle(a))
                k += 1

    else:
        # Two interleaved chains, useful near n around 10.
        gapx = 1.50
        gapy = 1.30
        a1 = 0.0
        a2 = math.pi
        left_count = (n + 1) // 2
        right_count = n // 2
        for i in range(left_count):
            x = -((left_count - 1) / 2.0 - i) * gapx
            y = (i - (left_count - 1) / 2.0) * (gapy * 0.55)
            centers.append((x - 0.8, y))
            angles.append(normalize_angle(a1 + (rng.random() - 0.5) * 0.10))
        for i in range(right_count):
            x = -((right_count - 1) / 2.0 - i) * gapx
            y = (i - (right_count - 1) / 2.0) * (gapy * 0.55)
            centers.append((x + 0.8, y + 0.35))
            angles.append(normalize_angle(a2 + (rng.random() - 0.5) * 0.10))

    centers = center_layout(centers)
    return centers, angles


def objective(centers, angles, s_weight=4.0, overlap_weight=4000.0):
    s = enclosing_side(centers, angles)
    pen = pairwise_overlap_penalty(centers, angles)
    sq = square_violation(centers, angles, s)
    return s_weight * s + overlap_weight * pen + 700.0 * sq


def local_search(centers, angles, seed=0, max_iter=4000):
    rng = random.Random(1234567 + 97 * seed)
    n = len(centers)

    cur_c = centers[:]
    cur_a = angles[:]
    cur_c = repair_scale(cur_c, cur_a)
    cur_o = objective(cur_c, cur_a)

    best_c = cur_c[:]
    best_a = cur_a[:]
    best_s = enclosing_side(best_c, best_a)

    step_xy = max(0.02, 0.015 * best_s)
    step_a = 0.20

    for t in range(max_iter):
        i = rng.randrange(n)
        old_c = cur_c[i]
        old_a = cur_a[i]

        r = rng.random()
        if r < 0.42:
            dx = (rng.random() * 2.0 - 1.0) * step_xy
            dy = (rng.random() * 2.0 - 1.0) * step_xy
            cur_c[i] = (old_c[0] + dx, old_c[1] + dy)
        elif r < 0.80:
            da = (rng.random() * 2.0 - 1.0) * step_a
            cur_a[i] = normalize_angle(old_a + da)
        else:
            cur_a[i] = normalize_angle(old_a + math.pi + (rng.random() - 0.5) * 0.18)

        test_c = repair_scale(cur_c, cur_a)
        test_o = objective(test_c, cur_a)

        accept = False
        if test_o <= cur_o:
            accept = True
        else:
            temp = 0.08 * (1.0 - t / max_iter) + 0.0015
            if rng.random() < math.exp(-(test_o - cur_o) / temp):
                accept = True

        if accept:
            cur_c = test_c
            cur_o = test_o
            s = enclosing_side(cur_c, cur_a)
            if not has_overlap(cur_c, cur_a) and s < best_s:
                best_c = cur_c[:]
                best_a = cur_a[:]
                best_s = s
        else:
            cur_c[i] = old_c
            cur_a[i] = old_a

        if (t + 1) % 600 == 0:
            step_xy *= 0.84
            step_a *= 0.90

    return best_c, best_a, best_s


def coordinate_descent(centers, angles, seed=0, rounds=5):
    rng = random.Random(24680 + seed)
    n = len(centers)
    best_c = centers[:]
    best_a = angles[:]
    best_s = enclosing_side(best_c, best_a)

    for r in range(rounds):
        improved = False
        for i in range(n):
            local_best = (best_c[:], best_a[:], best_s)
            base_c = best_c[:]
            base_a = best_a[:]
            for _ in range(20):
                c = base_c[:]
                a = base_a[:]
                dx = (rng.random() * 2.0 - 1.0) * (0.055 / (r + 1))
                dy = (rng.random() * 2.0 - 1.0) * (0.055 / (r + 1))
                da = (rng.random() * 2.0 - 1.0) * (0.16 / (r + 1))
                c[i] = (c[i][0] + dx, c[i][1] + dy)
                a[i] = normalize_angle(a[i] + da)
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


def final_tighten(centers, angles, seed=0):
    # A small deterministic refinement near the end.
    c, a, s = local_search(centers, angles, seed=9000 + seed, max_iter=1400)
    c, a, s = coordinate_descent(c, a, seed=7000 + seed, rounds=4)
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

    # Multiple deterministic seeds and motifs.
    seeds = [0, 1, 2, 3, 5, 8, 13, 21, 34]
    best = None

    for base_seed in seeds:
        rng = random.Random(2024000 + 131 * n + base_seed)
        # For n=10 in particular, try several candidate patterns aggressively.
        modes = (0, 1, 2, 3)
        for mode in modes:
            centers, angles = initial_layout(n, mode, rng)
            centers = repair_scale(centers, angles)

            centers, angles, s = local_search(
                centers,
                angles,
                seed=base_seed + 19 * mode,
                max_iter=2400 if n <= 12 else 1600,
            )
            centers, angles, s = coordinate_descent(
                centers, angles, seed=base_seed + 97 * mode, rounds=5
            )
            centers = repair_scale(centers, angles)
            s = enclosing_side(centers, angles)

            if best is None or s < best[2]:
                best = (centers[:], angles[:], s)

    centers, angles, s = best
    centers, angles, s = final_tighten(centers, angles, seed=n)

    # Final safety pass.
    if has_overlap(centers, angles):
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)

    # One last attempt to reduce container size by small local cleanup.
    centers, angles, s = coordinate_descent(centers, angles, seed=4242 + n, rounds=3)
    centers = repair_scale(centers, angles)
    s = enclosing_side(centers, angles)

    return centers, angles, s
