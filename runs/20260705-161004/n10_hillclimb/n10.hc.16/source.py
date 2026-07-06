"""Heuristic optimizer for packing n unit regular pentagons into the smallest
origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s).
Container: point p is inside iff max(|px|, |py|) <= s/2.

This rewrite uses:
- a more accurate geometric model based on convex polygon SAT,
- deterministic initial patterns inspired by staggered/double-lattice layouts,
- multi-stage local search with annealing-like random perturbations,
- boundary tightening by direct constrained shrinking,
- final exact validity cleanup.

The program is designed to produce substantially tighter packings than the
previous version while preserving the required signature and conventions.
"""

import math
import random
import itertools

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
PHI = (1.0 + math.sqrt(5.0)) / 2.0
TWOPI = 2.0 * math.pi

# Some useful seed orientations for regular pentagons.
ANGLE_SET = [
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
    n = len(polys)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pa, pb = polys[i], polys[j]
            worst = 0.0
            sep_found = False
            for ax, ay in poly_axes(pa) + poly_axes(pb):
                amin, amax = project(pa, ax, ay)
                bmin, bmax = project(pb, ax, ay)
                gap = max(amin, bmin) - min(amax, bmax)
                if gap >= 0:
                    sep_found = True
                    break
                worst = max(worst, -gap)
            if not sep_found:
                pen += worst * worst
    return pen


def repair_scale(centers, angles):
    """Scale centers about origin until all overlaps vanish."""
    if not has_overlap(centers, angles):
        return centers

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.08
    for _ in range(80):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.15
    else:
        return centers

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi)


def small_clip(x, lim):
    if x < -lim:
        return -lim
    if x > lim:
        return lim
    return x


def initial_layout(n, mode, rng):
    """Generate staggered candidate packings with opposite-orientation motifs."""
    centers = []
    angles = []

    rows = max(1, int(math.ceil(math.sqrt(n))))
    cols = int(math.ceil(n / rows))

    if mode == 0:
        # Diagonal stagger with alternating opposite rotations.
        sx = 1.62
        sy = 1.42
        for k in range(n):
            r, c = divmod(k, cols)
            x = (c - (cols - 1) / 2.0) * sx + (0.5 * sx if (r & 1) else 0.0)
            y = (r - (rows - 1) / 2.0) * sy
            ang = math.pi / 2.0 if ((r + c) & 1) == 0 else -math.pi / 2.0
            ang += (rng.random() - 0.5) * 0.10
            centers.append((x, y))
            angles.append(normalize_angle(ang))

    elif mode == 1:
        # Column-based stagger, tends to help odd counts.
        sx = 1.40
        sy = 1.60
        cols2 = max(1, int(math.ceil(math.sqrt(n))))
        rows2 = int(math.ceil(n / cols2))
        for k in range(n):
            c, r = divmod(k, rows2)
            x = (c - (cols2 - 1) / 2.0) * sx
            y = (r - (rows2 - 1) / 2.0) * sy + (0.5 * sy if (c & 1) else 0.0)
            ang = 0.0 if ((r + c) & 1) == 0 else math.pi
            ang += (rng.random() - 0.5) * 0.10
            centers.append((x, y))
            angles.append(normalize_angle(ang))

    else:
        # Triangular-lattice inspired layout.
        dx = 1.44
        dy = 1.28
        k = 0
        for r in range(rows):
            for c in range(cols):
                if k >= n:
                    break
                x = (c - (cols - 1) / 2.0) * dx + (0.5 * dx if (r & 1) else 0.0)
                y = (r - (rows - 1) / 2.0) * dy
                ang = ANGLE_SET[(k + r + c) % len(ANGLE_SET)]
                if (r + c) % 3 == 1:
                    ang += math.pi
                ang += (rng.random() - 0.5) * 0.08
                centers.append((x, y))
                angles.append(normalize_angle(ang))
                k += 1

    # Center the layout around origin for a better start.
    mx = sum(x for x, _ in centers) / n
    my = sum(y for _, y in centers) / n
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def objective(centers, angles, s_weight=3.0, overlap_weight=2500.0):
    s = enclosing_side(centers, angles)
    pen = pairwise_overlap_penalty(centers, angles)
    sq = square_violation(centers, angles, s)
    return s_weight * s + overlap_weight * pen + 600.0 * sq


def local_search(centers, angles, seed=0, max_iter=3000):
    rng = random.Random(1234567 + 97 * seed)
    n = len(centers)

    cur_c = centers[:]
    cur_a = angles[:]
    cur_c = repair_scale(cur_c, cur_a)
    cur_o = objective(cur_c, cur_a)

    best_c = cur_c[:]
    best_a = cur_a[:]
    best_s = enclosing_side(best_c, best_a)

    step_xy = max(0.025, best_s * 0.012)
    step_a = 0.22

    for t in range(max_iter):
        i = rng.randrange(n)
        old_c = cur_c[i]
        old_a = cur_a[i]

        choice = rng.random()
        if choice < 0.48:
            dx = (rng.random() * 2.0 - 1.0) * step_xy
            dy = (rng.random() * 2.0 - 1.0) * step_xy
            cur_c[i] = (old_c[0] + dx, old_c[1] + dy)
        elif choice < 0.82:
            da = (rng.random() * 2.0 - 1.0) * step_a
            cur_a[i] = normalize_angle(old_a + da)
        else:
            cur_a[i] = normalize_angle(old_a + math.pi + (rng.random() - 0.5) * 0.15)

        test_c = repair_scale(cur_c, cur_a)
        test_o = objective(test_c, cur_a)

        accept = False
        if test_o <= cur_o:
            accept = True
        else:
            temp = 0.06 * (1.0 - t / max_iter) + 0.002
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

        if (t + 1) % 500 == 0:
            step_xy *= 0.86
            step_a *= 0.92

    return best_c, best_a, best_s


def coordinate_descent(centers, angles, seed=0, rounds=4):
    rng = random.Random(24680 + seed)
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
            for _ in range(16):
                c = base_c[:]
                a = base_a[:]
                j = i
                dx = (rng.random() * 2.0 - 1.0) * (0.06 / (r + 1))
                dy = (rng.random() * 2.0 - 1.0) * (0.06 / (r + 1))
                da = (rng.random() * 2.0 - 1.0) * (0.18 / (r + 1))
                c[j] = (c[j][0] + dx, c[j][1] + dy)
                a[j] = normalize_angle(a[j] + da)
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


def pack(n):
    if n <= 0:
        return [], [], 0.0

    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    def clamp_centers(cs, lim):
        return [(small_clip(x, lim), small_clip(y, lim)) for x, y in cs]

    def tangent_push(cs, angs, rounds=3):
        cs = cs[:]
        angs = angs[:]
        for _ in range(rounds):
            polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(cs, angs)]
            for i in range(n):
                fx = fy = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    pa, pb = polys[i], polys[j]
                    sep_best = None
                    sep_vec = (0.0, 0.0)
                    overlap = True
                    for ax, ay in poly_axes(pa) + poly_axes(pb):
                        amin, amax = project(pa, ax, ay)
                        bmin, bmax = project(pb, ax, ay)
                        gap = max(amin, bmin) - min(amax, bmax)
                        if gap >= 0:
                            overlap = False
                            break
                        if sep_best is None or gap > sep_best:
                            sep_best = gap
                            sep_vec = (ax, ay)
                    if overlap and sep_best is not None:
                        fx -= sep_vec[0] * sep_best
                        fy -= sep_vec[1] * sep_best
                cs[i] = (cs[i][0] + 0.08 * fx, cs[i][1] + 0.08 * fy)
                angs[i] = normalize_angle(angs[i] + 0.03 * (fx - fy))
        return cs, angs

    # Hand-tuned orientation sets that encourage opposite-orientation motifs.
    motif_angles = [
        (math.pi / 2.0, -math.pi / 2.0),
        (0.0, math.pi),
        (math.pi / 5.0, math.pi / 5.0 + math.pi),
        (2.0 * math.pi / 5.0, 2.0 * math.pi / 5.0 + math.pi),
        (-math.pi / 5.0, -math.pi / 5.0 + math.pi),
    ]

    # Deterministic starting arrangements for small/medium n.
    patterns = []
    if n <= 10:
        patterns.append((
            [(0.0, 0.0)] * n,
            [motif_angles[0][i & 1] for i in range(n)],
        ))
        # Compact 3-row stagger with alternating orientations.
        rows = 3
        cols = int(math.ceil(n / rows))
        c = []
        a = []
        dx, dy = 1.34, 1.17
        for k in range(n):
            r, col = divmod(k, cols)
            x = (col - (cols - 1) / 2.0) * dx + (0.5 * dx if r & 1 else 0.0)
            y = (r - (rows - 1) / 2.0) * dy
            c.append((x, y))
            a.append(motif_angles[(r + col) % len(motif_angles)][k & 1])
        patterns.append((c, a))

        # 2-row boundary-leaning strip.
        cols = int(math.ceil(n / 2))
        c = []
        a = []
        dx, dy = 1.28, 1.48
        for k in range(n):
            r, col = divmod(k, cols)
            x = (col - (cols - 1) / 2.0) * dx + (0.35 * dx if r else 0.0)
            y = (r - 0.5) * dy
            c.append((x, y))
            a.append(motif_angles[1][(k + r) & 1])
        patterns.append((c, a))

    # Generic patterns from the existing heuristics.
    seeds = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    best = None

    for seed in seeds:
        rng = random.Random(2024000 + 131 * n + seed)
        for mode in (0, 1, 2):
            centers, angles = initial_layout(n, mode, rng)
            patterns.append((centers, angles))

    # A short deterministic pipeline over all starts.
    for idx, (centers, angles) in enumerate(patterns):
        centers = repair_scale(centers, angles)
        centers, angles, s = local_search(
            centers, angles, seed=1000 + idx, max_iter=2600 if n <= 10 else (1800 if n <= 14 else 1300)
        )
        centers, angles, s = coordinate_descent(centers, angles, seed=3000 + idx, rounds=5)
        centers, angles = tangent_push(centers, angles, rounds=2)
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)
        if best is None or s < best[2]:
            best = (centers[:], angles[:], s)

    centers, angles, s = best

    # Multiscale refinements: alternate tiny angle flips and positional nudges.
    for phase in range(3):
        rng = random.Random(90000 + 97 * n + phase)
        for _ in range(900 if n <= 10 else 650):
            i = rng.randrange(n)
            old_c = centers[i]
            old_a = angles[i]
            if rng.random() < 0.5:
                centers[i] = (
                    old_c[0] + (rng.random() * 2.0 - 1.0) * (0.045 / (phase + 1)),
                    old_c[1] + (rng.random() * 2.0 - 1.0) * (0.045 / (phase + 1)),
                )
            else:
                da = (rng.random() * 2.0 - 1.0) * (0.16 / (phase + 1))
                if rng.random() < 0.22:
                    da += math.pi
                angles[i] = normalize_angle(old_a + da)

            test_c = repair_scale(centers, angles)
            test_s = enclosing_side(test_c, angles)
            if test_s <= s:
                centers = test_c
                s = test_s
            else:
                centers[i] = old_c
                angles[i] = old_a

        centers, angles, s = coordinate_descent(centers, angles, seed=4000 + phase + n, rounds=4)
        centers, angles = tangent_push(centers, angles, rounds=1)
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)

    # Final greedy compression by scaling centers toward the origin, then repairing.
    for lam in [0.995, 0.99, 0.985, 0.98]:
        trial_c = [(x * lam, y * lam) for x, y in centers]
        trial_c = repair_scale(trial_c, angles)
        trial_s = enclosing_side(trial_c, angles)
        if trial_s < s:
            centers, s = trial_c, trial_s

    # One last cleanup pass.
    centers, angles, s = local_search(
        centers, angles, seed=9999 + n, max_iter=1400 if n <= 10 else 900
    )
    centers = repair_scale(centers, angles)
    s = enclosing_side(centers, angles)

    if has_overlap(centers, angles):
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)

    return centers, angles, s
