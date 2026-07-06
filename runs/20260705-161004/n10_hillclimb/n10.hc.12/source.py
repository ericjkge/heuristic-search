"""Heuristic optimizer for packing n unit regular pentagons into the smallest
origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s).
Container: point p is inside iff max(|px|, |py|) <= s/2.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # inradius
PHI = (1.0 + math.sqrt(5.0)) / 2.0

# A unit regular pentagon has 10-fold symmetry in pairwise avoidance when
# alternating opposite orientations; these angles are useful seeds.
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


def pentagon_vertices(cx, cy, angle):
    return [
        (cx + R * math.cos(angle + k * 2.0 * math.pi / 5.0),
         cy + R * math.sin(angle + k * 2.0 * math.pi / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    if not centers:
        return 0.0
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, ang)
    )


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


def polygons_overlap(pa, pb, eps=1e-10):
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


def repair_scale(centers, angles):
    """Scale centers about origin until all overlaps vanish."""
    if not has_overlap(centers, angles):
        return centers

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.15
    for _ in range(80):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.18
    else:
        return centers

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi)


def normalize_angle(a):
    twopi = 2.0 * math.pi
    a = math.fmod(a, twopi)
    if a <= -math.pi:
        a += twopi
    elif a > math.pi:
        a -= twopi
    return a


def candidate_orientations(k):
    # Mix opposite orientations and pentagon symmetries.
    base = ANGLE_SET[k % len(ANGLE_SET)]
    return base


def build_lattice_layout(n, mode, rng):
    """
    Construct a dense staggered arrangement inspired by double-lattice packing.
    Returns centers, angles.
    """
    centers = []
    angles = []

    # Good starting scales; later repair_scale will shrink/expand just enough.
    # Use a slightly compressed spacing for boundary tightening.
    sx = 0.92 * WIDTH()
    sy = 0.90 * HEIGHT()

    if mode == 0:
        # Alternating rows, opposite orientations.
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        for k in range(n):
            r, c = divmod(k, cols)
            x = (c - (cols - 1) / 2.0) * sx
            y = (r - (rows - 1) / 2.0) * sy
            if r % 2:
                x += 0.5 * sx
            # Opposite orientations encouraged in dense packings.
            ang = math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0
            ang += (rng.random() - 0.5) * 0.08
            centers.append((x, y))
            angles.append(ang)

    elif mode == 1:
        # Column-based stagger, useful for odd n.
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        for k in range(n):
            c, r = divmod(k, rows)
            x = (c - (cols - 1) / 2.0) * sx
            y = (r - (rows - 1) / 2.0) * sy
            if c % 2:
                y += 0.5 * sy
            ang = 0.0 if (r + c) % 2 == 0 else math.pi
            ang += (rng.random() - 0.5) * 0.08
            centers.append((x, y))
            angles.append(ang)

    else:
        # Hex-like packing using a triangular lattice basis.
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        dx = 0.92 * WIDTH()
        dy = 0.82 * HEIGHT()
        for k in range(n):
            r, c = divmod(k, cols)
            x = (c - (cols - 1) / 2.0) * dx + (0.5 * dx if (r % 2) else 0.0)
            y = (r - (rows - 1) / 2.0) * dy
            ang = candidate_orientations(k)
            if (r + c) % 3 == 1:
                ang += math.pi
            ang += (rng.random() - 0.5) * 0.06
            centers.append((x, y))
            angles.append(normalize_angle(ang))

    return centers, angles


def WIDTH():
    return 2.0 * R * math.sin(2.0 * math.pi / 5.0)


def HEIGHT():
    return R + APOTHEM


def square_penalty(centers, angles, s):
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


def pairwise_penalty(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    n = len(polys)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # Approximate overlap penalty via separating axes distances.
            overlap = True
            sep = 1e9
            for ax, ay in poly_axes(polys[i]) + poly_axes(polys[j]):
                amin, amax = project(polys[i], ax, ay)
                bmin, bmax = project(polys[j], ax, ay)
                gap = max(amin, bmin) - min(amax, bmax)
                if gap >= 0:
                    overlap = False
                    sep = min(sep, gap)
                    break
                sep = min(sep, -gap)
            if overlap:
                pen += sep * sep
    return pen


def local_optimize(centers, angles, max_iter=4000, seed=0):
    rng = random.Random(912345 + seed)
    n = len(centers)

    cur_c = centers[:]
    cur_a = angles[:]
    cur_s = enclosing_side(cur_c, cur_a)
    best = (cur_c[:], cur_a[:], cur_s)

    step_xy = max(0.006, cur_s * 0.006)
    step_a = 0.18

    def obj(c, a):
        s = enclosing_side(c, a)
        if has_overlap(c, a):
            return s + 10.0 * pairwise_penalty(c, a)
        return s

    cur_obj = obj(cur_c, cur_a)

    for t in range(max_iter):
        i = rng.randrange(n)
        old_c = cur_c[i]
        old_a = cur_a[i]

        move = rng.random()

        if move < 0.50:
            dx = (rng.random() * 2.0 - 1.0) * step_xy
            dy = (rng.random() * 2.0 - 1.0) * step_xy
            cur_c[i] = (old_c[0] + dx, old_c[1] + dy)
        elif move < 0.85:
            da = (rng.random() * 2.0 - 1.0) * step_a
            cur_a[i] = normalize_angle(old_a + da)
        else:
            # Bias toward opposite orientation to help interlocking.
            cur_a[i] = normalize_angle(old_a + math.pi + (rng.random() - 0.5) * 0.12)

        test_c = repair_scale(cur_c, cur_a)
        test_s = enclosing_side(test_c, cur_a)
        test_obj = obj(test_c, cur_a)

        accept = False
        if test_obj < cur_obj:
            accept = True
        else:
            temp = max(0.001, 0.03 * (1.0 - t / max_iter))
            delta = test_obj - cur_obj
            if rng.random() < math.exp(-delta / temp):
                accept = True

        if accept:
            cur_c = test_c
            cur_s = test_s
            cur_obj = test_obj
            if not has_overlap(cur_c, cur_a) and cur_s < best[2]:
                best = (cur_c[:], cur_a[:], cur_s)
        else:
            cur_c[i] = old_c
            cur_a[i] = old_a

        if (t + 1) % 700 == 0:
            step_xy *= 0.88
            step_a *= 0.93

    return best


def refine_with_global_shifts(centers, angles, seed=0):
    """Try translating subsets and slight global rescaling to tighten boundaries."""
    rng = random.Random(777 + seed)
    n = len(centers)
    best_c = centers[:]
    best_a = angles[:]
    best_s = enclosing_side(best_c, best_a)

    for _ in range(60):
        c = best_c[:]
        a = best_a[:]
        idx = rng.randrange(n)
        dx = (rng.random() * 2.0 - 1.0) * 0.05
        dy = (rng.random() * 2.0 - 1.0) * 0.05
        da = (rng.random() * 2.0 - 1.0) * 0.10
        c[idx] = (c[idx][0] + dx, c[idx][1] + dy)
        a[idx] = normalize_angle(a[idx] + da)
        c = repair_scale(c, a)
        s = enclosing_side(c, a)
        if not has_overlap(c, a) and s < best_s:
            best_c, best_a, best_s = c, a, s
    return best_c, best_a, best_s


def pack(n):
    """Return a valid packing of n unit pentagons into a minimum-ish square."""
    if n <= 0:
        return [], [], 0.0

    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    # More diverse multi-start search with stronger local refinement.
    best = None

    def transform_layout(centers, angles, rng):
        # Global random rotation + optional mirror + tiny jitter.
        rot = rng.uniform(-math.pi, math.pi)
        flip = -1.0 if rng.random() < 0.5 else 1.0
        c2 = []
        a2 = []
        for (x, y), a in zip(centers, angles):
            xr = x * math.cos(rot) - y * math.sin(rot)
            yr = x * math.sin(rot) + y * math.cos(rot)
            xr *= flip
            c2.append((xr, yr))
            a2.append(normalize_angle(a + rot + (math.pi if flip < 0 else 0.0)))
        # Recentre and tiny random perturbation to escape lattice symmetry.
        mx = sum(x for x, _ in c2) / len(c2)
        my = sum(y for _, y in c2) / len(c2)
        c2 = [((x - mx) + rng.uniform(-0.03, 0.03), (y - my) + rng.uniform(-0.03, 0.03)) for x, y in c2]
        a2 = [normalize_angle(a + rng.uniform(-0.05, 0.05)) for a in a2]
        return c2, a2

    def anneal(centers, angles, seed):
        rng = random.Random(9001 + 97 * n + seed)
        c = centers[:]
        a = angles[:]
        s = enclosing_side(c, a)
        best_local = (c[:], a[:], s)
        temp0 = 0.08
        for t in range(6000 if n <= 12 else 4200):
            i = rng.randrange(n)
            old_c = c[i]
            old_a = a[i]
            old_s = s

            # Adaptive move set: translation, rotation, or correlated row-like drift.
            m = rng.random()
            if m < 0.45:
                scale = 0.08 * (1.0 - t / (6000 if n <= 12 else 4200)) + 0.004
                c[i] = (old_c[0] + rng.uniform(-scale, scale), old_c[1] + rng.uniform(-scale, scale))
            elif m < 0.85:
                da = rng.uniform(-0.22, 0.22)
                a[i] = normalize_angle(old_a + da)
            else:
                a[i] = normalize_angle(old_a + math.pi + rng.uniform(-0.10, 0.10))

            # Occasionally shift a small cluster together to model boundary row tilt.
            if rng.random() < 0.18:
                j = (i + rng.randrange(1, max(2, n // 3 + 1))) % n
                c[j] = (c[j][0] + rng.uniform(-0.03, 0.03), c[j][1] + rng.uniform(-0.03, 0.03))
                a[j] = normalize_angle(a[j] + rng.uniform(-0.07, 0.07))

            c = repair_scale(c, a)
            s = enclosing_side(c, a)

            # Hard feasibility guard.
            if has_overlap(c, a):
                c[i] = old_c
                a[i] = old_a
                s = old_s
                continue

            # Soft score emphasizes bounding square size.
            score = s + 0.02 * pairwise_penalty(c, a)
            best_score = best_local[2] + 0.02 * pairwise_penalty(best_local[0], best_local[1])
            accept = score < best_score
            if not accept:
                temp = temp0 * (1.0 - t / (6000 if n <= 12 else 4200)) + 0.001
                if rng.random() < math.exp(-(score - best_score) / max(1e-9, temp)):
                    accept = True

            if accept:
                if s < best_local[2]:
                    best_local = (c[:], a[:], s)
            else:
                c[i] = old_c
                a[i] = old_a
                s = old_s

            if (t + 1) % 900 == 0:
                # Shrink-step plus more decisive orientation perturbations.
                c = repair_scale(c, a)
                for k in range(n):
                    if rng.random() < 0.12:
                        a[k] = normalize_angle(a[k] + rng.choice([0.0, math.pi, math.pi / 5.0, -math.pi / 5.0]) + rng.uniform(-0.03, 0.03))

        return best_local

    # High-diversity initial families: lattice, perturbed lattice, and handcrafted motifs.
    seeds = list(range(18))
    for seed in seeds:
        rng = random.Random(202406 + 97 * n + seed)

        # Baseline generated layouts.
        layouts = []
        for mode in (0, 1, 2):
            c, a = build_lattice_layout(n, mode, rng)
            layouts.append((c, a))

        # Add transformed variants.
        layouts = [transform_layout(c, a, rng) for c, a in layouts] + layouts

        # Add a few motif-heavy starting points for small/medium n.
        if n >= 2:
            c = []
            a = []
            # Start with a compact zig-zag strip; effective for boundary-tuned assemblies.
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            dx = 0.78 * WIDTH()
            dy = 0.76 * HEIGHT()
            for k in range(n):
                r, cidx = divmod(k, cols)
                x = (cidx - (cols - 1) / 2.0) * dx + (0.5 * dx if r % 2 else 0.0)
                y = (r - (rows - 1) / 2.0) * dy
                c.append((x, y))
                a.append(normalize_angle((math.pi / 2.0 if (r + cidx) % 2 == 0 else -math.pi / 2.0) + rng.uniform(-0.08, 0.08)))
            layouts.append((c, a))

        for centers, angles in layouts:
            centers = repair_scale(centers, angles)
            centers, angles, s = anneal(centers, angles, seed=seed)
            centers = repair_scale(centers, angles)
            s = enclosing_side(centers, angles)

            # Extra coordinate descent around the best local result.
            for _ in range(3 if n <= 12 else 2):
                centers, angles, s2 = local_optimize(
                    centers, angles,
                    max_iter=5000 if n <= 12 else 2800,
                    seed=seed * 37 + _
                )
                centers = repair_scale(centers, angles)
                s2 = enclosing_side(centers, angles)
                if s2 < s:
                    s = s2

            centers, angles, s = refine_with_global_shifts(centers, angles, seed=seed)
            centers = repair_scale(centers, angles)
            s = enclosing_side(centers, angles)

            if best is None or s < best[2]:
                best = (centers, angles, s)

    centers, angles, s = best

    # Final tightening: alternate between scale repair and targeted local annealing.
    for _ in range(4):
        centers = repair_scale(centers, angles)
        centers, angles, s2 = local_optimize(
            centers, angles,
            max_iter=2000 if n <= 12 else 1200,
            seed=999 + n + _
        )
        centers = repair_scale(centers, angles)
        s2 = enclosing_side(centers, angles)
        if s2 < s:
            s = s2

    return centers, angles, s
