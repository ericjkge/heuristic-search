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
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # width of a unit pentagon
HEIGHT = R + APOTHEM                               # height of point-up orientation


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


def polygons_overlap(pa, pb, eps=1e-9):
    # Touching is allowed; positive-area intersection is not.
    for ax, ay in poly_axes(pa) + poly_axes(pb):
        amin, amax = project(pa, ax, ay)
        bmin, bmax = project(pb, ax, ay)
        if min(amax, bmax) - max(amin, bmin) <= eps:
            return False
    return True


def min_pairwise_gap(centers, angles):
    """Lower bound on scale factor needed to remove overlaps under radial scaling."""
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    m = len(polys)
    worst = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            pa, pb = polys[i], polys[j]
            local = float("inf")
            for ax, ay in poly_axes(pa) + poly_axes(pb):
                amin, amax = project(pa, ax, ay)
                bmin, bmax = project(pb, ax, ay)
                ov = min(amax, bmax) - max(amin, bmin)
                if ov <= 0.0:
                    local = 0.0
                    break
                local = min(local, ov)
            worst = max(worst, local)
    return worst


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

    lo, hi = 1.0, 1.12
    for _ in range(50):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.18
    else:
        return centers

    for _ in range(48):
        mid = (lo + hi) * 0.5
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi)


def best_orientation(n, rng):
    """Provide a diverse initial angle pattern using mixed orientations."""
    angles = []
    for k in range(n):
        # Bias toward the two opposite orientations known to be useful in dense packings.
        base = math.pi / 2.0 if (k % 2 == 0) else -math.pi / 2.0
        # Small deterministic/random perturbations help avoid symmetry lock-in.
        jitter = (rng.random() - 0.5) * 0.22
        angles.append(base + jitter)
    return angles


def layout_candidates(n):
    """Generate several plausible center layouts."""
    cands = []

    # 1) Near-square grid with staggered rows
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    cands.append(("grid", cols, rows))

    # 2) Slightly rectangular grids in both directions
    for cols in range(max(1, int(math.floor(math.sqrt(n))) - 2), int(math.ceil(math.sqrt(n))) + 3):
        rows = int(math.ceil(n / cols))
        cands.append(("grid", cols, rows))

    # 3) Hex-like staggered rows
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    cands.append(("stagger", cols, rows))

    return cands


def initial_pack(n, seed=0):
    rng = random.Random(1234567 + 37 * n + seed)

    best = None

    # Hand-tuned orientation presets: opposite pairs and boundary tilts.
    presets = [
        [math.pi / 2.0, -math.pi / 2.0],
        [math.pi / 2.0 + 0.22, -math.pi / 2.0 - 0.22],
        [math.pi / 2.0 - 0.22, -math.pi / 2.0 + 0.22],
        [0.0, math.pi],
        [0.12, math.pi + 0.12],
    ]

    for mode, cols, rows in layout_candidates(n):
        for preset in presets:
            centers = []
            angles = []
            for k in range(n):
                angles.append(preset[k % len(preset)] + (rng.random() - 0.5) * 0.08)

            if mode == "grid":
                pitch_x = WIDTH * 0.66
                pitch_y = HEIGHT * 0.68
                for k in range(n):
                    i, j = k % cols, k // cols
                    x = (i - (cols - 1) / 2.0) * pitch_x
                    y = (j - (rows - 1) / 2.0) * pitch_y

                    # Boundary rows tilt toward walls.
                    if j == 0:
                        angles[k] += 0.16
                    elif j == rows - 1:
                        angles[k] -= 0.16

                    if (i + j) % 2 == 1:
                        angles[k] += math.pi

                    x += (j % 2) * 0.06 - 0.03
                    centers.append((x, y))

            else:  # stagger
                pitch_x = WIDTH * 0.64
                pitch_y = HEIGHT * 0.66
                for k in range(n):
                    j, i = divmod(k, cols)
                    x = (i - (cols - 1) / 2.0) * pitch_x
                    if j % 2:
                        x += pitch_x * 0.5
                    y = (j - (rows - 1) / 2.0) * pitch_y
                    if j == 0:
                        angles[k] += 0.18
                    elif j == rows - 1:
                        angles[k] -= 0.18
                    if (i + 2 * j) % 3 == 0:
                        angles[k] += math.pi
                    centers.append((x, y))

            centers = repair_scale(centers, angles)
            s = enclosing_side(centers, angles)
            if best is None or s < best[2]:
                best = (centers, angles, s)

    return best


def improve_by_local_search(centers, angles, steps=3000, seed=0):
    """Random local search on angles and center coordinates with feasibility repair."""
    rng = random.Random(99991 + seed)
    n = len(centers)

    cur_centers = centers[:]
    cur_angles = angles[:]
    cur_s = enclosing_side(cur_centers, cur_angles)

    step_xy = max(0.008, cur_s * 0.006)
    step_a = 0.22

    for t in range(steps):
        i = rng.randrange(n)
        old_c = cur_centers[i]
        old_a = cur_angles[i]

        mode = rng.random()
        if mode < 0.42:
            dx = (rng.random() * 2.0 - 1.0) * step_xy
            dy = (rng.random() * 2.0 - 1.0) * step_xy
            cur_centers[i] = (old_c[0] + dx, old_c[1] + dy)
        elif mode < 0.82:
            cur_angles[i] = old_a + (rng.random() * 2.0 - 1.0) * step_a
        else:
            cur_centers[i] = (-old_c[0], -old_c[1])
            cur_angles[i] = old_a + math.pi

        test_centers = repair_scale(cur_centers, cur_angles)
        test_s = enclosing_side(test_centers, cur_angles)

        if test_s < cur_s:
            cur_centers = test_centers
            cur_s = test_s
        else:
            cur_centers[i] = old_c
            cur_angles[i] = old_a

        if (t + 1) % 350 == 0:
            step_xy *= 0.90
            step_a *= 0.94

    return cur_centers, cur_angles, cur_s


def pack(n):
    """Return a valid packing of n unit pentagons into a minimum-ish square."""
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    best = None

    # Stronger multi-start strategy.
    for seed in range(24):
        centers, angles, s = initial_pack(n, seed=seed)
        centers, angles, s = improve_by_local_search(centers, angles, steps=2400, seed=seed)

        # Pairwise refinement: use small global rescaling only after local moves settle.
        if has_overlap(centers, angles):
            centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)

        # Additional centroid-free boundary tightening by random restarts from current best.
        for _ in range(2):
            jittered = []
            for (x, y), a in zip(centers, angles):
                jittered.append(((x + (random.Random(seed).random() - 0.5) * 0.03),
                                 (y + (random.Random(seed + 17).random() - 0.5) * 0.03)))
            jcenters = repair_scale(jittered, angles)
            js = enclosing_side(jcenters, angles)
            if js < s:
                centers, s = jcenters, js

        if best is None or s < best[2]:
            best = (centers, angles, s)

    centers, angles, s = best
    centers = repair_scale(centers, angles)
    s = enclosing_side(centers, angles)
    return centers, angles, s
