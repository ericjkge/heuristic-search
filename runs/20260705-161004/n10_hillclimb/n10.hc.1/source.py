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

    lo, hi = 1.0, 1.2
    for _ in range(60):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.25
    else:
        return centers

    for _ in range(70):
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

    for mode, cols, rows in layout_candidates(n):
        centers = []
        angles = best_orientation(n, rng)

        if mode == "grid":
            # Row/column pitches tuned for pentagon dimensions.
            pitch_x = WIDTH * 0.74
            pitch_y = HEIGHT * 0.74

            for k in range(n):
                i, j = k % cols, k // cols
                x = (i - (cols - 1) / 2.0) * pitch_x
                y = (j - (rows - 1) / 2.0) * pitch_y

                # Alternate flips by column and row to encourage interlocking.
                if (i + j) % 2 == 1:
                    angles[k] += math.pi

                # Small deterministic wobble.
                x += (j % 2) * 0.08 - 0.04
                centers.append((x, y))

        else:  # stagger
            pitch_x = WIDTH * 0.70
            pitch_y = HEIGHT * 0.72
            for k in range(n):
                j, i = divmod(k, cols)
                x = (i - (cols - 1) / 2.0) * pitch_x
                if j % 2:
                    x += pitch_x * 0.5
                y = (j - (rows - 1) / 2.0) * pitch_y
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

    # Search parameters
    step_xy = max(0.01, cur_s * 0.008)
    step_a = 0.18

    for t in range(steps):
        i = rng.randrange(n)
        old_c = cur_centers[i]
        old_a = cur_angles[i]

        # Propose a change
        if rng.random() < 0.55:
            dx = (rng.random() * 2.0 - 1.0) * step_xy
            dy = (rng.random() * 2.0 - 1.0) * step_xy
            cur_centers[i] = (old_c[0] + dx, old_c[1] + dy)
        else:
            cur_angles[i] = old_a + (rng.random() * 2.0 - 1.0) * step_a

        # Fast shrink: local changes may allow a smaller global scale after repair.
        test_centers = repair_scale(cur_centers, cur_angles)
        test_s = enclosing_side(test_centers, cur_angles)

        if test_s < cur_s:
            cur_centers = test_centers
            cur_s = test_s
        else:
            # Revert
            cur_centers[i] = old_c
            cur_angles[i] = old_a

        # Gradually anneal
        if (t + 1) % 500 == 0:
            step_xy *= 0.92
            step_a *= 0.96

    return cur_centers, cur_angles, cur_s


def pack(n):
    """Return a valid packing of n unit pentagons into a minimum-ish square."""
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    # Multiple randomized starts.
    best = None
    for seed in range(10):
        centers, angles, s = initial_pack(n, seed=seed)
        centers, angles, s = improve_by_local_search(centers, angles, steps=1800, seed=seed)
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)
        if best is None or s < best[2]:
            best = (centers, angles, s)

    # Final cleanup.
    centers, angles, s = best
    centers = repair_scale(centers, angles)
    s = enclosing_side(centers, angles)
    return centers, angles, s
