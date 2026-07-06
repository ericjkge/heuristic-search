"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles: list[float]
- s: side length of the enclosing square centered at the origin
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius ~ 0.8507
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # ~ 0.6882
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # widest extent (diagonal) ~ 1.618
HEIGHT = R + APOTHEM                              # point-up bounding height ~ 1.539

TWO_PI = 2.0 * math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0

# Small tolerance for geometric tests
EPS = 1e-10


def pentagon_vertices(cx, cy, angle):
    return [
        (cx + R * math.cos(angle + k * TWO_PI / 5.0),
         cy + R * math.sin(angle + k * TWO_PI / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, ang)
    ) if centers else 0.0


def polygon_axes(poly):
    axes = []
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        nx, ny = -(y2 - y1), (x2 - x1)
        norm = math.hypot(nx, ny)
        if norm > 0:
            axes.append((nx / norm, ny / norm))
    return axes


def project(poly, ax, ay):
    vals = [x * ax + y * ay for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb):
    # Separating axis theorem
    for ax, ay in polygon_axes(pa) + polygon_axes(pb):
        amin, amax = project(pa, ax, ay)
        bmin, bmax = project(pb, ax, ay)
        if amax <= bmin + EPS or bmax <= amin + EPS:
            return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def repair_scale(centers, angles, max_iter=80):
    """Scale all centers about the origin minimally until all overlaps vanish."""
    if not centers:
        return centers, 0.0

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    if not has_overlap(centers, angles):
        return centers, enclosing_side(centers, angles)

    lo, hi = 1.0, 1.02
    for _ in range(60):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.15
    else:
        return centers, enclosing_side(centers, angles)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    final_centers = scaled(hi)
    return final_centers, enclosing_side(final_centers, angles)


def square_side_for_rhombus(n, dx, dy, angle_a, angle_b):
    """A very compact double-lattice-inspired pattern: two rows with opposite orientations."""
    centers = []
    angles = []

    rows = 2 if n > 1 else 1
    cols = (n + rows - 1) // rows

    # Centered grid with row offset half a period
    idx = 0
    for r in range(rows):
        y = (r - (rows - 1) / 2.0) * dy
        x_shift = 0.5 * dx if (r % 2 == 1) else 0.0
        for c in range(cols):
            if idx >= n:
                break
            x = (c - (cols - 1) / 2.0) * dx + x_shift
            centers.append((x, y))
            angles.append(angle_a if ((c + r) % 2 == 0) else angle_b)
            idx += 1

    return centers, angles


def dense_candidates(n):
    """A small family of hand-tuned motif families, then local refinement by scaling."""
    candidates = []

    # Family 1: interlocking 2-row layout, opposite orientations.
    # Tuned to exploit pentagon anisotropy.
    candidates.append((
        square_side_for_rhombus(
            n,
            dx=WIDTH * 0.92,
            dy=HEIGHT * 0.83,
            angle_a=math.pi / 10.0,
            angle_b=math.pi / 10.0 + math.pi,
        ),
        "two_row_opposite"
    ))

    # Family 2: three staggered rows, alternating flips.
    rows = min(3, n)
    cols = (n + rows - 1) // rows
    centers, angles = [], []
    idx = 0
    dx = WIDTH * 0.88
    dy = HEIGHT * 0.79
    for r in range(rows):
        y = (r - (rows - 1) / 2.0) * dy
        x_shift = (r % 2) * dx * 0.5
        for c in range(cols):
            if idx >= n:
                break
            x = (c - (cols - 1) / 2.0) * dx + x_shift
            centers.append((x, y))
            angles.append((math.pi / 10.0 + math.pi) if (idx % 2) else (math.pi / 10.0))
            idx += 1
    candidates.append(((centers, angles), "three_row_stagger"))

    # Family 3: one central and a surrounding ring, useful for small n.
    if n <= 10:
        m = n
        centers, angles = [], []
        if m == 1:
            centers = [(0.0, 0.0)]
            angles = [math.pi / 10.0]
        else:
            centers.append((0.0, 0.0))
            angles.append(math.pi / 10.0)
            ring = m - 1
            rad = 1.05 * R * 1.95
            for k in range(ring):
                t = TWO_PI * k / ring
                centers.append((rad * math.cos(t), rad * math.sin(t)))
                angles.append((math.pi / 10.0 + math.pi) if (k % 2) else (math.pi / 10.0))
        candidates.append(((centers, angles), "ring"))

    return candidates


def optimize_layout(centers, angles, iters=200, seed=0):
    """Simple stochastic improvement: perturb positions/angles, keep best feasible under scaling."""
    rng = random.Random(seed)
    best_centers = [tuple(p) for p in centers]
    best_angles = list(angles)
    best_centers, best_s = repair_scale(best_centers, best_angles)

    if len(best_centers) <= 1:
        return best_centers, best_angles, best_s

    cur_centers = [tuple(p) for p in best_centers]
    cur_angles = list(best_angles)
    cur_s = best_s

    for t in range(iters):
        # Decrease mutation scale over time
        frac = 1.0 - t / max(1, iters - 1)
        pos_sigma = 0.16 * frac
        ang_sigma = 0.25 * frac

        cand_centers = []
        cand_angles = []
        for (x, y), a in zip(cur_centers, cur_angles):
            nx = x + rng.gauss(0.0, pos_sigma)
            ny = y + rng.gauss(0.0, pos_sigma)
            na = a + rng.gauss(0.0, ang_sigma)
            cand_centers.append((nx, ny))
            cand_angles.append(na)

        # Recentering helps the final square side
        mx = sum(x for x, _ in cand_centers) / len(cand_centers)
        my = sum(y for _, y in cand_centers) / len(cand_centers)
        cand_centers = [(x - mx, y - my) for x, y in cand_centers]

        cand_centers, cand_s = repair_scale(cand_centers, cand_angles)
        if cand_s < cur_s - 1e-8:
            cur_centers, cur_angles, cur_s = cand_centers, cand_angles, cand_s
            if cur_s < best_s - 1e-8:
                best_centers, best_angles, best_s = cur_centers, cur_angles, cur_s

    return best_centers, best_angles, best_s


def pack(n):
    """Construct a valid packing of n unit pentagons into a minimum-ish square."""
    if n <= 0:
        return [], [], 0.0

    # Keep a stable set of seeds for reproducibility, but explore a few layouts.
    layouts = dense_candidates(n)

    best = None
    best_s = float("inf")

    for idx, (layout, tag) in enumerate(layouts):
        centers, angles = layout
        # Normalize orientation angles
        angles = [((a + math.pi) % TWO_PI) - math.pi for a in angles]

        # First try as-is
        centers2, s2 = repair_scale(centers, angles)
        if s2 < best_s:
            best = (centers2, angles, s2)
            best_s = s2

        # Then local stochastic refinement
        refined_centers, refined_angles, refined_s = optimize_layout(
            centers2, angles, iters=220 if n <= 10 else 260, seed=1000 + 37 * n + idx
        )
        if refined_s < best_s:
            best = (refined_centers, refined_angles, refined_s)
            best_s = refined_s

    centers, angles, s = best

    # Final safety pass: tiny outward scaling if needed, then exact enclosing side.
    if has_overlap(centers, angles):
        centers, s = repair_scale(centers, angles)

    # Ensure all centers are list of tuples, angles are floats.
    centers = [(float(x), float(y)) for x, y in centers]
    angles = [float(a) for a in angles]
    s = float(enclosing_side(centers, angles))
    return centers, angles, s
