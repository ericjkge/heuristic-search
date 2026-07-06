"""Seed program: pack n unit regular pentagons into a square, minimizing side s.

Contract: pack(n) -> (centers, angles, s). Container = origin-centered
axis-aligned square, side s: point p inside iff max(|px|, |py|) <= s/2.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # inradius
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # widest extent
HEIGHT = R + APOTHEM                              # point-up bounding height

TWO_PI = 2.0 * math.pi


def pentagon_vertices(cx, cy, angle):
    """The 5 vertices of a unit pentagon centered at (cx, cy)."""
    return [
        (cx + R * math.cos(angle + k * TWO_PI / 5.0),
         cy + R * math.sin(angle + k * TWO_PI / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    """Smallest origin-centered square side enclosing all pentagons."""
    m = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            m = max(m, abs(vx), abs(vy))
    return 2.0 * m


# EVOLVE-BLOCK-START
EPS = 1e-9


def polygon_axes(poly):
    axes = []
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        ux, uy = -(y2 - y1), (x2 - x1)
        n = math.hypot(ux, uy)
        if n > 0:
            axes.append((ux / n, uy / n))
    return axes


def project(poly, ax, ay):
    lo = hi = poly[0][0] * ax + poly[0][1] * ay
    for x, y in poly[1:]:
        v = x * ax + y * ay
        if v < lo:
            lo = v
        elif v > hi:
            hi = v
    return lo, hi


def polygons_overlap(pa, pb):
    """Separating axis theorem for convex polygons."""
    for ax, ay in polygon_axes(pa) + polygon_axes(pb):
        amin, amax = project(pa, ax, ay)
        bmin, bmax = project(pb, ax, ay)
        if min(amax, bmax) - max(amin, bmin) <= EPS:
            return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def repair(centers, angles):
    """Smallest uniform dilation about origin that clears all overlaps."""
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.1
    for _ in range(60):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.25
    else:
        return centers

    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def _grid_candidates():
    """Generate good base lattice parameters for mixed-orientation rows."""
    # These are heuristic, from the fact that pentagons pack well in staggered,
    # opposite-orientation motifs. We search a small family of row/column spacings.
    for dx in [1.16, 1.18, 1.20, 1.22, 1.24, 1.26]:
        for dy in [0.92, 0.96, 1.00, 1.04, 1.08]:
            for shift in [0.0, 0.5 * dx, 0.33 * dx, 0.67 * dx]:
                yield dx, dy, shift


def _layout_for(n, dx, dy, shift, mode):
    """Create a compact staggered layout with orientations chosen for interlock."""
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))

    # Center the grid around origin.
    centers, angles = [], []
    idx = 0

    # Two possible scan orders: row-major and snake; choose the denser one by s.
    row_indices = list(range(rows))
    col_indices = list(range(cols))

    if mode == 1:
        # snake-like ordering reduces long boundary swings
        for j in row_indices:
            cols_it = col_indices if j % 2 == 0 else list(reversed(col_indices))
            for i in cols_it:
                if idx >= n:
                    break
                x = (i - (cols - 1) / 2.0) * dx + (shift if j % 2 else 0.0)
                y = (j - (rows - 1) / 2.0) * dy
                # Alternate opposite orientations, with a row-dependent flip.
                ang = math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0
                centers.append((x, y))
                angles.append(ang)
                idx += 1
            if idx >= n:
                break
    else:
        # row-major with column phase shift
        for k in range(n):
            i, j = k % cols, k // cols
            x = (i - (cols - 1) / 2.0) * dx + (shift if j % 2 else 0.0)
            y = (j - (rows - 1) / 2.0) * dy
            ang = math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0
            centers.append((x, y))
            angles.append(ang)

    return centers, angles


def _ring_layout(n):
    """Fallback for small n and a few awkward counts: distribute on concentric rings."""
    centers, angles = [], []
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]

    # Rough ring capacities.
    ring_radii = [0.0, 1.05, 2.05, 3.05]
    ring_counts = [1, 5, 10, 20]
    remaining = n

    # First center.
    centers.append((0.0, 0.0))
    angles.append(math.pi / 2.0)
    remaining -= 1
    if remaining <= 0:
        return centers, angles

    ring = 1
    while remaining > 0 and ring < len(ring_radii):
        c = min(remaining, ring_counts[ring])
        r = ring_radii[ring]
        for t in range(c):
            th = TWO_PI * t / c + (0.5 * math.pi / c)
            x = r * math.cos(th)
            y = r * math.sin(th)
            # Alternate orientation every other placement and ring.
            ang = math.pi / 2.0 if (t + ring) % 2 == 0 else -math.pi / 2.0
            centers.append((x, y))
            angles.append(ang)
        remaining -= c
        ring += 1

    return centers[:n], angles[:n]


def _maximize_compactness(n):
    """Search a small parameter family and keep the best valid packing."""
    best = None
    best_s = float("inf")

    # Deterministic exploration of several motif families.
    for mode in [0, 1]:
        for dx, dy, shift in _grid_candidates():
            centers, angles = _layout_for(n, dx, dy, shift, mode)
            centers = repair(centers, angles)
            s = enclosing_side(centers, angles)
            if s < best_s:
                best_s = s
                best = (centers, angles, s)

    # For tiny n and some awkward counts, ring layouts can beat grids.
    if n <= 12 or n in (13, 14, 15, 16, 17):
        centers, angles = _ring_layout(n)
        centers = repair(centers, angles)
        s = enclosing_side(centers, angles)
        if s < best_s:
            best_s = s
            best = (centers, angles, s)

    return best


def _local_optimize(centers, angles, rounds=2500, seed=0):
    """Lightweight randomized improvement: nudge positions and sometimes flip orientation.
    Validity is enforced by repair and final overlap checks."""
    rnd = random.Random(seed)
    best_centers = list(centers)
    best_angles = list(angles)
    best_s = enclosing_side(best_centers, best_angles)

    for step in range(rounds):
        cur_centers = [list(p) for p in best_centers]
        cur_angles = list(best_angles)

        # Randomly perturb a small subset.
        m = max(1, len(cur_centers) // 4)
        for _ in range(m):
            i = rnd.randrange(len(cur_centers))
            cur_centers[i][0] += rnd.uniform(-0.10, 0.10)
            cur_centers[i][1] += rnd.uniform(-0.10, 0.10)
            if rnd.random() < 0.30:
                cur_angles[i] = (cur_angles[i] + math.pi) % TWO_PI

        cur_centers = [(x, y) for x, y in cur_centers]
        cur_centers = repair(cur_centers, cur_angles)
        s = enclosing_side(cur_centers, cur_angles)
        if s < best_s and not has_overlap(cur_centers, cur_angles):
            best_s = s
            best_centers = list(cur_centers)
            best_angles = list(cur_angles)

        # Occasional re-symmetrization helps keep the packing centered and compact.
        if step % 50 == 0:
            cx = sum(x for x, _ in best_centers) / len(best_centers)
            cy = sum(y for _, y in best_centers) / len(best_centers)
            shifted = [(x - cx, y - cy) for x, y in best_centers]
            shifted = repair(shifted, best_angles)
            s2 = enclosing_side(shifted, best_angles)
            if s2 < best_s and not has_overlap(shifted, best_angles):
                best_s = s2
                best_centers = list(shifted)

    return best_centers, best_angles, best_s


def pack(n):
    """Pack n regular pentagons into the smallest square found by our heuristic search."""
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    # Deterministic base search.
    centers, angles, s = _maximize_compactness(n)

    # Randomized refinement around the best deterministic candidate.
    # Seed depends only on n for reproducibility.
    centers2, angles2, s2 = _local_optimize(centers, angles, rounds=1800 + 40 * n, seed=1234567 + n)
    if s2 < s:
        centers, angles, s = centers2, angles2, s2

    # Final safety pass.
    centers = repair(centers, angles)
    s = enclosing_side(centers, angles)

    return centers, angles, s
# EVOLVE-BLOCK-END
