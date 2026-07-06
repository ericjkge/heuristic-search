"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles:  list[float]
- s: outer square side length
Square convention: point p is inside iff max(|px|, |py|) <= s/2.

This implementation uses a hand-tuned template family for n=10 and a few
small n, plus a geometric validity check and local refinement by shrinking
the container. The goal is not a proof of optimality, but a substantially
better packing than a naive grid.
"""

import math
from itertools import combinations

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # inradius
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

EPS = 1e-9


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
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        ux, uy = -(y2 - y1), (x2 - x1)
        norm = math.hypot(ux, uy)
        if norm > 0:
            axes.append((ux / norm, uy / norm))
    return axes


def project(poly, ax, ay):
    vals = [x * ax + y * ay for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb):
    # Standard SAT. If any axis separates, they do not overlap.
    for ax, ay in poly_axes(pa) + poly_axes(pb):
        amin, amax = project(pa, ax, ay)
        bmin, bmax = project(pb, ax, ay)
        if amax <= bmin + EPS or bmax <= amin + EPS:
            return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i, j in combinations(range(len(polys)), 2):
        if polygons_overlap(polys[i], polys[j]):
            return True
    return False


def inside_square(centers, angles, s):
    half = s / 2.0 + 1e-12
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            if abs(vx) > half or abs(vy) > half:
                return False
    return True


def valid_packing(centers, angles, s):
    return inside_square(centers, angles, s) and not has_overlap(centers, angles)


def normalize_to_square(centers, angles):
    s = enclosing_side(centers, angles)
    return centers, angles, s


def shrink_toward_origin(centers, angles, factor):
    return [(x * factor, y * factor) for x, y in centers]


def binary_shrink(centers, angles, lo=0.0, hi=1.0):
    # Find maximum lambda in [lo, hi] such that scaled centers still valid.
    # We assume angles fixed and that lambda=hi is valid.
    assert valid_packing(shrink_toward_origin(centers, angles, hi), angles, enclosing_side(shrink_toward_origin(centers, angles, hi), angles))
    for _ in range(70):
        mid = (lo + hi) / 2.0
        c2 = shrink_toward_origin(centers, angles, mid)
        s2 = enclosing_side(c2, angles)
        if valid_packing(c2, angles, s2):
            lo = mid
        else:
            hi = mid
    return shrink_toward_origin(centers, angles, lo), enclosing_side(shrink_toward_origin(centers, angles, lo), angles)


def template_n10():
    """
    A 10-pentagon motif designed for a square container.
    Uses opposite orientations and slightly staggered rows to exploit the
    known tendency of pentagons to pack in mixed-orientation motifs.
    """
    ang_a = math.pi / 10.0
    ang_b = ang_a + math.pi

    # Base geometry: 5 pairs, arranged as 3+4+3 in y.
    # Hand-tuned coordinates; then we refine by shrinking if possible.
    centers = [
        (-1.55,  1.12), ( 0.00,  1.18), ( 1.55,  1.12),
        (-2.05,  0.00), (-0.68,  0.00), ( 0.68,  0.00), ( 2.05,  0.00),
        (-1.55, -1.12), ( 0.00, -1.18), ( 1.55, -1.12),
    ]
    angles = [
        ang_a, ang_b, ang_a,
        ang_b, ang_a, ang_b, ang_a,
        ang_b, ang_a, ang_b,
    ]
    return centers, angles


def template_n9():
    ang_a = math.pi / 10.0
    ang_b = ang_a + math.pi
    centers = [
        (-1.45,  1.08), ( 0.00,  1.12), ( 1.45,  1.08),
        (-1.95,  0.00), ( 0.00,  0.00), ( 1.95,  0.00),
        (-1.45, -1.08), ( 0.00, -1.12), ( 1.45, -1.08),
    ]
    angles = [ang_a, ang_b, ang_a, ang_b, ang_a, ang_b, ang_a, ang_b, ang_a]
    return centers, angles


def template_n8():
    ang_a = math.pi / 10.0
    ang_b = ang_a + math.pi
    centers = [
        (-1.45,  0.95), ( 0.00,  1.02), ( 1.45,  0.95),
        (-1.95, -0.05), (-0.50, -0.05), ( 0.95, -0.05), ( 2.40, -0.05),
        ( 0.00, -1.10),
    ]
    angles = [ang_a, ang_b, ang_a, ang_b, ang_a, ang_b, ang_a, ang_b]
    return centers, angles


def regular_grid_fallback(n):
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pitch_x = WIDTH + 0.05
    pitch_y = HEIGHT + 0.05
    centers, angles = [], []
    for k in range(n):
        i, j = k % cols, k // cols
        x = (i - (cols - 1) / 2.0) * pitch_x
        y = (j - (rows - 1) / 2.0) * pitch_y
        centers.append((x, y))
        angles.append((math.pi / 10.0) if (i + j) % 2 == 0 else (math.pi / 10.0 + math.pi))
    return centers, angles


def refine_by_global_shrink(centers, angles):
    # Start with a valid packing; then attempt to scale all centers toward the origin
    # as much as possible while preserving validity.
    s0 = enclosing_side(centers, angles)
    if not valid_packing(centers, angles, s0):
        # If invalid due to numerical issues, inflate slightly until valid.
        hi = 1.0
        for _ in range(40):
            c2 = shrink_toward_origin(centers, angles, hi)
            s2 = enclosing_side(c2, angles)
            if valid_packing(c2, angles, s2):
                centers = c2
                s0 = s2
                break
            hi *= 1.05

    lo, hi = 0.0, 1.0
    # ensure hi valid
    if not valid_packing(shrink_toward_origin(centers, angles, hi), angles, enclosing_side(shrink_toward_origin(centers, angles, hi), angles)):
        return centers, angles, enclosing_side(centers, angles)

    for _ in range(80):
        mid = (lo + hi) / 2.0
        c2 = shrink_toward_origin(centers, angles, mid)
        s2 = enclosing_side(c2, angles)
        if valid_packing(c2, angles, s2):
            hi = mid
        else:
            lo = mid
    c2 = shrink_toward_origin(centers, angles, hi)
    return c2, angles, enclosing_side(c2, angles)


def pack(n):
    """Return a packing of n unit regular pentagons in an axis-aligned square."""
    if n <= 0:
        return [], [], 0.0

    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 10.0]
        return centers, angles, enclosing_side(centers, angles)

    if n == 2:
        ang_a = math.pi / 10.0
        ang_b = ang_a + math.pi
        centers = [(-0.80, 0.0), (0.80, 0.0)]
        angles = [ang_a, ang_b]
        return refine_by_global_shrink(centers, angles)

    if n == 3:
        ang_a = math.pi / 10.0
        ang_b = ang_a + math.pi
        centers = [(-1.05, 0.0), (0.0, 0.0), (1.05, 0.0)]
        angles = [ang_a, ang_b, ang_a]
        return refine_by_global_shrink(centers, angles)

    if n == 4:
        ang_a = math.pi / 10.0
        ang_b = ang_a + math.pi
        centers = [(-0.95, -0.7), (0.95, -0.7), (-0.95, 0.7), (0.95, 0.7)]
        angles = [ang_a, ang_b, ang_b, ang_a]
        return refine_by_global_shrink(centers, angles)

    if n == 5:
        ang_a = math.pi / 10.0
        ang_b = ang_a + math.pi
        centers = [(-1.35, 0.0), (-0.45, 0.88), (0.45, 0.0), (1.35, 0.88), (0.0, -0.95)]
        angles = [ang_a, ang_b, ang_a, ang_b, ang_a]
        return refine_by_global_shrink(centers, angles)

    if n == 6:
        ang_a = math.pi / 10.0
        ang_b = ang_a + math.pi
        centers = [(-1.45, 0.9), (0.0, 0.95), (1.45, 0.9), (-1.45, -0.9), (0.0, -0.95), (1.45, -0.9)]
        angles = [ang_a, ang_b, ang_a, ang_b, ang_a, ang_b]
        return refine_by_global_shrink(centers, angles)

    if n == 7:
        ang_a = math.pi / 10.0
        ang_b = ang_a + math.pi
        centers = [(-1.55, 1.0), (0.0, 1.05), (1.55, 1.0), (-2.0, 0.0), (0.0, 0.0), (2.0, 0.0), (0.0, -1.05)]
        angles = [ang_a, ang_b, ang_a, ang_b, ang_a, ang_b, ang_a]
        return refine_by_global_shrink(centers, angles)

    if n == 8:
        centers, angles = template_n8()
        return refine_by_global_shrink(centers, angles)

    if n == 9:
        centers, angles = template_n9()
        return refine_by_global_shrink(centers, angles)

    if n == 10:
        centers, angles = template_n10()
        # Small local improvement: try a few mirrored / shifted variants and keep the best valid one.
        candidates = []
        variants = [
            (0.0, 0.0, 0.0),
            (0.02, 0.00, 0.0),
            (-0.02, 0.00, 0.0),
            (0.00, 0.02, 0.0),
            (0.00, -0.02, 0.0),
            (0.01, 0.01, 0.02),
            (-0.01, -0.01, -0.02),
        ]
        for dx, dy, da in variants:
            c = [(x + dx, y + dy) for x, y in centers]
            a = [ang + da for ang in angles]
            c, a, s = refine_by_global_shrink(c, a)
            if valid_packing(c, a, s):
                candidates.append((s, c, a))
        if candidates:
            candidates.sort(key=lambda t: t[0])
            _, centers, angles = candidates[0]
            return centers, angles, enclosing_side(centers, angles)
        return refine_by_global_shrink(centers, angles)

    # Generic fallback for larger n: staggered rows with alternating orientations.
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pitch_x = WIDTH * 0.94
    pitch_y = HEIGHT * 0.93

    centers, angles = [], []
    ang_a = math.pi / 10.0
    ang_b = ang_a + math.pi
    for k in range(n):
        i, j = k % cols, k // cols
        x = (i - (cols - 1) / 2.0) * pitch_x + (0.18 if j % 2 else 0.0)
        y = (j - (rows - 1) / 2.0) * pitch_y
        centers.append((x, y))
        angles.append(ang_a if (i + j) % 2 == 0 else ang_b)

    centers, angles, _ = refine_by_global_shrink(centers, angles)
    return centers, angles, enclosing_side(centers, angles)
