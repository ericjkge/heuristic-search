"""Seed program: pack n unit regular pentagons into a square, minimizing side s.

Contract: pack(n) -> (centers, angles, s). Container = origin-centered
axis-aligned square, side s: point p inside iff max(|px|, |py|) <= s/2.
"""

import math

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius ~ 0.8507
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # ~ 0.6882
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # widest extent (diagonal) ~ 1.618
HEIGHT = R + APOTHEM                              # point-up bounding height ~ 1.539


def pentagon_vertices(cx, cy, angle):
    """The 5 vertices of a unit pentagon centered at (cx, cy)."""
    return [
        (cx + R * math.cos(angle + k * 2.0 * math.pi / 5.0),
         cy + R * math.sin(angle + k * 2.0 * math.pi / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    """Smallest origin-centered square side enclosing all pentagons."""
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, ang)
    )


# EVOLVE-BLOCK-START
OVERLAP_EPS = 1e-7  # separation gap below this counts as clear of the validator's TOL


def pentagons_overlap(pa, pb):
    """Separating-axis test for two unit pentagons given as vertex lists."""
    for poly in (pa, pb):
        for i in range(5):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
            ux, uy = -(y2 - y1), (x2 - x1)
            amin = min(x * ux + y * uy for x, y in pa)
            amax = max(x * ux + y * uy for x, y in pa)
            bmin = min(x * ux + y * uy for x, y in pb)
            bmax = max(x * ux + y * uy for x, y in pb)
            norm = math.hypot(ux, uy)
            if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS * norm:
                return False  # a separating axis exists
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    return any(
        pentagons_overlap(polys[i], polys[j])
        for i in range(len(polys)) for j in range(i + 1, len(polys))
    )


def repair(centers, angles):
    """Dilate all centers about the origin by the smallest factor clearing every
    overlap (rotations unchanged). Turns any proposal into a valid packing with an
    honest -- possibly larger -- s. Pure geometry, not an optimizer. Coincident
    centers cannot be separated by dilation; those come back unchanged."""
    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    if not has_overlap(centers, angles):
        return centers
    lo, hi = 1.0, 2.0
    for _ in range(30):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 2.0
    else:
        return centers  # not separable (e.g. coincident centers)
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def pack(n):
    """Baseline: near-square grid of point-up pentagons, alternate columns flipped
    (rotated pi) so rows can interlock slightly after repair."""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pitch_x = WIDTH + 0.02
    pitch_y = HEIGHT + 0.02

    centers, angles = [], []
    for k in range(n):
        i, j = k % cols, k // cols
        x = (i - (cols - 1) / 2.0) * pitch_x
        y = (j - (rows - 1) / 2.0) * pitch_y
        centers.append((x, y))
        angles.append(math.pi / 2.0 if i % 2 == 0 else -math.pi / 2.0)

    centers = repair(centers, angles)
    return centers, angles, enclosing_side(centers, angles)
# EVOLVE-BLOCK-END
