"""Seed program: pack n unit regular hexagons into a regular hexagon, minimizing side s.

Contract: pack(n) -> (centers, angles, s). Container = origin-centered regular
hexagon, flat-top (edge-normals at 30/90/150 deg), side s.
"""

import math

SIDE = 1.0
R = SIDE  # circumradius of a unit hexagon = its side
PITCH = math.sqrt(3.0) * SIDE  # center distance of two edge-sharing unit hexagons

_CONTAINER_NORMALS = [
    (math.cos(a), math.sin(a)) for a in (math.pi / 6.0, math.pi / 2.0, 5.0 * math.pi / 6.0)
]


def hex_vertices(cx, cy, angle):
    """The 6 vertices of a unit hexagon centered at (cx, cy)."""
    return [
        (cx + R * math.cos(angle + k * math.pi / 3.0),
         cy + R * math.sin(angle + k * math.pi / 3.0))
        for k in range(6)
    ]


def enclosing_side(centers, angles):
    """Smallest container side enclosing all hexagons (exact support-function calc)."""
    max_support = max(
        abs(vx * ux + vy * uy)
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in hex_vertices(cx, cy, ang)
        for ux, uy in _CONTAINER_NORMALS
    )
    return max_support * 2.0 / math.sqrt(3.0)


# EVOLVE-BLOCK-START
OVERLAP_EPS = 1e-7  # separation gap below this counts as clear of the validator's TOL


def hexes_overlap(pa, pb):
    """Separating-axis test for two unit hexagons given as vertex lists."""
    for poly in (pa, pb):
        for i in range(6):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 6]
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
    polys = [hex_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    return any(
        hexes_overlap(polys[i], polys[j])
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
    """Baseline: the n triangular-lattice cells nearest the origin, axis-aligned."""
    v1 = (PITCH * math.cos(math.pi / 6.0), PITCH * math.sin(math.pi / 6.0))
    v2 = (0.0, PITCH)

    span = int(math.ceil(math.sqrt(n))) + 3
    pts = sorted(
        (x * x + y * y, x, y)
        for i in range(-span, span + 1)
        for j in range(-span, span + 1)
        for x, y in [(i * v1[0] + j * v2[0], i * v1[1] + j * v2[1])]
    )
    chosen = [(x, y) for _, x, y in pts[:n]]

    # center the cluster on the origin
    gx = sum(x for x, _ in chosen) / n
    gy = sum(y for _, y in chosen) / n
    centers = [(x - gx, y - gy) for x, y in chosen]
    angles = [0.0] * n

    centers = repair(centers, angles)
    return centers, angles, enclosing_side(centers, angles)
# EVOLVE-BLOCK-END
