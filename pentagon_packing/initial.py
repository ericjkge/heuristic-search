"""Seed program: pack n unit regular pentagons into a square, minimizing side s.

Contract: pack(n) -> (centers, angles, s). Container = origin-centered
axis-aligned square, side s: point p inside iff max(|px|, |py|) <= s/2.

Baseline: pentagonal ice-ray double lattice (Hales-Kuperberg-Kusner optimal
plane packing, density (5-sqrt(5))/3 ~ 0.92131).
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


# --- pentagonal ice-ray double lattice --------------------------------------
# The optimal plane packing of regular pentagons (Kuperberg^2 construction,
# proved optimal by Hales-Kusner; density (5-sqrt(5))/3 ~ 0.92131) is a double
# lattice: point-up pentagons on lattice {j*V1 + k*V2}, plus their half-turn
# images offset by T_Q. Derived here by optimizing the general double lattice
# (SLSQP on cell area with separating-axis contact constraints) to machine
# precision, where every constant snapped to closed form:
#   V1 = (3*phi/2)*(cos 108, sin 108), with x-component exactly -3/4
#   V2 = (R + APOTHEM)*(cos 18, sin 18)   (pentagon min-width step; V1 _|_ V2)
#   T_Q = (-cos 72, -2*APOTHEM)           (anti-parallel edge-flush, off-midpoint)
# Cell area (R+APOTHEM)*(3*phi/2) = 3.7348... holds the density identity. All
# contacts are exactly flush; the validator's TOL absorbs float error.

PHI = (1.0 + math.sqrt(5.0)) / 2.0
UP = math.pi / 2.0     # point-up
DOWN = -math.pi / 2.0  # its half-turn image
_C18, _S18 = math.cos(math.pi / 10.0), math.sin(math.pi / 10.0)
V1 = (-0.75, 1.5 * PHI * _C18)
V2 = (HEIGHT * _C18, HEIGHT * _S18)
T_Q = (-math.cos(0.4 * math.pi), -2.0 * APOTHEM)


def _sites(m):
    """Ice-ray sites (cx, cy, angle) for lattice indices |j|, |k| <= m."""
    out = []
    for j in range(-m, m + 1):
        for k in range(-m, m + 1):
            x, y = j * V1[0] + k * V2[0], j * V1[1] + k * V2[1]
            out.append((x, y, UP))
            out.append((x + T_Q[0], y + T_Q[1], DOWN))
    return out


def pack(n):
    """Best n-pentagon window of the ice-ray lattice: over a grid of global
    rotations (square is 90deg-symmetric) and lattice phases, rank sites by the
    Chebyshev radius of their vertices about the window center, keep the n
    smallest, recenter the bounding box on the origin (enclosing_side is only
    honest for centered packings). The lattice is collision-free by
    construction; has_overlap/repair stay as a safety net."""
    # Patch size: an n-site window has side ~ sqrt(n / site_density) =
    # sqrt(1.87n); the patch's inscribed disc radius is m*min(|V1|,|V2|) =
    # 1.539m (V1 _|_ V2). Containing the window's circumscribed disc needs
    # m >= sqrt(n)/1.5; +1 margin covers the T_Q offset and bbox recentering.
    m = math.ceil(math.sqrt(n) / 1.5) + 1
    sites = _sites(m)
    best = None
    for ri in range(72):
        th = (math.pi / 2.0) * ri / 72.0
        c, s_ = math.cos(th), math.sin(th)
        rsites = [(x * c - y * s_, x * s_ + y * c, a + th) for x, y, a in sites]
        verts = [pentagon_vertices(x, y, a) for x, y, a in rsites]
        for fa in (0.0, 1.0 / 3.0, 2.0 / 3.0):
            for fb in (0.0, 1.0 / 3.0, 2.0 / 3.0):
                px = fa * V1[0] + fb * V2[0]
                py = fa * V1[1] + fb * V2[1]
                cx, cy = px * c - py * s_, px * s_ + py * c
                ranked = sorted(range(len(rsites)), key=lambda i: max(
                    max(abs(vx - cx), abs(vy - cy)) for vx, vy in verts[i]))
                pick = ranked[:n]
                vs = [v for i in pick for v in verts[i]]
                ox = (min(v[0] for v in vs) + max(v[0] for v in vs)) / 2.0
                oy = (min(v[1] for v in vs) + max(v[1] for v in vs)) / 2.0
                side = 2.0 * max(max(abs(vx - ox), abs(vy - oy)) for vx, vy in vs)
                if best is None or side < best[0]:
                    best = (side,
                            [(rsites[i][0] - ox, rsites[i][1] - oy) for i in pick],
                            [rsites[i][2] for i in pick])
    _, centers, angles = best
    if has_overlap(centers, angles):
        centers = repair(centers, angles)
    return centers, angles, enclosing_side(centers, angles)
# EVOLVE-BLOCK-END
