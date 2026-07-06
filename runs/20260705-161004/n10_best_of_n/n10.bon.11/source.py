"""Pack n unit regular pentagons into the smallest origin-centered square.

Contract: pack(n) -> (centers, angles, s). Container = origin-centered
axis-aligned square, side s: point p inside iff max(|px|, |py|) <= s/2.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius ~ 0.8507
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # ~ 0.6882


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
    ) if centers else 0.0


# EVOLVE-BLOCK-START
OVERLAP_EPS = 1e-9


def _poly_axes(poly):
    axes = []
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        ux, uy = -(y2 - y1), (x2 - x1)
        norm = math.hypot(ux, uy)
        if norm > 0:
            axes.append((ux / norm, uy / norm))
    return axes


def _proj(poly, ax, ay):
    vals = [x * ax + y * ay for x, y in poly]
    return min(vals), max(vals)


def pentagons_overlap(pa, pb):
    for ax, ay in _poly_axes(pa):
        amin, amax = _proj(pa, ax, ay)
        bmin, bmax = _proj(pb, ax, ay)
        if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS:
            return False
    for ax, ay in _poly_axes(pb):
        amin, amax = _proj(pa, ax, ay)
        bmin, bmax = _proj(pb, ax, ay)
        if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS:
            return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if pentagons_overlap(polys[i], polys[j]):
                return True
    return False


def feasible(centers, angles, s):
    half = s / 2.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            if abs(vx) > half + 1e-9 or abs(vy) > half + 1e-9:
                return False
    return not has_overlap(centers, angles)


def _scale_centers(centers, lam):
    return [(x * lam, y * lam) for x, y in centers]


def repair_scale(centers, angles):
    """Scale centers about origin until overlaps disappear, if possible."""
    if not has_overlap(centers, angles):
        return centers
    lo, hi = 1.0, 1.2
    for _ in range(50):
        if not has_overlap(_scale_centers(centers, hi), angles):
            break
        lo, hi = hi, hi * 1.35
    else:
        return centers
    for _ in range(70):
        mid = (lo + hi) / 2.0
        if has_overlap(_scale_centers(centers, mid), angles):
            lo = mid
        else:
            hi = mid
    return _scale_centers(centers, hi)


def _evaluate(centers, angles):
    s = enclosing_side(centers, angles)
    return s, centers, angles


def _clip_to_square(centers, angles, s):
    """Move centers toward origin just enough to fit in the square, preserving angles."""
    half = s / 2.0
    new_centers = []
    for (cx, cy), ang in zip(centers, angles):
        verts = pentagon_vertices(cx, cy, ang)
        mx = max(abs(x) for x, y in verts)
        my = max(abs(y) for x, y in verts)
        scale = 1.0
        if mx > half:
            scale = min(scale, (half - 1e-9) / mx)
        if my > half:
            scale = min(scale, (half - 1e-9) / my)
        new_centers.append((cx * scale, cy * scale))
    return new_centers


def _pattern_rows(n, w=2, h=2):
    """A compact mixed-orientation motif inspired by double-lattice packing."""
    # 4-pentagon motif: two opposite orientations, slightly staggered.
    # Chosen to be conservative, then tightened by search.
    motif = [
        (0.0000, 0.0000, math.pi / 2.0),
        (0.7200, 0.2200, -math.pi / 2.0),
        (0.3600, 1.1700, math.pi / 2.0),
        (1.0800, 1.3900, -math.pi / 2.0),
    ]
    pts = []
    # generate a hex-like staggered lattice of motifs
    sx = 1.430
    sy = 1.520
    for j in range(h):
        for i in range(w):
            ox = i * sx + (j % 2) * 0.715
            oy = j * sy
            for dx, dy, a in motif:
                pts.append((ox + dx, oy + dy, a))
    pts = pts[:n]
    return pts


def _center_and_pack(raw):
    xs = [x for x, y, a in raw]
    ys = [y for x, y, a in raw]
    mx = (min(xs) + max(xs)) / 2.0
    my = (min(ys) + max(ys)) / 2.0
    centers = [(x - mx, y - my) for x, y, a in raw]
    angles = [a for x, y, a in raw]
    return centers, angles


def _try_construction(n):
    # Several candidate constructions; choose the best feasible one.
    candidates = []

    # 1) Mixed-orientation staggered rows.
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    dx = 1.18
    dy = 1.30
    raw = []
    for j in range(rows):
        for i in range(cols):
            if len(raw) >= n:
                break
            x = (i - (cols - 1) / 2.0) * dx + (j % 2) * 0.55
            y = (j - (rows - 1) / 2.0) * dy
            ang = math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0
            raw.append((x, y, ang))
    candidates.append(_center_and_pack(raw))

    # 2) Diagonal chain alternating orientations.
    raw = []
    step = 1.05
    for k in range(n):
        t = k - (n - 1) / 2.0
        x = t * step
        y = ((k % 3) - 1) * 0.62 + t * 0.32
        ang = math.pi / 2.0 if k % 2 == 0 else -math.pi / 2.0
        raw.append((x, y, ang))
    candidates.append(_center_and_pack(raw))

    # 3) Motif grid.
    w = max(1, int(math.ceil(math.sqrt(n / 4.0))))
    h = int(math.ceil(n / (4.0 * w)))
    raw = _pattern_rows(n, w=w, h=h)
    candidates.append(_center_and_pack(raw))

    best = None
    for centers, angles in candidates:
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)
        if feasible(centers, angles, s):
            if best is None or s < best[0]:
                best = (s, centers, angles)
    return best


def _local_search(centers, angles, iterations=4000):
    """Simple randomized hill-climb on a fixed orientation pattern."""
    rng = random.Random(123456789)
    n = len(centers)

    def score(c):
        return enclosing_side(c, angles)

    best = list(centers)
    best_s = score(best)

    # Progressive step sizes.
    steps = [0.20, 0.12, 0.07, 0.04, 0.02, 0.01]
    for step in steps:
        for _ in range(iterations // len(steps)):
            c = list(best)
            i = rng.randrange(n)
            dx = rng.uniform(-step, step)
            dy = rng.uniform(-step, step)
            c[i] = (c[i][0] + dx, c[i][1] + dy)

            # keep centered
            mx = sum(x for x, y in c) / n
            my = sum(y for x, y in c) / n
            c = [(x - mx, y - my) for x, y in c]

            # gently shrink if possible, then verify
            s = enclosing_side(c, angles)
            if s >= best_s + 1e-12:
                continue
            if not feasible(c, angles, s):
                continue
            best = c
            best_s = s

    return best, best_s
# EVOLVE-BLOCK-END


def pack(n):
    """Construct a reasonably tight packing using mixed orientations and local search."""
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        return [(0.0, 0.0)], [0.0], 2.0 * R

    best = _try_construction(n)

    # Fallback: simple compact rows if heuristics fail.
    if best is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        dx = 1.25
        dy = 1.34
        centers, angles = [], []
        for k in range(n):
            i, j = k % cols, k // cols
            centers.append(((i - (cols - 1) / 2.0) * dx,
                            (j - (rows - 1) / 2.0) * dy))
            angles.append(math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0)
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)
        return centers, angles, s

    s, centers, angles = best

    # Short local improvement on center positions.
    centers, s2 = _local_search(centers, angles, iterations=5000)
    if s2 < s:
        s = s2

    # Final guard: ensure containment and no overlap.
    s = enclosing_side(centers, angles)
    if not feasible(centers, angles, s):
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)

    return centers, angles, s
