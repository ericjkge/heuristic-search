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
OVERLAP_EPS = 1e-8  # tighter tolerance for robust geometry


def pentagon_vertices(cx, cy, angle):
    """The 5 vertices of a unit pentagon centered at (cx, cy)."""
    ca, sa = math.cos(angle), math.sin(angle)
    verts = []
    for k in range(5):
        t = angle + k * 2.0 * math.pi / 5.0
        verts.append((cx + R * math.cos(t), cy + R * math.sin(t)))
    return verts


def poly_axes(poly):
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        yield (-(y2 - y1), x2 - x1)


def project(poly, ux, uy):
    vals = [x * ux + y * uy for x, y in poly]
    return min(vals), max(vals)


def pentagons_overlap(pa, pb):
    """Separating-axis test for two unit pentagons given as vertex lists."""
    for poly in (pa, pb):
        for ux, uy in poly_axes(poly):
            amin, amax = project(pa, ux, uy)
            bmin, bmax = project(pb, ux, uy)
            norm = math.hypot(ux, uy)
            if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS * norm:
                return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if pentagons_overlap(polys[i], polys[j]):
                return True
    return False


def enclosing_side(centers, angles):
    """Smallest origin-centered square side enclosing all pentagons."""
    m = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            m = max(m, abs(vx), abs(vy))
    return 2.0 * m


def translate_to_origin(centers):
    mx = max(abs(x) for x, _ in centers) if centers else 0.0
    my = max(abs(y) for _, y in centers) if centers else 0.0
    return [(x, y) for x, y in centers], mx, my


def eval_packing(centers, angles):
    return enclosing_side(centers, angles)


def seed_layouts(n):
    """Generate several structured orientation/placement templates."""
    layouts = []

    # 1) Hex-like staggered rows with alternating orientations.
    for rows in range(1, n + 1):
        cols = int(math.ceil(n / rows))
        centers, angles = [], []
        dx = WIDTH * 0.82
        dy = HEIGHT * 0.84
        idx = 0
        for r in range(rows):
            count = cols if r < rows - 1 else n - idx
            xoff = -0.5 * (count - 1) * dx
            y = (r - (rows - 1) / 2.0) * dy
            for c in range(count):
                x = xoff + c * dx + (0.5 * dx if (r % 2 == 1) else 0.0)
                centers.append((x, y))
                angles.append(0.5 * math.pi if ((r + c) % 2 == 0) else -0.5 * math.pi)
                idx += 1
        layouts.append((centers, angles))

    # 2) Compact rectangle grids with boundary rows tilted.
    for cols in range(1, n + 1):
        rows = int(math.ceil(n / cols))
        centers, angles = [], []
        dx = WIDTH * 0.80
        dy = HEIGHT * 0.82
        for k in range(n):
            i, j = k % cols, k // cols
            x = (i - (cols - 1) / 2.0) * dx + (0.35 * dx if (j % 2 == 1) else 0.0)
            y = (j - (rows - 1) / 2.0) * dy
            centers.append((x, y))
            if j == 0 or j == rows - 1:
                angles.append(0.5 * math.pi if i % 2 == 0 else -0.5 * math.pi)
            else:
                angles.append(math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0)
        layouts.append((centers, angles))

    # 3) Double-lattice inspired 2-orientation motifs with a few discrete offsets.
    a = math.pi / 2.0
    b = -math.pi / 2.0
    shifts = [
        (0.0, 0.0),
        (0.45 * WIDTH, 0.18 * HEIGHT),
        (-0.45 * WIDTH, 0.18 * HEIGHT),
        (0.0, -0.36 * HEIGHT),
    ]
    for sx, sy in shifts:
        centers, angles = [], []
        for k in range(n):
            row = k // 2
            col = k % 2
            x = (col - 0.5) * WIDTH * 0.72 + (row % 2) * sx
            y = (row - (n // 2)) * HEIGHT * 0.38 + (col * 0.5 - 0.25) * sy
            centers.append((x, y))
            angles.append(a if (k % 2 == 0) else b)
        layouts.append((centers, angles))

    return layouts


def optimize_layout(centers, angles, steps=2500):
    """Simple local search on centers/angles followed by scale-down repair."""
    pts = [list(p) for p in centers]
    ang = list(angles)

    def score(ps, as_):
        if has_overlap(ps, as_):
            return float("inf")
        return enclosing_side(ps, as_)

    best = score(pts, ang)
    step = 0.15
    astep = 0.12
    rng = 1

    for t in range(steps):
        i = t % len(pts)
        oldp = pts[i][:]
        olda = ang[i]
        for dx, dy, da in (
            (step, 0.0, 0.0), (-step, 0.0, 0.0),
            (0.0, step, 0.0), (0.0, -step, 0.0),
            (0.0, 0.0, astep), (0.0, 0.0, -astep),
            (step * 0.7, step * 0.7, 0.0), (-step * 0.7, step * 0.7, 0.0),
            (step * 0.7, -step * 0.7, 0.0), (-step * 0.7, -step * 0.7, 0.0),
        ):
            pts[i][0] = oldp[0] + dx
            pts[i][1] = oldp[1] + dy
            ang[i] = olda + da
            s = score(pts, ang)
            if s < best:
                best = s
                oldp = pts[i][:]
                olda = ang[i]
                break
            pts[i][0], pts[i][1], ang[i] = oldp[0], oldp[1], olda
        if t and t % len(pts) == 0:
            step *= 0.985
            astep *= 0.985
            if step < 1e-3:
                break

    return [(x, y) for x, y in pts], ang


def pack(n):
    """Search over several structured motifs, then locally improve the best one."""
    if n <= 0:
        return [], [], 0.0

    best_centers = None
    best_angles = None
    best_s = float("inf")

    candidates = seed_layouts(n)
    # Add a few hand-crafted balanced seeds.
    for k in range(6):
        centers, angles = [], []
        for i in range(n):
            x = (i - (n - 1) / 2.0) * (WIDTH * 0.55)
            y = (((i * 3 + k) % n) - (n - 1) / 2.0) * (HEIGHT * 0.18)
            centers.append((x, y))
            angles.append(math.pi / 2.0 if (i + k) % 2 == 0 else -math.pi / 2.0)
        candidates.append((centers, angles))

    for centers, angles in candidates:
        centers, angles = optimize_layout(centers, angles, steps=1800)
        if has_overlap(centers, angles):
            continue
        s = enclosing_side(centers, angles)
        if s < best_s:
            best_s = s
            best_centers = centers
            best_angles = angles

    if best_centers is None:
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
        if has_overlap(centers, angles):
            centers = repair(centers, angles)
        return centers, angles, enclosing_side(centers, angles)

    # Final shrink by radial scaling about the origin.
    lo, hi = 0.8, 1.0
    while True:
        shrunk = [(x * lo, y * lo) for x, y in best_centers]
        if has_overlap(shrunk, best_angles):
            break
        hi = lo
        lo *= 0.95
        if lo < 1e-4:
            break
    for _ in range(55):
        mid = (lo + hi) / 2.0
        cand = [(x * mid, y * mid) for x, y in best_centers]
        if has_overlap(cand, best_angles):
            lo = mid
        else:
            hi = mid
    best_centers = [(x * hi, y * hi) for x, y in best_centers]
    return best_centers, best_angles, enclosing_side(best_centers, best_angles)
# EVOLVE-BLOCK-END
