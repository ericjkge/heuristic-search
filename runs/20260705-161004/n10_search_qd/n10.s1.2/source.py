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
OVERLAP_EPS = 1e-9  # small cushion for numerical robustness


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


def polygon_axes(poly):
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        yield (-(y2 - y1), x2 - x1)


def project(poly, ux, uy):
    vals = [x * ux + y * uy for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb):
    """Strict overlap test for convex polygons."""
    for ux, uy in polygon_axes(pa):
        amin, amax = project(pa, ux, uy)
        bmin, bmax = project(pb, ux, uy)
        if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS * math.hypot(ux, uy):
            return False
    for ux, uy in polygon_axes(pb):
        amin, amax = project(pa, ux, uy)
        bmin, bmax = project(pb, ux, uy)
        if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS * math.hypot(ux, uy):
            return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def bounds_ok(centers, angles, s):
    h = s / 2.0 + 1e-10
    for (cx, cy), a in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, a):
            if abs(vx) > h or abs(vy) > h:
                return False
    return True


def local_search(centers, angles, s0, rounds=1800):
    """Coordinate search on centers and a small discrete angle set."""
    import random
    random.seed(0)

    angleset = [
        0.0, math.pi / 5.0, 2.0 * math.pi / 5.0, 3.0 * math.pi / 5.0,
        4.0 * math.pi / 5.0, math.pi,
        -math.pi / 5.0, -2.0 * math.pi / 5.0, -3.0 * math.pi / 5.0, -4.0 * math.pi / 5.0,
    ]

    best_c = [tuple(c) for c in centers]
    best_a = list(angles)
    best_s = s0

    # Multi-scale step schedule.
    for t in range(rounds):
        if t < 500:
            step = 0.12
            astep = math.pi / 10.0
        elif t < 1100:
            step = 0.05
            astep = math.pi / 20.0
        else:
            step = 0.02
            astep = math.pi / 40.0

        idx = t % len(best_c)
        ox, oy = best_c[idx]
        oa = best_a[idx]

        # Greedy move: bias toward the boundary to utilize the square, while
        # staying valid.
        candidates = [(0.0, 0.0, oa)]
        candidates += [
            ( step, 0.0, oa), (-step, 0.0, oa),
            (0.0,  step, oa), (0.0, -step, oa),
            ( step,  step, oa), ( step, -step, oa),
            (-step,  step, oa), (-step, -step, oa),
        ]
        for da in (-astep, astep):
            candidates.append((0.0, 0.0, oa + da))
        for ang in angleset:
            candidates.append((0.0, 0.0, ang))

        random.shuffle(candidates)
        improved = False
        for dx, dy, na in candidates:
            nc = list(best_c)
            nc[idx] = (ox + dx, oy + dy)
            na_list = list(best_a)
            na_list[idx] = na

            s = enclosing_side(nc, na_list)
            if s + 1e-12 >= best_s:
                continue
            if not bounds_ok(nc, na_list, s):
                continue
            if has_overlap(nc, na_list):
                continue

            best_c, best_a, best_s = nc, na_list, s
            improved = True
            break

        if not improved and t % 37 == 0:
            # Randomized nudge for escaping symmetric basins.
            idx = random.randrange(len(best_c))
            nc = list(best_c)
            na_list = list(best_a)
            nc[idx] = (nc[idx][0] + random.uniform(-step, step),
                       nc[idx][1] + random.uniform(-step, step))
            na_list[idx] = random.choice(angleset)
            s = enclosing_side(nc, na_list)
            if s < best_s and bounds_ok(nc, na_list, s) and not has_overlap(nc, na_list):
                best_c, best_a, best_s = nc, na_list, s

    return best_c, best_a, best_s


def pack(n):
    """Structured double-lattice-inspired starter plus local search."""
    if n <= 0:
        return [], [], 0.0

    # Double-lattice motif: alternating orientations with staggered rows.
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # Start from a compact, boundary-friendly arrangement.
    base_x = 0.92
    base_y = 0.80
    centers, angles = [], []

    for k in range(n):
        i, j = k % cols, k // cols
        parity = (i + j) & 1
        x = (i - (cols - 1) / 2.0) * base_x
        y = (j - (rows - 1) / 2.0) * base_y

        # Stagger alternate rows to create interlocks and push boundary rows.
        x += (0.28 if (j & 1) else -0.28) * (1 if i % 2 == 0 else -1)
        y += (0.10 if parity == 0 else -0.10)

        ang = (math.pi / 2.0) if parity == 0 else (-math.pi / 2.0)
        if j == 0 or j == rows - 1:
            ang += (math.pi / 10.0 if i % 2 == 0 else -math.pi / 10.0)
        elif i == 0 or i == cols - 1:
            ang += (math.pi / 20.0 if j % 2 == 0 else -math.pi / 20.0)

        centers.append((x, y))
        angles.append(ang)

    # Normalize to origin-centered square and shrink until feasible.
    s = enclosing_side(centers, angles)
    if s > 0:
        scale = min(1.0, 4.8 / s)
        centers = [(x * scale, y * scale) for x, y in centers]

    # A light repair shrink only if needed.
    if has_overlap(centers, angles):
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            trial = [(x * mid, y * mid) for x, y in centers]
            if has_overlap(trial, angles):
                hi = mid
            else:
                lo = mid
        centers = [(x * lo, y * lo) for x, y in centers]

    s0 = enclosing_side(centers, angles)
    centers, angles, s = local_search(centers, angles, s0)

    # Final safety shrink if search left a tiny overlap due to numerical issues.
    if has_overlap(centers, angles) or not bounds_ok(centers, angles, s):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            trial = [(x * mid, y * mid) for x, y in centers]
            ss = enclosing_side(trial, angles)
            if has_overlap(trial, angles) or not bounds_ok(trial, angles, ss):
                hi = mid
            else:
                lo = mid
        centers = [(x * lo, y * lo) for x, y in centers]
        s = enclosing_side(centers, angles)

    return centers, angles, s
# EVOLVE-BLOCK-END
