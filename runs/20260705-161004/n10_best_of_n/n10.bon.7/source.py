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
HEIGHT = R + APOTHEM                               # point-up bounding height

# --- geometry helpers -------------------------------------------------------

def pentagon_vertices(cx, cy, angle):
    """Vertices of a unit regular pentagon centered at (cx, cy)."""
    return [
        (cx + R * math.cos(angle + k * 2.0 * math.pi / 5.0),
         cy + R * math.sin(angle + k * 2.0 * math.pi / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    """Smallest origin-centered square side enclosing all pentagons."""
    m = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            m = max(m, abs(vx), abs(vy))
    return 2.0 * m


def poly_bounds(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def polygon_overlap(poly1, poly2, eps=1e-10):
    """Separating axis theorem for convex polygons."""
    for poly in (poly1, poly2):
        for i in range(len(poly)):
            (x1, y1) = poly[i]
            (x2, y2) = poly[(i + 1) % len(poly)]
            ux, uy = -(y2 - y1), (x2 - x1)
            a = [x * ux + y * uy for x, y in poly1]
            b = [x * ux + y * uy for x, y in poly2]
            amin, amax = min(a), max(a)
            bmin, bmax = min(b), max(b)
            if min(amax, bmax) - max(amin, bmin) <= eps * math.hypot(ux, uy):
                return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygon_overlap(polys[i], polys[j]):
                return True
    return False


def inside_square(centers, angles, s):
    h = s / 2.0 + 1e-9
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            if abs(vx) > h or abs(vy) > h:
                return False
    return True


def pack_by_layout(layout):
    """Build centers/angles from a list of (x, y, angle)."""
    centers = [(x, y) for x, y, _ in layout]
    angles = [a for _, _, a in layout]
    return centers, angles


# --- optimization / search --------------------------------------------------

def score_layout(layout):
    centers, angles = pack_by_layout(layout)
    s = enclosing_side(centers, angles)
    if has_overlap(centers, angles):
        return s + 1000.0
    return s


def scale_layout(layout, lam):
    return [(x * lam, y * lam, a) for x, y, a in layout]


def recenter_layout(layout):
    if not layout:
        return layout
    cx = sum(x for x, _, _ in layout) / len(layout)
    cy = sum(y for _, y, _ in layout) / len(layout)
    return [(x - cx, y - cy, a) for x, y, a in layout]


def best_of_orientations(n, seed=0):
    """Construct a few promising families and choose the best."""
    rng = random.Random(seed)
    candidates = []

    # 1) Mixed-orientation staggered grid, with opposite orientations in adjacent rows.
    #    This tends to exploit the double-lattice character of pentagon packings.
    for cols in range(1, n + 1):
        rows = (n + cols - 1) // cols
        if cols * rows < n:
            continue

        for dy_scale in (0.92, 0.96, 1.00, 1.04):
            for dx_scale in (0.90, 0.95, 1.00, 1.05):
                pitch_x = WIDTH * dx_scale
                pitch_y = HEIGHT * dy_scale

                layout = []
                k = 0
                for j in range(rows):
                    for i in range(cols):
                        if k >= n:
                            break
                        # alternate row offsets and orientations
                        x = (i - (cols - 1) / 2.0) * pitch_x
                        if j % 2 == 1:
                            x += 0.5 * pitch_x
                        y = (j - (rows - 1) / 2.0) * pitch_y
                        # mix point-up and point-down; boundary rows slightly tilted
                        if j == 0 or j == rows - 1:
                            a = math.pi / 2.0 + (0.08 if (i % 2 == 0) else -0.08)
                        else:
                            a = math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0
                        layout.append((x, y, a))
                        k += 1

                layout = recenter_layout(layout)
                candidates.append(layout)

    # 2) Diamond / hex-like shell with alternating orientations.
    #    Good for small n and boundary-dominated cases.
    ring_steps = []
    for radius in range(1, 6):
        steps = max(6, 6 * radius)
        ring_steps.append((radius, steps))
    pts = []
    for radius, steps in ring_steps:
        for t in range(steps):
            theta = 2.0 * math.pi * t / steps
            x = radius * 0.62 * math.cos(theta)
            y = radius * 0.62 * math.sin(theta)
            a = math.pi / 2.0 if (t % 2 == 0) else -math.pi / 2.0
            pts.append((x, y, a))
    if len(pts) >= n:
        candidates.append(recenter_layout(pts[:n]))

    # 3) A small randomized local jitter around the most promising grid family.
    #    Jitter is tiny; we keep only valid improvements.
    if candidates:
        base = min(candidates, key=score_layout)
        for _ in range(24):
            layout = []
            for x, y, a in base:
                dx = rng.uniform(-0.03, 0.03)
                dy = rng.uniform(-0.03, 0.03)
                da = rng.uniform(-0.08, 0.08)
                layout.append((x + dx, y + dy, a + da))
            layout = recenter_layout(layout)
            candidates.append(layout)

    best = None
    best_s = float("inf")
    for layout in candidates:
        centers, angles = pack_by_layout(layout)
        s = enclosing_side(centers, angles)
        if has_overlap(centers, angles):
            continue
        if s < best_s:
            best_s = s
            best = layout

    return best


def local_improve(layout, iters=1200, seed=0):
    """Simple stochastic hill-climb on centers/angles, minimizing square side."""
    rng = random.Random(seed)
    best = [(x, y, a) for x, y, a in layout]
    best_s = score_layout(best)
    cur = [(x, y, a) for x, y, a in best]
    cur_s = best_s

    step_xy = 0.06
    step_a = 0.12

    for t in range(iters):
        frac = 1.0 - t / max(1, iters)
        sx = step_xy * (0.25 + 0.75 * frac)
        sa = step_a * (0.25 + 0.75 * frac)

        cand = []
        for x, y, a in cur:
            if rng.random() < 0.55:
                x += rng.gauss(0.0, sx)
                y += rng.gauss(0.0, sx)
            if rng.random() < 0.75:
                a += rng.gauss(0.0, sa)
            cand.append((x, y, a))

        cand = recenter_layout(cand)
        centers, angles = pack_by_layout(cand)
        s = enclosing_side(centers, angles)

        if not has_overlap(centers, angles) and s <= cur_s + 1e-12:
            cur = cand
            cur_s = s
            if s < best_s:
                best = cand
                best_s = s
        else:
            # occasional simulated annealing acceptance for exploration
            if rng.random() < math.exp(-max(0.0, s - cur_s) / (0.03 + 0.02 * frac)):
                cur = cand
                cur_s = s

    return best, best_s


def squeeze_to_square(layout):
    """Uniformly scale down until just non-overlapping and inside a square."""
    centers, angles = pack_by_layout(layout)
    if has_overlap(centers, angles):
        # binary search for minimal safe expansion away from origin
        lo, hi = 1.0, 2.0
        while True:
            test = scale_layout(layout, hi)
            c, a = pack_by_layout(test)
            if not has_overlap(c, a):
                break
            lo, hi = hi, hi * 2.0
            if hi > 64:
                return layout
        for _ in range(50):
            mid = (lo + hi) / 2.0
            test = scale_layout(layout, mid)
            c, a = pack_by_layout(test)
            if has_overlap(c, a):
                lo = mid
            else:
                hi = mid
        layout = scale_layout(layout, hi)

    # now shrink to touch the square boundary, but keep valid
    c, a = pack_by_layout(layout)
    s = enclosing_side(c, a)
    if s <= 0:
        return layout
    # Conservative shrink: the square side is exactly determined by vertices, so
    # scaling down is safe. We attempt a small improvement and revalidate.
    target = max(0.0, s - 1e-12)
    lam = target / s if s > 0 else 1.0
    cand = scale_layout(layout, lam)
    c2, a2 = pack_by_layout(cand)
    if not has_overlap(c2, a2) and inside_square(c2, a2, enclosing_side(c2, a2)):
        return cand
    return layout


def pack(n):
    """Pack n unit regular pentagons into the smallest axis-aligned origin-centered square."""
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0], HEIGHT

    # Start from the best of several structured families.
    layout = best_of_orientations(n, seed=12345 + n)

    if layout is None:
        # Fallback: simple staggered grid.
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        pitch_x = WIDTH * 0.98
        pitch_y = HEIGHT * 0.98
        layout = []
        k = 0
        for j in range(rows):
            for i in range(cols):
                if k >= n:
                    break
                x = (i - (cols - 1) / 2.0) * pitch_x + (0.5 * pitch_x if j % 2 else 0.0)
                y = (j - (rows - 1) / 2.0) * pitch_y
                a = math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0
                layout.append((x, y, a))
                k += 1

    # Local improvement stage.
    for phase, iters in enumerate((1200, 1600, 2200)):
        layout, _ = local_improve(layout, iters=iters, seed=999 + 17 * n + phase)
        layout = recenter_layout(layout)

    # Try a few rotated copies of the whole configuration; the container is axis-aligned,
    # so rotating the entire packing may reduce the enclosing square.
    best_layout = layout
    best_s = float("inf")
    for global_rot in [0.0, math.pi / 10.0, math.pi / 8.0, math.pi / 5.0, -math.pi / 10.0]:
        cg = math.cos(global_rot)
        sg = math.sin(global_rot)
        trial = []
        for x, y, a in layout:
            xr = cg * x - sg * y
            yr = sg * x + cg * y
            trial.append((xr, yr, a + global_rot))
        trial = recenter_layout(trial)
        centers, angles = pack_by_layout(trial)
        s = enclosing_side(centers, angles)
        if not has_overlap(centers, angles) and s < best_s:
            best_s = s
            best_layout = trial

    # Final squeeze and correctness pass.
    best_layout = squeeze_to_square(best_layout)
    centers, angles = pack_by_layout(best_layout)

    # If numerical issues remain, enlarge slightly until valid.
    if has_overlap(centers, angles) or not inside_square(centers, angles, enclosing_side(centers, angles)):
        lam_lo, lam_hi = 1.0, 1.02
        while True:
            test = scale_layout(best_layout, lam_hi)
            c, a = pack_by_layout(test)
            if (not has_overlap(c, a)) and inside_square(c, a, enclosing_side(c, a)):
                best_layout = test
                centers, angles = c, a
                break
            lam_lo, lam_hi = lam_hi, lam_hi * 1.02
            if lam_hi > 4.0:
                break

    centers = [(float(x), float(y)) for x, y, _ in best_layout]
    angles = [float(a) for _, _, a in best_layout]
    s = enclosing_side(centers, angles)
    return centers, angles, s
