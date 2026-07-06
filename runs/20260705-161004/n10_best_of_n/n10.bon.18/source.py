"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list of (x, y)
- angles:  list of rotations in radians
- s: outer side length of origin-centered square
"""

import math

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # inradius
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # widest extent
HEIGHT = R + APOTHEM                              # point-up bounding height

TAU = 2.0 * math.pi
EPS = 1e-9


def pentagon_vertices(cx, cy, angle):
    return [
        (cx + R * math.cos(angle + k * TAU / 5.0),
         cy + R * math.sin(angle + k * TAU / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, ang)
    ) if centers else 0.0


def poly_axes(poly):
    axes = []
    for i in range(5):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
        ex, ey = x2 - x1, y2 - y1
        nx, ny = -ey, ex
        norm = math.hypot(nx, ny)
        if norm > 0:
            axes.append((nx / norm, ny / norm))
    return axes


def project(poly, ax, ay):
    vals = [x * ax + y * ay for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(a, b, tol=1e-10):
    axes = poly_axes(a) + poly_axes(b)
    for ax, ay in axes:
        amin, amax = project(a, ax, ay)
        bmin, bmax = project(b, ax, ay)
        if min(amax, bmax) - max(amin, bmin) <= tol:
            return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, ang) for (cx, cy), ang in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def boundary_side(centers, angles):
    return enclosing_side(centers, angles)


def random(seed):
    seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
    return seed, seed / 0x7FFFFFFF


def rotate_point(x, y, ang):
    c, s = math.cos(ang), math.sin(ang)
    return c * x - s * y, s * x + c * y


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def cost(centers, angles):
    s = boundary_side(centers, angles)
    penalty = 0.0
    if has_overlap(centers, angles):
        penalty += 1000.0
    return s + penalty


def translate(centers, dx, dy):
    return [(x + dx, y + dy) for x, y in centers]


def scale_about_origin(centers, lam):
    return [(x * lam, y * lam) for x, y in centers]


def fit_to_square(centers, angles):
    if not centers:
        return centers, 0.0
    s = enclosing_side(centers, angles)
    if s <= 0:
        return centers, 0.0
    lam = (1.0 - 5e-10) * 1.0
    # Keep origin-centered convention; only scale down if needed to fit within a target unit square.
    # Here this helper is used only as a normalization for search heuristics.
    mx = max(max(abs(vx), abs(vy)) for (cx, cy), ang in zip(centers, angles)
             for vx, vy in pentagon_vertices(cx, cy, ang))
    if mx > 0:
        lam = 1.0 / mx
    return scale_about_origin(centers, lam), s * lam


def generate_pattern(n, mode=0):
    """Heuristic initial layouts based on rows/columns, mixed orientations, and boundary tilts."""
    if n == 0:
        return [], []

    # Candidate grid shapes
    shapes = []
    for cols in range(1, n + 1):
        rows = (n + cols - 1) // cols
        if cols * rows >= n and abs(cols - rows) <= max(2, n // 4 + 1):
            shapes.append((cols, rows))
    shapes = sorted(set(shapes), key=lambda cr: (abs(cr[0] - cr[1]), cr[0] * cr[1]))

    best = None
    for idx, (cols, rows) in enumerate(shapes[: min(len(shapes), 12)]):
        # Base pitches: slightly compressed relative to bounding boxes to encourage interlocking.
        px_base = 0.90 * WIDTH
        py_base = 0.84 * HEIGHT

        # Boundary tilt angles: interior near +/- pi/2, edge rows slightly rotated.
        centers, angles = [], []
        for k in range(n):
            i, j = k % cols, k // cols

            x = (i - (cols - 1) / 2.0) * px_base
            y = (j - (rows - 1) / 2.0) * py_base

            # Stagger every other row to create better interlocks.
            if j % 2 == 1:
                x += 0.18 * WIDTH

            # Compress odd/even columns a bit differently.
            y += 0.04 * HEIGHT * math.sin((i + 1) * 1.7)

            # Mix orientations: paired opposites are generally favorable.
            if ((i + j + mode) % 2) == 0:
                ang = math.pi / 2.0
            else:
                ang = -math.pi / 2.0

            # Boundary-row tilts
            if j == 0:
                ang += 0.18
            elif j == rows - 1:
                ang -= 0.18
            if i == 0:
                ang += 0.10
            elif i == cols - 1:
                ang -= 0.10

            centers.append((x, y))
            angles.append(ang)

        # Center at origin
        mx = sum(x for x, y in centers) / n
        my = sum(y for x, y in centers) / n
        centers = [(x - mx, y - my) for x, y in centers]

        # A few mode-specific perturbations
        if mode % 3 == 1:
            centers = [(x + 0.06 * math.sin(2.3 * i), y + 0.06 * math.cos(1.7 * j))
                       for k, ((x, y), ang) in enumerate(zip(centers, angles))
                       for i, j in [(k % cols, k // cols)]]
        elif mode % 3 == 2:
            centers = [(x * (1.0 + 0.02 * math.sin(i + j)),
                        y * (1.0 - 0.02 * math.sin(i + j)))
                       for k, ((x, y), ang) in enumerate(zip(centers, angles))
                       for i, j in [(k % cols, k // cols)]]

        # Keep as a candidate
        if best is None:
            best = (centers, angles)
        else:
            if cost(centers, angles) < cost(best[0], best[1]):
                best = (centers, angles)

    return best


def local_search(centers, angles, steps=5000, seed=1):
    """Coordinate descent / random hill-climb on centers and angles."""
    if not centers:
        return centers, angles

    n = len(centers)
    best_c = [list(p) for p in centers]
    best_a = list(angles)
    best_s = enclosing_side(best_c, best_a)

    # Current state
    cur_c = [list(p) for p in best_c]
    cur_a = list(best_a)
    cur_s = best_s

    # Annealing schedule
    step_xy = max(0.02, best_s * 0.03)
    step_ang = 0.22

    for t in range(steps):
        seed, r1 = random(seed)
        seed, r2 = random(seed)
        seed, r3 = random(seed)
        idx = int(r1 * n) % n

        proposal_c = [p[:] for p in cur_c]
        proposal_a = cur_a[:]

        # Mixed move types
        move_type = int(r2 * 5.0)
        if move_type == 0:
            proposal_c[idx][0] += (2.0 * r3 - 1.0) * step_xy
            proposal_c[idx][1] += (2.0 * (1.0 - r3) - 1.0) * step_xy
        elif move_type == 1:
            proposal_a[idx] += (2.0 * r3 - 1.0) * step_ang
        elif move_type == 2:
            dx = (2.0 * r3 - 1.0) * step_xy * 0.8
            dy = (2.0 * ((r3 * 1.7) % 1.0) - 1.0) * step_xy * 0.8
            for k in range(n):
                proposal_c[k][0] += dx
                proposal_c[k][1] += dy
        elif move_type == 3:
            lam = 1.0 + (2.0 * r3 - 1.0) * 0.015
            proposal_c = [[x * lam, y * lam] for x, y in proposal_c]
        else:
            # Rotate a small cluster of 2-4 items around their centroid
            m = 2 + int(r3 * 3.0)
            ids = [(idx + j) % n for j in range(m)]
            cx = sum(proposal_c[i][0] for i in ids) / m
            cy = sum(proposal_c[i][1] for i in ids) / m
            da = (2.0 * r2 - 1.0) * 0.16
            for i in ids:
                x, y = proposal_c[i][0] - cx, proposal_c[i][1] - cy
                x, y = rotate_point(x, y, da)
                proposal_c[i][0] = cx + x
                proposal_c[i][1] = cy + y
                proposal_a[i] += da * 0.5

        # Re-center softly
        mx = sum(p[0] for p in proposal_c) / n
        my = sum(p[1] for p in proposal_c) / n
        proposal_c = [[x - mx, y - my] for x, y in proposal_c]

        s = enclosing_side(proposal_c, proposal_a)
        if s < cur_s - 1e-12 and not has_overlap(proposal_c, proposal_a):
            cur_c, cur_a, cur_s = proposal_c, proposal_a, s
            if s < best_s - 1e-12:
                best_c, best_a, best_s = [p[:] for p in proposal_c], proposal_a[:], s

        # Occasionally accept slightly worse states to escape local minima.
        temp = max(0.01, 1.0 - t / max(1, steps))
        if not has_overlap(proposal_c, proposal_a):
            accept = s <= cur_s or (math.exp((cur_s - s) / (0.01 + temp)) > r1)
            if accept:
                cur_c, cur_a, cur_s = proposal_c, proposal_a, s

        # Gradually reduce move sizes
        if (t + 1) % 400 == 0:
            step_xy *= 0.86
            step_ang *= 0.90

    return [(x, y) for x, y in best_c], best_a


def deterministic_layout(n):
    """A few handcrafted structures for small and medium n."""
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]
    if n == 2:
        d = 0.95 * WIDTH
        return [(-d / 2.0, 0.0), (d / 2.0, 0.0)], [math.pi / 2.0, -math.pi / 2.0]
    if n == 3:
        return [(-0.55, -0.18), (0.55, -0.18), (0.0, 0.56)], [
            -math.pi / 2.0, math.pi / 2.0, math.pi / 2.0
        ]
    if n == 4:
        return [(-0.55, -0.55), (0.55, -0.55), (-0.55, 0.55), (0.55, 0.55)], [
            math.pi / 2.0, -math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0
        ]
    return None, None


def pack(n):
    """Heuristic optimizer returning a valid packing."""
    if n <= 0:
        return [], [], 0.0

    dc, da = deterministic_layout(n)
    candidates = []
    if dc is not None:
        candidates.append((dc, da))

    # Several seeded starting patterns
    for mode in range(6):
        c, a = generate_pattern(n, mode=mode)
        candidates.append((c, a))

    best_c, best_a, best_s = None, None, float("inf")

    for i, (c, a) in enumerate(candidates):
        c2, a2 = [tuple(p) for p in c], list(a)

        # Local optimization
        steps = 3000 + 700 * n
        c3, a3 = local_search(c2, a2, steps=steps, seed=12345 + 97 * i + n * 1000)

        if has_overlap(c3, a3):
            continue
        s = enclosing_side(c3, a3)
        if s < best_s:
            best_c, best_a, best_s = c3, a3, s

    if best_c is None:
        # Fallback: a safe, very loose packing.
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        px = WIDTH + 0.1
        py = HEIGHT + 0.1
        best_c, best_a = [], []
        for k in range(n):
            i, j = k % cols, k // cols
            best_c.append(((i - (cols - 1) / 2.0) * px, (j - (rows - 1) / 2.0) * py))
            best_a.append(math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0)
        best_s = enclosing_side(best_c, best_a)

    # Final tiny shrink attempt with overlap-preserving scaling by binary search
    # if possible.
    lo, hi = 0.0, 1.0
    if not has_overlap(best_c, best_a):
        for _ in range(50):
            mid = (lo + hi) / 2.0
            test_c = scale_about_origin(best_c, mid)
            if has_overlap(test_c, best_a):
                lo = mid
            else:
                hi = mid
        best_c = scale_about_origin(best_c, hi)
        best_s = enclosing_side(best_c, best_a)

    return best_c, best_a, best_s
