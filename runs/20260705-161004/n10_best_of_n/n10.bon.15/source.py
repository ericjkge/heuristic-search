"""Pack n unit regular pentagons into the smallest axis-aligned origin-centered square.

Contract: pack(n) -> (centers, angles, s), where the container is the
origin-centered axis-aligned square of side s. A point p is inside iff
max(abs(px), abs(py)) <= s/2.

This version uses a constructive multi-start local search with exact geometry
checks. It is designed to produce substantially tighter packings than a simple
grid, especially for n=10.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # inradius
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # widest extent
HEIGHT = R + APOTHEM                               # point-up bounding height

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
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, ang)
    )


# --- Exact overlap testing ---------------------------------------------------

def _project(poly, ux, uy):
    vals = [x * ux + y * uy for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb, eps=1e-10):
    """Separating-axis test for convex polygons."""
    for poly in (pa, pb):
        for i in range(len(poly)):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
            ux, uy = -(y2 - y1), (x2 - x1)
            amin, amax = _project(pa, ux, uy)
            bmin, bmax = _project(pb, ux, uy)
            norm = math.hypot(ux, uy)
            if min(amax, bmax) - max(amin, bmin) <= eps * norm:
                return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def feasible(centers, angles):
    return not has_overlap(centers, angles)


# --- Geometry helpers --------------------------------------------------------

def rotate_point(x, y, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (c * x - s * y, s * x + c * y)


def bounding_box_of_pentagon(angle):
    """Return (minx, maxx, miny, maxy) for a pentagon centered at origin."""
    pts = pentagon_vertices(0.0, 0.0, angle)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def normalize_solution(centers, angles):
    """Translate to keep the origin-centered square tight and return best side."""
    s = enclosing_side(centers, angles)
    return centers, angles, s


# --- Construction ------------------------------------------------------------

def initial_layout(n, rng):
    """
    Build a layered layout with alternating orientations and some row tilt.
    This is a much better starting point than a square grid.
    """
    if n <= 0:
        return [], []

    # Candidate layer shapes for n=10 and near values.
    # We use rows with horizontal staggering and opposite orientations.
    best = None

    def make_rows(row_counts, row_dx, row_dy, theta0, theta1):
        centers = []
        angles = []
        y = 0.0
        # Center rows around 0
        total_h = (len(row_counts) - 1) * row_dy
        y0 = -total_h / 2.0
        for r, cnt in enumerate(row_counts):
            yy = y0 + r * row_dy
            offset = 0.0 if (r % 2 == 0) else row_dx / 2.0
            start = -((cnt - 1) * row_dx) / 2.0
            ang = theta0 if (r % 2 == 0) else theta1
            for i in range(cnt):
                x = start + i * row_dx + offset
                centers.append((x, yy))
                # Alternate within row to create a "double-lattice" feel
                angles.append(ang if (i % 2 == 0) else (ang + math.pi))
        return centers, angles

    # Deterministic row patterns
    patterns = []
    if n == 10:
        patterns = [
            [4, 3, 3],
            [3, 4, 3],
            [3, 3, 4],
            [5, 5],
            [4, 2, 4],
        ]
    else:
        # Generic near-rectangular row decompositions
        r = int(math.ceil(math.sqrt(n)))
        patterns.append([n // r + (1 if i < n % r else 0) for i in range(r)])
        c = int(math.ceil(n / r))
        patterns.append([c] * (n // c) + ([n % c] if n % c else []))

    # Candidate spacings based on pentagon extents
    # Tighter than the naive WIDTH/HEIGHT grid, but still conservative.
    row_dx_candidates = [0.95 * WIDTH, 1.00 * WIDTH, 1.03 * WIDTH]
    row_dy_candidates = [0.78 * HEIGHT, 0.84 * HEIGHT, 0.90 * HEIGHT]
    theta_candidates = [
        (math.pi / 2.0, -math.pi / 2.0),
        (math.pi / 2.0 + 0.10, -math.pi / 2.0 - 0.10),
        (math.pi / 2.0 - 0.08, -math.pi / 2.0 + 0.08),
        (0.0, math.pi),
        (0.20, math.pi + 0.20),
    ]

    for row_counts in patterns:
        for row_dx in row_dx_candidates:
            for row_dy in row_dy_candidates:
                for theta0, theta1 in theta_candidates:
                    centers, angles = make_rows(row_counts, row_dx, row_dy, theta0, theta1)
                    if len(centers) != n:
                        continue
                    # Random small jitter to escape symmetric traps
                    jitter = 0.015
                    centers2 = [
                        (x + jitter * (rng.random() - 0.5), y + jitter * (rng.random() - 0.5))
                        for x, y in centers
                    ]
                    # Add a global slight skew/rotation as well
                    phi = 0.03 * (rng.random() - 0.5)
                    centers2 = [rotate_point(x, y, phi) for x, y in centers2]
                    angles2 = [a + phi for a in angles]
                    cand = (centers2, angles2)
                    if best is None:
                        best = cand
                    else:
                        # Prefer smaller immediate bounding box, even if overlapped;
                        # we will repair via optimization.
                        if enclosing_side(*cand) < enclosing_side(*best):
                            best = cand
    return best


# --- Local optimization ------------------------------------------------------

def objective(params, n, fixed_angles_mask=None):
    """
    Soft objective for optimization:
      - minimize enclosing square side
      - penalize overlaps strongly
      - penalize being outside the square by the enclosing side definition
    """
    centers = [(params[3 * i], params[3 * i + 1]) for i in range(n)]
    angles = [params[3 * i + 2] for i in range(n)]

    # Keep angles wrapped to a manageable range
    angles = [((a + math.pi) % TWO_PI) - math.pi for a in angles]

    s = enclosing_side(centers, angles)
    poly = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]

    # Overlap penalty
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(poly[i], poly[j], eps=1e-11):
                # Estimate a large penalty from center distance
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d2 = dx * dx + dy * dy
                overlap_pen += 10.0 + 3.0 / (1e-6 + d2)

    return s + 50.0 * overlap_pen


def try_scipy_refine(centers, angles, time_budget=1.0):
    """
    Optional SciPy-based refinement if available.
    Keeps the solution valid by re-checking and accepting only improvements.
    """
    try:
        import scipy.optimize as opt
    except Exception:
        return centers, angles

    n = len(centers)
    x0 = []
    for (cx, cy), a in zip(centers, angles):
        x0.extend([cx, cy, a])

    bounds = []
    s0 = enclosing_side(centers, angles)
    lim = max(3.0, s0)
    for _ in range(n):
        bounds.extend([(-lim, lim), (-lim, lim), (-math.pi, math.pi)])

    best = (centers, angles, s0)

    def fun(x):
        return objective(x, n)

    try:
        res = opt.minimize(
            fun,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(200 * time_budget), "ftol": 1e-12},
        )
        if res.success:
            x = res.x
            c2 = [(x[3 * i], x[3 * i + 1]) for i in range(n)]
            a2 = [((x[3 * i + 2] + math.pi) % TWO_PI) - math.pi for i in range(n)]
            if feasible(c2, a2):
                s2 = enclosing_side(c2, a2)
                if s2 < best[2]:
                    best = (c2, a2, s2)
    except Exception:
        pass

    return best[0], best[1]


def shrink_to_fit(centers, angles):
    """
    Deterministically shrink the packing by scaling centers toward the origin
    as much as possible while preserving non-overlap.
    """
    if not centers:
        return centers, angles

    # If already overlap-free, we can binary search a global dilation factor.
    # Scaling centers inward makes overlaps more likely, so we search downward.
    if has_overlap(centers, angles):
        return centers, angles

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 0.0, 1.0
    # Find a lower feasible bound close to 1.
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi), angles


def pack(n):
    """Return a valid packing as (centers, angles, s)."""
    if n <= 0:
        return [], [], 0.0

    rng = random.Random(123456789 + 97 * n)

    # Start with a few diverse constructions.
    candidates = []

    # 1) Strong layered start.
    centers, angles = initial_layout(n, rng)
    candidates.append((centers, angles))

    # 2) A compact hex-like staggered start.
    if n >= 2:
        row_counts = []
        rem = n
        a, b = (n + 1) // 2, n // 2
        row_counts = [a, b] if b > 0 else [a]
        dx = 0.965 * WIDTH
        dy = 0.82 * HEIGHT
        centers2, angles2 = [], []
        y0 = -((len(row_counts) - 1) * dy) / 2.0
        idx = 0
        for r, cnt in enumerate(row_counts):
            yy = y0 + r * dy
            offset = 0.0 if r % 2 == 0 else dx / 2.0
            start = -((cnt - 1) * dx) / 2.0
            for i in range(cnt):
                if idx >= n:
                    break
                centers2.append((start + i * dx + offset, yy))
                angles2.append(math.pi / 2.0 if (i + r) % 2 == 0 else -math.pi / 2.0)
                idx += 1
        # Fill remaining if any
        while len(centers2) < n:
            centers2.append((0.0, 0.0))
            angles2.append(0.0)
        candidates.append((centers2, angles2))

    # 3) A radial/clustered start.
    centers3, angles3 = [], []
    ring_r = 0.95
    for i in range(n):
        if i == 0:
            centers3.append((0.0, 0.0))
            angles3.append(0.0)
        else:
            t = TWO_PI * i / max(1, n - 1)
            rr = ring_r * (0.7 + 0.3 * (i % 3))
            centers3.append((rr * math.cos(t), rr * math.sin(t)))
            angles3.append((math.pi / 2.0 if i % 2 == 0 else -math.pi / 2.0) + 0.2 * math.sin(t))
    candidates.append((centers3, angles3))

    # Evaluate and optionally refine.
    best = None
    for cc, aa in candidates:
        # Jitter/rotation variants
        for _ in range(8):
            phi = 0.12 * (rng.random() - 0.5)
            tx = 0.08 * (rng.random() - 0.5)
            ty = 0.08 * (rng.random() - 0.5)
            c = [rotate_point(x + tx, y + ty, phi) for x, y in cc]
            a = [((ang + phi + 0.07 * (rng.random() - 0.5)) + math.pi) % TWO_PI - math.pi for ang in aa]

            # If overlapped, perform a modest "repulsion" pre-pass.
            for _step in range(60):
                polys = [pentagon_vertices(cx, cy, ang) for (cx, cy), ang in zip(c, a)]
                moved = False
                for i in range(n):
                    fx = fy = 0.0
                    for j in range(n):
                        if i == j:
                            continue
                        if polygons_overlap(polys[i], polys[j], eps=1e-11):
                            dx = c[i][0] - c[j][0]
                            dy = c[i][1] - c[j][1]
                            d = math.hypot(dx, dy) + 1e-9
                            push = 0.015 / d
                            fx += push * dx
                            fy += push * dy
                            moved = True
                    if moved:
                        c[i] = (c[i][0] + fx, c[i][1] + fy)
                        a[i] += 0.01 * (fx - fy)
                if not moved:
                    break

            if not feasible(c, a):
                continue

            # Gentle shrink toward origin.
            c, a = shrink_to_fit(c, a)

            s = enclosing_side(c, a)
            if best is None or s < best[2]:
                best = (c, a, s)

    if best is None:
        # Fallback to a safe non-overlapping loose arrangement.
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        pitch_x = WIDTH + 0.15
        pitch_y = HEIGHT + 0.15
        c, a = [], []
        for k in range(n):
            i, j = k % cols, k // cols
            x = (i - (cols - 1) / 2.0) * pitch_x
            y = (j - (rows - 1) / 2.0) * pitch_y
            c.append((x, y))
            a.append(math.pi / 2.0 if i % 2 == 0 else -math.pi / 2.0)
        return c, a, enclosing_side(c, a)

    # Optional optimizer refinement if available.
    c, a = best[0], best[1]
    c2, a2 = try_scipy_refine(c, a, time_budget=1.5)
    if feasible(c2, a2):
        # Final shrink-and-accept loop.
        c2, a2 = shrink_to_fit(c2, a2)
        if feasible(c2, a2) and enclosing_side(c2, a2) < enclosing_side(c, a):
            c, a = c2, a2

    return c, a, enclosing_side(c, a)
