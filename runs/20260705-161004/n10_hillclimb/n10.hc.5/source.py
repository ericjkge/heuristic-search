"""Heuristic optimizer for packing n unit regular pentagons into the smallest
origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s).
Container: point p is inside iff max(|px|, |py|) <= s/2.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # inradius
PHI = (1.0 + math.sqrt(5.0)) / 2.0
TAU = 2.0 * math.pi


def pentagon_vertices(cx, cy, angle):
    return [
        (cx + R * math.cos(angle + k * TAU / 5.0),
         cy + R * math.sin(angle + k * TAU / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    m = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            m = max(m, abs(vx), abs(vy))
    return 2.0 * m


def poly_axes(poly):
    axes = []
    for i in range(5):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
        ux, uy = -(y2 - y1), (x2 - x1)
        norm = math.hypot(ux, uy)
        if norm > 0:
            axes.append((ux / norm, uy / norm))
    return axes


def project(poly, ax, ay):
    vals = [x * ax + y * ay for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb, eps=1e-10):
    for ax, ay in poly_axes(pa) + poly_axes(pb):
        amin, amax = project(pa, ax, ay)
        bmin, bmax = project(pb, ax, ay)
        if min(amax, bmax) - max(amin, bmin) <= eps:
            return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    m = len(polys)
    for i in range(m):
        for j in range(i + 1, m):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def bounding_square_side(centers, angles):
    return enclosing_side(centers, angles)


def translate_to_origin(centers):
    if not centers:
        return centers
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    return [(x - mx, y - my) for x, y in centers]


def scale_centers(centers, lam):
    return [(x * lam, y * lam) for x, y in centers]


def greedy_shrink(centers, angles, max_iter=120):
    """Uniformly shrink centers about origin until just before overlap."""
    if not centers:
        return centers
    centers = translate_to_origin(centers)

    # First ensure feasibility by expanding if needed.
    lam = 1.0
    if has_overlap(centers, angles):
        lo, hi = 1.0, 1.1
        while has_overlap(scale_centers(centers, hi), angles) and hi < 50.0:
            lo, hi = hi, hi * 1.25
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if has_overlap(scale_centers(centers, mid), angles):
                lo = mid
            else:
                hi = mid
        centers = scale_centers(centers, hi)

    # Then attempt to shrink.
    lo = 0.0
    hi = 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        test = scale_centers(centers, mid)
        if has_overlap(test, angles):
            lo = mid
        else:
            hi = mid
    return scale_centers(centers, hi)


def mix_angle(base, i, seed):
    rng = random.Random((seed + 1) * 1000003 + i * 9176)
    return base + (rng.random() - 0.5) * 0.12


def plane_packing_seed(n, seed=0):
    """
    Construct a dense finite motif inspired by opposite-orientation pairing:
    rows of alternating up/down pentagons, with stagger and boundary tilt.
    """
    rng = random.Random(123456789 + 10007 * n + seed)

    # Choose a compact near-square row/column shape.
    rows = max(1, int(round(math.sqrt(n))))
    cols = (n + rows - 1) // rows
    if cols < rows:
        rows, cols = cols, rows

    # Use a motif scale tuned to pentagon dimensions.
    # These are intentionally aggressive; later repair handles overlap.
    dx = 0.92 * R * 1.618
    dy = 0.88 * (APOTHEM + 0.50)

    centers = []
    angles = []

    # Alternate orientations strongly.
    for k in range(n):
        r = k // cols
        c = k % cols
        if r >= rows:
            r = rows - 1

        # Staggered rows with small irrational wobble.
        x = (c - (cols - 1) / 2.0) * dx
        if r % 2 == 1:
            x += 0.5 * dx
        x += ((r * 7 + c * 11 + seed) % 5 - 2) * 0.02

        y = (r - (rows - 1) / 2.0) * dy
        y += (((r + 1) * (c + 2) + seed) % 7 - 3) * 0.015

        # Opposite orientations: 90 degrees apart.
        if (r + c) % 2 == 0:
            ang = math.pi / 2.0
        else:
            ang = -math.pi / 2.0

        # Boundary tilts: rows at the top/bottom and columns at the left/right
        # get a stronger nudge to help fit against square walls.
        if r == 0:
            ang += 0.20
        elif r == rows - 1:
            ang -= 0.20
        if c == 0:
            ang -= 0.11
        elif c == cols - 1:
            ang += 0.11

        ang = mix_angle(ang, k, seed)
        centers.append((x, y))
        angles.append(ang)

    # For incomplete last row, pull the final few pieces inward in a controlled way.
    extra = rows * cols - n
    if extra > 0:
        for i in range(n - extra, n):
            x, y = centers[i]
            centers[i] = (0.93 * x, 0.93 * y)

    # Center the motif and repair feasibility.
    centers = translate_to_origin(centers)
    centers = greedy_shrink(centers, angles)
    s = enclosing_side(centers, angles)
    return centers, angles, s


def candidate_layouts(n):
    """Generate several dense row/column motifs."""
    cands = []

    # Near-square
    r = max(1, int(round(math.sqrt(n))))
    c = (n + r - 1) // r
    cands.append((r, c))

    # Slightly wider/taller variants
    for dr in range(-2, 3):
        rr = max(1, r + dr)
        cc = (n + rr - 1) // rr
        cands.append((rr, cc))
    for dc in range(-2, 3):
        cc = max(1, c + dc)
        rr = (n + cc - 1) // cc
        cands.append((rr, cc))

    # Deduplicate
    seen = set()
    out = []
    for rr, cc in cands:
        if (rr, cc) not in seen:
            seen.add((rr, cc))
            out.append((rr, cc))
    return out


def build_from_layout(n, rows, cols, seed=0):
    rng = random.Random(987654321 + 131 * n + 17 * seed)

    centers = []
    angles = []

    # Base spacing derived from geometry but then optimized by shrink.
    dx = 0.80 * (2.0 * APOTHEM + 0.42)
    dy = 0.78 * (R + APOTHEM + 0.12)

    # Boundary compression factors.
    edge_x = 0.92
    edge_y = 0.92

    for idx in range(n):
        r = idx // cols
        c = idx % cols

        x = (c - (cols - 1) / 2.0) * dx
        y = (r - (rows - 1) / 2.0) * dy

        # Alternating offsets encourage interlocking.
        if r % 2:
            x += 0.5 * dx
        if (r + c) % 3 == 1:
            y += 0.06 * dy

        # Strong opposite orientations, plus some seed-based micro-variation.
        ang = math.pi / 2.0 if ((r + c) & 1) == 0 else -math.pi / 2.0
        ang += (rng.random() - 0.5) * 0.16

        # Nudge edge pentagons to tilt against walls.
        if r in (0, rows - 1):
            ang += 0.14 if r == 0 else -0.14
            y *= edge_y
        if c in (0, cols - 1):
            ang += -0.08 if c == 0 else 0.08
            x *= edge_x

        centers.append((x, y))
        angles.append(ang)

    centers = translate_to_origin(centers)
    centers = greedy_shrink(centers, angles)
    s = enclosing_side(centers, angles)
    return centers, angles, s


def optimize_locally(centers, angles, steps=2500, seed=0):
    rng = random.Random(24681357 + 31 * seed)
    n = len(centers)

    best_centers = centers[:]
    best_angles = angles[:]
    best_s = enclosing_side(best_centers, best_angles)

    cur_centers = centers[:]
    cur_angles = angles[:]
    cur_s = best_s

    step_xy = max(0.003, best_s * 0.010)
    step_a = 0.11

    for t in range(steps):
        i = rng.randrange(n)
        old_c = cur_centers[i]
        old_a = cur_angles[i]

        if rng.random() < 0.60:
            dx = (rng.random() * 2.0 - 1.0) * step_xy
            dy = (rng.random() * 2.0 - 1.0) * step_xy
            cur_centers[i] = (old_c[0] + dx, old_c[1] + dy)
        else:
            cur_angles[i] = old_a + (rng.random() * 2.0 - 1.0) * step_a

        # Recenter occasionally to keep the origin-centered square tight.
        if t % 25 == 0:
            cur_centers = translate_to_origin(cur_centers)

        test_centers = greedy_shrink(cur_centers, cur_angles, max_iter=50)
        test_s = enclosing_side(test_centers, cur_angles)

        if test_s < cur_s:
            cur_centers = test_centers
            cur_s = test_s
            if cur_s < best_s:
                best_centers = cur_centers[:]
                best_angles = cur_angles[:]
                best_s = cur_s
        else:
            cur_centers[i] = old_c
            cur_angles[i] = old_a

        if (t + 1) % 500 == 0:
            step_xy *= 0.86
            step_a *= 0.90

    return best_centers, best_angles, best_s


def pack(n):
    """Return a valid packing of n unit pentagons into a minimum-ish square."""
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    best = None

    # Multiple structured starts.
    for seed in range(12):
        for rows, cols in candidate_layouts(n):
            centers, angles, s = build_from_layout(n, rows, cols, seed=seed)
            centers, angles, s = optimize_locally(centers, angles, steps=1800, seed=seed + 17 * rows + cols)
            centers = greedy_shrink(centers, angles, max_iter=90)
            s = enclosing_side(centers, angles)
            if best is None or s < best[2]:
                best = (centers, angles, s)

        # Also try a direct dense motif.
        centers, angles, s = plane_packing_seed(n, seed=seed)
        centers, angles, s = optimize_locally(centers, angles, steps=1600, seed=seed + 999)
        centers = greedy_shrink(centers, angles, max_iter=90)
        s = enclosing_side(centers, angles)
        if best is None or s < best[2]:
            best = (centers, angles, s)

    centers, angles, s = best

    # Final consistency pass.
    centers = greedy_shrink(centers, angles, max_iter=120)
    s = enclosing_side(centers, angles)

    return centers, angles, s
