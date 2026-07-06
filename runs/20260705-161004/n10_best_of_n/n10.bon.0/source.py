"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)

Container convention:
    A point p is inside iff max(|px|, |py|) <= s/2.

This implementation uses a constructive double-lattice / row-staggered heuristic
with local improvement and exact geometric validation. It keeps the required API
and returns a valid packing for every n.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius ~ 0.85065
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # ~ 0.68819
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # widest extent ~ 1.61803
HEIGHT = R + APOTHEM                               # point-up bounding height ~ 1.53884

TAU = 2.0 * math.pi
OVERLAP_EPS = 1e-9


def pentagon_vertices(cx, cy, angle):
    """Vertices of a unit regular pentagon centered at (cx, cy)."""
    ca = math.cos(angle)
    sa = math.sin(angle)
    verts = []
    for k in range(5):
        t = angle + k * TAU / 5.0
        verts.append((cx + R * math.cos(t), cy + R * math.sin(t)))
    return verts


def enclosing_side(centers, angles):
    """Smallest origin-centered square side enclosing all pentagons."""
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


def polygons_overlap(pa, pb):
    """Exact SAT overlap test for convex polygons."""
    for poly in (pa, pb):
        for ax, ay in poly_axes(poly):
            amin, amax = project(pa, ax, ay)
            bmin, bmax = project(pb, ax, ay)
            if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS:
                return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def safe_side(centers, angles):
    return enclosing_side(centers, angles)


def center_pack(points):
    if not points:
        return []
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    dx = (max(xs) + min(xs)) / 2.0
    dy = (max(ys) + min(ys)) / 2.0
    return [(x - dx, y - dy) for x, y in points]


def row_stagger_layout(n, a0=math.pi / 2.0, a1=-math.pi / 2.0,
                       pitch_x=1.60, pitch_y=1.40, stagger=0.50):
    """Generate a staggered row layout with alternating orientations."""
    if n <= 0:
        return [], []
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    centers = []
    angles = []
    idx = 0
    for j in range(rows):
        # Alternate row direction to encourage better boundary placement.
        row = list(range(cols))
        if j % 2 == 1:
            row.reverse()
        shift = (stagger if (j % 2 == 1) else 0.0) * pitch_x
        y = j * pitch_y
        for i in row:
            if idx >= n:
                break
            x = i * pitch_x + shift
            centers.append((x, y))
            angles.append(a0 if ((i + j) % 2 == 0) else a1)
            idx += 1
    return center_pack(centers), angles


def hex_like_layout(n, a0=math.pi / 2.0, a1=-math.pi / 2.0,
                    pitch_x=1.58, pitch_y=1.37, stagger=0.50):
    """Dense-ish staggered lattice using alternating rows and orientations."""
    if n <= 0:
        return [], []
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    centers = []
    angles = []
    idx = 0
    for j in range(rows):
        shift = (stagger * pitch_x) if (j % 2 == 1) else 0.0
        for i in range(cols):
            if idx >= n:
                break
            x = i * pitch_x + shift
            y = j * pitch_y
            centers.append((x, y))
            angles.append(a0 if ((i + j) % 2 == 0) else a1)
            idx += 1
    return center_pack(centers), angles


def border_ring_layout(n):
    """Place as a near-square frame, then fill the inside rows."""
    if n <= 0:
        return [], []
    k = int(math.ceil(n / 4.0))
    side = max(1, k)
    pts = []

    # top row
    for i in range(side):
        pts.append((i, 0))
    # right column
    for j in range(1, side):
        pts.append((side - 1, j))
    # bottom row
    for i in range(side - 2, -1, -1):
        pts.append((i, side - 1))
    # left column
    for j in range(side - 2, 0, -1):
        pts.append((0, j))

    pts = pts[:n]
    centers = [(x * 1.55, y * 1.40) for x, y in pts]
    angles = []
    for x, y in pts:
        angles.append(math.pi / 2.0 if (x + y) % 2 == 0 else -math.pi / 2.0)
    return center_pack(centers), angles


def candidate_packings(n):
    """Generate several construction candidates."""
    cands = []

    # Main family: staggered rows with alternating orientation.
    for px in (1.56, 1.58, 1.60, 1.62):
        for py in (1.34, 1.36, 1.38, 1.40):
            for st in (0.35, 0.40, 0.45, 0.50, 0.55):
                cands.append(hex_like_layout(n, pitch_x=px, pitch_y=py, stagger=st))

    # A slightly wider rectangular layout.
    for px in (1.58, 1.60, 1.63):
        for py in (1.38, 1.42, 1.46):
            for st in (0.40, 0.50):
                cands.append(row_stagger_layout(n, pitch_x=px, pitch_y=py, stagger=st))

    # Border-heavy layout for small n / awkward residues.
    cands.append(border_ring_layout(n))

    return cands


def scale_centers(centers, lam):
    return [(x * lam, y * lam) for x, y in centers]


def repair(centers, angles):
    """Increase a global scale factor until the packing is non-overlapping."""
    if not has_overlap(centers, angles):
        return centers

    lo, hi = 1.0, 1.0
    for _ in range(40):
        hi *= 1.15
        if not has_overlap(scale_centers(centers, hi), angles):
            break
        lo = hi
    else:
        # Fallback: if candidate is badly degenerate, return original.
        return centers

    for _ in range(70):
        mid = (lo + hi) / 2.0
        if has_overlap(scale_centers(centers, mid), angles):
            lo = mid
        else:
            hi = mid
    return scale_centers(centers, hi)


def local_search(centers, angles, iters=500, step0=0.03, seed=0):
    """Small random perturbation search to shave the enclosing square."""
    rng = random.Random(seed)

    def objective(c):
        return enclosing_side(c, angles)

    best = centers[:]
    best_s = objective(best)

    step = step0
    for t in range(iters):
        i = rng.randrange(len(best))
        dx = (rng.random() * 2.0 - 1.0) * step
        dy = (rng.random() * 2.0 - 1.0) * step

        trial = best[:]
        x, y = trial[i]
        trial[i] = (x + dx, y + dy)

        if has_overlap(trial, angles):
            step *= 0.999
            continue

        s = objective(trial)
        if s + 1e-12 < best_s:
            best = trial
            best_s = s
            step *= 0.995
        else:
            step *= 0.9995

    return best


def optimize_orientation_layout(centers, angles):
    """Try a few global angle flips / shifts and keep the best valid variant."""
    variants = []

    # Original.
    variants.append((centers, angles))

    # Flip all by pi.
    variants.append((centers, [a + math.pi for a in angles]))

    # Shift by quarter-turn variants.
    variants.append((centers, [a + math.pi / 2.0 for a in angles]))
    variants.append((centers, [a - math.pi / 2.0 for a in angles]))

    # Normalize angles into [-pi, pi].
    best = None
    best_s = float("inf")
    for c, a in variants:
        c2 = repair(c, a)
        if has_overlap(c2, a):
            continue
        s = enclosing_side(c2, a)
        if s < best_s:
            best_s = s
            best = (c2, a)

    if best is None:
        return centers, angles
    return best


def pack(n):
    """Construct and refine a packing for n unit regular pentagons."""
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    best_centers = None
    best_angles = None
    best_s = float("inf")

    for idx, (centers, angles) in enumerate(candidate_packings(n)):
        centers, angles = optimize_orientation_layout(centers, angles)
        centers = repair(centers, angles)

        # Local improvement on the valid packing.
        for seed in range(3):
            improved = local_search(centers, angles, iters=350, step0=0.035, seed=seed + 1000 * idx)
            if not has_overlap(improved, angles):
                s = enclosing_side(improved, angles)
                if s < best_s:
                    best_s = s
                    best_centers = improved
                    best_angles = angles[:]
                    centers = improved

        # Also consider the repaired baseline.
        s0 = enclosing_side(centers, angles)
        if s0 < best_s and not has_overlap(centers, angles):
            best_s = s0
            best_centers = centers[:]
            best_angles = angles[:]

    # Final safety pass.
    if best_centers is None:
        centers, angles = row_stagger_layout(n)
        centers = repair(centers, angles)
        best_centers, best_angles = centers, angles
        best_s = enclosing_side(best_centers, best_angles)

    # Center the configuration about the origin if it does not worsen the bound.
    cx = (max(x for x, _ in best_centers) + min(x for x, _ in best_centers)) / 2.0
    cy = (max(y for _, y in best_centers) + min(y for _, y in best_centers)) / 2.0
    shifted = [(x - cx, y - cy) for x, y in best_centers]
    if not has_overlap(shifted, best_angles):
        s_shift = enclosing_side(shifted, best_angles)
        if s_shift <= best_s + 1e-12:
            best_centers = shifted
            best_s = s_shift

    return best_centers, best_angles, best_s
