"""Improved packer for unit regular pentagons in an origin-centered square.

Contract: pack(n) -> (centers, angles, s)
- centers: list of (x, y)
- angles: list of rotations in radians
- s: side length of the smallest origin-centered axis-aligned square containing
     all pentagons.

The implementation uses a small library of hand-tuned patterns for n <= 10,
then runs a light numerical refinement (translation/rotation/scale) to reduce
the enclosing square while preserving non-overlap and containment.
"""

import math

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

OVERLAP_EPS = 1e-8
CONTAIN_EPS = 1e-10


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


def pentagons_overlap(pa, pb):
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
                return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if pentagons_overlap(polys[i], polys[j]):
                return True
    return False


def contains_all(centers, angles, s):
    half = s / 2.0 + CONTAIN_EPS
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            if max(abs(vx), abs(vy)) > half:
                return False
    return True


def normalize_to_origin(centers, angles):
    if not centers:
        return centers
    xs = []
    ys = []
    for (cx, cy), a in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, a):
            xs.append(vx)
            ys.append(vy)
    shift_x = (max(xs) + min(xs)) * 0.5
    shift_y = (max(ys) + min(ys)) * 0.5
    return [(x - shift_x, y - shift_y) for x, y in centers]


def scale_centers(centers, lam):
    return [(x * lam, y * lam) for x, y in centers]


def refine_scale(centers, angles):
    """Binary search the best scale <= 1 that preserves validity."""
    if not centers:
        return centers, 0.0
    lo, hi = 0.0, 1.0
    # Ensure hi valid
    if has_overlap(centers, angles) or not contains_all(centers, angles, enclosing_side(centers, angles)):
        return centers, enclosing_side(centers, angles)
    for _ in range(55):
        mid = (lo + hi) * 0.5
        c = scale_centers(centers, mid)
        s = enclosing_side(c, angles)
        if has_overlap(c, angles) or not contains_all(c, angles, s):
            hi = mid
        else:
            lo = mid
    c = scale_centers(centers, lo)
    return c, enclosing_side(c, angles)


def best_rotation_for_set(centers, base_angles):
    """Try a few global rotations, keep the best valid one."""
    best = None
    for t in [i * math.pi / 60.0 for i in range(60)]:
        angles = [a + t for a in base_angles]
        c = normalize_to_origin(centers, angles)
        if has_overlap(c, angles):
            continue
        s = enclosing_side(c, angles)
        if not contains_all(c, angles, s):
            continue
        if best is None or s < best[2]:
            best = (c, angles, s)
    return best


def pattern_grid(n):
    """Compact grid with mixed orientations and slight staggering."""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    # Slightly compressed pitches; safe because repair/refine can scale out if needed.
    pitch_x = 1.42
    pitch_y = 1.34

    centers = []
    angles = []
    for k in range(n):
        i, j = k % cols, k // cols
        x = (i - (cols - 1) / 2.0) * pitch_x
        y = (j - (rows - 1) / 2.0) * pitch_y
        if j % 2 == 1:
            x += 0.42
        if i % 2 == 0:
            ang = math.pi / 2.0
        else:
            ang = -math.pi / 2.0
        if (i + j) % 3 == 0:
            ang += math.pi / 5.0
        centers.append((x, y))
        angles.append(ang)
    return centers, angles


def pattern_hexish(n):
    """A tighter hand-crafted layout used for several n values."""
    pts = [
        (0.00, 0.00, math.pi / 2.0),
        (1.34, 0.06, -math.pi / 2.0),
        (-1.30, -0.02, -math.pi / 2.0),
        (0.68, 1.18, math.pi / 2.0),
        (-0.72, 1.14, math.pi / 2.0),
        (0.70, -1.15, -math.pi / 2.0),
        (-0.72, -1.12, -math.pi / 2.0),
        (2.05, 1.15, math.pi / 2.0),
        (-2.00, 1.15, -math.pi / 2.0),
        (0.02, 2.25, math.pi / 2.0),
    ]
    centers = [(x, y) for x, y, _ in pts[:n]]
    angles = [a for _, _, a in pts[:n]]
    return centers, angles


def pattern_ring(n):
    """Points on concentric rings with alternating orientations."""
    if n == 0:
        return [], []
    centers = [(0.0, 0.0)]
    angles = [math.pi / 2.0]
    if n == 1:
        return centers, angles
    m = n - 1
    r1 = 1.50
    r2 = 2.55
    ring1 = min(5, m)
    ring2 = m - ring1
    for k in range(ring1):
        t = 2.0 * math.pi * k / ring1 + 0.15
        centers.append((r1 * math.cos(t), r1 * math.sin(t)))
        angles.append((-1 if k % 2 else 1) * math.pi / 2.0)
    for k in range(ring2):
        t = 2.0 * math.pi * k / max(1, ring2) + 0.35
        centers.append((r2 * math.cos(t), r2 * math.sin(t)))
        angles.append((1 if k % 2 else -1) * math.pi / 2.0)
    return centers[:n], angles[:n]


def candidate_patterns(n):
    # A small curated set; enough for n <= 10.
    cands = []
    if n <= 10:
        cands.append(pattern_hexish(n))
        cands.append(pattern_grid(n))
        cands.append(pattern_ring(n))
        # Slightly perturbed grid to help asymmetries.
        c, a = pattern_grid(n)
        c = [(x + (0.08 if i % 2 == 0 else -0.05), y + (0.03 if i % 3 == 0 else 0.0))
             for i, (x, y) in enumerate(c)]
        cands.append((c, a))
    return cands


def optimize_pattern(centers, angles):
    """Try global rotation, recentring, and scale refinement."""
    best = None
    # Use multiple global rotations.
    for t in [i * math.pi / 120.0 for i in range(120)]:
        angs = [a + t for a in angles]
        cc = normalize_to_origin(centers, angs)
        if has_overlap(cc, angs):
            continue
        s0 = enclosing_side(cc, angs)
        if not contains_all(cc, angs, s0):
            continue
        cc2, s2 = refine_scale(cc, angs)
        if has_overlap(cc2, angs):
            continue
        if best is None or s2 < best[2]:
            best = (cc2, angs, s2)

    if best is None:
        # Fallback: just return normalized original.
        angs = angles[:]
        cc = normalize_to_origin(centers, angs)
        s = enclosing_side(cc, angs)
        return cc, angs, s
    return best


def pack(n):
    """Return a valid packing of n unit regular pentagons in the smallest square found."""
    if n <= 0:
        return [], [], 0.0

    best = None
    for centers, angles in candidate_patterns(n):
        packed = optimize_pattern(centers, angles)
        if best is None or packed[2] < best[2]:
            best = packed

    centers, angles, s = best

    # Final safety pass: if the best pattern is numerically borderline, gently inflate.
    if has_overlap(centers, angles) or not contains_all(centers, angles, s):
        lam = 1.0005
        while True:
            c2 = scale_centers(centers, lam)
            s2 = enclosing_side(c2, angles)
            if not has_overlap(c2, angles) and contains_all(c2, angles, s2):
                centers, s = c2, s2
                break
            lam *= 1.001

    return centers, angles, s
