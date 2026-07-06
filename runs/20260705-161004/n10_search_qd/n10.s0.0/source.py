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


def pack(n):
    """Heuristic search over small double-lattice-like layouts plus local
    optimization, then a final feasibility repair if needed."""
    import random

    def wrap(a):
        tau = 2.0 * math.pi
        return a % tau

    def objective(vec):
        # vec = [cx0, cy0, a0, cx1, cy1, a1, ...]
        centers = [(vec[3 * i], vec[3 * i + 1]) for i in range(n)]
        angles = [vec[3 * i + 2] for i in range(n)]
        s = enclosing_side(centers, angles)

        # Smooth-ish penalties using only vertex distances to square and pairwise AABB gaps.
        polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
        pen = 0.0
        for poly in polys:
            for x, y in poly:
                m = max(abs(x), abs(y)) - s / 2.0
                if m > 0.0:
                    pen += 1000.0 * m * m
        for i in range(n):
            for j in range(i + 1, n):
                pa, pb = polys[i], polys[j]
                for poly in (pa, pb):
                    for k in range(5):
                        (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % 5]
                        ux, uy = -(y2 - y1), (x2 - x1)
                        amin = min(x * ux + y * uy for x, y in pa)
                        amax = max(x * ux + y * uy for x, y in pa)
                        bmin = min(x * ux + y * uy for x, y in pb)
                        bmax = max(x * ux + y * uy for x, y in pb)
                        gap = min(amax, bmax) - max(amin, bmin)
                        if gap > 0.0:
                            pen += 1000.0 * gap * gap
        return s + pen

    # Candidate families: strip/paired motifs with opposite orientations.
    best = None
    best_s = float("inf")

    def try_layout(centers, angles):
        nonlocal best, best_s
        if has_overlap(centers, angles):
            centers = repair(centers, angles)
        s = enclosing_side(centers, angles)
        if s < best_s:
            best_s = s
            best = (centers, angles, s)

    rng = random.Random(0)

    # 1) Hand-tuned mixed-orientation rows.
    for cols in range(2, n + 1):
        rows = int(math.ceil(n / cols))
        if cols * rows < n:
            continue
        for sx in (0.82, 0.88, 0.94):
            for sy in (0.80, 0.86, 0.92):
                for phase in (0.0, 0.5):
                    centers, angles = [], []
                    pitch_x = WIDTH * sx
                    pitch_y = HEIGHT * sy
                    for k in range(n):
                        i, j = k % cols, k // cols
                        x = (i - (cols - 1) / 2.0) * pitch_x
                        y = (j - (rows - 1) / 2.0) * pitch_y
                        if j % 2 == 1:
                            x += phase * pitch_x * 0.5
                        centers.append((x, y))
                        angles.append((math.pi / 2.0) if ((i + j) % 2 == 0) else (-math.pi / 2.0))
                    try_layout(centers, angles)

    # 2) Double-lattice inspired: alternating rows of opposite orientation,
    # with row offsets and tilt.
    for cols in range(2, n + 1):
        rows = int(math.ceil(n / cols))
        if cols * rows < n:
            continue
        for ox in (-0.22, -0.12, 0.0, 0.12, 0.22):
            for tilt in (-0.18, -0.08, 0.0, 0.08, 0.18):
                for sx in (0.84, 0.90, 0.96):
                    for sy in (0.82, 0.88, 0.94):
                        centers, angles = [], []
                        pitch_x = WIDTH * sx
                        pitch_y = HEIGHT * sy
                        for k in range(n):
                            i, j = k % cols, k // cols
                            x = (i - (cols - 1) / 2.0) * pitch_x
                            y = (j - (rows - 1) / 2.0) * pitch_y
                            x += (j - (rows - 1) / 2.0) * ox * pitch_x
                            a = (math.pi / 2.0 + tilt) if (j % 2 == 0) else (-math.pi / 2.0 - tilt)
                            centers.append((x, y))
                            angles.append(wrap(a))
                        try_layout(centers, angles)

    # 3) One-dimensional boundary-friendly strip for small n / odd n.
    for cols in range(1, n + 1):
        rows = int(math.ceil(n / cols))
        if cols * rows < n:
            continue
        for sx in (0.82, 0.88, 0.94):
            for sy in (0.76, 0.84, 0.92):
                centers, angles = [], []
                pitch_x = WIDTH * sx
                pitch_y = HEIGHT * sy
                for k in range(n):
                    i, j = k % cols, k // cols
                    x = (i - (cols - 1) / 2.0) * pitch_x
                    y = (j - (rows - 1) / 2.0) * pitch_y
                    if j % 2 == 1:
                        x += 0.33 * pitch_x
                    ang = math.pi / 2.0 if ((i + 2 * j) % 3 != 0) else -math.pi / 2.0
                    centers.append((x, y))
                    angles.append(ang)
                try_layout(centers, angles)

    # 4) Local randomized coordinate search from the best discrete candidate.
    if best is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        centers, angles = [], []
        for k in range(n):
            i, j = k % cols, k // cols
            centers.append(((i - (cols - 1) / 2.0) * WIDTH, (j - (rows - 1) / 2.0) * HEIGHT))
            angles.append(math.pi / 2.0 if i % 2 == 0 else -math.pi / 2.0)
        best = (centers, angles, enclosing_side(centers, angles))
        best_s = best[2]

    centers, angles, _ = best
    vec = []
    for (x, y), a in zip(centers, angles):
        vec.extend([x, y, a])

    step = 0.25
    for _ in range(220):
        improved = False
        base = objective(vec)
        for _trial in range(40):
            cand = vec[:]
            idx = rng.randrange(len(cand))
            if idx % 3 == 2:
                cand[idx] = wrap(cand[idx] + rng.uniform(-0.35, 0.35) * step)
            else:
                cand[idx] += rng.uniform(-1.0, 1.0) * step
            val = objective(cand)
            if val < base:
                vec = cand
                base = val
                improved = True
        step *= 0.93
        if not improved and step < 1e-3:
            break

    centers = [(vec[3 * i], vec[3 * i + 1]) for i in range(n)]
    angles = [wrap(vec[3 * i + 2]) for i in range(n)]

    if has_overlap(centers, angles):
        centers = repair(centers, angles)

    return centers, angles, enclosing_side(centers, angles)
# EVOLVE-BLOCK-END
