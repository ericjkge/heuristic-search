"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles: list[float]
- s: outer side of the enclosing square
Container convention: point p is inside iff max(|px|, |py|) <= s/2.

This implementation uses a geometry-first construction:
- several deterministic candidate templates inspired by double-lattice packing,
  staggered rows, and boundary rows with opposing orientations
- exact polygon overlap checking via SAT
- local optimization with coordinate/angle perturbations
- conservative scaling to guarantee validity

The goal is to produce smaller square packings for up to 10 unit regular pentagons.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))  # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

OVERLAP_EPS = 1e-10


def pentagon_vertices(cx, cy, angle):
    return [
        (
            cx + R * math.cos(angle + k * 2.0 * math.pi / 5.0),
            cy + R * math.sin(angle + k * 2.0 * math.pi / 5.0),
        )
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    if not centers:
        return 0.0
    m = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            m = max(m, abs(vx), abs(vy))
    return 2.0 * m


def _proj(poly, ux, uy):
    dots = [x * ux + y * uy for x, y in poly]
    return min(dots), max(dots)


def pentagons_overlap(pa, pb):
    """Separating axis test; returns True if polygons overlap with positive area."""
    for poly in (pa, pb):
        for i in range(5):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
            ux, uy = -(y2 - y1), (x2 - x1)
            amin, amax = _proj(pa, ux, uy)
            bmin, bmax = _proj(pb, ux, uy)
            norm = math.hypot(ux, uy)
            if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS * max(1.0, norm):
                return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if pentagons_overlap(polys[i], polys[j]):
                return True
    return False


def _center(centers):
    if not centers:
        return centers
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    return [(x - mx, y - my) for x, y in centers]


def repair(centers, angles):
    """Radially dilate centers until overlaps vanish. Rotations stay fixed."""
    if not centers:
        return centers
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.02
    for _ in range(70):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.25
    else:
        return centers

    for _ in range(70):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def _template_rows(n, row_lengths, pitch_x, pitch_y, row_phase=0.0, tilt_edges=True):
    centers, angles = [], []
    rows = len(row_lengths)
    idx = 0
    for r, m in enumerate(row_lengths):
        y = (r - (rows - 1) / 2.0) * pitch_y
        x_shift = 0.5 * pitch_x if ((r + int(row_phase)) % 2 == 1) else 0.0
        x0 = -((m - 1) * pitch_x) / 2.0 + x_shift

        edge = (r == 0 or r == rows - 1)
        base = 0.0
        if edge and tilt_edges:
            base = math.pi / 20.0 * (1.0 if r == 0 else -1.0)

        for c in range(m):
            if idx >= n:
                break
            x = x0 + c * pitch_x
            ang = base + (math.pi / 2.0 if ((r + c) % 2 == 0) else -math.pi / 2.0)
            centers.append((x, y))
            angles.append(ang)
            idx += 1
    return _center(centers), angles


def _balanced_row_lengths(n, rows):
    base = n // rows
    rem = n % rows
    arr = [base + (1 if i < rem else 0) for i in range(rows)]
    # Enforce monotone-ish envelope
    while len(arr) > 1 and arr[-1] == 0:
        arr.pop()
    return arr


def _spiral_template(n):
    centers, angles = [], []
    if n <= 0:
        return centers, angles
    placed = 0
    ring = 0
    while placed < n:
        if ring == 0:
            count = 1
        else:
            count = 6 * ring
        rx = ring * WIDTH * 0.44
        ry = ring * HEIGHT * 0.37
        for t in range(count):
            if placed >= n:
                break
            th = 2.0 * math.pi * t / count + (0.19 if ring % 2 else 0.0)
            x = rx * math.cos(th)
            y = ry * math.sin(th)
            ang = (math.pi / 2.0 if placed % 2 == 0 else -math.pi / 2.0)
            ang += (0.10 if ring % 2 else -0.10)
            centers.append((x, y))
            angles.append(ang)
            placed += 1
        ring += 1
    return _center(centers), angles


def _double_lattice_template(n):
    """Two interleaved sublattices with opposite orientations."""
    if n <= 0:
        return [], []
    m = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / m))
    pitch_x = WIDTH * 0.84
    pitch_y = HEIGHT * 0.80
    centers, angles = [], []
    k = 0
    for r in range(rows):
        for c in range(m):
            if k >= n:
                break
            x = (c - (m - 1) / 2.0) * pitch_x + (0.42 * pitch_x if r % 2 else 0.0)
            y = (r - (rows - 1) / 2.0) * pitch_y
            edge = (r == 0 or r == rows - 1 or c == 0 or c == m - 1)
            ang = math.pi / 2.0 if ((r + c) % 2 == 0) else -math.pi / 2.0
            if edge:
                ang += (math.pi / 22.0 if ((r + c) % 2 == 0) else -math.pi / 22.0)
            centers.append((x, y))
            angles.append(ang)
            k += 1
    return _center(centers), angles


def _candidate_templates(n):
    if n == 1:
        return [([(0.0, 0.0)], [math.pi / 2.0])]

    cands = []
    # row-based variants
    for rows in range(1, min(n, 5) + 1):
        row_lengths = _balanced_row_lengths(n, rows)
        if rows == 1:
            pitch_x = WIDTH * 0.82
            pitch_y = HEIGHT
        else:
            pitch_x = WIDTH * (0.80 + 0.03 * rows)
            pitch_y = HEIGHT * (0.72 + 0.035 * rows)
        cands.append(_template_rows(n, row_lengths, pitch_x, pitch_y, row_phase=0.0, tilt_edges=True))
        cands.append(_template_rows(n, row_lengths, pitch_x * 0.98, pitch_y * 1.01, row_phase=1.0, tilt_edges=True))

    cands.append(_double_lattice_template(n))
    cands.append(_spiral_template(n))
    return cands


def _evaluate(centers, angles):
    return enclosing_side(centers, angles)


def _objective(centers, angles):
    return _evaluate(centers, angles)


def _local_search(centers, angles, seed=0):
    rng = random.Random(seed)
    centers = repair(list(centers), list(angles))
    centers = _center(centers)
    s = _objective(centers, angles)

    best_centers = centers[:]
    best_angles = angles[:]
    best_s = s

    step_pos = max(0.015, s * 0.04)
    step_ang = 0.14

    def try_state(cc, aa):
        if has_overlap(cc, aa):
            return None
        return _objective(cc, aa)

    for phase in range(8):
        for _ in range(260):
            i = rng.randrange(len(best_centers))
            cx, cy = best_centers[i]
            a = best_angles[i]

            # generate move set
            moves = [
                (0.0, 0.0, 0.0),
                (step_pos, 0.0, 0.0),
                (-step_pos, 0.0, 0.0),
                (0.0, step_pos, 0.0),
                (0.0, -step_pos, 0.0),
                (0.7 * step_pos, 0.7 * step_pos, 0.0),
                (0.7 * step_pos, -0.7 * step_pos, 0.0),
                (0.0, 0.0, step_ang),
                (0.0, 0.0, -step_ang),
                (0.5 * step_pos, 0.0, step_ang),
                (-0.5 * step_pos, 0.0, -step_ang),
            ]

            bias_x = -math.copysign(1.0, cx) if abs(cx) > 1e-10 else 0.0
            bias_y = -math.copysign(1.0, cy) if abs(cy) > 1e-10 else 0.0

            improved = False
            for dx, dy, da in moves:
                cc = best_centers[:]
                aa = best_angles[:]
                cc[i] = (cx + dx + bias_x * step_pos * 0.10, cy + dy + bias_y * step_pos * 0.10)
                aa[i] = a + da + (0.02 if (abs(cx) > abs(cy) and i % 2 == 0) else 0.0)
                cc = _center(cc)
                val = try_state(cc, aa)
                if val is not None and val + 1e-12 < best_s:
                    best_centers, best_angles, best_s = cc, aa, val
                    improved = True
                    break

            if not improved and rng.random() < 0.10:
                cc = best_centers[:]
                aa = best_angles[:]
                j = rng.randrange(len(cc))
                cc[j] = (cc[j][0] + rng.uniform(-step_pos, step_pos) * 0.25, cc[j][1] + rng.uniform(-step_pos, step_pos) * 0.25)
                aa[j] = aa[j] + rng.uniform(-step_ang, step_ang) * 0.25
                cc = _center(cc)
                cc = repair(cc, aa)
                val = try_state(cc, aa)
                if val is not None and val <= best_s * 1.01:
                    best_centers, best_angles, best_s = cc, aa, val

        step_pos *= 0.58
        step_ang *= 0.68

    best_centers = repair(best_centers, best_angles)
    best_centers = _center(best_centers)
    best_s = _objective(best_centers, best_angles)
    return best_centers, best_angles, best_s


def _shrink_uniform(centers, angles):
    """Try to shrink by scaling centers toward origin while staying non-overlapping."""
    if not centers:
        return centers, 0.0
    lo, hi = 0.0, 1.0
    # only shrink if valid
    if has_overlap(centers, angles):
        centers = repair(centers, angles)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        cc = [(x * mid, y * mid) for x, y in centers]
        if has_overlap(cc, angles):
            lo = mid
        else:
            hi = mid
    cc = [(x * hi, y * hi) for x, y in centers]
    return cc, _objective(cc, angles)


def pack(n):
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    templates = _candidate_templates(n)

    best = None
    best_s = float("inf")

    for seed, (c0, a0) in enumerate(templates):
        c, a, s = _local_search(c0, a0, seed=seed + 123)
        c, s2 = _shrink_uniform(c, a)
        if s2 < s:
            s = s2
        if s < best_s:
            best = (c, a, s)
            best_s = s

    centers, angles, s = best
    centers = repair(centers, angles)
    centers = _center(centers)
    s = enclosing_side(centers, angles)
    return centers, angles, s
