"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles: list[float]
- s: outer side of the enclosing square
Container convention: point p is inside iff max(|px|, |py|) <= s/2.

Strategy:
- Use a geometry-first search over a family of structured templates inspired by
  double-lattice motifs, boundary rows, staggered rows, and ring-like layouts.
- Refine the best candidates with a lightweight numerical local search.
- Guarantee validity by checking polygon overlap exactly via SAT and, if needed,
  conservative scaling / repair.
- The search is deterministic and designed to improve the square side compared
  with naive row packing for n <= 10.
"""

import math
import random

SIDE = 1.0
TAU = 2.0 * math.pi
R = SIDE / (2.0 * math.sin(math.pi / 5.0))  # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM
OVERLAP_EPS = 1e-11


def pentagon_vertices(cx, cy, angle):
    return [
        (
            cx + R * math.cos(angle + k * TAU / 5.0),
            cy + R * math.sin(angle + k * TAU / 5.0),
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
    """Separating axis theorem: True iff polygons overlap with positive area."""
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


def _canonical_angle(a):
    # Keep angles numerically tame.
    a = (a + math.pi) % TAU - math.pi
    return a


def repair(centers, angles):
    """Radially dilate centers until overlaps vanish. Rotations stay fixed."""
    if not centers:
        return centers
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.01
    for _ in range(80):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.2
    else:
        return centers

    for _ in range(80):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def _template_rows(row_lengths, pitch_x, pitch_y, shift_alt=True, edge_tilt=True, base_flip=True):
    centers, angles = [], []
    rows = len(row_lengths)
    idx = 0
    for r, m in enumerate(row_lengths):
        y = (r - (rows - 1) / 2.0) * pitch_y
        x_shift = (0.5 * pitch_x if (shift_alt and (r % 2 == 1)) else 0.0)
        x0 = -((m - 1) * pitch_x) / 2.0 + x_shift

        base = 0.0
        if edge_tilt and (r == 0 or r == rows - 1):
            base = (math.pi / 18.0) * (1.0 if r == 0 else -1.0)
        if base_flip and rows > 2 and r == rows // 2:
            base *= -0.6

        for c in range(m):
            x = x0 + c * pitch_x
            ang = base + (math.pi / 2.0 if ((r + c) % 2 == 0) else -math.pi / 2.0)
            if rows >= 4 and (r in (0, rows - 1)) and (c == 0 or c == m - 1):
                ang += (math.pi / 42.0) * (-1.0 if (r + c) % 2 else 1.0)
            centers.append((x, y))
            angles.append(_canonical_angle(ang))
            idx += 1
    return _center(centers), angles


def _balanced_row_lengths(n, rows):
    base = n // rows
    rem = n % rows
    arr = [base + (1 if i < rem else 0) for i in range(rows)]
    while len(arr) > 1 and arr[-1] == 0:
        arr.pop()
    return arr


def _double_lattice_template(n, m=None, rows=None, shift=0.42, px_scale=0.84, py_scale=0.80):
    if n <= 0:
        return [], []
    if m is None:
        m = int(math.ceil(math.sqrt(n)))
    if rows is None:
        rows = int(math.ceil(n / m))
    pitch_x = WIDTH * px_scale
    pitch_y = HEIGHT * py_scale
    centers, angles = [], []
    k = 0
    for r in range(rows):
        for c in range(m):
            if k >= n:
                break
            x = (c - (m - 1) / 2.0) * pitch_x + (shift * pitch_x if r % 2 else 0.0)
            y = (r - (rows - 1) / 2.0) * pitch_y
            ang = math.pi / 2.0 if ((r + c) % 2 == 0) else -math.pi / 2.0
            edge = (r in (0, rows - 1) or c in (0, m - 1))
            if edge:
                ang += (math.pi / 24.0) * (1.0 if ((r + c) % 2 == 0) else -1.0)
            centers.append((x, y))
            angles.append(_canonical_angle(ang))
            k += 1
    return _center(centers), angles


def _spiral_template(n):
    centers, angles = [], []
    if n <= 0:
        return centers, angles
    placed = 0
    ring = 0
    # Shape loosely follows a stretched spiral, useful for boundary-friendly placements.
    while placed < n:
        count = 1 if ring == 0 else 6 * ring
        rx = ring * WIDTH * 0.44
        ry = ring * HEIGHT * 0.37
        for t in range(count):
            if placed >= n:
                break
            th = TAU * t / count + (0.19 if ring % 2 else 0.0)
            x = rx * math.cos(th)
            y = ry * math.sin(th)
            ang = (math.pi / 2.0 if placed % 2 == 0 else -math.pi / 2.0)
            ang += (0.12 if ring % 2 else -0.12)
            centers.append((x, y))
            angles.append(_canonical_angle(ang))
            placed += 1
        ring += 1
    return _center(centers), angles


def _boundary_lattice_template(n):
    """A rectangle-like packing with a more deliberate boundary emphasis."""
    if n <= 0:
        return [], []
    # Try several aspect ratios and choose the one with best pre-optimization score.
    best = None
    best_s = float("inf")
    for rows in range(1, min(n, 5) + 1):
        cols = int(math.ceil(n / rows))
        pitch_x = WIDTH * (0.78 + 0.05 * cols)
        pitch_y = HEIGHT * (0.68 + 0.05 * rows)
        centers, angles = [], []
        k = 0
        for r in range(rows):
            y = (r - (rows - 1) / 2.0) * pitch_y
            shift = 0.5 * pitch_x if (r % 2 == 1) else 0.0
            x0 = -((cols - 1) * pitch_x) / 2.0 + shift
            for c in range(cols):
                if k >= n:
                    break
                x = x0 + c * pitch_x
                ang = math.pi / 2.0 if ((r + c) % 2 == 0) else -math.pi / 2.0
                if r in (0, rows - 1) or c in (0, cols - 1):
                    ang += (math.pi / 20.0) * (1.0 if (r + c) % 2 == 0 else -1.0)
                centers.append((x, y))
                angles.append(_canonical_angle(ang))
                k += 1
        centers = _center(centers)
        s = enclosing_side(centers, angles)
        if s < best_s:
            best = (centers, angles)
            best_s = s
    return best


def _ring_template(n):
    """Small n often benefits from a center + ring arrangement."""
    if n <= 0:
        return [], []
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]

    centers = [(0.0, 0.0)]
    angles = [math.pi / 2.0]
    placed = 1
    ring = 1
    while placed < n:
        count = min(6 * ring, n - placed)
        rx = ring * WIDTH * 0.46
        ry = ring * HEIGHT * 0.38
        for t in range(count):
            th = TAU * t / count + (0.11 if ring % 2 else 0.0)
            x = rx * math.cos(th)
            y = ry * math.sin(th)
            ang = (math.pi / 2.0 if (t + ring) % 2 == 0 else -math.pi / 2.0)
            ang += (0.15 if ring % 2 else -0.15)
            centers.append((x, y))
            angles.append(_canonical_angle(ang))
            placed += 1
            if placed >= n:
                break
        ring += 1
    return _center(centers), angles


def _candidate_templates(n):
    if n <= 0:
        return [([], [])]
    if n == 1:
        return [([(0.0, 0.0)], [math.pi / 2.0])]

    cands = []

    # Hand-tuned row partitions for n <= 10.
    row_sets = {
        2: [[1, 1], [2]],
        3: [[1, 2], [3], [2, 1]],
        4: [[2, 2], [1, 3], [3, 1], [4]],
        5: [[2, 3], [3, 2], [1, 4], [4, 1], [5]],
        6: [[2, 2, 2], [3, 3], [1, 2, 3], [1, 5], [5, 1], [6]],
        7: [[2, 2, 3], [3, 2, 2], [1, 3, 3], [4, 3], [3, 4], [7]],
        8: [[2, 3, 3], [3, 3, 2], [2, 2, 2, 2], [1, 2, 2, 3], [4, 4], [8]],
        9: [[3, 3, 3], [2, 3, 4], [4, 3, 2], [1, 2, 3, 3], [3, 2, 2, 2], [9]],
        10: [[2, 2, 3, 3], [3, 2, 2, 3], [2, 4, 4], [4, 3, 3], [5, 5], [1, 3, 3, 3], [10]],
    }

    # Add row-based templates with varied pitches and phase shifts.
    for rows in range(1, min(n, 5) + 1):
        partitions = row_sets.get(n, [])
        if not partitions:
            partitions = [_balanced_row_lengths(n, rows)]
        for row_lengths in partitions:
            if sum(row_lengths) != n:
                continue
            px_base = WIDTH * (0.77 + 0.03 * rows)
            py_base = HEIGHT * (0.70 + 0.045 * rows)
            cands.append(_template_rows(row_lengths, px_base, py_base, shift_alt=True, edge_tilt=True, base_flip=True))
            cands.append(_template_rows(row_lengths, px_base * 0.98, py_base * 1.02, shift_alt=False, edge_tilt=True, base_flip=True))
            cands.append(_template_rows(row_lengths, px_base * 1.01, py_base * 0.96, shift_alt=True, edge_tilt=False, base_flip=False))

    cands.append(_double_lattice_template(n, shift=0.40, px_scale=0.82, py_scale=0.78))
    cands.append(_double_lattice_template(n, shift=0.46, px_scale=0.80, py_scale=0.82))
    cands.append(_boundary_lattice_template(n))
    cands.append(_spiral_template(n))
    cands.append(_ring_template(n))

    # Deduplicate trivial exact duplicates by coarse signature.
    uniq = []
    seen = set()
    for c, a in cands:
        sig = tuple((round(x, 4), round(y, 4), round(ang, 4)) for (x, y), ang in zip(c, a))
        if sig not in seen:
            seen.add(sig)
            uniq.append((c, a))
    return uniq


def _objective(centers, angles):
    return enclosing_side(centers, angles)


def _local_search(centers, angles, seed=0):
    rng = random.Random(seed)
    centers = repair(list(centers), list(angles))
    centers = _center(centers)
    angles = [_canonical_angle(a) for a in angles]
    best_centers = centers[:]
    best_angles = angles[:]
    best_s = _objective(best_centers, best_angles)

    step_pos = max(0.02, best_s * 0.03)
    step_ang = 0.18

    def try_state(cc, aa):
        if has_overlap(cc, aa):
            return None
        return _objective(cc, aa)

    for phase in range(10):
        for _ in range(220):
            i = rng.randrange(len(best_centers))
            cx, cy = best_centers[i]
            a = best_angles[i]

            neighborhood = [
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
            for dx, dy, da in neighborhood:
                cc = best_centers[:]
                aa = best_angles[:]
                cc[i] = (
                    cx + dx + bias_x * step_pos * 0.08,
                    cy + dy + bias_y * step_pos * 0.08,
                )
                aa[i] = _canonical_angle(a + da + (0.02 if (abs(cx) > abs(cy) and i % 2 == 0) else 0.0))
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
                cc[j] = (
                    cc[j][0] + rng.uniform(-step_pos, step_pos) * 0.30,
                    cc[j][1] + rng.uniform(-step_pos, step_pos) * 0.30,
                )
                aa[j] = _canonical_angle(aa[j] + rng.uniform(-step_ang, step_ang) * 0.25)
                cc = _center(cc)
                cc = repair(cc, aa)
                val = try_state(cc, aa)
                if val is not None and val <= best_s * 1.02:
                    best_centers, best_angles, best_s = cc, aa, val

        step_pos *= 0.62
        step_ang *= 0.70

    best_centers = repair(best_centers, best_angles)
    best_centers = _center(best_centers)
    best_angles = [_canonical_angle(a) for a in best_angles]
    best_s = _objective(best_centers, best_angles)
    return best_centers, best_angles, best_s


def _shrink_uniform(centers, angles):
    """Try to shrink by scaling centers toward origin while staying non-overlapping."""
    if not centers:
        return centers, 0.0
    if has_overlap(centers, angles):
        centers = repair(centers, angles)
    lo, hi = 0.0, 1.0
    for _ in range(70):
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

    # Multiple deterministic refinement passes.
    for seed, (c0, a0) in enumerate(templates):
        c, a, s = _local_search(c0, a0, seed=seed + 1234)
        c, s2 = _shrink_uniform(c, a)
        if s2 < s:
            s = s2
        if s < best_s:
            best = (c, a, s)
            best_s = s

    # A final conservative repair and exact measurement.
    centers, angles, _ = best
    centers = repair(centers, angles)
    centers = _center(centers)
    angles = [_canonical_angle(a) for a in angles]
    s = enclosing_side(centers, angles)
    return centers, angles, s
