"""Improved packing of n unit regular pentagons into the smallest
origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s). Container = origin-centered
axis-aligned square, side s: point p inside iff max(|px|, |py|) <= s/2.

Strategy:
- Use a hand-tuned family of mixed-orientation motifs inspired by double-lattice
  pentagon packings.
- Generate several candidate layouts for n, score them by enclosing square side,
  and locally refine with coordinate-descent nudges while preserving validity.
- Fall back to robust grid-like placements if a candidate fails.

The code avoids external dependencies and remains fully deterministic.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

PHI = (1.0 + 5.0 ** 0.5) / 2.0
TAU = 2.0 * math.pi


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
    )


# Geometry / overlap ---------------------------------------------------------

OVERLAP_EPS = 1e-9


def _project(poly, ux, uy):
    vals = [x * ux + y * uy for x, y in poly]
    return min(vals), max(vals)


def pentagons_overlap(pa, pb):
    """True iff polygons intersect or touch within tolerance."""
    for poly in (pa, pb):
        for i in range(5):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
            ux, uy = -(y2 - y1), (x2 - x1)
            amin, amax = _project(pa, ux, uy)
            bmin, bmax = _project(pb, ux, uy)
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


def within_square(centers, angles, s):
    half = s / 2.0 + 1e-12
    for (cx, cy), a in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, a):
            if max(abs(vx), abs(vy)) > half:
                return False
    return True


def repair_dilate(centers, angles):
    """Scale centers about origin until overlaps vanish, keeping rotations fixed."""
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.05
    for _ in range(40):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.35
    else:
        return centers

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


# Candidate construction -----------------------------------------------------

def _motif_offsets():
    """A small set of relative placements that often interlock well."""
    # Chosen by manual search and approximate geometric intuition.
    return [
        (0.0, 0.0, 0.0),
        (0.93, 0.06, math.pi),
        (-0.89, 0.18, math.pi),
        (0.45, 0.88, math.pi),
        (-0.48, -0.84, math.pi),
    ]


def _double_lattice_rows(n, row_shift=0.46, col_shift=0.0):
    """Alternating-orientation staggered rows."""
    # Dense-ish row pitch based on pentagon height and point overlap.
    pitch_y = 1.34
    pitch_x = 1.50
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    centers, angles = [], []
    k = 0
    for j in range(rows):
        y = (j - (rows - 1) / 2.0) * pitch_y
        shift = row_shift if (j % 2 == 1) else 0.0
        for i in range(cols):
            if k >= n:
                break
            x = (i - (cols - 1) / 2.0) * pitch_x + shift
            centers.append((x + col_shift, y))
            angles.append(0.0 if (i + j) % 2 == 0 else math.pi)
            k += 1
    return centers, angles


def _spiral_mix(n):
    """Seed points on a golden-angle spiral with alternating orientations."""
    centers, angles = [], []
    a = 0.52
    b = 0.52
    for k in range(n):
        t = k + 1
        r = a * math.sqrt(t)
        ang = t * (TAU / PHI)
        x = r * math.cos(ang)
        y = r * math.sin(ang)
        # Flatten slightly to fit a square better.
        y *= 0.88
        centers.append((x, y))
        angles.append((k % 2) * math.pi)
    # center
    mx = sum(x for x, _ in centers) / n
    my = sum(y for _, y in centers) / n
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def _hex_like(n):
    """Triangular lattice with two orientations; boundary rows slightly tilted."""
    pitch_x = 1.42
    pitch_y = 1.23
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    centers, angles = [], []
    idx = 0
    for j in range(rows):
        y = (j - (rows - 1) / 2.0) * pitch_y
        xshift = (j % 2) * (pitch_x / 2.0)
        for i in range(cols):
            if idx >= n:
                break
            x = (i - (cols - 1) / 2.0) * pitch_x + xshift
            centers.append((x, y))
            if j in (0, rows - 1):
                angles.append(math.pi / 10.0 if i % 2 == 0 else math.pi + math.pi / 10.0)
            else:
                angles.append(0.0 if (i + j) % 2 == 0 else math.pi)
            idx += 1
    # Recentre.
    mx = sum(x for x, _ in centers) / n
    my = sum(y for _, y in centers) / n
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def _motif_pack(n):
    """Pack pentagons using repeated 5-pentagon motifs on a coarse grid."""
    motif = _motif_offsets()
    centers, angles = [], []
    # layout motifs in a near-square arrangement
    m = int(math.ceil(n / 5.0))
    cols = int(math.ceil(math.sqrt(m)))
    rows = int(math.ceil(m / cols))
    gx, gy = 2.15, 1.95
    motif_idx = 0
    for j in range(rows):
        for i in range(cols):
            if motif_idx >= m:
                break
            ox = (i - (cols - 1) / 2.0) * gx
            oy = (j - (rows - 1) / 2.0) * gy
            if j % 2 == 1:
                ox += 0.35
            for dx, dy, ang in motif:
                if len(centers) >= n:
                    break
                centers.append((ox + dx, oy + dy))
                angles.append(ang + (math.pi / 5.0 if (motif_idx % 2 == 1 and ang == 0.0) else 0.0))
            motif_idx += 1
    # trim and recentre
    centers = centers[:n]
    angles = angles[:n]
    mx = sum(x for x, _ in centers) / n
    my = sum(y for _, y in centers) / n
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


# Local refinement -----------------------------------------------------------

def _all_polys(centers, angles):
    return [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]


def _min_pairwise_separation(centers, angles):
    polys = _all_polys(centers, angles)
    best = 1e9
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            pa, pb = polys[i], polys[j]
            sep = 1e9
            for poly in (pa, pb):
                for t in range(5):
                    (x1, y1), (x2, y2) = poly[t], poly[(t + 1) % 5]
                    ux, uy = -(y2 - y1), (x2 - x1)
                    amin, amax = _project(pa, ux, uy)
                    bmin, bmax = _project(pb, ux, uy)
                    norm = math.hypot(ux, uy)
                    sep = min(sep, (min(amax, bmax) - max(amin, bmin)) / (norm + 1e-15))
            best = min(best, sep)
    return best


def _refine(centers, angles, rounds=180):
    """Coordinate descent on centers and occasional angle tweaks."""
    centers = list(centers)
    angles = list(angles)

    def score(c, a):
        return enclosing_side(c, a)

    best_s = score(centers, angles)
    best_c = centers[:]
    best_a = angles[:]

    step_xy = 0.08
    step_ang = math.pi / 30.0

    for r in range(rounds):
        improved = False
        order = list(range(len(centers)))
        random.shuffle(order)

        for i in order:
            base_x, base_y = centers[i]
            base_a = angles[i]
            best_local = None
            candidates = [
                (0.0, 0.0, 0.0),
                ( step_xy, 0.0, 0.0),
                (-step_xy, 0.0, 0.0),
                (0.0,  step_xy, 0.0),
                (0.0, -step_xy, 0.0),
                ( step_xy,  step_xy, 0.0),
                ( step_xy, -step_xy, 0.0),
                (-step_xy,  step_xy, 0.0),
                (-step_xy, -step_xy, 0.0),
                (0.0, 0.0, step_ang),
                (0.0, 0.0, -step_ang),
            ]
            for dx, dy, da in candidates:
                centers[i] = (base_x + dx, base_y + dy)
                angles[i] = base_a + da
                if not has_overlap(centers, angles):
                    s = score(centers, angles)
                    if best_local is None or s < best_local[0]:
                        best_local = (s, centers[i], angles[i])
            centers[i] = (base_x, base_y)
            angles[i] = base_a
            if best_local is not None and best_local[0] + 1e-12 < best_s:
                best_s = best_local[0]
                centers[i] = best_local[1]
                angles[i] = best_local[2]
                best_c = centers[:]
                best_a = angles[:]
                improved = True

        if not improved:
            step_xy *= 0.72
            step_ang *= 0.72
            if step_xy < 1e-4:
                break

    return best_c, best_a


# Main pack -----------------------------------------------------------------

def pack(n):
    if n <= 0:
        return [], [], 0.0

    random.seed(123456789 + 1009 * n)

    candidates = []

    # A few structured starts
    candidates.append(_double_lattice_rows(n))
    candidates.append(_hex_like(n))
    candidates.append(_spiral_mix(n))
    candidates.append(_motif_pack(n))

    # Slightly varied parameterizations
    for shift in (0.34, 0.46, 0.58):
        c, a = _double_lattice_rows(n, row_shift=shift)
        candidates.append((c, a))
    for _ in range(3):
        # jittered spiral for escape from symmetric local minima
        c, a = _spiral_mix(n)
        c = [(x + random.uniform(-0.04, 0.04), y + random.uniform(-0.04, 0.04)) for x, y in c]
        a = [ang + random.choice([0.0, math.pi]) for ang in a]
        candidates.append((c, a))

    best = None
    best_s = float("inf")

    for centers, angles in candidates:
        # Center the candidate
        mx = sum(x for x, _ in centers) / n
        my = sum(y for _, y in centers) / n
        centers = [(x - mx, y - my) for x, y in centers]

        # Resolve any initial overlaps by dilation
        centers = repair_dilate(centers, angles)

        # Local improvement
        centers, angles = _refine(centers, angles, rounds=120)

        # If refinement introduced issues, fix by tiny dilation
        if has_overlap(centers, angles):
            centers = repair_dilate(centers, angles)
            if has_overlap(centers, angles):
                continue

        s = enclosing_side(centers, angles)
        if s < best_s:
            best_s = s
            best = (centers, angles)

    if best is None:
        # Robust fallback: conservative square-ish grid with alternating flips.
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        pitch_x = WIDTH * 1.02
        pitch_y = HEIGHT * 1.02
        centers, angles = [], []
        for k in range(n):
            i, j = k % cols, k // cols
            x = (i - (cols - 1) / 2.0) * pitch_x
            y = (j - (rows - 1) / 2.0) * pitch_y
            centers.append((x, y))
            angles.append(0.0 if (i + j) % 2 == 0 else math.pi)
        centers = repair_dilate(centers, angles)
        best = (centers, angles)
        best_s = enclosing_side(centers, angles)

    centers, angles = best

    # Final safety pass: ensure all are inside and non-overlapping.
    if has_overlap(centers, angles):
        centers = repair_dilate(centers, angles)
    s = enclosing_side(centers, angles)

    # Slight uniform shrink of centers if possible? Not safe without rechecking.
    # Return the valid packing.
    return centers, angles, s
