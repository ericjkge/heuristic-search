"""Improved packer for unit regular pentagons in the smallest origin-centered
axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles:  list[float] (radians)
- s: side length of the enclosing square

The construction below uses a hand-built family of mixed-orientation templates
with local numerical refinement. It keeps all pentagons inside the square and
non-overlapping, and aims for substantially smaller s than the seed program.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

TAU = 2.0 * math.pi
PHI = (1.0 + 5.0 ** 0.5) / 2.0

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def pentagon_vertices(cx, cy, angle):
    return [
        (cx + R * math.cos(angle + k * TAU / 5.0),
         cy + R * math.sin(angle + k * TAU / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), a in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, a)
    ) if centers else 0.0


OVERLAP_EPS = 1e-8


def polygon_axes(poly):
    axes = []
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        ux, uy = -(y2 - y1), (x2 - x1)
        norm = math.hypot(ux, uy)
        if norm > 0:
            axes.append((ux / norm, uy / norm))
    return axes


def project(poly, axis):
    ux, uy = axis
    vals = [x * ux + y * uy for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb):
    # SAT
    for axis in polygon_axes(pa) + polygon_axes(pb):
        amin, amax = project(pa, axis)
        bmin, bmax = project(pb, axis)
        if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS:
            return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def inside_square(centers, angles, s):
    h = s / 2.0 + 1e-9
    for (cx, cy), a in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, a):
            if abs(vx) > h or abs(vy) > h:
                return False
    return True


def valid_pack(centers, angles, s=None):
    if s is None:
        s = enclosing_side(centers, angles)
    return inside_square(centers, angles, s) and not has_overlap(centers, angles)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def add_vec(p, q):
    return (p[0] + q[0], p[1] + q[1])


def mul_vec(a, p):
    return (a * p[0], a * p[1])


def rot(angle, radius):
    return (radius * math.cos(angle), radius * math.sin(angle))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def normalize_angle(a):
    while a <= -math.pi:
        a += TAU
    while a > math.pi:
        a -= TAU
    return a


# ---------------------------------------------------------------------------
# Local optimization by random search / coordinate descent
# ---------------------------------------------------------------------------

def score(centers, angles):
    # Smaller is better; overlap/containment violations are heavily penalized.
    s = enclosing_side(centers, angles)
    penalty = 0.0
    if has_overlap(centers, angles):
        penalty += 10.0
    if not inside_square(centers, angles, s):
        penalty += 10.0
    return s + penalty


def refine(centers, angles, rounds=1200, seed=0):
    rng = random.Random(seed)
    n = len(centers)
    if n == 0:
        return centers, angles

    best_centers = list(centers)
    best_angles = list(angles)
    best_s = enclosing_side(best_centers, best_angles)
    if not valid_pack(best_centers, best_angles, best_s):
        return centers, angles

    # Adaptive step sizes
    step_xy = max(0.03, best_s * 0.01)
    step_a = 0.12
    temp = 1.0

    for t in range(rounds):
        idx = rng.randrange(n)
        cx, cy = best_centers[idx]
        a = best_angles[idx]

        # Occasional broader perturbation
        if t % 150 == 0:
            dx = rng.uniform(-step_xy * 4.0, step_xy * 4.0)
            dy = rng.uniform(-step_xy * 4.0, step_xy * 4.0)
            da = rng.uniform(-step_a * 4.0, step_a * 4.0)
        else:
            dx = rng.gauss(0.0, step_xy)
            dy = rng.gauss(0.0, step_xy)
            da = rng.gauss(0.0, step_a)

        cand_centers = list(best_centers)
        cand_angles = list(best_angles)
        cand_centers[idx] = (cx + dx, cy + dy)
        cand_angles[idx] = normalize_angle(a + da)

        # Gentle recentering to keep the packing around origin
        mx = sum(x for x, _ in cand_centers) / n
        my = sum(y for _, y in cand_centers) / n
        cand_centers = [(x - 0.15 * mx, y - 0.15 * my) for x, y in cand_centers]

        # scale down if clearly outside or overlapping
        cand_s = enclosing_side(cand_centers, cand_angles)
        if not inside_square(cand_centers, cand_angles, cand_s):
            continue

        # Accept only if not worse and valid
        if not has_overlap(cand_centers, cand_angles):
            cand_s = enclosing_side(cand_centers, cand_angles)
            if cand_s < best_s - 1e-10:
                best_centers, best_angles, best_s = cand_centers, cand_angles, cand_s
                step_xy = max(0.01, step_xy * 0.999)
                step_a = max(0.03, step_a * 0.999)
                temp = 1.0
            else:
                # Sometimes accept sideways moves to escape local minima
                if rng.random() < math.exp(-(cand_s - best_s) / max(1e-9, temp * 0.002)):
                    best_centers, best_angles, best_s = cand_centers, cand_angles, cand_s
                temp *= 0.9995
        else:
            step_xy *= 0.9998
            step_a *= 0.9998

    return best_centers, best_angles


# ---------------------------------------------------------------------------
# Template families
# ---------------------------------------------------------------------------

def pack_row_of_k(k, xgap=0.0, ygap=0.0, tilt=0.0):
    # Mixed orientations along a row; alternating flip helps interlock.
    centers = []
    angles = []
    pitch = WIDTH * 0.88 + xgap
    for i in range(k):
        x = (i - (k - 1) / 2.0) * pitch
        y = 0.0
        centers.append((x, y))
        angles.append(tilt if i % 2 == 0 else normalize_angle(tilt + math.pi))
    # Small stagger to exploit pentagon asymmetry
    for i in range(k):
        x, y = centers[i]
        centers[i] = (x, y + ((-1) ** i) * ygap)
    return centers, angles


def pack_two_rows(k1, k2, dx=0.0, dy=0.0, ang1=0.0, ang2=math.pi):
    # A compact two-row motif with opposing orientations.
    c1, a1 = pack_row_of_k(k1, xgap=dx, ygap=0.0, tilt=ang1)
    c2, a2 = pack_row_of_k(k2, xgap=dx, ygap=0.0, tilt=ang2)
    # vertical separation tuned for interlocking
    ysep = HEIGHT * 0.62 + dy
    c1 = [(x, y - ysep / 2.0) for x, y in c1]
    c2 = [(x + 0.18 * WIDTH, y + ysep / 2.0) for x, y in c2]
    return c1 + c2, a1 + a2


def pack_hexish(n):
    """
    Build from a small palette of motifs:
    - 2-row opposite-orientation blocks
    - single tilted boundary pentagons
    - short interlocking rows
    """
    if n == 0:
        return [], []
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]

    centers = []
    angles = []

    # Partition n into compact blocks.
    blocks = []
    remaining = n
    while remaining > 0:
        if remaining >= 6:
            blocks.append(6)
            remaining -= 6
        elif remaining == 5:
            blocks.append(5)
            remaining -= 5
        elif remaining == 4:
            blocks.append(4)
            remaining -= 4
        elif remaining == 3:
            blocks.append(3)
            remaining -= 3
        elif remaining == 2:
            blocks.append(2)
            remaining -= 2
        else:
            blocks.append(1)
            remaining -= 1

    # Arrange blocks in a near-square macro-grid.
    m = len(blocks)
    cols = int(math.ceil(math.sqrt(m)))
    rows = int(math.ceil(m / cols))
    macro_x = WIDTH * 1.05
    macro_y = HEIGHT * 0.88

    coords = []
    for k in range(m):
        i, j = k % cols, k // cols
        coords.append(((i - (cols - 1) / 2.0) * macro_x,
                       (j - (rows - 1) / 2.0) * macro_y))

    for idx, b in enumerate(blocks):
        ox, oy = coords[idx]
        if b == 6:
            c, a = pack_two_rows(3, 3, dx=-0.03, dy=-0.02, ang1=math.pi / 2.0, ang2=-math.pi / 2.0)
        elif b == 5:
            c, a = pack_two_rows(3, 2, dx=-0.02, dy=0.0, ang1=math.pi / 2.0, ang2=-math.pi / 2.0)
        elif b == 4:
            c, a = pack_two_rows(2, 2, dx=-0.01, dy=0.0, ang1=math.pi / 2.0, ang2=-math.pi / 2.0)
        elif b == 3:
            c = [(-WIDTH * 0.25, 0.0), (0.0, 0.0), (WIDTH * 0.25, 0.0)]
            a = [math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0]
        elif b == 2:
            c = [(-WIDTH * 0.18, 0.0), (WIDTH * 0.18, 0.0)]
            a = [math.pi / 2.0, -math.pi / 2.0]
        else:
            c = [(0.0, 0.0)]
            a = [math.pi / 2.0]

        # slight random-like but deterministic per block offset/tilt
        jitter = (idx * 0.07) % 0.12 - 0.06
        for p in range(len(c)):
            x, y = c[p]
            c[p] = (x + ox + ((p % 2) * 0.03 - 0.015), y + oy + jitter)
            a[p] = normalize_angle(a[p] + (0.04 if (idx + p) % 3 == 0 else -0.03))
        centers.extend(c)
        angles.extend(a)

    return centers, angles


# ---------------------------------------------------------------------------
# Optional exact-ish compression by global binary search on scaling
# ---------------------------------------------------------------------------

def compress_to_valid(centers, angles):
    if not centers:
        return centers, 0.0
    s = enclosing_side(centers, angles)
    if valid_pack(centers, angles, s):
        return centers, s

    # If something went wrong, dilate outward until valid then binary search inward.
    lo, hi = 1.0, 1.2
    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    for _ in range(50):
        c = scaled(hi)
        if valid_pack(c, angles, enclosing_side(c, angles)):
            break
        lo, hi = hi, hi * 1.25
    else:
        return centers, enclosing_side(centers, angles)

    for _ in range(50):
        mid = (lo + hi) / 2.0
        c = scaled(mid)
        if valid_pack(c, angles, enclosing_side(c, angles)):
            hi = mid
        else:
            lo = mid
    c = scaled(hi)
    return c, enclosing_side(c, angles)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def pack(n):
    """
    Mixed-orientation heuristic packer.

    Strategy:
    1) Start from a compact macro-tiling of small motifs with alternating
       orientations.
    2) Apply local randomized refinement while preserving validity.
    3) Return the best packing found.
    """
    if n <= 0:
        return [], [], 0.0

    centers, angles = pack_hexish(n)

    # Center the configuration around the origin.
    if centers:
        mx = sum(x for x, _ in centers) / len(centers)
        my = sum(y for _, y in centers) / len(centers)
        centers = [(x - mx, y - my) for x, y in centers]

    # A few refinement passes with different seeds.
    best_centers = centers
    best_angles = angles
    best_s = enclosing_side(centers, angles)
    if not valid_pack(best_centers, best_angles, best_s):
        # fallback to conservative grid-like layout if template failed
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        pitch_x = WIDTH * 0.93
        pitch_y = HEIGHT * 0.83
        centers = []
        angles = []
        for k in range(n):
            i, j = k % cols, k // cols
            x = (i - (cols - 1) / 2.0) * pitch_x
            y = (j - (rows - 1) / 2.0) * pitch_y
            centers.append((x, y))
            angles.append(math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0)
        best_centers, best_angles = centers, angles
        best_s = enclosing_side(centers, angles)

    for seed in range(8):
        c, a = refine(best_centers, best_angles, rounds=1200, seed=seed * 9173 + n * 13)
        s = enclosing_side(c, a)
        if valid_pack(c, a, s) and s < best_s:
            best_centers, best_angles, best_s = c, a, s

    # Final tiny compression safeguard: if valid, keep as is.
    best_s = enclosing_side(best_centers, best_angles)
    return best_centers, best_angles, best_s
