"""Pack n unit regular pentagons into the smallest axis-aligned origin-centered square.

Contract: pack(n) -> (centers, angles, s), where the square is
origin-centered and axis-aligned, and a point p is inside iff
max(|px|, |py|) <= s/2.

This version uses a hand-tuned mixed-orientation family for n=10 and a small
local search / scaling repair. For other n, it falls back to a simple heuristic.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM


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
    ) if centers else 0.0


OVERLAP_EPS = 1e-9


def poly_axes(poly):
    for i in range(5):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
        ux, uy = -(y2 - y1), (x2 - x1)
        yield ux, uy


def project(poly, ux, uy):
    dots = [x * ux + y * uy for x, y in poly]
    return min(dots), max(dots)


def pentagons_overlap(pa, pb):
    """Separating-axis test for two unit pentagons given as vertex lists."""
    for poly in (pa, pb):
        for ux, uy in poly_axes(poly):
            amin, amax = project(pa, ux, uy)
            bmin, bmax = project(pb, ux, uy)
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


def repair(centers, angles):
    """Dilate all centers about the origin until no overlaps remain."""
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.05
    for _ in range(60):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.15
    else:
        return centers

    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def local_optimize(centers, angles, iters=4000, seed=0):
    """Small randomized hill-climber on positions and rotations."""
    rng = random.Random(seed)

    def score(c, a):
        return enclosing_side(c, a)

    best_c = [tuple(p) for p in centers]
    best_a = list(angles)
    best_s = score(best_c, best_a)

    # Step sizes tuned for n=10. The search is very conservative: only accept
    # valid improvements.
    pos_steps = [0.06, 0.03, 0.015, 0.008, 0.004]
    ang_steps = [0.20, 0.10, 0.05, 0.025, 0.01]

    for t in range(iters):
        k = min(t * len(pos_steps) // max(1, iters), len(pos_steps) - 1)
        ps = pos_steps[k]
        as_ = ang_steps[k]

        cand_c = [list(p) for p in best_c]
        cand_a = list(best_a)

        move_type = rng.random()
        if move_type < 0.45:
            i = rng.randrange(len(cand_c))
            cand_c[i][0] += rng.uniform(-ps, ps)
            cand_c[i][1] += rng.uniform(-ps, ps)
        elif move_type < 0.80:
            i = rng.randrange(len(cand_a))
            cand_a[i] += rng.uniform(-as_, as_)
        else:
            # paired opposite-orientation nudge
            i = rng.randrange(len(cand_c))
            dx = rng.uniform(-ps, ps)
            dy = rng.uniform(-ps, ps)
            cand_c[i][0] += dx
            cand_c[i][1] += dy
            j = (i + len(cand_c) // 2) % len(cand_c)
            cand_c[j][0] -= dx
            cand_c[j][1] -= dy

        # Recentering to keep the square balanced around origin
        mx = sum(x for x, y in cand_c) / len(cand_c)
        my = sum(y for x, y in cand_c) / len(cand_c)
        cand_c = [(x - mx, y - my) for x, y in cand_c]

        # Try tiny angle normalization around canonical orientations
        for i, a in enumerate(cand_a):
            # Snap near the two commonly useful orientations if close
            for target in (math.pi / 2.0, -math.pi / 2.0, math.pi / 10.0, -math.pi / 10.0):
                if abs((a - target + math.pi) % (2.0 * math.pi) - math.pi) < 0.03:
                    if rng.random() < 0.2:
                        cand_a[i] = target + rng.uniform(-0.01, 0.01)

        cand_c = [tuple(p) for p in cand_c]
        cand_c = repair(cand_c, cand_a)
        if has_overlap(cand_c, cand_a):
            continue

        s = score(cand_c, cand_a)
        if s + 1e-12 < best_s:
            best_c, best_a, best_s = cand_c, cand_a, s

    return best_c, best_a, best_s


def pack10():
    """
    Hand-tuned 10-pentagon motif:
    two interlocking rows of 5, with mixed orientations.
    This is intentionally compact and then lightly optimized.
    """
    # Base layout: a slightly staggered double row. Orientations alternate
    # between up/down to mimic the known good double-lattice structure.
    a0 = math.pi / 2.0
    a1 = -math.pi / 2.0

    # Initial centers chosen to be fairly tight but still safe after repair.
    dx = 0.92
    dy = 0.84

    centers = []
    angles = []

    # Bottom row
    for i in range(5):
        x = (i - 2) * dx
        y = -dy * 0.95
        centers.append((x, y))
        angles.append(a0 if i % 2 == 0 else a1)

    # Top row offset to interlock with bottom row
    for i in range(5):
        x = (i - 2) * dx + 0.46
        y = dy * 0.95
        centers.append((x, y))
        angles.append(a1 if i % 2 == 0 else a0)

    # Normalize around origin
    mx = sum(x for x, y in centers) / 10.0
    my = sum(y for x, y in centers) / 10.0
    centers = [(x - mx, y - my) for x, y in centers]

    centers = repair(centers, angles)
    centers, angles, s = local_optimize(centers, angles, iters=7000, seed=10)

    # Final safety repair and a tiny scale-up if necessary
    centers = repair(centers, angles)
    if has_overlap(centers, angles):
        # Last-resort conservative fallback.
        centers = repair(centers, angles)

    s = enclosing_side(centers, angles)
    return centers, angles, s


def generic_pack(n):
    """Fallback heuristic for other n: mixed-orientation staggered rows."""
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0], enclosing_side([(0.0, 0.0)], [math.pi / 2.0])

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pitch_x = 0.96
    pitch_y = 0.86

    centers, angles = [], []
    for k in range(n):
        i, j = k % cols, k // cols
        x = (i - (cols - 1) / 2.0) * pitch_x + (0.34 if j % 2 else 0.0)
        y = (j - (rows - 1) / 2.0) * pitch_y
        centers.append((x, y))
        angles.append(math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0)

    mx = sum(x for x, y in centers) / n
    my = sum(y for x, y in centers) / n
    centers = [(x - mx, y - my) for x, y in centers]

    centers = repair(centers, angles)
    centers, angles, s = local_optimize(centers, angles, iters=3000, seed=n)
    centers = repair(centers, angles)
    return centers, angles, enclosing_side(centers, angles)


def pack(n):
    """Return (centers, angles, s) for n unit regular pentagons."""
    if n == 10:
        return pack10()
    return generic_pack(n)
