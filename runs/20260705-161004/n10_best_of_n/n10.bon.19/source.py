"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned
square.

Contract: pack(n) -> (centers, angles, s), where the square is
max(|x|, |y|) <= s/2.
"""

import math
import random
from typing import List, Tuple

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # ~ 1.618
HEIGHT = R + APOTHEM                              # ~ 1.539

# Geometry / validation tolerances
EPS = 1e-9
OVERLAP_EPS = 1e-10


def pentagon_vertices(cx, cy, angle):
    """Vertices of a unit regular pentagon centered at (cx, cy)."""
    return [
        (cx + R * math.cos(angle + k * 2.0 * math.pi / 5.0),
         cy + R * math.sin(angle + k * 2.0 * math.pi / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    """Smallest origin-centered square side enclosing all pentagons."""
    mx = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            mx = max(mx, abs(vx), abs(vy))
    return 2.0 * mx


def polygon_axes(poly):
    axes = []
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        ux, uy = -(y2 - y1), (x2 - x1)
        norm = math.hypot(ux, uy)
        if norm > 0:
            axes.append((ux / norm, uy / norm))
    return axes


def project(poly, ax, ay):
    vals = [x * ax + y * ay for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb):
    """Separating-axis test: True iff polygons overlap/touch."""
    for poly in (pa, pb):
        for ax, ay in polygon_axes(poly):
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


def validate_inside(centers, angles, s):
    half = s / 2.0 + 1e-8
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            if abs(vx) > half or abs(vy) > half:
                return False
    return True


def repair_scale_to_fit(centers, angles):
    """Scale about the origin until no overlaps remain and return scaled centers."""
    if not has_overlap(centers, angles):
        return centers

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.05
    for _ in range(60):
        if not has_overlap(scaled(hi), angles):
            break
        lo = hi
        hi *= 1.05
        if hi > 1000:
            return centers

    for _ in range(60):
        mid = (lo + hi) * 0.5
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi)


# ---- Packing templates ----------------------------------------------------

def _template_10():
    """
    A hand-crafted 10-pentagon template with mixed orientations.
    The idea is a 3+4+3 staggered layout with opposing rotations to exploit
    interlocking rather than a rectangular grid.
    """
    # Angles chosen from two opposite orientations and one slight tilt family.
    # These are good starting values; positions are optimized numerically.
    angles = [
        math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2,
        -math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2,
    ]

    # Initial centers in three staggered rows.
    # Row lengths 3,4,3.
    centers = [
        (-1.60,  1.10), (0.00,  1.10), (1.60,  1.10),
        (-2.35,  0.00), (-0.78, 0.00), (0.78,  0.00), (2.35, 0.00),
        (-1.60, -1.10), (0.00, -1.10), (1.60, -1.10),
    ]
    return centers, angles


def _template_n(n):
    """General mixed-orientation staggered rows for any n."""
    if n <= 0:
        return [], []

    # Choose rows to be close to a triangular arrangement.
    rows = int(math.floor(math.sqrt(n * 0.9))) or 1
    while rows * math.ceil(n / rows) < n:
        rows += 1
    cols = int(math.ceil(n / rows))

    # Better aspect for a square container: use near-square rows/cols.
    # We'll lay out on a staggered grid and then optimize.
    pitch_x = 1.62
    pitch_y = 1.45

    centers, angles = [], []
    idx = 0
    for j in range(rows):
        count = cols if (j < rows - 1 or n % cols == 0) else n - cols * (rows - 1)
        # Stagger alternate rows.
        shift = 0.5 * pitch_x if (j % 2 == 1) else 0.0
        x0 = -0.5 * (count - 1) * pitch_x
        y = (j - 0.5 * (rows - 1)) * pitch_y
        for i in range(count):
            if idx >= n:
                break
            x = x0 + i * pitch_x + shift
            centers.append((x, y))
            # Alternate orientations to improve interlocking.
            angles.append((math.pi / 2) if ((i + j) % 2 == 0) else (-math.pi / 2))
            idx += 1

    return centers, angles


# ---- Numerical improvement -----------------------------------------------

def _energy(centers, angles):
    """Objective: minimize required square side, with penalties for overlap."""
    s = enclosing_side(centers, angles)
    # Overlap penalty via pairwise SAT margin approximation.
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    penalty = 0.0
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            pa, pb = polys[i], polys[j]
            # quick bounding box reject
            aminx = min(x for x, y in pa); amaxx = max(x for x, y in pa)
            aminy = min(y for x, y in pa); amaxy = max(y for x, y in pa)
            bminx = min(x for x, y in pb); bmaxx = max(x for x, y in pb)
            bminy = min(y for x, y in pb); bmaxy = max(y for x, y in pb)
            dx = min(amaxx, bmaxx) - max(aminx, bminx)
            dy = min(amaxy, bmaxy) - max(aminy, bminy)
            if dx > 0 and dy > 0:
                penalty += dx * dy
    return s + 50.0 * penalty


def _random_perturb(centers, angles, step_xy, step_ang):
    c2 = []
    a2 = []
    for (x, y), a in zip(centers, angles):
        c2.append((x + random.uniform(-step_xy, step_xy),
                   y + random.uniform(-step_xy, step_xy)))
        a2.append(a + random.uniform(-step_ang, step_ang))
    return c2, a2


def optimize_pack(centers, angles, iters=6000):
    """Simple stochastic hill-climbing with occasional restarts."""
    best_c = [tuple(p) for p in centers]
    best_a = list(angles)

    # Start with a slight global shrink to encourage feasibility, then repair.
    best_c = [(x * 0.95, y * 0.95) for x, y in best_c]
    best_c = repair_scale_to_fit(best_c, best_a)

    best_e = _energy(best_c, best_a)

    step_xy = 0.08
    step_ang = 0.08

    for t in range(iters):
        # Anneal slowly.
        frac = t / max(1, iters - 1)
        sx = step_xy * (0.2 + 0.8 * (1.0 - frac))
        sa = step_ang * (0.2 + 0.8 * (1.0 - frac))

        cand_c, cand_a = _random_perturb(best_c, best_a, sx, sa)

        # Recentering helps the origin-centered square.
        mx = sum(x for x, y in cand_c) / len(cand_c)
        my = sum(y for x, y in cand_c) / len(cand_c)
        cand_c = [(x - mx, y - my) for x, y in cand_c]

        # Small global contraction; if necessary repair by expanding only.
        cand_c = [(x * 0.995, y * 0.995) for x, y in cand_c]
        cand_c = repair_scale_to_fit(cand_c, cand_a)

        e = _energy(cand_c, cand_a)
        if e + 1e-12 < best_e:
            best_e = e
            best_c, best_a = cand_c, cand_a

        # Occasionally escape local minima.
        if (t + 1) % 1200 == 0:
            best_c, best_a = _random_perturb(best_c, best_a, 0.02, 0.15)
            mx = sum(x for x, y in best_c) / len(best_c)
            my = sum(y for x, y in best_c) / len(best_c)
            best_c = [(x - mx, y - my) for x, y in best_c]
            best_c = repair_scale_to_fit(best_c, best_a)
            best_e = _energy(best_c, best_a)

    # Final local coordinate search on centers only.
    for scale in [0.02, 0.01, 0.005, 0.0025]:
        improved = True
        while improved:
            improved = False
            for i in range(len(best_c)):
                x, y = best_c[i]
                a = best_a[i]
                for dx, dy in [(scale, 0), (-scale, 0), (0, scale), (0, -scale),
                               (scale, scale), (scale, -scale), (-scale, scale), (-scale, -scale)]:
                    cand_c = list(best_c)
                    cand_c[i] = (x + dx, y + dy)
                    mx = sum(px for px, py in cand_c) / len(cand_c)
                    my = sum(py for px, py in cand_c) / len(cand_c)
                    cand_c = [(px - mx, py - my) for px, py in cand_c]
                    cand_c = repair_scale_to_fit(cand_c, best_a)
                    e = _energy(cand_c, best_a)
                    if e + 1e-12 < best_e:
                        best_e = e
                        best_c = cand_c
                        improved = True
                        break
                if improved:
                    break

    return best_c, best_a


def pack(n):
    """Return a valid packing of n unit regular pentagons inside the square."""
    if n <= 0:
        return [], [], 0.0

    random.seed(1234567 + n * 10007)

    if n == 10:
        centers, angles = _template_10()
    else:
        centers, angles = _template_n(n)

    # Optimize from a few randomized seeds, keep the best.
    best = None
    best_s = float("inf")

    seeds = 4 if n <= 20 else 2
    for seed in range(seeds):
        if seed > 0:
            # Jitter template differently each run.
            c0, a0 = centers[:], angles[:]
            for i in range(n):
                x, y = c0[i]
                c0[i] = (x + random.uniform(-0.08, 0.08), y + random.uniform(-0.08, 0.08))
                a0[i] = a0[i] + random.uniform(-0.12, 0.12)
            mx = sum(x for x, y in c0) / n
            my = sum(y for x, y in c0) / n
            c0 = [(x - mx, y - my) for x, y in c0]
        else:
            c0, a0 = [tuple(p) for p in centers], list(angles)

        c1, a1 = optimize_pack(c0, a0, iters=4500 if n <= 12 else 3000)
        if has_overlap(c1, a1):
            # As a final fallback, scale out until overlaps vanish.
            c1 = repair_scale_to_fit(c1, a1)
        s1 = enclosing_side(c1, a1)

        if validate_inside(c1, a1, s1) and not has_overlap(c1, a1) and s1 < best_s:
            best_s = s1
            best = (c1, a1, s1)

    if best is None:
        # Guaranteed feasible fallback: loose staggered grid, then repair.
        centers, angles = _template_n(n)
        centers = repair_scale_to_fit(centers, angles)
        s = enclosing_side(centers, angles)
        return centers, angles, s

    centers, angles, s = best

    # Small final centering tweak to reduce enclosure.
    mx = sum(x for x, y in centers) / n
    my = sum(y for x, y in centers) / n
    centers2 = [(x - mx, y - my) for x, y in centers]
    centers2 = repair_scale_to_fit(centers2, angles)
    s2 = enclosing_side(centers2, angles)
    if validate_inside(centers2, angles, s2) and not has_overlap(centers2, angles) and s2 <= s:
        centers, s = centers2, s2

    return centers, angles, s
