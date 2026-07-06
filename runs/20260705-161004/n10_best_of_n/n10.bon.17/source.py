"""Pack n unit regular pentagons into the smallest possible origin-centered
axis-aligned square.

Contract: pack(n) -> (centers, angles, s)

Container convention:
A point p is inside the square iff max(|px|, |py|) <= s/2.

This version uses a compact hand-tuned construction family inspired by
double-lattice / staggered-row packings, then performs a local search with
feasibility checks and adaptive shrinking.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

OVERLAP_EPS = 1e-9


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


def _poly_axes(poly):
    axes = []
    for i in range(5):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
        ux, uy = -(y2 - y1), x2 - x1
        norm = math.hypot(ux, uy)
        if norm > 0:
            axes.append((ux / norm, uy / norm))
    return axes


def pentagons_overlap(pa, pb):
    for poly in (pa, pb):
        for ux, uy in _poly_axes(poly):
            a = [x * ux + y * uy for x, y in pa]
            b = [x * ux + y * uy for x, y in pb]
            if min(max(a), max(b)) - max(min(a), min(b)) <= OVERLAP_EPS:
                return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if pentagons_overlap(polys[i], polys[j]):
                return True
    return False


def inside_square(centers, angles, s):
    half = s / 2.0 + 1e-10
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            if abs(vx) > half or abs(vy) > half:
                return False
    return True


def feasible(centers, angles, s):
    return (not has_overlap(centers, angles)) and inside_square(centers, angles, s)


def shrink_to_fit(centers, angles, s):
    """Uniformly scale centers toward the origin until just feasible."""
    if not centers:
        return centers, s
    if feasible(centers, angles, s):
        hi = 1.0
    else:
        hi = 1.0
        while hi > 1e-8 and not feasible([(x * hi, y * hi) for x, y in centers], angles, s):
            hi *= 0.9
        if hi <= 1e-8:
            return centers, s

    lo = hi
    if feasible([(x * lo, y * lo) for x, y in centers], angles, s):
        pass
    else:
        return centers, s

    # Try to reduce side by shrinking centers and then recomputing the enclosing side.
    # Binary search a uniform scale factor making the configuration feasible in a tighter box.
    best_centers = centers
    best_s = enclosing_side(centers, angles)

    # Search over scale factors and evaluate induced side.
    lo, hi = 0.0, 1.0
    for _ in range(36):
        mid = (lo + hi) / 2.0
        cand = [(x * mid, y * mid) for x, y in centers]
        if feasible(cand, angles, s):
            hi = mid
            best_centers = cand
            best_s = enclosing_side(cand, angles)
        else:
            lo = mid
    return best_centers, best_s


def candidate_layouts(n):
    """Generate several promising staggered-row layouts."""
    layouts = []

    # Family 1: staggered rows, alternating orientation by row and column.
    for rows in range(1, n + 1):
        cols = int(math.ceil(n / rows))
        if rows * cols < n:
            continue
        layouts.append(("grid", rows, cols))

    # Family 2: near-hexagonal dimensions.
    for cols in range(1, n + 1):
        rows = int(math.ceil(n / cols))
        layouts.append(("hex", rows, cols))

    return layouts


def build_layout(n, kind, rows, cols):
    """Construct a compact staggered arrangement."""
    centers = []
    angles = []

    # Base pitches tuned for regular pentagons; staggering helps interlock.
    px = 0.95 * WIDTH
    py = 0.90 * HEIGHT

    # If rows/cols are extreme, slightly compress one axis.
    aspect = cols / max(rows, 1)
    if aspect > 1.2:
        px *= 0.94
        py *= 1.02
    elif aspect < 0.8:
        px *= 1.02
        py *= 0.94

    for k in range(n):
        r = k // cols
        c = k % cols

        if kind == "hex":
            # Axial stagger: every other row is shifted.
            x = (c - (cols - 1) / 2.0 + 0.5 * (r % 2)) * px
            y = (r - (rows - 1) / 2.0) * py
            ang = math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0
            # Boundary tilt on outer rows to reduce square span.
            if r == 0 or r == rows - 1:
                ang += 0.12 if c % 2 == 0 else -0.12
            if c == 0 or c == cols - 1:
                ang += 0.08 if r % 2 == 0 else -0.08
        else:
            # Slightly denser alternating row offset.
            x = (c - (cols - 1) / 2.0 + 0.5 * (r % 2)) * px
            y = (r - (rows - 1) / 2.0) * py
            ang = math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0
            if r == 0 or r == rows - 1:
                ang += 0.10 if c % 2 == 0 else -0.10

        centers.append((x, y))
        angles.append(ang)

    return centers, angles


def random_perturb(centers, angles, scale=0.08):
    c2 = []
    a2 = []
    for (x, y), a in zip(centers, angles):
        c2.append((x + random.uniform(-scale, scale), y + random.uniform(-scale, scale)))
        a2.append(a + random.uniform(-0.18, 0.18))
    return c2, a2


def local_search(centers, angles, s, steps=2500):
    """Simple stochastic improvement: perturb, accept if feasible and improves s."""
    best_c = [tuple(p) for p in centers]
    best_a = list(angles)
    best_s = enclosing_side(best_c, best_a)

    # Use a scale that gradually cools.
    for t in range(steps):
        temp = max(0.02, 0.22 * (1.0 - t / steps))
        cand_c, cand_a = random_perturb(best_c, best_a, scale=temp)
        cand_s = enclosing_side(cand_c, cand_a)

        if cand_s >= best_s - 1e-10:
            continue
        if feasible(cand_c, cand_a, cand_s):
            best_c, best_a, best_s = cand_c, cand_a, cand_s

    return best_c, best_a, best_s


def normalize_to_origin(centers):
    if not centers:
        return centers
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    return [(x - mx, y - my) for x, y in centers]


def pack(n):
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    random.seed(123456789 + n)

    best = None

    # Evaluate multiple structured initial layouts.
    layouts = candidate_layouts(n)
    for idx, (kind, rows, cols) in enumerate(layouts[: min(len(layouts), 20)]):
        centers, angles = build_layout(n, kind, rows, cols)
        centers = normalize_to_origin(centers)

        # Initial inflation margin to ensure clean start.
        s0 = enclosing_side(centers, angles) * 1.02
        if not feasible(centers, angles, s0):
            # Push outward slightly if needed.
            scale = 1.02
            while scale < 2.5 and not feasible([(x * scale, y * scale) for x, y in centers], angles, s0 * scale):
                scale += 0.03
            centers = [(x * scale, y * scale) for x, y in centers]
            s0 = enclosing_side(centers, angles)

        # Local search refinement.
        c2, a2, s2 = local_search(centers, angles, s0, steps=1200 if n <= 20 else 1800)

        # Try a small deterministic shrink by uniform scaling of centers.
        # Since the square is centered at origin, shrinking centers monotonically helps.
        cur_c = c2
        cur_a = a2
        cur_s = s2
        for _ in range(18):
            scale = 0.995
            trial = [(x * scale, y * scale) for x, y in cur_c]
            trial_s = enclosing_side(trial, cur_a)
            if feasible(trial, cur_a, trial_s):
                cur_c, cur_s = trial, trial_s
            else:
                break

        cand = (cur_c, cur_a, cur_s)
        if best is None or cand[2] < best[2]:
            best = cand

    centers, angles, s = best

    # Final cleanup: small coordinate centering adjustment and validation.
    centers = normalize_to_origin(centers)
    s = enclosing_side(centers, angles)

    # If numerical noise caused an issue, back off slightly.
    if not feasible(centers, angles, s):
        scale = 1.01
        centers = [(x * scale, y * scale) for x, y in centers]
        s = enclosing_side(centers, angles)

    return centers, angles, s
