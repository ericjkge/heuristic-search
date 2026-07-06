"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles: list[float]
- s: outer side of the enclosing square
Container convention: point p is inside iff max(|px|, |py|) <= s/2.

This version uses:
- multiple geometry-inspired initial motifs
- a deterministic local search over center positions and rotations
- pairwise separation-based relaxation
- final tightening by binary search scaling
- exact overlap checks using the Separating Axis Theorem

The implementation is self-contained and deterministic.
"""

import math
import random
from typing import List, Tuple

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))  # circumradius of unit regular pentagon
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

EPS = 1e-12
OVERLAP_EPS = 1e-11


def pentagon_vertices(cx: float, cy: float, angle: float):
    return [
        (
            cx + R * math.cos(angle + k * 2.0 * math.pi / 5.0),
            cy + R * math.sin(angle + k * 2.0 * math.pi / 5.0),
        )
        for k in range(5)
    ]


def _proj(poly, ux, uy):
    dots = [x * ux + y * uy for x, y in poly]
    return min(dots), max(dots)


def has_overlap(centers, angles) -> bool:
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    n = len(polys)
    for i in range(n):
        pi = polys[i]
        for j in range(i + 1, n):
            pj = polys[j]
            # Separating Axis Theorem on edge normals of both polygons
            sep = False
            for poly in (pi, pj):
                for k in range(5):
                    (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % 5]
                    ux, uy = -(y2 - y1), (x2 - x1)
                    amin, amax = _proj(pi, ux, uy)
                    bmin, bmax = _proj(pj, ux, uy)
                    norm = math.hypot(ux, uy)
                    if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS * max(1.0, norm):
                        sep = True
                        break
                if sep:
                    break
            if not sep:
                return True
    return False


def enclosing_side(centers, angles) -> float:
    if not centers:
        return 0.0
    m = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            m = max(m, abs(vx), abs(vy))
    return 2.0 * m


def _center(centers):
    if not centers:
        return centers
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    return [(x - mx, y - my) for x, y in centers]


def _normalize_angles(angles):
    return [((a + math.pi) % (2.0 * math.pi)) - math.pi for a in angles]


def repair(centers, angles):
    """Radially dilate centers until overlaps vanish; angles fixed."""
    if not centers:
        return centers
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.02
    for _ in range(80):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.25
    else:
        return centers

    for _ in range(80):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def _tighten_to_valid(centers, angles):
    """Scale centers toward the origin as much as possible while preserving non-overlap."""
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
    return cc, enclosing_side(cc, angles)


def _side_constraints(centers, angles):
    """Return max excess over square boundary along x/y after centering."""
    if not centers:
        return 0.0
    mx = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            mx = max(mx, abs(vx), abs(vy))
    return mx


def _pairwise_min_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _template_double_lattice(n):
    """Opposite-orientation rows with stagger, inspired by double-lattice motifs."""
    centers, angles = [], []
    if n <= 0:
        return centers, angles

    cols = max(1, int(round(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    # Start with a moderately tight lattice and rely on local optimization.
    dx = WIDTH * 0.78
    dy = HEIGHT * 0.76

    k = 0
    for r in range(rows):
        m = min(cols, n - k)
        y = (r - (rows - 1) / 2.0) * dy
        shift = 0.5 * dx if (r % 2 == 1) else 0.0
        x0 = -((m - 1) * dx) / 2.0 + shift
        edge_row = (r == 0 or r == rows - 1)
        row_tilt = math.pi / 26.0 if edge_row else 0.0
        if r == rows - 1:
            row_tilt *= -1.0
        for c in range(m):
            x = x0 + c * dx
            ang = math.pi / 2.0 if ((r + c) % 2 == 0) else -math.pi / 2.0
            if edge_row and (c == 0 or c == m - 1):
                ang += row_tilt
            centers.append((x, y))
            angles.append(ang)
            k += 1
    return _center(centers), _normalize_angles(angles)


def _template_boundary_ring(n):
    """Put some pentagons on a ring to improve boundary utilization."""
    centers, angles = [], []
    if n <= 0:
        return centers, angles

    ring1 = max(5, n // 2)
    r1 = 0.9 * WIDTH
    r2 = 0.55 * WIDTH
    k = 0
    for idx in range(n):
        if idx < ring1:
            th = 2.0 * math.pi * idx / ring1
            x = r1 * math.cos(th)
            y = r2 * math.sin(th)
            ang = (math.pi / 2.0 if idx % 2 == 0 else -math.pi / 2.0) + (0.11 if idx % 3 == 0 else -0.06)
        else:
            j = idx - ring1
            m = n - ring1
            th = 2.0 * math.pi * j / max(1, m)
            x = 0.48 * WIDTH * math.cos(th + 0.17)
            y = 0.42 * HEIGHT * math.sin(th - 0.09)
            ang = (0.8 if j % 2 == 0 else -0.8)
        centers.append((x, y))
        angles.append(ang)
        k += 1
    return _center(centers), _normalize_angles(angles)


def _template_triangular(n):
    """Skewed triangular grid with alternating orientations."""
    centers, angles = [], []
    if n <= 0:
        return centers, angles

    cols = min(n, max(2, int(math.ceil(math.sqrt(n) * 1.2))))
    rows = int(math.ceil(n / cols))
    dx = WIDTH * 0.74
    dy = HEIGHT * 0.69

    k = 0
    for r in range(rows):
        m = min(cols, n - k)
        y = (r - (rows - 1) / 2.0) * dy
        xshift = (0.5 * dx if r % 2 else 0.0) + (0.1 * dx if r % 3 == 2 else 0.0)
        x0 = -((m - 1) * dx) / 2.0 + xshift
        for c in range(m):
            x = x0 + c * dx
            ang = math.pi / 2.0 if ((r + c) % 2 == 0) else -math.pi / 2.0
            if r in (0, rows - 1):
                ang += (math.pi / 30.0 if r == 0 else -math.pi / 30.0)
            if c in (0, m - 1):
                ang += (0.05 if (r + c) % 2 == 0 else -0.05)
            centers.append((x, y))
            angles.append(ang)
            k += 1
            if k >= n:
                break
    return _center(centers), _normalize_angles(angles)


def _template_spiral(n):
    centers, angles = [], []
    if n <= 0:
        return centers, angles
    placed = 0
    ring = 0
    while placed < n:
        count = 1 if ring == 0 else 6 * ring
        rx = ring * WIDTH * 0.42
        ry = ring * HEIGHT * 0.34
        for t in range(count):
            if placed >= n:
                break
            th = 2.0 * math.pi * t / count + (0.21 if ring % 2 else 0.0)
            x = rx * math.cos(th)
            y = ry * math.sin(th)
            ang = (math.pi / 2.0 if placed % 2 == 0 else -math.pi / 2.0) + (0.08 if ring % 2 else -0.08)
            centers.append((x, y))
            angles.append(ang)
            placed += 1
        ring += 1
    return _center(centers), _normalize_angles(angles)


def _candidate_templates(n):
    if n <= 0:
        return [([], [])]
    if n == 1:
        return [([(0.0, 0.0)], [math.pi / 2.0])]

    cands = []
    cands.append(_template_double_lattice(n))
    cands.append(_template_boundary_ring(n))
    cands.append(_template_triangular(n))
    cands.append(_template_spiral(n))

    # Slightly perturbed variants for exploration.
    for base in [_template_double_lattice, _template_triangular]:
        c0, a0 = base(n)
        for rot in (0.03, -0.03, 0.07, -0.07):
            aa = _normalize_angles([a + rot for a in a0])
            cands.append((c0[:], aa))

    return cands


def _pair_energy(centers, angles):
    """Heuristic energy: overlap penalty + boundary penalty + weak compactness."""
    n = len(centers)
    if n == 0:
        return 0.0

    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    energy = 0.0

    # Boundary penalty
    for poly in polys:
        for x, y in poly:
            m = max(abs(x), abs(y))
            if m > 0:
                energy += 0.02 * m * m

    # Pairwise overlap proxy / spacing penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            d2 = dx * dx + dy * dy
            # soft repulsion at short distances
            if d2 < (WIDTH * 1.1) ** 2:
                energy += 0.6 / (d2 + 1e-9)

            # SAT distance proxy: encourage separation along edge normals
            pi = polys[i]
            pj = polys[j]
            local_pen = 0.0
            for poly in (pi, pj):
                for k in range(5):
                    (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % 5]
                    ux, uy = -(y2 - y1), (x2 - x1)
                    amin, amax = _proj(pi, ux, uy)
                    bmin, bmax = _proj(pj, ux, uy)
                    overlap = min(amax, bmax) - max(amin, bmin)
                    if overlap > 0:
                        local_pen += overlap * overlap
            energy += 12.0 * local_pen

    return energy


def _local_relax(centers, angles, seed=0, rounds=10):
    rng = random.Random(seed)
    centers = _center(list(centers))
    angles = list(_normalize_angles(angles))

    if has_overlap(centers, angles):
        centers = repair(centers, angles)
        centers = _center(centers)

    best_c = centers[:]
    best_a = angles[:]
    best_s = enclosing_side(best_c, best_a)
    best_e = _pair_energy(best_c, best_a)

    step_pos = max(0.03, best_s * 0.045)
    step_ang = 0.18

    def try_state(cc, aa):
        if has_overlap(cc, aa):
            return None
        return enclosing_side(cc, aa), _pair_energy(cc, aa)

    for phase in range(rounds):
        for _ in range(360):
            i = rng.randrange(len(best_c))
            cx, cy = best_c[i]
            a = best_a[i]

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
                (0.45 * step_pos, 0.0, step_ang),
                (-0.45 * step_pos, 0.0, -step_ang),
                (0.0, 0.45 * step_pos, step_ang),
                (0.0, -0.45 * step_pos, -step_ang),
            ]

            improved = False
            for dx, dy, da in moves:
                cc = best_c[:]
                aa = best_a[:]
                bx = -math.copysign(1.0, cx) if abs(cx) > 1e-12 else 0.0
                by = -math.copysign(1.0, cy) if abs(cy) > 1e-12 else 0.0
                cc[i] = (cx + dx + bx * step_pos * 0.06, cy + dy + by * step_pos * 0.06)
                aa[i] = a + da + (0.02 if (abs(cx) > abs(cy) and i % 2 == 0) else 0.0)
                cc = _center(cc)
                res = try_state(cc, aa)
                if res is None:
                    continue
                s, e = res
                if (e + 1e-12 < best_e) or (abs(e - best_e) <= 1e-12 and s + 1e-12 < best_s):
                    best_c, best_a, best_s, best_e = cc, aa, s, e
                    improved = True
                    break

            if not improved and rng.random() < 0.10:
                cc = best_c[:]
                aa = best_a[:]
                j = rng.randrange(len(cc))
                cc[j] = (
                    cc[j][0] + rng.uniform(-step_pos, step_pos) * 0.18,
                    cc[j][1] + rng.uniform(-step_pos, step_pos) * 0.18,
                )
                aa[j] = aa[j] + rng.uniform(-step_ang, step_ang) * 0.18
                cc = _center(cc)
                cc = repair(cc, aa)
                res = try_state(cc, aa)
                if res is not None:
                    s, e = res
                    if (e + 1e-12 < best_e) or (abs(e - best_e) <= 1e-12 and s <= best_s * 1.01):
                        best_c, best_a, best_s, best_e = cc, aa, s, e

        step_pos *= 0.63
        step_ang *= 0.72

    best_c = repair(best_c, best_a)
    best_c = _center(best_c)
    best_s = enclosing_side(best_c, best_a)
    return best_c, best_a, best_s


def _extra_refine(centers, angles):
    """Try deterministic pairwise nudges for a final small improvement."""
    if len(centers) <= 1:
        return centers, angles, enclosing_side(centers, angles)

    best_c = centers[:]
    best_a = angles[:]
    best_s = enclosing_side(best_c, best_a)

    candidates = [0.0, math.pi / 72.0, -math.pi / 72.0, math.pi / 48.0, -math.pi / 48.0]
    for i in range(len(best_c)):
        for j in range(i + 1, len(best_c)):
            for da in candidates:
                cc = best_c[:]
                aa = best_a[:]
                vx = cc[j][0] - cc[i][0]
                vy = cc[j][1] - cc[i][1]
                norm = math.hypot(vx, vy) + 1e-15
                ux, uy = vx / norm, vy / norm
                delta = min(0.016, 0.012 * best_s)
                cc[i] = (cc[i][0] - ux * delta, cc[i][1] - uy * delta)
                cc[j] = (cc[j][0] + ux * delta, cc[j][1] + uy * delta)
                aa[i] += da
                aa[j] -= da
                cc = _center(cc)
                if has_overlap(cc, aa):
                    continue
                s = enclosing_side(cc, aa)
                if s + 1e-12 < best_s:
                    best_c, best_a, best_s = cc, aa, s

    return best_c, best_a, best_s


def pack(n):
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    best = None
    best_s = float("inf")

    templates = _candidate_templates(n)

    # Explore templates with deterministic local optimization.
    for seed, (c0, a0) in enumerate(templates):
        c, a, s = _local_relax(c0, a0, seed=seed + 137, rounds=11)
        c, s2 = _tighten_to_valid(c, a)
        if s2 < s:
            s = s2
        c, a, s = _extra_refine(c, a)
        c, s2 = _tighten_to_valid(c, a)
        if s2 < s:
            s = s2
        if s < best_s:
            best = (c, a, s)
            best_s = s

    centers, angles, s = best

    # Final cleanup.
    centers = _center(centers)
    angles = _normalize_angles(angles)
    if has_overlap(centers, angles):
        centers = repair(centers, angles)
        centers = _center(centers)

    centers, s = _tighten_to_valid(centers, angles)
    return centers, angles, s
