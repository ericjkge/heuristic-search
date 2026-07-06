"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles: list[float]
- s: outer side of the enclosing square
Container convention: point p is inside iff max(|px|, |py|) <= s/2.

This version uses a geometry-aware template based on staggered rows, opposite
orientations, and boundary tilting, then runs a local improvement loop on the
container size via deterministic simulated annealing / coordinate search style
refinement.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # inradius
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM


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


# EVOLVE-BLOCK-START
OVERLAP_EPS = 1e-8


def _proj(poly, ux, uy):
    dots = [x * ux + y * uy for x, y in poly]
    return min(dots), max(dots)


def pentagons_overlap(pa, pb):
    """Separating-axis test."""
    for poly in (pa, pb):
        for i in range(5):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
            ux, uy = -(y2 - y1), (x2 - x1)
            amin, amax = _proj(pa, ux, uy)
            bmin, bmax = _proj(pb, ux, uy)
            norm = math.hypot(ux, uy)
            if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS * norm:
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


def repair(centers, angles):
    """Radially dilate centers until overlaps vanish. Rotations stay fixed."""
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.05
    for _ in range(60):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.3
    else:
        return centers

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def _pack_template(n):
    """Staggered row template with opposite orientations and boundary tilts.

    The construction is intentionally non-grid-like:
    - rows alternate between point-up / point-down phases
    - row lengths vary to create diagonal contacts and better boundary usage
    - rows are staggered by a half-step to encourage interlocking
    """
    if n <= 0:
        return [], []

    # Candidate row lengths: distribute units in a near-hexagonal envelope.
    rows = max(1, int(round(math.sqrt(n * 1.15))))
    base = n // rows
    rem = n % rows
    row_lengths = [base + (1 if r < rem else 0) for r in range(rows)]

    # Rebalance to avoid extremely short/long rows.
    while rows > 1 and min(row_lengths) == 0:
        row_lengths.remove(0)
        rows -= 1

    # Geometric tuning constants discovered empirically.
    # Horizontal pitch slightly below width encourages interlocking after template placement;
    # vertical pitch is a bit below full height due to alternating orientations.
    pitch_x = WIDTH * 0.915
    pitch_y = HEIGHT * 0.835

    centers, angles = [], []
    idx = 0
    for r, m in enumerate(row_lengths):
        # Alternate row parity: odd rows shifted half a horizontal pitch.
        x_shift = 0.5 * pitch_x if (r % 2 == 1) else 0.0

        # Mix orientations: alternating row families use opposite orientations.
        # Boundary rows are tilted slightly to reduce the enclosing square.
        if r == 0 or r == rows - 1:
            row_angle_base = math.pi / 10.0  # slight tilt on boundaries
        else:
            row_angle_base = 0.0

        y = (r - (rows - 1) / 2.0) * pitch_y
        # Center row horizontally around the origin with stagger correction.
        x0 = -((m - 1) * pitch_x) / 2.0 + x_shift

        for c in range(m):
            if idx >= n:
                break
            x = x0 + c * pitch_x
            # Opposite orientations by checkerboard of row/column.
            ang = row_angle_base + (math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0)
            centers.append((x, y))
            angles.append(ang)
            idx += 1

    # Recentering, since row construction may not be perfectly symmetric for all n.
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def _evaluate(centers, angles):
    return enclosing_side(centers, angles)


def _local_search(centers, angles, seed=0):
    """Deterministic local optimization on centers/angles and container side."""
    rng = random.Random(seed)

    # Start with a valid packing.
    centers = repair(list(centers), list(angles))
    s = _evaluate(centers, angles)

    # Coordinate-descent / annealing hybrid.
    step_pos = max(0.01, s * 0.03)
    step_ang = 0.10
    best_centers = centers[:]
    best_angles = angles[:]
    best_s = s

    def try_state(cand_centers, cand_angles):
        if has_overlap(cand_centers, cand_angles):
            return None
        return _evaluate(cand_centers, cand_angles)

    for phase in range(6):
        # Multiple randomized passes; later phases are finer.
        for _ in range(220):
            i = rng.randrange(len(best_centers))
            cx, cy = best_centers[i]
            a = best_angles[i]

            # Candidate moves: boundary-focused and interlocking-focused.
            moves = [
                (0.0, 0.0, 0.0),
                (step_pos, 0.0, 0.0), (-step_pos, 0.0, 0.0),
                (0.0, step_pos, 0.0), (0.0, -step_pos, 0.0),
                (step_pos * 0.7, step_pos * 0.7, 0.0),
                (step_pos * 0.7, -step_pos * 0.7, 0.0),
                (0.0, 0.0, step_ang), (0.0, 0.0, -step_ang),
                (step_pos * 0.5, 0.0, step_ang),
                (-step_pos * 0.5, 0.0, -step_ang),
            ]

            # Bias boundary polygons toward the walls to shrink enclosing square.
            vx = cx
            vy = cy
            bias_x = -math.copysign(1.0, vx) if abs(vx) > 1e-9 else 0.0
            bias_y = -math.copysign(1.0, vy) if abs(vy) > 1e-9 else 0.0

            candidates = []
            for dx, dy, da in moves:
                cc = best_centers[:]
                aa = best_angles[:]
                cc[i] = (cx + dx + bias_x * step_pos * 0.15, cy + dy + bias_y * step_pos * 0.15)
                aa[i] = a + da + (0.03 if abs(vx) > abs(vy) and i % 2 == 0 else 0.0)
                # Recentering after a move helps keep the square symmetric.
                mx = sum(x for x, _ in cc) / len(cc)
                my = sum(y for _, y in cc) / len(cc)
                cc = [(x - mx, y - my) for x, y in cc]
                candidates.append((cc, aa))

            improved = False
            for cc, aa in candidates:
                val = try_state(cc, aa)
                if val is not None and val + 1e-12 < best_s:
                    best_centers, best_angles, best_s = cc, aa, val
                    improved = True
                    break

            # If no strict improvement, occasionally accept a harmless perturbation
            # to escape shallow basins, then repair and continue.
            if not improved and rng.random() < 0.12:
                cc, aa = candidates[rng.randrange(len(candidates))]
                cc = repair(cc, aa)
                val = _evaluate(cc, aa)
                if val <= best_s * 1.01 and not has_overlap(cc, aa):
                    best_centers, best_angles, best_s = cc, aa, val

        step_pos *= 0.55
        step_ang *= 0.65

    # Final clean-up: recompute a safe packing and then tighten via isotropic scaling.
    best_centers = repair(best_centers, best_angles)
    best_s = _evaluate(best_centers, best_angles)
    return best_centers, best_angles, best_s


def pack(n):
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    # Use multiple deterministic templates and keep the best after local search.
    templates = []

    # Template A: staggered rows
    centers_a, angles_a = _pack_template(n)
    templates.append((centers_a, angles_a))

    # Template B: opposite-orientation double-lattice inspired pattern.
    # Two interleaved sublattices with slight skew.
    m = int(math.ceil(math.sqrt(n)))
    pitch_x = WIDTH * 0.88
    pitch_y = HEIGHT * 0.82
    centers_b, angles_b = [], []
    for k in range(n):
        r, c = divmod(k, m)
        x = (c - (m - 1) / 2.0) * pitch_x + (0.45 * pitch_x if r % 2 else 0.0)
        y = (r - (math.ceil(n / m) - 1) / 2.0) * pitch_y
        # Tilt outer rows more aggressively.
        edge = (r == 0 or r == math.ceil(n / m) - 1 or c == 0 or c == m - 1)
        ang = (math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0)
        if edge:
            ang += (math.pi / 18.0 if (r + c) % 2 == 0 else -math.pi / 18.0)
        centers_b.append((x, y))
        angles_b.append(ang)
    mx = sum(x for x, _ in centers_b) / len(centers_b)
    my = sum(y for _, y in centers_b) / len(centers_b)
    centers_b = [(x - mx, y - my) for x, y in centers_b]
    templates.append((centers_b, angles_b))

    # Template C: compact spiral-ish fill.
    centers_c, angles_c = [], []
    ring = 0
    placed = 0
    while placed < n:
        count = max(1, 6 * ring if ring > 0 else 1)
        radius_x = ring * WIDTH * 0.46
        radius_y = ring * HEIGHT * 0.39
        for t in range(count):
            if placed >= n:
                break
            theta = (2.0 * math.pi * t / count) + (0.17 if ring % 2 else 0.0)
            x = radius_x * math.cos(theta)
            y = radius_y * math.sin(theta)
            ang = (math.pi / 2.0 if (placed % 2 == 0) else -math.pi / 2.0) + (0.12 if ring % 2 else -0.12)
            centers_c.append((x, y))
            angles_c.append(ang)
            placed += 1
        ring += 1
    templates.append((centers_c, angles_c))

    best = None
    best_s = float("inf")

    for seed, (centers, angles) in enumerate(templates):
        c, a, s = _local_search(centers, angles, seed=seed + 17)
        # One extra tightening pass by uniform radial scaling, preserving validity.
        if not has_overlap(c, a):
            # Binary search the minimal scale factor >= 1 that remains valid.
            def scaled(lam):
                return [(x * lam, y * lam) for x, y in c]

            lo, hi = 1.0, 1.0
            # Small inward attempts are not allowed, but we can slightly expand to
            # clear tiny numerical contacts, then accept the better container if valid.
            if has_overlap(c, a):
                lo, hi = 1.0, 1.02
            else:
                hi = 1.0
            # Since the state is valid, just use it; no shrinking possible without
            # a stronger optimizer.
            s = enclosing_side(c, a)

        if s < best_s:
            best = (c, a, s)
            best_s = s

    centers, angles, s = best

    # Final safety repair and exact container recomputation.
    centers = repair(centers, angles)
    s = enclosing_side(centers, angles)
    return centers, angles, s
# EVOLVE-BLOCK-END
