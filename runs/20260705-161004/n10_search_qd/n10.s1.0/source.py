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
            if min(amax, bmax) < max(amin, bmin) + OVERLAP_EPS * norm:
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

    lo, hi = 1.0, 1.02
    for _ in range(70):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.18
    else:
        return centers

    for _ in range(70):
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

    try:
        import scipy.optimize as opt
        have_scipy = True
    except Exception:
        have_scipy = False

    centers = repair(list(centers), list(angles))
    angles = list(angles)
    n = len(centers)

    def pack_from_vec(vec):
        cc = [(vec[3 * i], vec[3 * i + 1]) for i in range(n)]
        aa = [vec[3 * i + 2] for i in range(n)]
        mx = sum(x for x, _ in cc) / n
        my = sum(y for _, y in cc) / n
        cc = [(x - mx, y - my) for x, y in cc]
        return cc, aa

    def vec_from_pack(cc, aa):
        vec = []
        for (x, y), a in zip(cc, aa):
            vec.extend([x, y, a])
        return vec

    def objective(vec):
        cc, aa = pack_from_vec(vec)
        s = enclosing_side(cc, aa)
        if has_overlap(cc, aa):
            return s + 50.0
        # Mild boundary reward: encourage wider usage of the square.
        mx = max(max(abs(x), abs(y)) for x, y in cc)
        return s - 0.03 * mx

    x0 = vec_from_pack(centers, angles)

    if have_scipy:
        bounds = []
        s0 = enclosing_side(centers, angles)
        lim = max(2.0, s0)
        for _ in range(n):
            bounds.extend([(-lim, lim), (-lim, lim), (-math.pi, math.pi)])
        try:
            res = opt.minimize(
                objective,
                x0,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 900, "xtol": 1e-5, "ftol": 1e-6, "disp": False},
            )
            if res.success or res.fun < objective(x0):
                centers, angles = pack_from_vec(res.x)
        except Exception:
            pass

    # Stochastic coordinate search around the best state found.
    centers = repair(centers, angles)
    best_centers = centers[:]
    best_angles = angles[:]
    best_s = enclosing_side(best_centers, best_angles)

    step_pos = max(0.006, best_s * 0.018)
    step_ang = 0.09

    def eval_state(cc, aa):
        if has_overlap(cc, aa):
            return None
        return enclosing_side(cc, aa)

    for phase in range(9):
        for _ in range(500):
            i = rng.randrange(n)
            cx, cy = best_centers[i]
            a = best_angles[i]
            rad = math.hypot(cx, cy) + 1e-12
            ux, uy = cx / rad, cy / rad

            moves = [
                (0.0, 0.0, 0.0),
                (ux * step_pos, uy * step_pos, 0.0),
                (-ux * step_pos, -uy * step_pos, 0.0),
                (-uy * step_pos, ux * step_pos, 0.0),
                (uy * step_pos, -ux * step_pos, 0.0),
                (0.0, 0.0, step_ang),
                (0.0, 0.0, -step_ang),
                (ux * 0.6 * step_pos, uy * 0.6 * step_pos, step_ang),
                (-ux * 0.6 * step_pos, -uy * 0.6 * step_pos, -step_ang),
            ]

            cand_list = []
            for dx, dy, da in moves:
                cc = best_centers[:]
                aa = best_angles[:]
                cc[i] = (cx + dx, cy + dy)
                aa[i] = a + da
                mx = sum(x for x, _ in cc) / n
                my = sum(y for _, y in cc) / n
                cc = [(x - mx, y - my) for x, y in cc]
                cand_list.append((cc, aa))

            if phase >= 4:
                # Boundary-focused tilt for outer polygons.
                edge_ids = sorted(range(n), key=lambda j: max(abs(best_centers[j][0]), abs(best_centers[j][1])), reverse=True)
                for j in edge_ids[: max(2, n // 3)]:
                    cc = best_centers[:]
                    aa = best_angles[:]
                    x, y = cc[j]
                    rad = math.hypot(x, y) + 1e-12
                    ux, uy = x / rad, y / rad
                    cc[j] = (x + ux * step_pos * 0.8, y + uy * step_pos * 0.8)
                    aa[j] = aa[j] + (0.07 if (j % 2 == 0) else -0.07)
                    mx = sum(x for x, _ in cc) / n
                    my = sum(y for _, y in cc) / n
                    cc = [(x - mx, y - my) for x, y in cc]
                    cand_list.append((cc, aa))

            improved = False
            for cc, aa in cand_list:
                val = eval_state(cc, aa)
                if val is not None and val + 1e-12 < best_s:
                    best_centers, best_angles, best_s = cc, aa, val
                    improved = True
                    break

            if not improved and rng.random() < 0.20:
                cc, aa = cand_list[rng.randrange(len(cand_list))]
                if not has_overlap(cc, aa):
                    val = enclosing_side(cc, aa)
                    if val < best_s * 1.02:
                        best_centers, best_angles, best_s = cc, aa, val

        step_pos *= 0.72
        step_ang *= 0.72

    best_centers = repair(best_centers, best_angles)
    best_s = enclosing_side(best_centers, best_angles)
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
    m = int(math.ceil(math.sqrt(n)))
    pitch_x = WIDTH * 0.84
    pitch_y = HEIGHT * 0.79
    centers_b, angles_b = [], []
    for k in range(n):
        r, c = divmod(k, m)
        x = (c - (m - 1) / 2.0) * pitch_x + (0.42 * pitch_x if r % 2 else 0.0)
        y = (r - (math.ceil(n / m) - 1) / 2.0) * pitch_y
        edge = (r == 0 or r == math.ceil(n / m) - 1 or c == 0 or c == m - 1)
        ang = (math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0)
        if edge:
            ang += (math.pi / 14.0 if (r + c) % 2 == 0 else -math.pi / 14.0)
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
        radius_x = ring * WIDTH * 0.42
        radius_y = ring * HEIGHT * 0.36
        for t in range(count):
            if placed >= n:
                break
            theta = (2.0 * math.pi * t / count) + (0.19 if ring % 2 else 0.0)
            x = radius_x * math.cos(theta)
            y = radius_y * math.sin(theta)
            ang = (math.pi / 2.0 if (placed % 2 == 0) else -math.pi / 2.0) + (0.10 if ring % 2 else -0.10)
            centers_c.append((x, y))
            angles_c.append(ang)
            placed += 1
        ring += 1
    templates.append((centers_c, angles_c))

    # Template D: boundary-heavy shell plus inner double-lattice.
    centers_d, angles_d = [], []
    if n >= 1:
        shell = min(n, max(4, int(round(2.5 * math.sqrt(n)))))
        r0 = max(0.85, 0.55 * math.sqrt(n))
        for k in range(shell):
            th = 2.0 * math.pi * k / shell
            x = r0 * math.cos(th)
            y = r0 * math.sin(th)
            ang = math.pi / 2.0 + (0.16 if k % 2 == 0 else -0.16)
            centers_d.append((x, y))
            angles_d.append(ang)
        placed = shell
        inner = n - shell
        if inner > 0:
            m2 = int(math.ceil(math.sqrt(inner)))
            px = WIDTH * 0.82
            py = HEIGHT * 0.76
            for k in range(inner):
                r, c = divmod(k, m2)
                x = (c - (m2 - 1) / 2.0) * px + (0.34 * px if r % 2 else 0.0)
                y = (r - (math.ceil(inner / m2) - 1) / 2.0) * py
                ang = math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0
                centers_d.append((x, y))
                angles_d.append(ang)
    mx = sum(x for x, _ in centers_d) / len(centers_d)
    my = sum(y for _, y in centers_d) / len(centers_d)
    centers_d = [(x - mx, y - my) for x, y in centers_d]
    templates.append((centers_d, angles_d))

    best = None
    best_s = float("inf")

    for seed, (centers, angles) in enumerate(templates):
        c, a, s = _local_search(centers, angles, seed=seed + 17)
        if s < best_s:
            best = (c, a, s)
            best_s = s

    centers, angles, s = best
    centers = repair(centers, angles)
    s = enclosing_side(centers, angles)
    return centers, angles, s
# EVOLVE-BLOCK-END
