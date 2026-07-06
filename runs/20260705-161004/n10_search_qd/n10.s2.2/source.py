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
    """More compact multistart template pool for 10 pentagons.

    Builds several high-contact candidate layouts:
    - double-lattice rows with tunable pitch
    - mixed-orientation shell/inner patterns
    - a mild boundary-biased row family
    The best candidate is selected later by local search.
    """
    if n <= 0:
        return [], []

    def center_pack(cc, aa):
        if not cc:
            return cc, aa
        mx = sum(x for x, _ in cc) / len(cc)
        my = sum(y for _, y in cc) / len(cc)
        return [(x - mx, y - my) for x, y in cc], aa

    best = None
    best_s = float("inf")

    # Candidate 1: dense staggered double-lattice
    for px_mul, py_mul, shift_mul, tilt in [
        (0.845, 0.775, 0.50, 0.00),
        (0.820, 0.760, 0.42, 0.05),
        (0.865, 0.790, 0.36, -0.04),
    ]:
        m = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / m))
        cc, aa = [], []
        for k in range(n):
            r, c = divmod(k, m)
            x = (c - (m - 1) / 2.0) * (WIDTH * px_mul) + (shift_mul * WIDTH * px_mul if r % 2 else 0.0)
            y = (r - (rows - 1) / 2.0) * (HEIGHT * py_mul)
            ang = (math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0) + (tilt if r % 2 == 0 else -tilt)
            cc.append((x, y))
            aa.append(ang)
        cc, aa = center_pack(cc, aa)
        s = enclosing_side(cc, aa)
        if s < best_s:
            best, best_s = (cc, aa), s

    # Candidate 2: shell + compact inner fill.
    shell = min(n, max(4, int(round(2.2 * math.sqrt(n)))))
    cc, aa = [], []
    if n > 0:
        r0 = max(0.80, 0.48 * math.sqrt(n))
        for k in range(shell):
            th = 2.0 * math.pi * k / shell + (0.14 if k % 2 else 0.0)
            cc.append((r0 * math.cos(th), r0 * math.sin(th)))
            aa.append((math.pi / 2.0 if k % 2 == 0 else -math.pi / 2.0) + (0.11 if k % 3 == 0 else -0.07))
        inner = n - shell
        if inner > 0:
            m = int(math.ceil(math.sqrt(inner)))
            rows = int(math.ceil(inner / m))
            px = WIDTH * 0.80
            py = HEIGHT * 0.74
            for k in range(inner):
                r, c = divmod(k, m)
                x = (c - (m - 1) / 2.0) * px + (0.40 * px if r % 2 else 0.0)
                y = (r - (rows - 1) / 2.0) * py
                cc.append((x, y))
                aa.append(math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0)
    cc, aa = center_pack(cc, aa)
    s = enclosing_side(cc, aa)
    if s < best_s:
        best, best_s = (cc, aa), s

    # Candidate 3: boundary-weighted rows.
    row_lengths = []
    rows = max(2, int(round(math.sqrt(n * 1.4))))
    base = n // rows
    rem = n % rows
    for r in range(rows):
        row_lengths.append(base + (1 if r < rem else 0))
    cc, aa = [], []
    idx = 0
    py = HEIGHT * 0.81
    px = WIDTH * 0.84
    for r, m in enumerate(row_lengths):
        x_shift = 0.5 * px if r % 2 else 0.0
        y = (r - (rows - 1) / 2.0) * py
        ang0 = math.pi / 10.0 if (r == 0 or r == rows - 1) else 0.0
        x0 = -((m - 1) * px) / 2.0 + x_shift
        for c in range(m):
            if idx >= n:
                break
            cc.append((x0 + c * px, y))
            aa.append(ang0 + (math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0))
            idx += 1
    cc, aa = center_pack(cc, aa)
    s = enclosing_side(cc, aa)
    if s < best_s:
        best = (cc, aa)

    return best


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

    templates = []

    c0, a0 = _pack_template(n)
    templates.append((c0, a0))

    # Extra handcrafted 10-gon focused layouts.
    if n == 10:
        # 1) 3-4-3 staggered rows
        centers, angles = [], []
        xs = [(-1.0, 0), (0.0, 0), (1.0, 0)]
        rows = [3, 4, 3]
        py = HEIGHT * 0.735
        px = WIDTH * 0.790
        y0 = -py
        for r, m in enumerate(rows):
            x_shift = 0.5 * px if r == 1 else 0.0
            x0 = -((m - 1) * px) / 2.0 + x_shift
            y = y0 + r * py
            for c in range(m):
                centers.append((x0 + c * px, y))
                angles.append((math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0) + (0.07 if r == 0 else -0.07 if r == 2 else 0.0))
        templates.append((centers, angles))

        # 2) two interleaved pentads on offset ellipses
        centers, angles = [], []
        for k in range(5):
            th = 2.0 * math.pi * k / 5.0 + 0.18
            centers.append((1.23 * math.cos(th), 0.95 * math.sin(th)))
            angles.append(math.pi / 2.0 if k % 2 == 0 else -math.pi / 2.0)
        for k in range(5):
            th = 2.0 * math.pi * k / 5.0 + math.pi / 5.0 - 0.12
            centers.append((1.23 * math.cos(th), 0.95 * math.sin(th)))
            angles.append(-math.pi / 2.0 if k % 2 == 0 else math.pi / 2.0)
        templates.append((centers, angles))

        # 3) tight shell + center pair
        centers, angles = [], []
        for k in range(8):
            th = 2.0 * math.pi * k / 8.0 + 0.10 * (k % 2)
            centers.append((1.72 * math.cos(th), 1.55 * math.sin(th)))
            angles.append((math.pi / 2.0 if k % 2 == 0 else -math.pi / 2.0) + (0.12 if k < 4 else -0.12))
        centers.extend([(-0.36, 0.0), (0.36, 0.0)])
        angles.extend([0.0, math.pi])
        templates.append((centers, angles))

    # Generic fallback templates for other n.
    m = int(math.ceil(math.sqrt(n)))
    pitch_x = WIDTH * 0.82
    pitch_y = HEIGHT * 0.76
    centers_b, angles_b = [], []
    for k in range(n):
        r, c = divmod(k, m)
        x = (c - (m - 1) / 2.0) * pitch_x + (0.38 * pitch_x if r % 2 else 0.0)
        y = (r - (math.ceil(n / m) - 1) / 2.0) * pitch_y
        ang = (math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0) + (0.06 if r % 2 == 0 else -0.06)
        centers_b.append((x, y))
        angles_b.append(ang)
    mx = sum(x for x, _ in centers_b) / len(centers_b)
    my = sum(y for _, y in centers_b) / len(centers_b)
    templates.append(([(x - mx, y - my) for x, y in centers_b], angles_b))

    best = None
    best_s = float("inf")
    for seed, (centers, angles) in enumerate(templates):
        c, a, s = _local_search(centers, angles, seed=seed + 31)
        if s < best_s:
            best = (c, a, s)
            best_s = s

    centers, angles, s = best
    centers = repair(centers, angles)
    s = enclosing_side(centers, angles)
    return centers, angles, s
# EVOLVE-BLOCK-END
