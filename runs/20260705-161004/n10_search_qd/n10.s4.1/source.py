"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles: list[float]
- s: outer side of the enclosing square
Container convention: point p is inside iff max(|px|, |py|) <= s/2.

Strategy:
- Use several deterministic geometric templates inspired by double-lattice and
  boundary-assisted motifs.
- Refine with a compact but stronger numerical optimizer when SciPy is available.
- Validate non-overlap and containment exactly by polygon tests.
"""

import math
import random

SIDE = 1.0
TAU = 2.0 * math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0

R = SIDE / (2.0 * math.sin(math.pi / 5.0))
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

OVERLAP_EPS = 1e-10


def pentagon_vertices(cx, cy, angle):
    return [
        (cx + R * math.cos(angle + k * TAU / 5.0),
         cy + R * math.sin(angle + k * TAU / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    if not centers:
        return 0.0
    m = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            t = max(abs(vx), abs(vy))
            if t > m:
                m = t
    return 2.0 * m


def _proj(poly, ux, uy):
    dots = [x * ux + y * uy for x, y in poly]
    return min(dots), max(dots)


def pentagons_overlap(pa, pb):
    """True iff polygons overlap with positive area (touching is allowed)."""
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


def _center_pack(centers):
    if not centers:
        return centers
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    return [(x - mx, y - my) for x, y in centers]


def _wrap_angle(a):
    while a <= -math.pi:
        a += TAU
    while a > math.pi:
        a -= TAU
    return a


def _safe_minimize(fun, x0, bounds=None, method="Powell", maxiter=1200):
    try:
        import scipy.optimize as opt
    except Exception:
        return None
    try:
        return opt.minimize(
            fun,
            x0,
            method=method,
            bounds=bounds,
            options={"maxiter": maxiter, "xtol": 1e-5, "ftol": 1e-7, "disp": False},
        )
    except Exception:
        return None


def _vec_from_pack(centers, angles):
    vec = []
    for (x, y), a in zip(centers, angles):
        vec.extend([x, y, a])
    return vec


def _pack_from_vec(vec):
    n = len(vec) // 3
    centers = [(vec[3 * i], vec[3 * i + 1]) for i in range(n)]
    angles = [_wrap_angle(vec[3 * i + 2]) for i in range(n)]
    return _center_pack(centers), angles


def _objective_factory():
    def objective(vec):
        centers, angles = _pack_from_vec(vec)
        s = enclosing_side(centers, angles)
        if has_overlap(centers, angles):
            return s + 50.0
        # Gentle compactness penalty.
        rad = max((math.hypot(x, y) for x, y in centers), default=0.0)
        return s + 0.0005 * rad
    return objective


def _repair_scale(centers, angles):
    """Increase scale until all overlaps disappear."""
    centers = list(centers)
    angles = list(angles)
    if not has_overlap(centers, angles):
        return centers

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.02
    for _ in range(90):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.15
    else:
        return centers

    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi)


def _template_double_lattice(n, shift=0.0, tilt=0.0, edge_tilt=0.0, compact=1.0):
    """Opposite-orientation motif with staggered rows."""
    if n <= 0:
        return [], []
    m = int(math.ceil(math.sqrt(n)))
    r = int(math.ceil(n / m))

    px = WIDTH * (0.76 * compact)
    py = HEIGHT * (0.76 * compact)

    centers, angles = [], []
    for k in range(n):
        row, col = divmod(k, m)
        x = (col - (m - 1) / 2.0) * px + (shift if row % 2 else 0.0)
        y = (row - (r - 1) / 2.0) * py
        ang = (math.pi / 2.0 if (row + col) % 2 == 0 else -math.pi / 2.0)
        ang += tilt if (row + col) % 2 == 0 else -tilt
        if row == 0 or row == r - 1 or col == 0 or col == m - 1:
            ang += edge_tilt if (row + col) % 2 == 0 else -edge_tilt
        centers.append((x, y))
        angles.append(ang)
    return _center_pack(centers), angles


def _template_rows(n, row_bias=0.0, alternation=0.0):
    if n <= 0:
        return [], []
    rows = max(1, int(round(math.sqrt(n * (1.0 + row_bias)))))
    base = n // rows
    rem = n % rows
    row_lengths = [base + (1 if i < rem else 0) for i in range(rows)]

    px = WIDTH * 0.79
    py = HEIGHT * 0.75
    centers, angles = [], []
    idx = 0
    for r, m in enumerate(row_lengths):
        y = (r - (rows - 1) / 2.0) * py
        xshift = 0.5 * px if (r % 2) else 0.0
        x0 = -((m - 1) * px) / 2.0 + xshift
        boundary = (r == 0 or r == rows - 1)
        for c in range(m):
            if idx >= n:
                break
            x = x0 + c * px
            ang = (math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0)
            if boundary:
                ang += alternation if (r + c) % 2 == 0 else -alternation
            centers.append((x, y))
            angles.append(ang)
            idx += 1
    return _center_pack(centers), angles


def _template_shell(n):
    if n <= 0:
        return [], []
    shell = min(n, max(4, int(round(2.8 * math.sqrt(n)))))
    inner = n - shell
    centers, angles = [], []
    r0 = max(0.85, 0.56 * math.sqrt(n))
    for k in range(shell):
        th = TAU * k / shell
        centers.append((r0 * math.cos(th), r0 * math.sin(th)))
        angles.append((math.pi / 2.0 if k % 2 == 0 else -math.pi / 2.0) + (0.11 if k % 3 == 0 else -0.11))
    if inner > 0:
        c2, a2 = _template_double_lattice(inner, shift=0.33 * WIDTH, tilt=0.02, edge_tilt=0.04, compact=0.95)
        centers.extend(c2)
        angles.extend(a2)
    return _center_pack(centers), angles


def _template_spiral(n):
    if n <= 0:
        return [], []
    centers, angles = [], []
    placed = 0
    ring = 0
    while placed < n:
        count = 1 if ring == 0 else 6 * ring
        rx = ring * WIDTH * 0.38
        ry = ring * HEIGHT * 0.33
        phase = 0.18 if ring % 2 else 0.0
        for t in range(count):
            if placed >= n:
                break
            th = TAU * t / count + phase
            centers.append((rx * math.cos(th), ry * math.sin(th)))
            angles.append((math.pi / 2.0 if placed % 2 == 0 else -math.pi / 2.0) + (0.08 if ring % 2 else -0.08))
            placed += 1
        ring += 1
    return _center_pack(centers), angles


def _template_mixed(n):
    if n <= 0:
        return [], []
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]
    core_n = max(0, n - max(4, int(round(1.7 * math.sqrt(n)))))
    c1, a1 = _template_double_lattice(core_n, shift=0.29 * WIDTH, tilt=0.03, edge_tilt=0.05, compact=0.97)
    b = n - core_n
    c2, a2 = [], []
    if b > 0:
        r = max(0.72, 0.49 * math.sqrt(n))
        for k in range(b):
            th = TAU * k / b + (0.20 if k % 2 else 0.0)
            c2.append((r * math.cos(th), r * math.sin(th)))
            a2.append((math.pi / 2.0 if k % 2 == 0 else -math.pi / 2.0) + (0.10 if k % 3 == 0 else -0.10))
    return _center_pack(c1 + c2), a1 + a2


def _local_search(centers, angles, seed=0):
    rng = random.Random(seed)
    centers = _center_pack(list(centers))
    angles = list(angles)
    n = len(centers)

    centers = _repair_scale(centers, angles)
    best_centers = centers[:]
    best_angles = angles[:]
    best_s = enclosing_side(best_centers, best_angles)

    # Global optimizer.
    x0 = _vec_from_pack(best_centers, best_angles)
    lim = max(2.5, best_s * 1.35)
    bounds = []
    for _ in range(n):
        bounds.extend([(-lim, lim), (-lim, lim), (-math.pi, math.pi)])

    res = _safe_minimize(_objective_factory(), x0, bounds=bounds, method="Powell", maxiter=1800)
    if res is not None:
        c, a = _pack_from_vec(res.x)
        if not has_overlap(c, a):
            s = enclosing_side(c, a)
            if s < best_s:
                best_centers, best_angles, best_s = c, a, s

    # Coordinate-wise improvement with anisotropic moves.
    step_pos = max(0.006, best_s * 0.012)
    step_ang = 0.08

    def eval_state(cc, aa):
        if has_overlap(cc, aa):
            return None
        return enclosing_side(cc, aa)

    for phase in range(12):
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            cx, cy = best_centers[i]
            a = best_angles[i]
            rad = math.hypot(cx, cy) + 1e-12
            ux, uy = cx / rad, cy / rad
            tx, ty = -uy, ux

            moves = [
                (0.0, 0.0, 0.0),
                (ux * step_pos, uy * step_pos, 0.0),
                (-ux * step_pos, -uy * step_pos, 0.0),
                (tx * step_pos, ty * step_pos, 0.0),
                (-tx * step_pos, -ty * step_pos, 0.0),
                (0.0, 0.0, step_ang),
                (0.0, 0.0, -step_ang),
                (ux * 0.55 * step_pos, uy * 0.55 * step_pos, step_ang),
                (-ux * 0.55 * step_pos, -uy * 0.55 * step_pos, -step_ang),
                (tx * 0.55 * step_pos, ty * 0.55 * step_pos, step_ang),
                (-tx * 0.55 * step_pos, -ty * 0.55 * step_pos, -step_ang),
            ]

            cand = []
            for dx, dy, da in moves:
                cc = best_centers[:]
                aa = best_angles[:]
                cc[i] = (cx + dx, cy + dy)
                aa[i] = _wrap_angle(a + da)
                cc = _center_pack(cc)
                cand.append((cc, aa))

            if phase >= 5:
                edge_ids = sorted(
                    range(n),
                    key=lambda j: max(abs(best_centers[j][0]), abs(best_centers[j][1])),
                    reverse=True,
                )
                for j in edge_ids[: max(2, n // 3)]:
                    x, y = best_centers[j]
                    rad = math.hypot(x, y) + 1e-12
                    ux, uy = x / rad, y / rad
                    cc = best_centers[:]
                    aa = best_angles[:]
                    cc[j] = (x + ux * step_pos * 0.8, y + uy * step_pos * 0.8)
                    aa[j] = _wrap_angle(aa[j] + (0.05 if j % 2 == 0 else -0.05))
                    cc = _center_pack(cc)
                    cand.append((cc, aa))

            improved = False
            for cc, aa in cand:
                val = eval_state(cc, aa)
                if val is not None and val + 1e-12 < best_s:
                    best_centers, best_angles, best_s = cc, aa, val
                    improved = True
                    break

            if not improved and rng.random() < 0.18:
                cc, aa = cand[rng.randrange(len(cand))]
                val = eval_state(cc, aa)
                if val is not None and val < best_s * 1.02:
                    best_centers, best_angles, best_s = cc, aa, val

        step_pos *= 0.72
        step_ang *= 0.72

    best_centers = _repair_scale(best_centers, best_angles)
    best_s = enclosing_side(best_centers, best_angles)
    return best_centers, best_angles, best_s


def pack(n):
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    templates = [
        _template_double_lattice(n, shift=0.27 * WIDTH, tilt=0.020, edge_tilt=0.045, compact=0.98),
        _template_double_lattice(n, shift=0.36 * WIDTH, tilt=-0.018, edge_tilt=0.060, compact=0.96),
        _template_rows(n, row_bias=0.06, alternation=0.08),
        _template_shell(n),
        _template_spiral(n),
        _template_mixed(n),
    ]

    best = None
    best_s = float("inf")

    for seed, (centers, angles) in enumerate(templates):
        c, a, s = _local_search(centers, angles, seed=seed + 101)
        if s < best_s:
            best = (c, a, s)
            best_s = s

    centers, angles, s = best

    if has_overlap(centers, angles):
        centers = _repair_scale(centers, angles)
        s = enclosing_side(centers, angles)

    return centers, angles, s
