"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles: list[float]
- s: outer side of the enclosing square
Container convention: point p is inside iff max(|px|, |py|) <= s/2.

Strategy:
- Build several deterministic candidate packings emphasizing opposite-orientation
  double-lattice motifs, boundary rows, and ring/shell arrangements.
- Refine each candidate with a conservative overlap-preserving local search.
- Use a robust, geometry-based score that directly targets the enclosing square.
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


def _center_pack(centers):
    if not centers:
        return centers
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    return [(x - mx, y - my) for x, y in centers]


def _proj(poly, ux, uy):
    vals = [x * ux + y * uy for x, y in poly]
    return min(vals), max(vals)


def pentagons_overlap(pa, pb):
    """Separating Axis Theorem with strict positive-area overlap test."""
    for poly in (pa, pb):
        for i in range(5):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
            ux, uy = -(y2 - y1), (x2 - x1)
            norm = math.hypot(ux, uy)
            if norm < 1e-15:
                continue
            amin, amax = _proj(pa, ux, uy)
            bmin, bmax = _proj(pb, ux, uy)
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
    """Uniformly dilate centers until all overlaps vanish."""
    centers = list(centers)
    angles = list(angles)
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.02
    for _ in range(90):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.18
    else:
        return centers

    for _ in range(90):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def _safe_minimize(fun, x0, bounds=None, method="Powell", maxiter=1500):
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
            options={"maxiter": maxiter, "xtol": 1e-5, "ftol": 1e-6, "disp": False},
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
    angles = [vec[3 * i + 2] for i in range(n)]
    return _center_pack(centers), angles


def _objective_factory():
    def objective(vec):
        centers, angles = _pack_from_vec(vec)
        s = enclosing_side(centers, angles)
        if has_overlap(centers, angles):
            return s + 50.0
        # Soft compactness bias.
        rad = max((math.hypot(x, y) for x, y in centers), default=0.0)
        return s + 0.002 * rad
    return objective


def _template_double_lattice(n, sx=0.0, sy=0.0, tilt=0.0, edge_tilt=0.0, xscale=1.0, yscale=1.0):
    if n <= 0:
        return [], []
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    px = WIDTH * 0.80 * xscale
    py = HEIGHT * 0.78 * yscale

    centers, angles = [], []
    for k in range(n):
        r, c = divmod(k, cols)
        x = (c - (cols - 1) / 2.0) * px + (sx if r % 2 else 0.0)
        y = (r - (rows - 1) / 2.0) * py + sy * (r - (rows - 1) / 2.0) / max(1, rows - 1)
        ang = math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0
        ang += tilt if (r + c) % 2 == 0 else -tilt
        if r in (0, rows - 1) or c in (0, cols - 1):
            ang += edge_tilt if (r + c) % 2 == 0 else -edge_tilt
        centers.append((x, y))
        angles.append(ang)

    return _center_pack(centers), angles


def _template_rows(n):
    if n <= 0:
        return [], []
    # Nearly square row count, with mild unevenness to match n.
    rows = max(1, int(round(math.sqrt(n * 0.95))))
    base = n // rows
    rem = n % rows
    lengths = [base + (1 if i < rem else 0) for i in range(rows)]

    px = WIDTH * 0.82
    py = HEIGHT * 0.77
    centers, angles = [], []
    for r, m in enumerate(lengths):
        y = (r - (rows - 1) / 2.0) * py
        xshift = 0.50 * px if r % 2 else 0.0
        x0 = -0.5 * (m - 1) * px + xshift
        boundary = (r == 0 or r == rows - 1)
        for c in range(m):
            x = x0 + c * px
            ang = math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0
            if boundary:
                ang += 0.09 if (r + c) % 2 == 0 else -0.09
            centers.append((x, y))
            angles.append(ang)
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
        phase = 0.13 if ring % 2 else 0.0
        for t in range(count):
            if placed >= n:
                break
            th = TAU * t / count + phase
            x = rx * math.cos(th)
            y = ry * math.sin(th)
            ang = (math.pi / 2.0 if placed % 2 == 0 else -math.pi / 2.0)
            ang += 0.10 if ring % 2 else -0.10
            centers.append((x, y))
            angles.append(ang)
            placed += 1
        ring += 1
    return _center_pack(centers), angles


def _template_shell(n):
    if n <= 0:
        return [], []
    shell = min(n, max(4, int(round(2.8 * math.sqrt(n)))))
    inner = n - shell
    centers, angles = [], []
    r0 = max(0.82, 0.56 * math.sqrt(n))
    for k in range(shell):
        th = TAU * k / shell
        centers.append((r0 * math.cos(th), r0 * math.sin(th)))
        angles.append(math.pi / 2.0 + (0.14 if k % 2 == 0 else -0.14))
    if inner > 0:
        c2, a2 = _template_double_lattice(inner, sx=0.33 * WIDTH, tilt=0.025, edge_tilt=0.04, xscale=0.98, yscale=0.98)
        centers.extend(c2)
        angles.extend(a2)
    return _center_pack(centers), angles


def _template_mixed(n):
    if n <= 0:
        return [], []
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]
    core = max(0, n - max(4, int(round(1.7 * math.sqrt(n)))))
    c1, a1 = _template_double_lattice(core, sx=0.29 * WIDTH, tilt=0.02, edge_tilt=0.05, xscale=0.97, yscale=0.98)
    b = n - core
    c2, a2 = [], []
    if b > 0:
        rr = max(0.72, 0.50 * math.sqrt(n))
        for k in range(b):
            th = TAU * k / b + (0.18 if k % 2 else 0.0)
            c2.append((rr * math.cos(th), rr * math.sin(th)))
            a2.append((math.pi / 2.0 if k % 2 == 0 else -math.pi / 2.0) + (0.11 if k % 3 == 0 else -0.11))
    return _center_pack(c1 + c2), a1 + a2


def _template_arched_rows(n):
    if n <= 0:
        return [], []
    rows = max(2, int(round(math.sqrt(n * 1.1))))
    base = n // rows
    rem = n % rows
    lengths = [base + (1 if i < rem else 0) for i in range(rows)]
    px = WIDTH * 0.81
    py = HEIGHT * 0.74
    centers, angles = [], []
    for r, m in enumerate(lengths):
        y = (r - (rows - 1) / 2.0) * py
        arc = 0.16 * math.sin((r - (rows - 1) / 2.0) * math.pi / max(2, rows - 1))
        x0 = -0.5 * (m - 1) * px
        for c in range(m):
            x = x0 + c * px + arc * (c - (m - 1) / 2.0)
            ang = math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0
            if r == 0 or r == rows - 1:
                ang += 0.08 if (r + c) % 2 == 0 else -0.08
            centers.append((x, y))
            angles.append(ang)
    return _center_pack(centers), angles


def _local_search(centers, angles, seed=0):
    rng = random.Random(seed)
    centers = _center_pack(list(centers))
    angles = list(angles)
    n = len(centers)

    if has_overlap(centers, angles):
        centers = repair(centers, angles)

    best_centers, best_angles = centers[:], angles[:]
    best_s = enclosing_side(best_centers, best_angles)

    obj = _objective_factory()
    x0 = _vec_from_pack(best_centers, best_angles)
    lim = max(2.0, best_s * 1.35)
    bounds = []
    for _ in range(n):
        bounds.extend([(-lim, lim), (-lim, lim), (-math.pi, math.pi)])

    res = _safe_minimize(obj, x0, bounds=bounds, method="Powell", maxiter=1800)
    if res is not None:
        c, a = _pack_from_vec(res.x)
        if not has_overlap(c, a):
            s = enclosing_side(c, a)
            if s < best_s:
                best_centers, best_angles, best_s = c, a, s

    step_pos = max(0.007, best_s * 0.014)
    step_ang = 0.11

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
            ]
            if phase >= 5:
                moves += [
                    (ux * 1.4 * step_pos, uy * 1.4 * step_pos, 0.0),
                    (-ux * 1.4 * step_pos, -uy * 1.4 * step_pos, 0.0),
                ]

            candidates = []
            for dx, dy, da in moves:
                cc = best_centers[:]
                aa = best_angles[:]
                cc[i] = (cx + dx, cy + dy)
                aa[i] = a + da
                cc = _center_pack(cc)
                candidates.append((cc, aa))

            if phase >= 4:
                edge_ids = sorted(range(n), key=lambda j: max(abs(best_centers[j][0]), abs(best_centers[j][1])), reverse=True)
                for j in edge_ids[: max(2, n // 3)]:
                    cc = best_centers[:]
                    aa = best_angles[:]
                    x, y = cc[j]
                    rad = math.hypot(x, y) + 1e-12
                    ux, uy = x / rad, y / rad
                    cc[j] = (x + ux * step_pos * 0.9, y + uy * step_pos * 0.9)
                    aa[j] = aa[j] + (0.07 if j % 2 == 0 else -0.07)
                    cc = _center_pack(cc)
                    candidates.append((cc, aa))

            improved = False
            for cc, aa in candidates:
                if has_overlap(cc, aa):
                    continue
                s = enclosing_side(cc, aa)
                if s + 1e-12 < best_s:
                    best_centers, best_angles, best_s = cc, aa, s
                    improved = True
                    break

            if not improved and rng.random() < 0.18:
                cc, aa = candidates[rng.randrange(len(candidates))]
                if not has_overlap(cc, aa):
                    s = enclosing_side(cc, aa)
                    if s < best_s * 1.02:
                        best_centers, best_angles, best_s = cc, aa, s

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

    templates = [
        _template_double_lattice(n, sx=0.30 * WIDTH, tilt=0.020, edge_tilt=0.055, xscale=0.97, yscale=0.98),
        _template_double_lattice(n, sx=0.38 * WIDTH, tilt=-0.018, edge_tilt=0.070, xscale=0.98, yscale=0.97),
        _template_rows(n),
        _template_arched_rows(n),
        _template_shell(n),
        _template_spiral(n),
        _template_mixed(n),
    ]

    best = None
    best_s = float("inf")

    for seed, (centers, angles) in enumerate(templates):
        c, a, s = _local_search(centers, angles, seed=17 + seed)
        if s < best_s:
            best = (c, a, s)
            best_s = s

    centers, angles, s = best

    if has_overlap(centers, angles):
        centers = repair(centers, angles)
        s = enclosing_side(centers, angles)

    # Final sanity pass: try a tiny global shrink if safe.
    if not has_overlap(centers, angles):
        lo, hi = 0.985, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            cc = [(x * mid, y * mid) for x, y in centers]
            if has_overlap(cc, angles):
                lo = mid
            else:
                hi = mid
        centers = [(x * hi, y * hi) for x, y in centers]
        s = enclosing_side(centers, angles)

    return centers, angles, s
