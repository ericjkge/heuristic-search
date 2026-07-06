"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles: list[float]
- s: outer side of the enclosing square
Container convention: point p is inside iff max(|px|, |py|) <= s/2.

This implementation uses a deterministic multi-start search built around
double-lattice / opposite-orientation motifs, boundary-aware row patterns,
and a robust numerical improvement phase when SciPy is available.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))          # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))    # inradius
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

TAU = 2.0 * math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0


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


# EVOLVE-BLOCK-START
OVERLAP_EPS = 1e-9


def _proj(poly, ux, uy):
    dots = [x * ux + y * uy for x, y in poly]
    return min(dots), max(dots)


def pentagons_overlap(pa, pb):
    """Return True iff polygons overlap with positive area."""
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


def repair(centers, angles):
    """Radially dilate centers until overlaps vanish; keep angles fixed."""
    centers = list(centers)
    angles = list(angles)
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.02
    for _ in range(80):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.18
    else:
        return centers

    for _ in range(80):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


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
    centers = _center_pack(centers)
    return centers, angles


def _objective_factory(n):
    def objective(vec):
        centers, angles = _pack_from_vec(vec)
        s = enclosing_side(centers, angles)
        if has_overlap(centers, angles):
            return s + 100.0
        # Mild penalty for drifting too far from origin; helps Powell stay compact.
        rad = max((math.hypot(x, y) for x, y in centers), default=0.0)
        return s + 0.001 * rad
    return objective


def _contact_heuristic_score(centers, angles):
    """Lower is better. Uses pairwise distances and boundary utilization."""
    if not centers:
        return 0.0
    s = enclosing_side(centers, angles)
    hw = s / 2.0
    pts = [p for c, a in zip(centers, angles) for p in pentagon_vertices(c[0], c[1], a)]
    bound = max(max(abs(x), abs(y)) for x, y in pts)
    return bound + 0.03 * s


def _template_double_lattice(n, axial_shift=0.0, tilt=0.0, edge_tilt=0.0):
    """Dense opposite-orientation lattice with rectangular envelope."""
    if n <= 0:
        return [], []
    m = int(math.ceil(math.sqrt(n)))
    r = int(math.ceil(n / m))
    px = WIDTH * 0.80
    py = HEIGHT * 0.77

    centers, angles = [], []
    for k in range(n):
        row, col = divmod(k, m)
        x = (col - (m - 1) / 2.0) * px + (axial_shift if row % 2 else 0.0)
        y = (row - (r - 1) / 2.0) * py
        ang = (math.pi / 2.0 if (row + col) % 2 == 0 else -math.pi / 2.0)
        ang += tilt if (row + col) % 2 == 0 else -tilt
        if row == 0 or row == r - 1 or col == 0 or col == m - 1:
            ang += edge_tilt if (row + col) % 2 == 0 else -edge_tilt
        centers.append((x, y))
        angles.append(ang)

    centers = _center_pack(centers)
    return centers, angles


def _template_staggered_rows(n):
    if n <= 0:
        return [], []
    rows = max(1, int(round(math.sqrt(n * 1.08))))
    base = n // rows
    rem = n % rows
    row_lengths = [base + (1 if i < rem else 0) for i in range(rows)]
    px = WIDTH * 0.83
    py = HEIGHT * 0.76

    centers, angles = [], []
    idx = 0
    for r, m in enumerate(row_lengths):
        xshift = 0.5 * px if (r % 2) else 0.0
        y = (r - (rows - 1) / 2.0) * py
        x0 = -((m - 1) * px) / 2.0 + xshift
        boundary = (r == 0 or r == rows - 1)
        for c in range(m):
            if idx >= n:
                break
            x = x0 + c * px
            ang = (math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0)
            if boundary:
                ang += 0.10 if (r + c) % 2 == 0 else -0.10
            centers.append((x, y))
            angles.append(ang)
            idx += 1
    return _center_pack(centers), angles


def _template_spiral(n):
    if n <= 0:
        return [], []
    centers, angles = [], []
    placed = 0
    ring = 0
    while placed < n:
        count = 1 if ring == 0 else 6 * ring
        rx = ring * WIDTH * 0.39
        ry = ring * HEIGHT * 0.34
        phase = 0.17 if ring % 2 else 0.0
        for t in range(count):
            if placed >= n:
                break
            th = TAU * t / count + phase
            x = rx * math.cos(th)
            y = ry * math.sin(th)
            ang = (math.pi / 2.0 if placed % 2 == 0 else -math.pi / 2.0) + (0.08 if ring % 2 else -0.08)
            centers.append((x, y))
            angles.append(ang)
            placed += 1
        ring += 1
    return _center_pack(centers), angles


def _template_shell(n):
    if n <= 0:
        return [], []
    shell = min(n, max(4, int(round(2.7 * math.sqrt(n)))))
    inner = n - shell
    centers, angles = [], []
    r0 = max(0.85, 0.58 * math.sqrt(n))
    for k in range(shell):
        th = TAU * k / shell
        centers.append((r0 * math.cos(th), r0 * math.sin(th)))
        angles.append(math.pi / 2.0 + (0.14 if k % 2 == 0 else -0.14))
    if inner > 0:
        c2, a2 = _template_double_lattice(inner, axial_shift=0.38 * WIDTH, tilt=0.03, edge_tilt=0.06)
        centers.extend(c2)
        angles.extend(a2)
    return _center_pack(centers), angles


def _template_mixed(n):
    """Mix of a lattice core and boundary-assisted outer ring."""
    if n <= 0:
        return [], []
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]

    core_n = max(0, n - max(4, int(round(1.8 * math.sqrt(n)))))
    c1, a1 = _template_double_lattice(core_n, axial_shift=0.33 * WIDTH, tilt=0.02, edge_tilt=0.03)
    b = n - core_n
    c2, a2 = [], []
    if b > 0:
        shell_r = max(0.7, 0.50 * math.sqrt(n))
        for k in range(b):
            th = TAU * k / b + (0.19 if k % 2 else 0.0)
            c2.append((shell_r * math.cos(th), shell_r * math.sin(th)))
            a2.append((math.pi / 2.0 if k % 2 == 0 else -math.pi / 2.0) + (0.12 if k % 3 == 0 else -0.12))
    centers = c1 + c2
    angles = a1 + a2
    return _center_pack(centers), angles


def _local_search(centers, angles, seed=0):
    rng = random.Random(seed)
    centers = _center_pack(list(centers))
    angles = list(angles)
    n = len(centers)

    centers = repair(centers, angles)
    best_centers = centers[:]
    best_angles = angles[:]
    best_s = enclosing_side(best_centers, best_angles)

    x0 = _vec_from_pack(best_centers, best_angles)
    obj = _objective_factory(n)

    lim = max(2.5, best_s * 1.7)
    bounds = []
    for _ in range(n):
        bounds.extend([(-lim, lim), (-lim, lim), (-math.pi, math.pi)])

    # Stronger continuous search with multiple methods.
    for method, maxiter in (("Powell", 2800), ("Nelder-Mead", 3200)):
        res = _safe_minimize(obj, x0, bounds=bounds, method=method, maxiter=maxiter)
        if res is not None:
            c, a = _pack_from_vec(res.x)
            if not has_overlap(c, a):
                s = enclosing_side(c, a)
                if s < best_s:
                    best_centers, best_angles, best_s = c, a, s
                    x0 = _vec_from_pack(best_centers, best_angles)

    def eval_state(cc, aa):
        if has_overlap(cc, aa):
            return None
        return enclosing_side(cc, aa)

    # Boundary-aware local improvement with pair moves.
    step_pos = max(0.005, best_s * 0.010)
    step_ang = 0.07

    for phase in range(16):
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

            if phase >= 4:
                edge_ids = sorted(range(n), key=lambda j: max(abs(best_centers[j][0]), abs(best_centers[j][1])), reverse=True)
                for j in edge_ids[: max(2, n // 3)]:
                    x, y = best_centers[j]
                    radj = math.hypot(x, y) + 1e-12
                    uxj, uyj = x / radj, y / radj
                    moves.extend([
                        (uxj * step_pos * 1.1, uyj * step_pos * 1.1, 0.0),
                        (-uxj * step_pos * 1.1, -uyj * step_pos * 1.1, 0.0),
                        (0.0, 0.0, 0.05 if j % 2 == 0 else -0.05),
                    ])

            best_local = None
            for dx, dy, da in moves:
                cc = best_centers[:]
                aa = best_angles[:]
                cc[i] = (cx + dx, cy + dy)
                aa[i] = a + da
                cc = _center_pack(cc)
                val = eval_state(cc, aa)
                if val is not None and (best_local is None or val < best_local[0]):
                    best_local = (val, cc, aa)

            if best_local is not None and best_local[0] + 1e-12 < best_s:
                best_s, best_centers, best_angles = best_local

        # Pairwise coupled moves for the boundary ring.
        if phase >= 3 and n >= 2:
            ids = sorted(range(n), key=lambda k: max(abs(best_centers[k][0]), abs(best_centers[k][1])), reverse=True)
            for u in ids[: min(n, 6)]:
                for v in ids[: min(n, 6)]:
                    if u >= v:
                        continue
                    cu = best_centers[u]
                    cv = best_centers[v]
                    au = best_angles[u]
                    av = best_angles[v]
                    for d in (-0.06, 0.0, 0.06):
                        cc = best_centers[:]
                        aa = best_angles[:]
                        mx = (cu[0] + cv[0]) * 0.5
                        my = (cu[1] + cv[1]) * 0.5
                        dx = (cu[0] - cv[0]) * 0.5
                        dy = (cu[1] - cv[1]) * 0.5
                        cc[u] = (mx + dx * (1.0 + d), my + dy * (1.0 + d))
                        cc[v] = (mx - dx * (1.0 + d), my - dy * (1.0 + d))
                        aa[u] = au + d
                        aa[v] = av - d
                        cc = _center_pack(cc)
                        val = eval_state(cc, aa)
                        if val is not None and val + 1e-12 < best_s:
                            best_centers, best_angles, best_s = cc, aa, val

        step_pos *= 0.78
        step_ang *= 0.78

    # Low-temperature random refinement.
    for _ in range(1200):
        i = rng.randrange(n)
        cc = best_centers[:]
        aa = best_angles[:]
        x, y = cc[i]
        a = aa[i]
        rad = math.hypot(x, y) + 1e-12
        ux, uy = x / rad, y / rad
        tx, ty = -uy, ux
        dx = (rng.random() * 2.0 - 1.0) * step_pos
        dy = (rng.random() * 2.0 - 1.0) * step_pos
        da = (rng.random() * 2.0 - 1.0) * step_ang
        cc[i] = (x + ux * dx + tx * dy, y + uy * dx + ty * dy)
        aa[i] = a + da
        cc = _center_pack(cc)
        val = eval_state(cc, aa)
        if val is not None and val < best_s:
            best_centers, best_angles, best_s = cc, aa, val

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

    # Strong double-lattice motifs, slightly perturbed to enable tighter square fit.
    templates.append(_template_double_lattice(n, axial_shift=0.31 * WIDTH, tilt=0.025, edge_tilt=0.055))
    templates.append(_template_double_lattice(n, axial_shift=0.39 * WIDTH, tilt=-0.020, edge_tilt=0.070))

    # Row-based boundary tilting and mixed motifs.
    templates.append(_template_staggered_rows(n))
    templates.append(_template_shell(n))
    templates.append(_template_spiral(n))
    templates.append(_template_mixed(n))

    best = None
    best_s = float("inf")

    for seed, (centers, angles) in enumerate(templates):
        c, a, s = _local_search(centers, angles, seed=seed + 19)
        if s < best_s:
            best = (c, a, s)
            best_s = s

    centers, angles, s = best

    # Final cleanup: a last feasibility check and tiny recovery if needed.
    if has_overlap(centers, angles):
        centers = repair(centers, angles)
        s = enclosing_side(centers, angles)

    return centers, angles, s
# EVOLVE-BLOCK-END
