"""Heuristic + numerical optimizer for packing n unit regular pentagons into the
smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s).
Container: point p is inside iff max(|px|, |py|) <= s/2.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # inradius
TAU = 2.0 * math.pi

PHI = (1.0 + math.sqrt(5.0)) / 2.0
ANGLE_SET = [
    0.0,
    math.pi / 5.0,
    2.0 * math.pi / 5.0,
    3.0 * math.pi / 5.0,
    4.0 * math.pi / 5.0,
    math.pi,
    -math.pi / 5.0,
    -2.0 * math.pi / 5.0,
    -3.0 * math.pi / 5.0,
    -4.0 * math.pi / 5.0,
]

try:
    import scipy.optimize as _spo
except Exception:
    _spo = None


def normalize_angle(a):
    a = math.fmod(a, TAU)
    if a <= -math.pi:
        a += TAU
    elif a > math.pi:
        a -= TAU
    return a


def pentagon_vertices(cx, cy, angle):
    return [
        (cx + R * math.cos(angle + k * TAU / 5.0),
         cy + R * math.sin(angle + k * TAU / 5.0))
        for k in range(5)
    ]


def poly_axes(poly):
    axes = []
    for i in range(5):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
        ux, uy = -(y2 - y1), (x2 - x1)
        norm = math.hypot(ux, uy)
        if norm > 0:
            axes.append((ux / norm, uy / norm))
    return axes


def project(poly, ax, ay):
    vals = [x * ax + y * ay for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb, eps=1e-10):
    for ax, ay in poly_axes(pa) + poly_axes(pb):
        amin, amax = project(pa, ax, ay)
        bmin, bmax = project(pb, ax, ay)
        if min(amax, bmax) - max(amin, bmin) <= eps:
            return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    m = len(polys)
    for i in range(m):
        for j in range(i + 1, m):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def enclosing_side(centers, angles):
    if not centers:
        return 0.0
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, ang)
    )


def box_bounds(centers, angles):
    xs = []
    ys = []
    for (cx, cy), a in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, a):
            xs.append(vx)
            ys.append(vy)
    return min(xs), max(xs), min(ys), max(ys)


def center_to_origin(centers):
    if not centers:
        return centers
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    return [(x - mx, y - my) for x, y in centers]


def repair_scale(centers, angles):
    if not has_overlap(centers, angles):
        return centers

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.05
    for _ in range(90):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.12
    else:
        return centers

    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi)


def safe_penalty(centers, angles, s):
    half = 0.5 * s
    pen = 0.0
    for (cx, cy), a in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, a):
            dx = abs(vx) - half
            dy = abs(vy) - half
            if dx > 0:
                pen += dx * dx
            if dy > 0:
                pen += dy * dy
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    m = len(polys)
    for i in range(m):
        for j in range(i + 1, m):
            for ax, ay in poly_axes(polys[i]) + poly_axes(polys[j]):
                amin, amax = project(polys[i], ax, ay)
                bmin, bmax = project(polys[j], ax, ay)
                gap = max(amin, bmin) - min(amax, bmax)
                if gap < 0:
                    pen += (-gap) ** 2
    return pen


def base_orientations(n, rng):
    if n <= 1:
        return [math.pi / 2.0] * n
    # Prefer opposite-orientation pairings, with symmetry-related seeds.
    if n <= 4:
        seq = [math.pi / 2.0, -math.pi / 2.0, 0.0, math.pi]
        return [normalize_angle(seq[i]) for i in range(n)]
    seq = [
        math.pi / 2.0, -math.pi / 2.0,
        0.0, math.pi,
        math.pi / 5.0, -math.pi / 5.0,
        2.0 * math.pi / 5.0, -2.0 * math.pi / 5.0,
        3.0 * math.pi / 5.0, -3.0 * math.pi / 5.0,
    ]
    out = []
    for i in range(n):
        a = seq[i % len(seq)]
        a += (rng.random() - 0.5) * 0.02
        out.append(normalize_angle(a))
    return out


def compact_rows(n, rng):
    """Start from a tight staggered row layout."""
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]

    rows = max(1, int(round(math.sqrt(n * 0.85))))
    cols = int(math.ceil(n / rows))
    # Rough initial spacing; later numerical optimization will tighten.
    dx = 1.75
    dy = 1.55

    centers = []
    angles = base_orientations(n, rng)
    idx = 0
    for r in range(rows):
        row_n = min(cols, n - idx)
        shift = 0.5 * dx if (r % 2 == 1) else 0.0
        x0 = -0.5 * (row_n - 1) * dx
        y = (r - 0.5 * (rows - 1)) * dy
        for c in range(row_n):
            x = x0 + c * dx + shift
            # Mild center jitter to avoid symmetric traps.
            x += (rng.random() - 0.5) * 0.02
            yj = y + (rng.random() - 0.5) * 0.02
            centers.append((x, yj))
            idx += 1
            if idx >= n:
                break
    centers = center_to_origin(centers)
    return centers, angles


def compact_columns(n, rng):
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]

    cols = max(1, int(round(math.sqrt(n * 0.85))))
    rows = int(math.ceil(n / cols))
    dx = 1.55
    dy = 1.75

    centers = []
    angles = base_orientations(n, rng)
    idx = 0
    for c in range(cols):
        col_n = min(rows, n - idx)
        shift = 0.5 * dy if (c % 2 == 1) else 0.0
        y0 = -0.5 * (col_n - 1) * dy
        x = (c - 0.5 * (cols - 1)) * dx
        for r in range(col_n):
            y = y0 + r * dy + shift
            xj = x + (rng.random() - 0.5) * 0.02
            y += (rng.random() - 0.5) * 0.0
            centers.append((xj, y))
            idx += 1
            if idx >= n:
                break
    centers = center_to_origin(centers)
    return centers, angles


def mixed_diamond(n, rng):
    """Create a skewed double-lattice-ish start with alternating opposite orientations."""
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]

    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    a = 1.55
    b = 1.35
    centers = []
    angles = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            x = (c - 0.5 * (cols - 1)) * a + (r % 2) * 0.5 * a
            y = (r - 0.5 * (rows - 1)) * b
            # Use paired opposite orientations and symmetry seeds.
            ang = math.pi / 2.0 if (r + c) % 2 == 0 else -math.pi / 2.0
            if (r + 2 * c) % 5 == 1:
                ang += math.pi / 5.0
            if (r + c) % 7 == 3:
                ang -= math.pi / 5.0
            ang += (rng.random() - 0.5) * 0.03
            centers.append((x, y))
            angles.append(normalize_angle(ang))
            idx += 1
    centers = center_to_origin(centers)
    return centers, angles


def init_layouts(n, seed):
    rng = random.Random(1000003 + 97 * n + seed)
    layouts = []
    layouts.append(compact_rows(n, rng))
    layouts.append(compact_columns(n, rng))
    layouts.append(mixed_diamond(n, rng))

    # Additional randomized row/column hybrid starts.
    for _ in range(3):
        rrng = random.Random(rng.randrange(10**9))
        if rrng.random() < 0.5:
            layouts.append(compact_rows(n, rrng))
        else:
            layouts.append(mixed_diamond(n, rrng))
    return layouts


def objective(vec, n):
    centers = [(vec[3 * i], vec[3 * i + 1]) for i in range(n)]
    angles = [normalize_angle(vec[3 * i + 2]) for i in range(n)]
    s = enclosing_side(centers, angles)
    pen = safe_penalty(centers, angles, s)
    return s + 2000.0 * pen


def pack_with_scipy(centers, angles, maxiter=600):
    if _spo is None:
        return centers, angles, enclosing_side(centers, angles)

    n = len(centers)
    x0 = []
    for (cx, cy), a in zip(centers, angles):
        x0.extend([cx, cy, a])

    res = _spo.minimize(
        lambda v: objective(v, n),
        x0,
        method="Powell",
        options={"maxiter": maxiter, "disp": False, "xtol": 1e-4, "ftol": 1e-4},
    )

    vec = res.x
    centers = [(vec[3 * i], vec[3 * i + 1]) for i in range(n)]
    angles = [normalize_angle(vec[3 * i + 2]) for i in range(n)]
    centers = center_to_origin(centers)
    centers = repair_scale(centers, angles)
    return centers, angles, enclosing_side(centers, angles)


def local_random_improve(centers, angles, seed=0, max_iter=5000):
    rng = random.Random(424242 + 131 * seed)
    n = len(centers)

    cur_c = [tuple(p) for p in centers]
    cur_a = [normalize_angle(a) for a in angles]
    cur_s = enclosing_side(cur_c, cur_a)
    best = (cur_c[:], cur_a[:], cur_s)

    step_xy = max(0.005, cur_s * 0.008)
    step_a = 0.16

    def score(c, a):
        s = enclosing_side(c, a)
        if has_overlap(c, a):
            return s + 1000.0 * safe_penalty(c, a, s)
        return s

    cur_score = score(cur_c, cur_a)

    for t in range(max_iter):
        i = rng.randrange(n)
        oldc = cur_c[i]
        olda = cur_a[i]

        mode = rng.random()
        if mode < 0.45:
            dx = (rng.random() * 2.0 - 1.0) * step_xy
            dy = (rng.random() * 2.0 - 1.0) * step_xy
            cur_c[i] = (oldc[0] + dx, oldc[1] + dy)
        elif mode < 0.80:
            da = (rng.random() * 2.0 - 1.0) * step_a
            cur_a[i] = normalize_angle(olda + da)
        else:
            cur_a[i] = normalize_angle(olda + math.pi + (rng.random() - 0.5) * 0.14)

        cur_c = center_to_origin(cur_c)
        cur_c = repair_scale(cur_c, cur_a)
        sc = score(cur_c, cur_a)

        accept = False
        if sc < cur_score:
            accept = True
        else:
            temp = max(1e-4, 0.04 * (1.0 - t / max_iter))
            if rng.random() < math.exp(-(sc - cur_score) / temp):
                accept = True

        if accept:
            cur_score = sc
            cur_s = enclosing_side(cur_c, cur_a)
            if not has_overlap(cur_c, cur_a) and cur_s < best[2]:
                best = (cur_c[:], cur_a[:], cur_s)
        else:
            cur_c[i] = oldc
            cur_a[i] = olda

        if (t + 1) % 800 == 0:
            step_xy *= 0.84
            step_a *= 0.90

    return best


def pair_adjustment_pass(centers, angles, seed=0):
    rng = random.Random(9001 + seed)
    n = len(centers)
    if n <= 1:
        return centers, angles, enclosing_side(centers, angles)

    best_c = centers[:]
    best_a = angles[:]
    best_s = enclosing_side(best_c, best_a)

    for _ in range(80):
        c = best_c[:]
        a = best_a[:]
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        dx = (rng.random() * 2.0 - 1.0) * 0.08
        dy = (rng.random() * 2.0 - 1.0) * 0.08
        da = (rng.random() * 2.0 - 1.0) * 0.18
        c[i] = (c[i][0] + dx, c[i][1] + dy)
        c[j] = (c[j][0] - dx, c[j][1] - dy)
        a[i] = normalize_angle(a[i] + da)
        a[j] = normalize_angle(a[j] - da)
        c = center_to_origin(c)
        c = repair_scale(c, a)
        s = enclosing_side(c, a)
        if not has_overlap(c, a) and s < best_s:
            best_c, best_a, best_s = c, a, s

    return best_c, best_a, best_s


def pack(n):
    if n <= 0:
        return [], [], 0.0

    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    best = None

    seeds = [0, 1, 2, 3, 4, 7, 11, 19, 29, 41]
    for seed in seeds:
        for centers, angles in init_layouts(n, seed):
            centers = center_to_origin(centers)
            centers = repair_scale(centers, angles)
            s = enclosing_side(centers, angles)

            if _spo is not None and n <= 14:
                centers2, angles2, s2 = pack_with_scipy(centers, angles, maxiter=500)
                if best is None or s2 < best[2]:
                    best = (centers2, angles2, s2)
            else:
                centers2, angles2, s2 = local_random_improve(
                    centers, angles, seed=seed, max_iter=5000 if n <= 12 else 3000
                )
                centers2, angles2, s2 = pair_adjustment_pass(centers2, angles2, seed=seed)
                centers2 = center_to_origin(centers2)
                centers2 = repair_scale(centers2, angles2)
                s2 = enclosing_side(centers2, angles2)
                if best is None or s2 < best[2]:
                    best = (centers2, angles2, s2)

    centers, angles, s = best

    # Final tightening loop: optimize scale through repeated local search.
    for k in range(3):
        centers, angles, s = pair_adjustment_pass(centers, angles, seed=100 + k)
        centers, angles, s = local_random_improve(
            centers, angles, seed=200 + k, max_iter=1800 if n <= 12 else 1000
        )
        centers = center_to_origin(centers)
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)

    # Absolute validity check fallback.
    if has_overlap(centers, angles):
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)

    return centers, angles, s
