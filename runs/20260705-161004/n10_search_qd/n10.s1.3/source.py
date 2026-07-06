"""Pack n unit regular pentagons into the smallest possible origin-centered square.

Contract: pack(n) -> (centers, angles, s)

Container convention:
    The square is axis-aligned and centered at the origin. A point p is inside iff
    max(|px|, |py|) <= s/2.

This implementation uses a small search over structured packing templates, then
applies a numerical improvement phase (Nelder-Mead when SciPy is available; a
fallback coordinate search otherwise). It keeps all pentagons non-overlapping and
inside the square.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius ~ 0.8507
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # ~ 0.6882
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # widest extent (diagonal) ~ 1.618
HEIGHT = R + APOTHEM                              # point-up bounding height ~ 1.539

TAU = 2.0 * math.pi
EPS = 1e-9


def pentagon_vertices(cx, cy, angle):
    return [
        (cx + R * math.cos(angle + k * TAU / 5.0),
         cy + R * math.sin(angle + k * TAU / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, ang)
    ) if centers else 0.0


def poly_bounds(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def polygons_overlap(pa, pb):
    # Separating Axis Theorem; if any axis separates, they do not overlap.
    for poly in (pa, pb):
        for i in range(5):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
            ux, uy = -(y2 - y1), (x2 - x1)
            amin = min(x * ux + y * uy for x, y in pa)
            amax = max(x * ux + y * uy for x, y in pa)
            bmin = min(x * ux + y * uy for x, y in pb)
            bmax = max(x * ux + y * uy for x, y in pb)
            if min(amax, bmax) - max(amin, bmin) <= EPS * math.hypot(ux, uy):
                return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, ang) for (cx, cy), ang in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False


def valid_packing(centers, angles):
    if has_overlap(centers, angles):
        return False
    s = enclosing_side(centers, angles)
    half = s / 2.0 + 1e-10
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            if abs(vx) > half or abs(vy) > half:
                return False
    return True


def square_span(centers, angles):
    if not centers:
        return 0.0
    half = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            half = max(half, abs(vx), abs(vy))
    return 2.0 * half


def shift_centers(centers, dx, dy):
    return [(x + dx, y + dy) for x, y in centers]


def translate_to_origin(centers, angles):
    # Center the bounding box as much as possible while preserving validity.
    halfx = halfy = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            halfx = max(halfx, abs(vx))
            halfy = max(halfy, abs(vy))
    dx = -(max(v[0] for c in centers for v in pentagon_vertices(c[0], c[1], angles[centers.index(c)])) +
           min(v[0] for c in centers for v in pentagon_vertices(c[0], c[1], angles[centers.index(c)]))) / 2.0
    dy = -(max(v[1] for c in centers for v in pentagon_vertices(c[0], c[1], angles[centers.index(c)])) +
           min(v[1] for c in centers for v in pentagon_vertices(c[0], c[1], angles[centers.index(c)]))) / 2.0
    return shift_centers(centers, dx, dy)


def objective_from_params(params, base_centers, base_angles, mode):
    # params: [dx, dy, theta, sx, sy, ...] depending on mode.
    if mode == "rigid":
        dx, dy, theta = params
        ct, st = math.cos(theta), math.sin(theta)
        centers = []
        angles = []
        for (x, y), a in zip(base_centers, base_angles):
            xr = ct * x - st * y + dx
            yr = st * x + ct * y + dy
            centers.append((xr, yr))
            angles.append(a + theta)
        return centers, angles
    elif mode == "rigid_mirror":
        dx, dy, theta = params
        ct, st = math.cos(theta), math.sin(theta)
        centers = []
        angles = []
        for (x, y), a in zip(base_centers, base_angles):
            # mirror x -> -x, then rotate and translate
            xm = -x
            xr = ct * xm - st * y + dx
            yr = st * xm + ct * y + dy
            centers.append((xr, yr))
            angles.append(-a + theta)
        return centers, angles
    elif mode == "affine":
        dx, dy, theta, sx, sy = params
        ct, st = math.cos(theta), math.sin(theta)
        centers = []
        angles = []
        for (x, y), a in zip(base_centers, base_angles):
            x2 = sx * x
            y2 = sy * y
            xr = ct * x2 - st * y2 + dx
            yr = st * x2 + ct * y2 + dy
            centers.append((xr, yr))
            angles.append(a + theta)
        return centers, angles
    else:
        raise ValueError("unknown mode")


def repair_by_scaling(centers, angles):
    if not has_overlap(centers, angles):
        return centers, angles
    lo, hi = 1.0, 1.2

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    for _ in range(30):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.5
    else:
        return centers, angles

    for _ in range(50):
        mid = (lo + hi) / 2.0
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi), angles


def make_base_pattern(n, variant=0):
    # Structured templates designed to introduce boundary utilization, tilt, and
    # opposite orientations. The goal is to get a good starting configuration for
    # local improvement.
    centers = []
    angles = []

    if n <= 0:
        return centers, angles

    # Several row/column layouts with alternating orientations and staggered rows.
    if variant % 4 == 0:
        cols = max(1, int(math.ceil(math.sqrt(n * 1.15))))
        rows = int(math.ceil(n / cols))
        dx = 0.82 * WIDTH
        dy = 0.84 * HEIGHT
        for k in range(n):
            i, j = k % cols, k // cols
            x = (i - (cols - 1) / 2.0) * dx
            y = (j - (rows - 1) / 2.0) * dy + (0.22 * dy if j % 2 else 0.0)
            centers.append((x, y))
            angles.append((math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0))
    elif variant % 4 == 1:
        # Two interleaved diagonal chains.
        cols = max(2, int(math.ceil(math.sqrt(n))))
        rows = int(math.ceil(n / cols))
        dx = 0.74 * WIDTH
        dy = 0.70 * HEIGHT
        for k in range(n):
            i, j = k % cols, k // cols
            x = (i - (cols - 1) / 2.0) * dx + (0.32 * dx if j % 2 else -0.32 * dx)
            y = (j - (rows - 1) / 2.0) * dy
            centers.append((x, y))
            angles.append((-math.pi / 2.0 if (i + j) % 3 == 0 else math.pi / 2.0))
    elif variant % 4 == 2:
        # Hex-like rings in square projection.
        r0 = 0.72 * WIDTH
        layers = []
        m = 0
        while m < n:
            cnt = 1 if not layers else 6 * len(layers)
            layers.append(cnt)
            m += cnt
        pts = []
        rad = 0.0
        total = 0
        for layer, cnt in enumerate(layers):
            rad = 0.86 * layer * 0.9
            for t in range(cnt):
                if total >= n:
                    break
                ang = TAU * t / cnt + (math.pi / 10.0 if layer % 2 else 0.0)
                x = rad * math.cos(ang)
                y = rad * math.sin(ang)
                pts.append((x, y))
                total += 1
        centers = pts[:n]
        angles = [math.pi / 2.0 if i % 2 == 0 else -math.pi / 2.0 for i in range(n)]
    else:
        # Cross + boundary-biased perimeter points.
        ring = max(4, int(math.ceil(0.6 * n)))
        inner = n - ring
        for i in range(ring):
            t = TAU * i / ring
            x = 0.92 * math.cos(t)
            y = 0.92 * math.sin(t)
            centers.append((x, y))
            angles.append(math.pi / 2.0 if i % 2 == 0 else -math.pi / 2.0)
        for i in range(inner):
            x = (i - (inner - 1) / 2.0) * 0.72
            y = 0.0 if i % 2 == 0 else 0.36
            centers.append((x, y))
            angles.append(-math.pi / 2.0 if i % 2 == 0 else math.pi / 2.0)

    return centers[:n], angles[:n]


def score_solution(centers, angles):
    if not valid_packing(centers, angles):
        return float("inf")
    return enclosing_side(centers, angles)


def local_improve(centers, angles, time_budget=2.0):
    # A simple, robust coordinate descent over translations and rotations.
    start = __import__("time").time()
    best_c = list(centers)
    best_a = list(angles)
    best_s = score_solution(best_c, best_a)

    step_xy = best_s * 0.02 if best_s > 0 else 0.1
    step_th = 0.08
    rng = random.Random(0xC0FFEE)

    while __import__("time").time() - start < time_budget:
        improved = False
        idxs = list(range(len(best_c)))
        rng.shuffle(idxs)

        for idx in idxs:
            for dtheta in (0.0, step_th, -step_th, 2 * step_th, -2 * step_th):
                for dx, dy in ((0.0, 0.0), (step_xy, 0.0), (-step_xy, 0.0), (0.0, step_xy), (0.0, -step_xy)):
                    cand_c = best_c[:]
                    cand_a = best_a[:]
                    x, y = cand_c[idx]
                    cand_c[idx] = (x + dx, y + dy)
                    cand_a[idx] = cand_a[idx] + dtheta

                    if valid_packing(cand_c, cand_a):
                        s = enclosing_side(cand_c, cand_a)
                        if s + 1e-12 < best_s:
                            best_c, best_a, best_s = cand_c, cand_a, s
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

        if not improved:
            step_xy *= 0.72
            step_th *= 0.72
            if step_xy < 1e-5:
                break

    return best_c, best_a


def global_refine(centers, angles):
    # Try SciPy if available, otherwise a deterministic fallback.
    try:
        import numpy as np
        from scipy.optimize import minimize
        use_scipy = True
    except Exception:
        use_scipy = False

    if not centers:
        return centers, angles

    # Fit a few global transformation modes to reduce square span.
    modes = ["rigid", "rigid_mirror", "affine"]
    best = (centers, angles, score_solution(centers, angles))

    for mode in modes:
        base_c = list(centers)
        base_a = list(angles)

        if mode == "rigid":
            x0 = [0.0, 0.0, 0.0]
        elif mode == "rigid_mirror":
            x0 = [0.0, 0.0, 0.0]
        else:
            x0 = [0.0, 0.0, 0.0, 1.0, 1.0]

        def obj(x):
            cand_c, cand_a = objective_from_params(list(x), base_c, base_a, mode)
            s = score_solution(cand_c, cand_a)
            if s == float("inf"):
                # Penalize invalids based on rough span and overlap.
                return 1e6
            return s

        if use_scipy:
            res = minimize(obj, x0, method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-6})
            xbest = res.x
        else:
            # Very small coordinate search fallback.
            xbest = list(x0)
            step = [0.05] * len(xbest)
            for _ in range(120):
                improved = False
                for i in range(len(xbest)):
                    for d in (-step[i], step[i]):
                        trial = xbest[:]
                        trial[i] += d
                        if obj(trial) < obj(xbest) - 1e-12:
                            xbest = trial
                            improved = True
                if not improved:
                    step = [v * 0.7 for v in step]
                    if max(step) < 1e-4:
                        break

        cand_c, cand_a = objective_from_params(list(xbest), base_c, base_a, mode)
        cand_c, cand_a = repair_by_scaling(cand_c, cand_a)
        s = score_solution(cand_c, cand_a)
        if s < best[2]:
            best = (cand_c, cand_a, s)

    return best[0], best[1]


def pack(n):
    if n <= 0:
        return [], [], 0.0

    # Search a small set of structured starts and keep the best after local refinement.
    starts = []
    for v in range(8):
        c, a = make_base_pattern(n, v)
        c, a = repair_by_scaling(c, a)
        starts.append((c, a))

    best_centers, best_angles, best_s = None, None, float("inf")

    # First pass: deterministic local improvement.
    for c, a in starts:
        c2, a2 = local_improve(c, a, time_budget=0.35)
        s2 = score_solution(c2, a2)
        if s2 < best_s:
            best_centers, best_angles, best_s = c2, a2, s2

    # Second pass: global refinement from the best candidate.
    best_centers, best_angles = global_refine(best_centers, best_angles)
    best_centers, best_angles = local_improve(best_centers, best_angles, time_budget=1.0)

    # Final safety check and recentring.
    if not valid_packing(best_centers, best_angles):
        # Fall back to the best original repaired start if something went wrong.
        for c, a in starts:
            if valid_packing(c, a):
                s = score_solution(c, a)
                if s < best_s:
                    best_centers, best_angles, best_s = c, a, s
        if not valid_packing(best_centers, best_angles):
            best_centers, best_angles = make_base_pattern(n, 0)

    # Shift to keep the square centered around the packing extent.
    # (The container is origin-centered; shifting the configuration preserves validity.)
    min_x = min(vx for (cx, cy), ang in zip(best_centers, best_angles) for vx, vy in pentagon_vertices(cx, cy, ang))
    max_x = max(vx for (cx, cy), ang in zip(best_centers, best_angles) for vx, vy in pentagon_vertices(cx, cy, ang))
    min_y = min(vy for (cx, cy), ang in zip(best_centers, best_angles) for vx, vy in pentagon_vertices(cx, cy, ang))
    max_y = max(vy for (cx, cy), ang in zip(best_centers, best_angles) for vx, vy in pentagon_vertices(cx, cy, ang))
    dx = -(min_x + max_x) / 2.0
    dy = -(min_y + max_y) / 2.0
    best_centers = shift_centers(best_centers, dx, dy)

    # Ensure final validity; if translation introduced numerical issues, keep original.
    if not valid_packing(best_centers, best_angles):
        best_centers, best_angles = starts[0]
        if not valid_packing(best_centers, best_angles):
            best_centers, best_angles = repair_by_scaling(best_centers, best_angles)

    s = enclosing_side(best_centers, best_angles)
    return best_centers, best_angles, s
