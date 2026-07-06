"""Pack n unit regular pentagons into the smallest axis-centered square.

Contract: pack(n) -> (centers, angles, s), where the container is the
origin-centered axis-aligned square of side s: a point p is inside iff
max(|px|, |py|) <= s/2.

This version uses a numerical search over several hand-designed motifs and
then locally improves the best candidate with a coordinate/rotation optimizer.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # inradius
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

PHI = (1.0 + math.sqrt(5.0)) / 2.0
TAU = 2.0 * math.pi


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
    )


# EVOLVE-BLOCK-START
OVERLAP_EPS = 1e-9


def _poly_axes(poly):
    axes = []
    for i in range(5):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
        ux, uy = -(y2 - y1), (x2 - x1)
        norm = math.hypot(ux, uy)
        axes.append((ux / norm, uy / norm))
    return axes


def pentagons_overlap(pa, pb):
    for poly in (pa, pb):
        for ux, uy in _poly_axes(poly):
            aproj = [x * ux + y * uy for x, y in pa]
            bproj = [x * ux + y * uy for x, y in pb]
            if min(max(aproj), max(bproj)) - max(min(aproj), min(bproj)) <= OVERLAP_EPS:
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


def repair_scale(centers, angles):
    if not has_overlap(centers, angles):
        return centers, 1.0

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.05
    while has_overlap(dilated(hi), angles):
        lo, hi = hi, hi * 1.15
        if hi > 20.0:
            return centers, 1.0

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi), hi


def _pack_to_params(params, n):
    centers = []
    angles = []
    for i in range(n):
        x = params[3 * i]
        y = params[3 * i + 1]
        a = params[3 * i + 2]
        centers.append((x, y))
        angles.append(a)
    return centers, angles


def _params_to_pack(centers, angles):
    params = []
    for (x, y), a in zip(centers, angles):
        params.extend([x, y, a])
    return params


def _score(centers, angles):
    if has_overlap(centers, angles):
        return 1e9 + enclosing_side(centers, angles)
    return enclosing_side(centers, angles)


def _random_motif(n, seed=0):
    rnd = random.Random(seed)

    centers = []
    angles = []

    # Several motif families: alternating-lattice, tilted rows, and perturbed
    # double-lattice-inspired arrangements.
    family = seed % 6

    if family == 0:
        # 2-row alternating opposite orientations
        cols = int(math.ceil(n / 2.0))
        dx = 0.86
        dy = 0.74
        for k in range(n):
            r = k // cols
            c = k % cols
            x = (c - (cols - 1) / 2.0) * dx + (0.5 * dy if r % 2 else 0.0)
            y = (r - 0.5) * dy
            centers.append((x, y))
            angles.append((math.pi / 2.0) if (c + r) % 2 == 0 else (-math.pi / 2.0))

    elif family == 1:
        # 3-row compact with slight shear
        rows = 3 if n >= 6 else 2
        cols = int(math.ceil(n / rows))
        dx = 0.84
        dy = 0.66
        shear = 0.22
        for k in range(n):
            r = k // cols
            c = k % cols
            x = (c - (cols - 1) / 2.0) * dx + (r - (rows - 1) / 2.0) * shear
            y = (r - (rows - 1) / 2.0) * dy
            centers.append((x, y))
            angles.append((math.pi / 2.0) if (r + c) % 2 == 0 else (-math.pi / 2.0))

    elif family == 2:
        # Hex-like cloud with opposite orientations
        a = 0.82
        b = 0.72
        pts = []
        for q in range(-4, 5):
            for r in range(-4, 5):
                x = a * (q + 0.5 * r)
                y = b * r
                pts.append((x, y))
        pts.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
        for i in range(n):
            x, y = pts[i]
            centers.append((x, y))
            angles.append((math.pi / 2.0) if i % 2 == 0 else (-math.pi / 2.0))

    elif family == 3:
        # Golden-ratio spiral with alternating orientation
        for i in range(n):
            t = i + 1
            r = 0.33 * math.sqrt(t)
            ang = TAU * (t / PHI)
            x = r * math.cos(ang)
            y = 0.86 * r * math.sin(ang)
            centers.append((x, y))
            angles.append((math.pi / 2.0) if i % 2 == 0 else (-math.pi / 2.0))

    elif family == 4:
        # Two interleaved tilted lines, good for small n boundary fitting
        dx = 0.86
        dy = 0.70
        for i in range(n):
            row = i % 2
            col = i // 2
            x = (col - (math.ceil(n / 2.0) - 1) / 2.0) * dx + (row * 0.38)
            y = (row - 0.5) * dy
            centers.append((x, y))
            angles.append((0.15 if row == 0 else math.pi + 0.15))

    else:
        # Randomized local cluster around a compact grid
        k = int(math.ceil(math.sqrt(n)))
        dx = 0.80
        dy = 0.70
        for i in range(n):
            r = i // k
            c = i % k
            x = (c - (k - 1) / 2.0) * dx + rnd.uniform(-0.05, 0.05)
            y = (r - (k - 1) / 2.0) * dy + rnd.uniform(-0.05, 0.05)
            centers.append((x, y))
            angles.append(rnd.choice([math.pi / 2.0, -math.pi / 2.0, 0.15, math.pi + 0.15]))

    # Center it
    mx = sum(x for x, _ in centers) / n
    my = sum(y for _, y in centers) / n
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def _optimize(centers, angles, time_budget=2.5):
    # Lightweight hill-climbing / coordinate descent with adaptive step.
    start = __import__("time").time()
    n = len(centers)
    params = _params_to_pack(centers, angles)
    best_centers, best_angles = centers, angles
    best = _score(best_centers, best_angles)

    step_pos = 0.12
    step_ang = 0.10
    rnd = random.Random(12345 + n)

    while __import__("time").time() - start < time_budget:
        improved = False
        order = list(range(3 * n))
        rnd.shuffle(order)

        for idx in order:
            old = params[idx]
            if idx % 3 == 2:
                candidates = [old - step_ang, old + step_ang, old - 0.5 * step_ang, old + 0.5 * step_ang]
            else:
                candidates = [old - step_pos, old + step_pos, old - 0.5 * step_pos, old + 0.5 * step_pos]

            local_best_val = best
            local_best = old
            local_best_pack = None

            for cand in candidates:
                params[idx] = cand
                c, a = _pack_to_params(params, n)
                val = _score(c, a)
                if val < local_best_val:
                    local_best_val = val
                    local_best = cand
                    local_best_pack = (c, a)

            params[idx] = local_best
            if local_best_pack is not None and local_best_val + 1e-12 < best:
                best = local_best_val
                best_centers, best_angles = local_best_pack
                improved = True

        # global random jitter around current best
        if not improved:
            for _ in range(max(6, n)):
                trial = params[:]
                for i in range(n):
                    trial[3 * i] += rnd.uniform(-step_pos, step_pos) * 0.4
                    trial[3 * i + 1] += rnd.uniform(-step_pos, step_pos) * 0.4
                    trial[3 * i + 2] += rnd.uniform(-step_ang, step_ang) * 0.4
                c, a = _pack_to_params(trial, n)
                val = _score(c, a)
                if val < best:
                    params = trial
                    best = val
                    best_centers, best_angles = c, a
                    improved = True
                    break

        if not improved:
            step_pos *= 0.82
            step_ang *= 0.82
            if step_pos < 1e-4:
                break

    return best_centers, best_angles


def _valid_solution(centers, angles):
    if has_overlap(centers, angles):
        return False
    s = enclosing_side(centers, angles)
    half = s / 2.0 + 1e-10
    for (cx, cy), a in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, a):
            if max(abs(vx), abs(vy)) > half + 1e-9:
                return False
    return True


def pack(n):
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    # Generate several candidate packings.
    candidates = []
    for seed in range(18):
        centers, angles = _random_motif(n, seed)
        centers, _ = repair_scale(centers, angles)
        candidates.append((centers, angles))

    # Add a few deterministic lattice variants.
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    for shear in (0.0, 0.10, 0.18, 0.25):
        for flip in (False, True):
            centers, angles = [], []
            dx = 0.80
            dy = 0.68
            for k in range(n):
                r = k // cols
                c = k % cols
                x = (c - (cols - 1) / 2.0) * dx + (r - (rows - 1) / 2.0) * shear
                y = (r - (rows - 1) / 2.0) * dy
                centers.append((x, y))
                if flip:
                    angles.append((math.pi / 2.0) if (r + c) % 2 == 0 else (-math.pi / 2.0))
                else:
                    angles.append((0.13 if (r + c) % 2 == 0 else math.pi + 0.13))
            mx = sum(x for x, _ in centers) / n
            my = sum(y for _, y in centers) / n
            centers = [(x - mx, y - my) for x, y in centers]
            centers, _ = repair_scale(centers, angles)
            candidates.append((centers, angles))

    # Optimize each candidate briefly and keep the best valid one.
    best_centers, best_angles = None, None
    best_s = float("inf")

    for i, (centers, angles) in enumerate(candidates):
        c2, a2 = _optimize(centers, angles, time_budget=0.18 if n >= 8 else 0.10)
        c2, scale = repair_scale(c2, a2)
        s = enclosing_side(c2, a2)
        if scale != 1.0:
            s = enclosing_side(c2, a2)
        if s < best_s and _valid_solution(c2, a2):
            best_s = s
            best_centers, best_angles = c2, a2

    # Fallback: ensure validity.
    if best_centers is None:
        best_centers, best_angles = candidates[0]
        best_centers, _ = repair_scale(best_centers, best_angles)
        if has_overlap(best_centers, best_angles):
            # Last resort: blow up slightly until valid.
            lam = 1.0
            while has_overlap([(x * lam, y * lam) for x, y in best_centers], best_angles):
                lam *= 1.05
            best_centers = [(x * lam, y * lam) for x, y in best_centers]

    return best_centers, best_angles, enclosing_side(best_centers, best_angles)
# EVOLVE-BLOCK-END
