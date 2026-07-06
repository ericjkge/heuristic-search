"""Heuristic optimizer for packing n unit regular pentagons into the smallest
origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s).
Container: point p is inside iff max(|px|, |py|) <= s/2.

Strategy:
- Use a small library of hand-crafted motif layouts inspired by dense pentagon packings.
- For each motif, build a candidate arrangement with mixed/opposite orientations.
- Refine with a penalty-based stochastic optimizer that simultaneously shrinks the
  enclosing square while pushing overlaps out.
- Keep the best valid packing found within the time budget.

This is a self-contained pure-Python solver; it does not depend on SciPy.
"""

import math
import random
import time

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # inradius
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # width of a unit pentagon
HEIGHT = R + APOTHEM                               # height of a point-up orientation

TAU = 2.0 * math.pi


def pentagon_vertices(cx, cy, angle):
    ca = math.cos(angle)
    sa = math.sin(angle)
    # Unrolled for speed
    verts = []
    for k in range(5):
        th = angle + k * TAU / 5.0
        verts.append((cx + R * math.cos(th), cy + R * math.sin(th)))
    return verts


def enclosing_side(centers, angles):
    m = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            t = max(abs(vx), abs(vy))
            if t > m:
                m = t
    return 2.0 * m


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


def polygons_overlap(pa, pb, eps=1e-9):
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
        pi = polys[i]
        for j in range(i + 1, m):
            if polygons_overlap(pi, polys[j]):
                return True
    return False


def clip_to_square(centers, angles, s):
    half = s * 0.5
    new_centers = []
    for (cx, cy), ang in zip(centers, angles):
        verts = pentagon_vertices(cx, cy, ang)
        minx = min(vx for vx, vy in verts)
        maxx = max(vx for vx, vy in verts)
        miny = min(vy for vx, vy in verts)
        maxy = max(vy for vx, vy in verts)
        dx = 0.0
        dy = 0.0
        if minx < -half:
            dx += -half - minx
        if maxx > half:
            dx -= maxx - half
        if miny < -half:
            dy += -half - miny
        if maxy > half:
            dy -= maxy - half
        new_centers.append((cx + dx, cy + dy))
    return new_centers


def repair_scale(centers, angles):
    """Scale centers about origin until all overlaps vanish."""
    if not has_overlap(centers, angles):
        return centers

    def scaled(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.15
    for _ in range(70):
        if not has_overlap(scaled(hi), angles):
            break
        lo, hi = hi, hi * 1.22
    else:
        return centers

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if has_overlap(scaled(mid), angles):
            lo = mid
        else:
            hi = mid
    return scaled(hi)


def normalize_angles(angles):
    return [((a + math.pi) % TAU) - math.pi for a in angles]


def base_angles(n, rng, mode=0):
    """Mixed-orientation seeds around the two 'opposite' motifs."""
    angles = []
    for k in range(n):
        if mode == 0:
            a = math.pi / 2.0 if (k % 2 == 0) else -math.pi / 2.0
        elif mode == 1:
            a = math.pi / 2.0 + (math.pi / 5.0 if (k % 3 == 0) else 0.0)
            if k % 2:
                a += math.pi
        elif mode == 2:
            a = -math.pi / 2.0 + (math.pi / 10.0 if (k % 4 in (1, 2)) else 0.0)
            if k % 2 == 0:
                a += math.pi
        else:
            a = (k % 5) * (TAU / 10.0) - math.pi / 2.0
            if k % 2:
                a += math.pi
        a += (rng.random() - 0.5) * 0.18
        angles.append(a)
    return normalize_angles(angles)


def motif_layout(n, kind, rng):
    """Generate initial centers/angles for several packing motifs."""
    centers = []
    angles = base_angles(n, rng, mode=kind % 4)

    if kind == 0:
        # Two-row staggered strip, good for small n and boundary tilt.
        rows = 2 if n > 3 else 1
        cols = int(math.ceil(n / rows))
        px = WIDTH * 0.70
        py = HEIGHT * 0.83
        for k in range(n):
            r = k // cols
            c = k % cols
            x = (c - (cols - 1) / 2.0) * px
            if r % 2:
                x += 0.5 * px
            y = (r - (rows - 1) / 2.0) * py
            if (r + c) % 2:
                angles[k] += math.pi
            centers.append((x, y))

    elif kind == 1:
        # Compact near-square grid with alternating flips.
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        px = WIDTH * 0.68
        py = HEIGHT * 0.75
        for k in range(n):
            c = k % cols
            r = k // cols
            x = (c - (cols - 1) / 2.0) * px
            y = (r - (rows - 1) / 2.0) * py
            if r % 2:
                x += 0.33 * px
            if (r + 2 * c) % 2 == 1:
                angles[k] += math.pi
            centers.append((x, y))

    elif kind == 2:
        # Hex-like row offsets; useful for boundary rows.
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        px = WIDTH * 0.66
        py = HEIGHT * 0.71
        for k in range(n):
            r, c = divmod(k, cols)
            x = (c - (cols - 1) / 2.0) * px
            if r % 2:
                x += 0.5 * px
            y = (r - (rows - 1) / 2.0) * py
            if (r + c) % 3 == 0:
                angles[k] += math.pi
            centers.append((x, y))

    elif kind == 3:
        # Radial shell with central points and opposite-orientation ring.
        if n == 1:
            centers = [(0.0, 0.0)]
        else:
            m = n - 1
            rings = max(1, int(math.ceil(math.sqrt(m / 5.0))))
            placed = 0
            centers.append((0.0, 0.0))
            angles[0] = math.pi / 2.0
            for r in range(1, rings + 1):
                cnt = min(m - placed, 5 * r)
                if cnt <= 0:
                    break
                radius = (R + APOTHEM) * 0.95 * r
                for j in range(cnt):
                    th = TAU * j / cnt + (r % 2) * (TAU / (2.0 * cnt))
                    x = radius * math.cos(th)
                    y = radius * math.sin(th)
                    centers.append((x, y))
                    idx = placed + 1 + j
                    if idx < n:
                        angles[idx] = (math.pi / 2.0 if (j % 2 == 0) else -math.pi / 2.0) + (r % 2) * 0.2
                placed += cnt
                if placed >= m:
                    break

    else:
        # Slanted lattice with alternating opposite orientations.
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        px = WIDTH * 0.64
        py = HEIGHT * 0.69
        shear = 0.18 * px
        for k in range(n):
            r = k // cols
            c = k % cols
            x = (c - (cols - 1) / 2.0) * px + (r - (rows - 1) / 2.0) * shear
            y = (r - (rows - 1) / 2.0) * py
            if r % 2:
                x += 0.45 * px
            if (r + c) % 2 == 0:
                angles[k] += math.pi
            centers.append((x, y))

    centers = repair_scale(centers, angles)
    return centers, normalize_angles(angles)


def score_state(centers, angles):
    s = enclosing_side(centers, angles)
    penalty = 0.0
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    m = len(polys)
    for i in range(m):
        pi = polys[i]
        for j in range(i + 1, m):
            pj = polys[j]
            # overlap depth approximation via separating axes
            ov = 0.0
            for ax, ay in poly_axes(pi) + poly_axes(pj):
                amin, amax = project(pi, ax, ay)
                bmin, bmax = project(pj, ax, ay)
                d = min(amax, bmax) - max(amin, bmin)
                if d <= 0.0:
                    ov = 0.0
                    break
                if d > ov:
                    ov = d
            if ov > 0.0:
                penalty += ov * ov
    return s + 20.0 * penalty, s, penalty


def local_optimize(centers, angles, deadline, rng, rounds=4):
    n = len(centers)
    best_centers = [tuple(c) for c in centers]
    best_angles = angles[:]
    best_score, best_s, _ = score_state(best_centers, best_angles)

    cur_centers = [tuple(c) for c in centers]
    cur_angles = angles[:]
    cur_score, cur_s, _ = score_state(cur_centers, cur_angles)

    step_xy = max(0.015, cur_s * 0.015)
    step_a = 0.22
    temp = 0.03

    for r in range(rounds):
        iters = 2400 if n <= 12 else 3200
        for t in range(iters):
            if time.time() >= deadline:
                break

            i = rng.randrange(n)
            old_c = cur_centers[i]
            old_a = cur_angles[i]

            move = rng.random()
            if move < 0.50:
                dx = (rng.random() * 2.0 - 1.0) * step_xy
                dy = (rng.random() * 2.0 - 1.0) * step_xy
                cur_centers[i] = (old_c[0] + dx, old_c[1] + dy)
            elif move < 0.80:
                cur_angles[i] = old_a + (rng.random() * 2.0 - 1.0) * step_a
            else:
                # paired-ish nudge on opposite orientation tendencies
                j = (i + n // 2) % n
                c2 = cur_centers[j]
                cur_centers[i] = (old_c[0] + (rng.random() - 0.5) * step_xy,
                                  old_c[1] + (rng.random() - 0.5) * step_xy)
                cur_centers[j] = (c2[0] + (rng.random() - 0.5) * step_xy,
                                  c2[1] + (rng.random() - 0.5) * step_xy)
                if rng.random() < 0.5:
                    cur_angles[i] += math.pi
                if rng.random() < 0.5:
                    cur_angles[j] += math.pi

            cur_angles = normalize_angles(cur_angles)

            # Mild recentering to keep origin symmetry useful.
            mx = sum(x for x, y in cur_centers) / n
            my = sum(y for x, y in cur_centers) / n
            if abs(mx) > 1e-12 or abs(my) > 1e-12:
                cur_centers = [(x - mx * 0.15, y - my * 0.15) for x, y in cur_centers]

            # Shrink step by current scale estimate
            cur_centers = clip_to_square(cur_centers, cur_angles, max(cur_s, 1e-9) * 0.999)

            test_centers = repair_scale(cur_centers, cur_angles)
            test_score, test_s, test_pen = score_state(test_centers, cur_angles)

            accept = False
            if test_score < cur_score:
                accept = True
            else:
                delta = test_score - cur_score
                if rng.random() < math.exp(-max(0.0, delta) / max(1e-9, temp)):
                    accept = True

            if accept:
                cur_centers = test_centers
                cur_score = test_score
                cur_s = test_s
                if test_score < best_score and test_pen <= 1e-12:
                    best_centers = [tuple(c) for c in cur_centers]
                    best_angles = cur_angles[:]
                    best_score = test_score
                    best_s = test_s
            else:
                cur_centers[i] = old_c
                cur_angles[i] = old_a
                cur_angles = normalize_angles(cur_angles)

            if (t + 1) % 600 == 0:
                step_xy *= 0.88
                step_a *= 0.92
                temp *= 0.85

        # After each round, do a deterministic tightening pass.
        if time.time() >= deadline:
            break
        cur_centers = repair_scale(cur_centers, cur_angles)
        cur_score, cur_s, cur_pen = score_state(cur_centers, cur_angles)
        if cur_pen <= 1e-12 and cur_s < best_s:
            best_centers = [tuple(c) for c in cur_centers]
            best_angles = cur_angles[:]
            best_score = cur_score
            best_s = cur_s

    return best_centers, best_angles, best_s


def pack(n):
    """Return a valid packing of n unit regular pentagons into an origin-centered square."""
    if n <= 0:
        return [], [], 0.0

    start = time.time()
    deadline = start + 295.0

    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    rng = random.Random(20240517 + 7919 * n)

    # A simple lower bound for initial scale, used to keep candidate magnitudes sane.
    area_lb = math.sqrt(n * 5.0 / (4.0 * math.tan(math.pi / 5.0)))
    best = None

    # Try multiple motif seeds.
    motif_ids = list(range(5))
    rng.shuffle(motif_ids)
    for mid in motif_ids:
        if time.time() >= deadline:
            break
        centers, angles = motif_layout(n, mid, rng)

        # Global scale heuristic: smaller for dense starts, but avoid immediate overlaps.
        s0 = max(area_lb * 1.05, enclosing_side(centers, angles))
        scale = s0 / max(enclosing_side(centers, angles), 1e-9)
        centers = [(x * scale, y * scale) for x, y in centers]

        # Add a tiny symmetry-breaking perturbation
        angles = normalize_angles([a + (rng.random() - 0.5) * 0.08 for a in angles])

        centers = repair_scale(centers, angles)
        centers, angles, s = local_optimize(centers, angles, deadline, rng, rounds=3)

        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)
        if not has_overlap(centers, angles):
            if best is None or s < best[2]:
                best = (centers, angles, s)

    # Fallback: if nothing valid was found, return the best repairable candidate.
    if best is None:
        centers, angles = motif_layout(n, 0, rng)
        centers = repair_scale(centers, angles)
        s = enclosing_side(centers, angles)
        best = (centers, angles, s)

    # Final polishing with a few more passes from the current best.
    centers, angles, s = best
    for extra_seed in range(3):
        if time.time() >= deadline:
            break
        rrng = random.Random(9001 + 97 * n + extra_seed)
        c2, a2, s2 = local_optimize(centers, angles, deadline, rrng, rounds=2)
        c2 = repair_scale(c2, a2)
        s2 = enclosing_side(c2, a2)
        if not has_overlap(c2, a2) and s2 < s:
            centers, angles, s = c2, a2, s2

    return centers, angles, s
