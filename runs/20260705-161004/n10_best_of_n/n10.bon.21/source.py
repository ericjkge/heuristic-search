"""Packing 10 unit regular pentagons into the smallest origin-centered square.

Contract: pack(n) -> (centers, angles, s), where the container is the
origin-centered axis-aligned square of side s, i.e.
max(|x|, |y|) <= s/2.

Strategy:
- Use a carefully chosen finite packing template for n=10 based on a
  staggered / mixed-orientation arrangement.
- Optimize the packing numerically with a score that prioritizes:
    1) no overlaps,
    2) small enclosing square,
    3) a mild preference for non-symmetric placements that can beat
       grid-like layouts.
- Keep the container convention and the public signature unchanged.

This module is self-contained and does not depend on scipy.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))


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
    )


OVERLAP_EPS = 1e-9


def _poly_axes(poly):
    axes = []
    for i in range(5):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
        ux, uy = -(y2 - y1), (x2 - x1)
        norm = math.hypot(ux, uy)
        if norm > 0:
            axes.append((ux / norm, uy / norm))
    return axes


def pentagons_overlap(pa, pb):
    """SAT overlap test. Returns True if polygons overlap (or touch within tol)."""
    for poly in (pa, pb):
        for ux, uy in _poly_axes(poly):
            a = [x * ux + y * uy for x, y in pa]
            b = [x * ux + y * uy for x, y in pb]
            if min(max(a), max(b)) - max(min(a), min(b)) <= OVERLAP_EPS:
                return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if pentagons_overlap(polys[i], polys[j]):
                return True
    return False


def repair(centers, angles):
    """If needed, uniformly dilate centers until all overlaps vanish."""
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.2
    for _ in range(40):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.35
    else:
        return centers

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def _packing_energy(centers, angles):
    """Soft objective: smaller square is better, overlaps are heavily penalized."""
    s = enclosing_side(centers, angles)
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]

    # Overlap penalty via minimum separating gap over all axes
    penalty = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            pa, pb = polys[i], polys[j]
            worst = 1e9
            for poly in (pa, pb):
                for ux, uy in _poly_axes(poly):
                    A = [x * ux + y * uy for x, y in pa]
                    B = [x * ux + y * uy for x, y in pb]
                    gap = min(max(A), max(B)) - max(min(A), min(B))
                    worst = min(worst, gap)
            if worst <= 0.0:
                penalty += 1e4 * (-worst + 1e-4) ** 2
            else:
                penalty += 1e-6 / (worst + 1e-9)

    # Mild centering regularizer to keep the square symmetric about origin.
    cx = sum(x for x, _ in centers) / len(centers)
    cy = sum(y for _, y in centers) / len(centers)
    center_pen = 0.03 * (cx * cx + cy * cy)
    return s + penalty + center_pen


def _initial_template(n):
    """Staggered finite template with mixed orientations."""
    # Hand-tuned rows based on point-up / point-down interlocking.
    # The final optimization will move and rotate these.
    if n == 10:
        centers = [
            (-2.10,  0.00),
            (-1.05,  0.00),
            ( 0.00,  0.00),
            ( 1.05,  0.00),
            ( 2.10,  0.00),
            (-1.57,  1.34),
            (-0.52,  1.34),
            ( 0.53,  1.34),
            ( 1.58,  1.34),
            ( 0.00, -1.34),
        ]
        angles = [
            math.pi / 2.0,
            -math.pi / 2.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            math.pi / 2.0,
            -math.pi / 2.0,
        ]
        return centers, angles

    # Generic fallback for other n: two staggered rows.
    cols = int(math.ceil(n / 2.0))
    pitch_x = 1.02 * 1.618033988749895
    pitch_y = 1.02 * (R + APOTHEM)
    centers, angles = [], []
    for k in range(n):
        row = k // cols
        col = k % cols
        x = (col - (cols - 1) / 2.0) * pitch_x + (0.5 * pitch_x if row % 2 else 0.0)
        y = (row - 0.5) * pitch_y
        centers.append((x, y))
        angles.append(math.pi / 2.0 if (k + row) % 2 == 0 else -math.pi / 2.0)
    return centers, angles


def _transform(centers, angles, dx=0.0, dy=0.0, scale=1.0, rot=0.0):
    c, s = math.cos(rot), math.sin(rot)
    out = []
    for (x, y), a in zip(centers, angles):
        xx = scale * (c * x - s * y) + dx
        yy = scale * (s * x + c * y) + dy
        out.append((xx, yy))
    return out, [(a + rot) for a in angles]


def _local_search(centers, angles, iters=12000, seed=0):
    rng = random.Random(seed)
    best_c = [tuple(p) for p in centers]
    best_a = list(angles)
    best_e = _packing_energy(best_c, best_a)

    step_pos = 0.08
    step_ang = 0.12
    step_rot = 0.08
    step_scale = 0.02

    for t in range(iters):
        cand_c = [list(p) for p in best_c]
        cand_a = best_a[:]

        move = rng.random()
        if move < 0.55:
            i = rng.randrange(len(cand_c))
            cand_c[i][0] += rng.uniform(-step_pos, step_pos)
            cand_c[i][1] += rng.uniform(-step_pos, step_pos)
            cand_a[i] += rng.uniform(-step_ang, step_ang)
        elif move < 0.78:
            # Global affine nudge.
            dx = rng.uniform(-step_pos, step_pos) * 0.5
            dy = rng.uniform(-step_pos, step_pos) * 0.5
            rot = rng.uniform(-step_rot, step_rot)
            scale = 1.0 + rng.uniform(-step_scale, step_scale)
            cand_c, cand_a = _transform(cand_c, cand_a, dx=dx, dy=dy, scale=scale, rot=rot)
        else:
            # Swap / flip a small neighborhood of orientations.
            i = rng.randrange(len(cand_a))
            cand_a[i] = -cand_a[i]

        cand_c = [(x, y) for x, y in cand_c]
        cand_c = repair(cand_c, cand_a)
        e = _packing_energy(cand_c, cand_a)

        if e < best_e:
            best_c, best_a, best_e = cand_c, cand_a, e
            step_pos = max(0.01, step_pos * 0.999)
            step_ang = max(0.02, step_ang * 0.999)
            step_rot = max(0.02, step_rot * 0.999)
            step_scale = max(0.005, step_scale * 0.999)
        elif t % 250 == 0:
            # Occasional diversification.
            step_pos = min(0.14, step_pos * 1.02)
            step_ang = min(0.20, step_ang * 1.01)

    return best_c, best_a


def _final_symmetrize(centers, angles):
    """Try a couple of cheap symmetry-preserving transforms and keep the best."""
    best = (centers, angles, enclosing_side(centers, angles))

    candidates = []

    # Identity
    candidates.append((centers, angles))

    # Flip orientations
    candidates.append((centers[:], [-a for a in angles]))

    # Rotate 180 degrees
    c2, a2 = _transform(centers, angles, rot=math.pi)
    candidates.append((c2, a2))

    # Reflect across y-axis and reverse angles
    c3 = [(-x, y) for x, y in centers]
    a3 = [math.pi - a for a in angles]
    candidates.append((c3, a3))

    for c, a in candidates:
        c = repair(c, a)
        s = enclosing_side(c, a)
        if s < best[2] and not has_overlap(c, a):
            best = (c, a, s)

    return best[0], best[1]


def pack(n):
    if n <= 0:
        return [], [], 0.0

    centers, angles = _initial_template(n)

    # A few deterministic restarts with different global transformations.
    restarts = []
    restarts.append((centers, angles))
    restarts.append(_transform(centers, angles, rot=math.pi / 5.0))
    restarts.append(_transform(centers, angles, rot=-math.pi / 7.0, scale=0.985))
    restarts.append(_transform(centers, angles, dx=0.07, dy=-0.05, rot=math.pi / 11.0, scale=1.01))
    restarts.append((centers[:], [a + (math.pi if i % 3 == 0 else 0.0) for i, a in enumerate(angles)]))

    best_c, best_a, best_s = None, None, float("inf")

    for seed, (c0, a0) in enumerate(restarts):
        c0 = repair([tuple(p) for p in c0], list(a0))
        c1, a1 = _local_search(c0, list(a0), iters=9000, seed=seed * 17 + 3)
        c1, a1 = _final_symmetrize(c1, a1)
        c1 = repair(c1, a1)
        s = enclosing_side(c1, a1)
        if (not has_overlap(c1, a1)) and s < best_s:
            best_c, best_a, best_s = c1, a1, s

    # Final cleanup.
    best_c = repair(best_c, best_a)
    best_s = enclosing_side(best_c, best_a)

    return best_c, best_a, best_s


if __name__ == "__main__":
    c, a, s = pack(10)
    print("s =", s)
    print("centers =", c)
    print("angles =", a)
