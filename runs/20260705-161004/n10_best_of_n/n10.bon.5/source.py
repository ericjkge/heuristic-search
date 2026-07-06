"""Improved packer for regular pentagons in an origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)

- centers: list of (x, y)
- angles:  list of rotation angles in radians
- s:       side length of the smallest origin-centered axis-aligned square enclosing
           the returned packing
"""

import math

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

TAU = 2.0 * math.pi
EPS = 1e-9
OVERLAP_EPS = 1e-8


def pentagon_vertices(cx, cy, angle):
    return [
        (cx + R * math.cos(angle + k * TAU / 5.0),
         cy + R * math.sin(angle + k * TAU / 5.0))
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    if not centers:
        return 0.0
    return 2.0 * max(
        max(abs(vx), abs(vy))
        for (cx, cy), ang in zip(centers, angles)
        for vx, vy in pentagon_vertices(cx, cy, ang)
    )


def poly_axes(poly):
    axes = []
    for i in range(5):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
        ux, uy = -(y2 - y1), (x2 - x1)
        n = math.hypot(ux, uy)
        if n > 0:
            axes.append((ux / n, uy / n))
    return axes


def project(poly, ax, ay):
    vals = [x * ax + y * ay for x, y in poly]
    return min(vals), max(vals)


def polygons_overlap(pa, pb):
    # SAT with strict separation threshold.
    for ax, ay in poly_axes(pa) + poly_axes(pb):
        amin, amax = project(pa, ax, ay)
        bmin, bmax = project(pb, ax, ay)
        if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS:
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


def repair(centers, angles):
    """Scale centers about the origin until overlaps disappear."""
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

    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def local_refine(centers, angles, iters=250):
    """Small deterministic hill-climb on centers/angles to shrink the enclosing square.

    This is conservative: it only accepts feasible improvements.
    """
    n = len(centers)
    if n == 0:
        return centers, angles

    best_c = [list(p) for p in centers]
    best_a = list(angles)
    best_s = enclosing_side(best_c, best_a)

    # Scale step sizes to the current bounding box.
    xs = [x for x, y in best_c]
    ys = [y for x, y in best_c]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    step_xy = 0.12 * span
    step_ang = 0.18

    # Candidate directions biased toward diagonal compression and wall-tightening.
    dirs = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
    ]
    dirs = [(dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)) for dx, dy in dirs]

    for t in range(iters):
        improved = False
        # Gradually reduce step sizes.
        if t and t % 50 == 0:
            step_xy *= 0.68
            step_ang *= 0.72

        # Try per-piece local moves.
        for i in range(n):
            base_x, base_y = best_c[i]
            base_a = best_a[i]
            trials = []

            # Rotate in both directions.
            trials.append((base_x, base_y, base_a + step_ang))
            trials.append((base_x, base_y, base_a - step_ang))

            # Move along a few useful directions.
            for dx, dy in dirs:
                trials.append((base_x + dx * step_xy, base_y + dy * step_xy, base_a))
                trials.append((base_x + 0.5 * dx * step_xy, base_y + 0.5 * dy * step_xy, base_a + 0.5 * step_ang))
                trials.append((base_x + 0.5 * dx * step_xy, base_y + 0.5 * dy * step_xy, base_a - 0.5 * step_ang))

            for x, y, a in trials:
                cand_c = [p[:] for p in best_c]
                cand_a = best_a[:]
                cand_c[i] = [x, y]
                cand_a[i] = a
                cand_centers = [(p[0], p[1]) for p in cand_c]
                if has_overlap(cand_centers, cand_a):
                    continue
                s = enclosing_side(cand_centers, cand_a)
                if s + 1e-12 < best_s:
                    best_c, best_a, best_s = cand_c, cand_a, s
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try a collective shrink toward the origin.
            shrink = 0.995
            cand_centers = [(x * shrink, y * shrink) for x, y in [(p[0], p[1]) for p in best_c]]
            if not has_overlap(cand_centers, best_a):
                s = enclosing_side(cand_centers, best_a)
                if s + 1e-12 < best_s:
                    best_c = [list(p) for p in cand_centers]
                    best_s = s
                    improved = True

        if not improved and step_xy < 1e-4:
            break

    return [(p[0], p[1]) for p in best_c], best_a


def initial_layout(n):
    """A compact double-lattice style seed.

    The search space is built from paired opposite orientations with slight row staggering.
    For finite n, rows near the boundary are allowed to tilt and compress.
    """
    if n <= 0:
        return [], []

    # Place in rows of paired pentagons, using a staggered triangular backbone.
    # The geometry constants are deliberately conservative; subsequent refinement
    # and repair shrink the packing substantially.
    pair_dx = 0.92 * WIDTH
    pair_dy = 0.92 * HEIGHT
    row_shift = 0.50 * pair_dx

    # Choose row/column counts by trying a few candidates.
    best = None

    candidates = []
    root = int(math.sqrt(n))
    for cols in range(max(1, root - 2), root + 4):
        rows = (n + cols - 1) // cols
        candidates.append((cols, rows))
    # Also try a few near-square factorizations.
    for rows in range(max(1, root - 2), root + 4):
        cols = (n + rows - 1) // rows
        candidates.append((cols, rows))

    seen = set()
    for cols, rows in candidates:
        if (cols, rows) in seen:
            continue
        seen.add((cols, rows))
        centers = []
        angles = []
        idx = 0
        for r in range(rows):
            # Alternate row offsets to encourage interlocking.
            xoff = (r % 2) * row_shift
            y = (r - (rows - 1) / 2.0) * pair_dy
            for c in range(cols):
                if idx >= n:
                    break
                # Opposite orientations are used throughout, with mild boundary variation.
                x = (c - (cols - 1) / 2.0) * pair_dx + xoff
                ang = math.pi / 2.0 if ((r + c) % 2 == 0) else -math.pi / 2.0
                # Boundary rows get a small tilt to better fit square walls.
                if r == 0:
                    ang += 0.09
                elif r == rows - 1:
                    ang -= 0.09
                if c == 0:
                    ang -= 0.05
                elif c == cols - 1:
                    ang += 0.05
                centers.append((x, y))
                angles.append(ang)
                idx += 1

        # Recentering.
        mx = sum(x for x, y in centers) / n
        my = sum(y for x, y in centers) / n
        centers = [(x - mx, y - my) for x, y in centers]

        # Quick objective.
        s = enclosing_side(centers, angles)
        if best is None or s < best[0]:
            best = (s, centers, angles)

    return best[1], best[2]


def pack(n):
    if n <= 0:
        return [], [], 0.0

    # Seed construction.
    centers, angles = initial_layout(n)

    # Repair any overlaps by uniform expansion of centers.
    centers = repair(centers, angles)

    # A few rounds of local improvement, each followed by a repair if needed.
    best_centers = centers
    best_angles = angles
    best_s = enclosing_side(best_centers, best_angles)

    # Deterministic multi-start perturbations: slight global rotation and row tilts.
    starts = []
    starts.append((centers, angles))

    # Additional starts: rotate positions a bit around origin and tweak angles.
    for phi in (0.0, 0.03, -0.03, 0.06, -0.06):
        c2 = []
        for x, y in centers:
            c2.append((x * math.cos(phi) - y * math.sin(phi),
                       x * math.sin(phi) + y * math.cos(phi)))
        a2 = [a + (0.08 if i % 2 == 0 else -0.08) for i, a in enumerate(angles)]
        starts.append((c2, a2))

    for c0, a0 in starts:
        c0 = repair(c0, a0)
        c1, a1 = local_refine(c0, a0, iters=220 if n <= 12 else 160)
        c1 = repair(c1, a1)
        s1 = enclosing_side(c1, a1)
        if s1 < best_s and not has_overlap(c1, a1):
            best_centers, best_angles, best_s = c1, a1, s1

    # Final small compression pass.
    if n > 1:
        for lam in (0.999, 0.997, 0.995, 0.993):
            c_try = [(x * lam, y * lam) for x, y in best_centers]
            if not has_overlap(c_try, best_angles):
                s_try = enclosing_side(c_try, best_angles)
                if s_try < best_s:
                    best_centers, best_s = c_try, s_try

    # Keep the square centered by subtracting centroid if that helps; do not worsen.
    mx = sum(x for x, y in best_centers) / n
    my = sum(y for x, y in best_centers) / n
    c_centered = [(x - mx, y - my) for x, y in best_centers]
    if not has_overlap(c_centered, best_angles):
        s_centered = enclosing_side(c_centered, best_angles)
        if s_centered <= best_s + 1e-12:
            best_centers, best_s = c_centered, s_centered

    return best_centers, best_angles, enclosing_side(best_centers, best_angles)
