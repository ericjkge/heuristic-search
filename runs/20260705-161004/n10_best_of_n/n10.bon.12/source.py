"""Improved seed program: pack n unit regular pentagons into a square, minimizing side s.

Contract: pack(n) -> (centers, angles, s). Container = origin-centered
axis-aligned square, side s: point p inside iff max(|px|, |py|) <= s/2.

Strategy:
- Build good geometric initial layouts from a tilted rectangular lattice of opposite
  orientations (a double-lattice style motif), with a compact centered arrangement.
- For small n, search a family of hand-tuned low-diameter templates.
- Use local numerical optimization on centers, angles, and scale to shrink the square.
- Keep every pentagon non-overlapping and inside the square.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
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
    ) if centers else 0.0


# -------------------- geometry / collision --------------------

EPS = 1e-10


def _poly_axes(poly):
    axes = []
    for i in range(5):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
        ux, uy = -(y2 - y1), (x2 - x1)
        n = math.hypot(ux, uy)
        if n > 0:
            axes.append((ux / n, uy / n))
    return axes


def pentagons_overlap(pa, pb):
    """Separating-axis test; returns True if polygons overlap or touch."""
    for poly in (pa, pb):
        for ux, uy in _poly_axes(poly):
            amin = min(x * ux + y * uy for x, y in pa)
            amax = max(x * ux + y * uy for x, y in pa)
            bmin = min(x * ux + y * uy for x, y in pb)
            bmax = max(x * ux + y * uy for x, y in pb)
            if min(amax, bmax) - max(amin, bmin) <= EPS:
                return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if pentagons_overlap(polys[i], polys[j]):
                return True
    return False


def inside_square(centers, angles, s):
    h = s / 2.0 + 1e-12
    for (cx, cy), a in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, a):
            if max(abs(vx), abs(vy)) > h:
                return False
    return True


def feasible(centers, angles, s):
    return (not has_overlap(centers, angles)) and inside_square(centers, angles, s)


# -------------------- template generation --------------------

def compact_lattice_template(n, cols=None, row_shift=0.5, angle_mode="alternate"):
    """Rectangular arrangement with opposite orientations.
    row_shift controls alternating horizontal offset per row.
    """
    if n == 0:
        return [], [], 0.0

    # Rough geometric extents of a unit pentagon.
    # In a "point-up" orientation, width is about 1.618 and height about 1.539.
    width = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
    height = R + (SIDE / (2.0 * math.tan(math.pi / 5.0)))

    # Slightly compact but safe base pitches; optimization will refine.
    pitch_x = width * 0.96
    pitch_y = height * 0.93

    if cols is None:
        cols = max(1, int(round(math.sqrt(n))))
    rows = int(math.ceil(n / cols))

    centers = []
    angles = []
    for k in range(n):
        i = k % cols
        j = k // cols
        x = (i - (cols - 1) / 2.0) * pitch_x
        if row_shift:
            x += (0.5 if (j % 2 == 1) else 0.0) * row_shift * pitch_x
        y = (j - (rows - 1) / 2.0) * pitch_y

        if angle_mode == "alternate":
            ang = math.pi / 2.0 if ((i + j) % 2 == 0) else -math.pi / 2.0
        elif angle_mode == "row":
            ang = math.pi / 2.0 if (j % 2 == 0) else -math.pi / 2.0
        else:
            ang = 0.0

        centers.append((x, y))
        angles.append(ang)

    # Center exactly at origin.
    mx = sum(x for x, _ in centers) / n
    my = sum(y for _, y in centers) / n
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def small_n_templates(n):
    """Hand-tuned compact templates for small n."""
    if n == 1:
        return [(0.0, 0.0)], [math.pi / 2.0]

    if n == 2:
        d = 1.55
        return [(-d / 2, 0.0), (d / 2, 0.0)], [math.pi / 2.0, -math.pi / 2.0]

    if n == 3:
        return [(-1.25, -0.35), (1.25, -0.35), (0.0, 1.05)], [
            math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0
        ]

    if n == 4:
        centers = [(-1.15, -0.90), (1.15, -0.90), (-1.15, 0.90), (1.15, 0.90)]
        angles = [math.pi / 2.0, -math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0]
        return centers, angles

    if n == 5:
        centers = [(-1.2, -1.1), (1.2, -1.1), (-1.35, 0.15), (1.35, 0.15), (0.0, 1.25)]
        angles = [math.pi / 2.0, -math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0, math.pi / 2.0]
        return centers, angles

    if n == 6:
        centers = [(-1.35, -1.05), (1.35, -1.05), (-1.2, 0.2),
                   (1.2, 0.2), (-0.65, 1.25), (0.65, 1.25)]
        angles = [math.pi / 2.0, -math.pi / 2.0, -math.pi / 2.0,
                  math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0]
        return centers, angles

    return None, None


def make_initial_candidates(n):
    cands = []

    cs, angs = small_n_templates(n)
    if cs is not None:
        cands.append((cs, angs))

    for cols in range(1, min(n, int(math.ceil(math.sqrt(n))) + 3) + 1):
        for rs in (0.0, 0.35, 0.5, 0.65):
            for mode in ("alternate", "row"):
                cands.append(compact_lattice_template(n, cols=cols, row_shift=rs, angle_mode=mode))

    # A few random perturbations of the best structured patterns.
    random.seed(12345 + n)
    base = cands[:]
    for centers, angles in base[: min(len(base), 8)]:
        for _ in range(3):
            cc = [(x + random.uniform(-0.05, 0.05), y + random.uniform(-0.05, 0.05)) for x, y in centers]
            aa = [a + random.uniform(-0.08, 0.08) for a in angles]
            cands.append((cc, aa))

    return cands


# -------------------- optimization --------------------

def normalize(centers, angles):
    if not centers:
        return centers, angles
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def scale_about_origin(centers, factor):
    return [(x * factor, y * factor) for x, y in centers]


def objective(centers, angles, s, w_overlap=2e4, w_out=2e4):
    """Penalized objective for optimization."""
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    h = s / 2.0
    val = s

    # Boundary penalty
    for poly in polys:
        for x, y in poly:
            dx = abs(x) - h
            dy = abs(y) - h
            if dx > 0:
                val += w_out * dx * dx
            if dy > 0:
                val += w_out * dy * dy

    # Overlap penalty
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            pa, pb = polys[i], polys[j]
            sep = 1e9
            for poly in (pa, pb):
                for ux, uy in _poly_axes(poly):
                    amin = min(x * ux + y * uy for x, y in pa)
                    amax = max(x * ux + y * uy for x, y in pa)
                    bmin = min(x * ux + y * uy for x, y in pb)
                    bmax = max(x * ux + y * uy for x, y in pb)
                    overlap = min(amax, bmax) - max(amin, bmin)
                    sep = min(sep, overlap)
            if sep > 0:
                val += w_overlap * sep * sep
    return val


def local_refine(centers, angles, time_budget=2.0):
    """Simple coordinate descent + shrinking."""
    import time
    start = time.time()

    centers, angles = normalize(list(centers), list(angles))
    s = enclosing_side(centers, angles)
    if s <= 0:
        return centers, angles, s

    # First make it feasible by uniform expansion if needed.
    if not feasible(centers, angles, s):
        lo, hi = 1.0, 1.2
        while not feasible(scale_about_origin(centers, hi), angles, s * hi):
            lo = hi
            hi *= 1.15
            if hi > 50:
                break
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if feasible(scale_about_origin(centers, mid), angles, s * mid):
                hi = mid
            else:
                lo = mid
        centers = scale_about_origin(centers, hi)
        s *= hi

    best_c = centers[:]
    best_a = angles[:]
    best_s = enclosing_side(best_c, best_a)

    # Multi-scale coordinate descent.
    step_pos = max(0.05, best_s * 0.03)
    step_ang = 0.12
    shrink = 0.985

    while time.time() - start < time_budget:
        improved = False

        # Try a systematic sweep over variables.
        for idx in range(len(best_c)):
            if time.time() - start >= time_budget:
                break

            x, y = best_c[idx]
            a = best_a[idx]

            candidates = [
                (x, y, a),
                (x + step_pos, y, a),
                (x - step_pos, y, a),
                (x, y + step_pos, a),
                (x, y - step_pos, a),
                (x + step_pos, y + step_pos, a),
                (x - step_pos, y - step_pos, a),
                (x + step_pos, y - step_pos, a),
                (x - step_pos, y + step_pos, a),
                (x, y, a + step_ang),
                (x, y, a - step_ang),
                (x, y, a + math.pi),
            ]

            cur_best = None
            cur_score = None
            for nx, ny, na in candidates:
                c2 = best_c[:]
                a2 = best_a[:]
                c2[idx] = (nx, ny)
                a2[idx] = na
                c2, a2 = normalize(c2, a2)
                ss = enclosing_side(c2, a2)
                sc = objective(c2, a2, ss)
                if cur_best is None or sc < cur_score:
                    cur_best = (c2, a2, ss)
                    cur_score = sc

            if cur_score is not None and cur_score < objective(best_c, best_a, best_s):
                best_c, best_a, best_s = cur_best
                improved = True

        # Try a shrinking move around origin.
        if time.time() - start < time_budget:
            lam = 0.995
            c2 = scale_about_origin(best_c, lam)
            s2 = enclosing_side(c2, best_a)
            if feasible(c2, best_a, s2):
                best_c, best_a, best_s = c2, best_a, s2
                improved = True

        step_pos *= shrink
        step_ang *= 0.995

        if not improved and step_pos < 1e-4:
            break

    # Final binary search shrink on scale only.
    if feasible(best_c, best_a, best_s):
        lo, hi = 0.98, 1.0
        base_c = best_c[:]
        for _ in range(35):
            mid = (lo + hi) / 2.0
            c2 = scale_about_origin(base_c, mid)
            s2 = enclosing_side(c2, best_a)
            if feasible(c2, best_a, s2):
                hi = mid
            else:
                lo = mid
        best_c = scale_about_origin(base_c, hi)
        best_s = enclosing_side(best_c, best_a)

    return best_c, best_a, best_s


def best_of_candidates(n):
    best = None
    best_s = float("inf")

    cands = make_initial_candidates(n)

    # Prefer more "double-lattice" looking patterns for larger n.
    for centers, angles in cands:
        centers, angles = normalize(centers, angles)
        s0 = enclosing_side(centers, angles)
        if s0 <= 0:
            continue

        # quick feasibility repair by isotropic dilation
        if not feasible(centers, angles, s0):
            lam = 1.0
            while not feasible(scale_about_origin(centers, lam), angles, s0 * lam):
                lam *= 1.08
                if lam > 20:
                    break
            centers = scale_about_origin(centers, lam)
            s0 = enclosing_side(centers, angles)

        # Light refinement on each candidate.
        t = 0.2 if n <= 6 else 0.9
        c2, a2, s2 = local_refine(centers, angles, time_budget=t)
        if s2 < best_s and feasible(c2, a2, s2):
            best = (c2, a2, s2)
            best_s = s2

    return best


# -------------------- public API --------------------

def pack(n):
    """Return centers, angles, and side length s for a valid packing."""
    if n <= 0:
        return [], [], 0.0

    # Special case tiny inputs.
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    # First take the best candidate from a diverse family.
    best = best_of_candidates(n)
    if best is None:
        # Fallback: simple compact lattice.
        centers, angles = compact_lattice_template(n)
        centers, angles, s = local_refine(centers, angles, time_budget=1.0)
        return centers, angles, s

    centers, angles, s = best

    # Final polish with a bit more time for larger n.
    centers, angles, s = local_refine(centers, angles, time_budget=2.0 if n <= 12 else 4.0)

    # Guarantee validity; if something went wrong, enlarge slightly until valid.
    if not feasible(centers, angles, s):
        lam = 1.0
        while not feasible(scale_about_origin(centers, lam), angles, s * lam):
            lam *= 1.01
            if lam > 20:
                break
        centers = scale_about_origin(centers, lam)
        s = enclosing_side(centers, angles)

    return centers, angles, s


if __name__ == "__main__":
    # Tiny self-check
    for n in [1, 2, 3, 4, 5, 6, 10]:
        c, a, s = pack(n)
        print(n, round(s, 6), feasible(c, a, s))
