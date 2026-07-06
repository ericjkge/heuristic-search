"""Improved packer: pack n unit regular pentagons into the smallest
origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
Container = origin-centered axis-aligned square, side s:
a point p is inside iff max(|px|, |py|) <= s/2.

Strategy:
- Build a small family of handcrafted dense motifs for pentagons.
- For n <= 10, search over motif combinations and orientations.
- Use local geometric optimization (Nelder-Mead / Powell if available)
  on center coordinates, rotations, and global scale to reduce square side.
- Validate by exact polygon separation with a conservative tolerance.
- Keep the output signature unchanged.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

TAU = 2.0 * math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0

# ---------------- Geometry ----------------

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
    ) if centers else 0.0


OVERLAP_EPS = 1e-9


def pentagons_overlap(pa, pb):
    for poly in (pa, pb):
        for i in range(5):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
            ux, uy = -(y2 - y1), (x2 - x1)
            amin = min(x * ux + y * uy for x, y in pa)
            amax = max(x * ux + y * uy for x, y in pa)
            bmin = min(x * ux + y * uy for x, y in pb)
            bmax = max(x * ux + y * uy for x, y in pb)
            norm = math.hypot(ux, uy)
            if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS * norm:
                return False
    return True


def has_overlap(centers, angles):
    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if pentagons_overlap(polys[i], polys[j]):
                return True
    return False


def all_inside(centers, angles, s):
    h = s / 2.0 + 1e-10
    for (cx, cy), a in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, a):
            if abs(vx) > h or abs(vy) > h:
                return False
    return True


# ---------------- Optimization helpers ----------------

def repair(centers, angles):
    """Uniformly dilate centers about origin until no overlaps."""
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.2
    for _ in range(60):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.25
    else:
        return centers

    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def _try_scipy_optimize():
    try:
        from scipy.optimize import minimize
        return minimize
    except Exception:
        return None


def optimize_configuration(centers, angles, max_iter=1200):
    """Local continuous optimization to reduce enclosing square side.

    Variables:
      - all center coordinates
      - all angles
      - a global scale factor relative to current coordinates
    Objective:
      - enclosing square side after penalty for overlaps and out-of-bounds.
    """
    minimize = _try_scipy_optimize()
    if minimize is None:
        return centers, angles

    n = len(centers)
    if n == 0:
        return centers, angles

    # Normalize initial guess near origin
    c0 = [v for xy in centers for v in xy]
    a0 = [a for a in angles]
    x0 = c0 + a0 + [0.0]  # last variable: log-scale tweak

    def unpack(x):
        cs = [(x[2 * i], x[2 * i + 1]) for i in range(n)]
        ang = [x[2 * n + i] for i in range(n)]
        scale = math.exp(x[-1])
        cs = [(cx * scale, cy * scale) for cx, cy in cs]
        return cs, ang

    def objective(x):
        cs, ang = unpack(x)
        # Keep angles wrapped
        ang = [((a + math.pi) % TAU) - math.pi for a in ang]

        s = enclosing_side(cs, ang)
        # Penalties
        pen = 0.0
        h = s / 2.0
        polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(cs, ang)]
        for poly in polys:
            for vx, vy in poly:
                pen += max(0.0, abs(vx) - h) ** 2 + max(0.0, abs(vy) - h) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                if pentagons_overlap(polys[i], polys[j]):
                    # approximate overlap penalty using center distance
                    dx = cs[i][0] - cs[j][0]
                    dy = cs[i][1] - cs[j][1]
                    d2 = dx * dx + dy * dy
                    pen += 10.0 / (1e-9 + d2)
        return s + 20.0 * pen

    # Randomized multistart around the seed
    best_x = x0[:]
    best_val = objective(best_x)
    rng = random.Random(1234567 + n)

    starts = [x0]
    for _ in range(min(10, 2 * n + 3)):
        x = x0[:]
        for i in range(len(x)):
            if i < 2 * n:
                x[i] += rng.uniform(-0.2, 0.2)
            elif i < 3 * n:
                x[i] += rng.uniform(-0.4, 0.4)
            else:
                x[i] += rng.uniform(-0.15, 0.15)
        starts.append(x)

    methods = ["Nelder-Mead", "Powell"]
    for xstart in starts:
        for method in methods:
            try:
                res = minimize(
                    objective,
                    xstart,
                    method=method,
                    options=dict(maxiter=max_iter, xatol=1e-10, fatol=1e-10, disp=False),
                )
                if res is not None and res.success:
                    val = objective(res.x)
                    if val < best_val:
                        best_val = val
                        best_x = res.x[:]
            except Exception:
                pass

    cs, ang = unpack(best_x)
    ang = [((a + math.pi) % TAU) - math.pi for a in ang]

    # Final repair/cleanup if needed
    cs = repair(cs, ang)
    return cs, ang


# ---------------- Construction library ----------------

def motif_double_lattice():
    """A compact 2-pentagon anti-parallel motif."""
    # Two pentagons facing opposite directions with a slight offset.
    return [(0.0, 0.0), (0.92, 0.08)], [math.pi / 2.0, -math.pi / 2.0]


def motif_five_ring():
    """Five pentagons arranged in a ring with alternating orientations."""
    centers = []
    angles = []
    rad = 1.15
    for k in range(5):
        t = TAU * k / 5.0
        centers.append((rad * math.cos(t), rad * math.sin(t)))
        angles.append((math.pi / 2.0 if k % 2 == 0 else -math.pi / 2.0) + 0.12 * math.sin(2.0 * t))
    return centers, angles


def motif_tilted_row(m, dy=0.0):
    """A short row with alternating flips and a gentle tilt."""
    centers, angles = [], []
    for i in range(m):
        x = i * 0.98
        y = dy + 0.12 * ((i % 2) * 2 - 1)
        centers.append((x, y))
        angles.append((math.pi / 2.0 if i % 2 == 0 else -math.pi / 2.0) + 0.07 * (i - (m - 1) / 2.0))
    return centers, angles


def motif_v_stack(m):
    """A vertical stack with alternating orientations and stagger."""
    centers, angles = [], []
    for j in range(m):
        x = 0.10 * ((j % 2) * 2 - 1)
        y = j * 0.96
        centers.append((x, y))
        angles.append((-math.pi / 2.0 if j % 2 == 0 else math.pi / 2.0) + 0.06 * (j - (m - 1) / 2.0))
    return centers, angles


def centered_layout_from_parts(parts):
    centers, angles = [], []
    for c, a in parts:
        centers.extend(c)
        angles.extend(a)
    if not centers:
        return centers, angles
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    centers = [(x - mx, y - my) for x, y in centers]
    return centers, angles


def generate_candidates(n):
    """Return a list of initial candidate packings."""
    cands = []

    # Basic grids with mixed orientations
    for cols in range(1, n + 1):
        rows = (n + cols - 1) // cols
        if cols * rows < n:
            continue
        pitch_x = 1.02 * WIDTH
        pitch_y = 0.96 * HEIGHT
        centers, angles = [], []
        for k in range(n):
            i, j = k % cols, k // cols
            x = (i - (cols - 1) / 2.0) * pitch_x
            y = (j - (rows - 1) / 2.0) * pitch_y
            centers.append((x, y))
            if (i + j) % 2 == 0:
                angles.append(math.pi / 2.0)
            else:
                angles.append(-math.pi / 2.0)
        cands.append((centers, angles))

    # Motif-based candidates
    if n >= 2:
        c, a = motif_double_lattice()
        parts = [(c, a)]
        remaining = n - 2
        if remaining > 0:
            rowc, rowa = motif_tilted_row(remaining)
            rowc = [(x + 2.1, y + 0.15) for x, y in rowc]
            parts.append((rowc, rowa))
        cands.append(centered_layout_from_parts(parts))

    if n >= 5:
        c1, a1 = motif_five_ring()
        cands.append((c1, a1))

    if n >= 3:
        # Two short rows
        m1 = n // 2
        m2 = n - m1
        c1, a1 = motif_tilted_row(m1, dy=-0.55)
        c2, a2 = motif_tilted_row(m2, dy=0.55)
        c2 = [(x + 0.48, y) for x, y in c2]
        cands.append(centered_layout_from_parts([(c1, a1), (c2, a2)]))

    if n >= 4:
        # Column-stack hybrid
        m1 = (n + 1) // 2
        m2 = n - m1
        c1, a1 = motif_v_stack(m1)
        c2, a2 = motif_v_stack(m2)
        c2 = [(x + 1.02, y + 0.28) for x, y in c2]
        cands.append(centered_layout_from_parts([(c1, a1), (c2, a2)]))

    # Randomized perturbations of motifs
    rng = random.Random(20240517 + 31 * n)
    base = cands[:]
    for centers, angles in base:
        for _ in range(10):
            cc = [(x + rng.uniform(-0.08, 0.08), y + rng.uniform(-0.08, 0.08)) for x, y in centers]
            aa = [a + rng.uniform(-0.12, 0.12) for a in angles]
            # recentre
            mx = sum(x for x, _ in cc) / len(cc)
            my = sum(y for _, y in cc) / len(cc)
            cc = [(x - mx, y - my) for x, y in cc]
            cands.append((cc, aa))

    return cands


def normalize_solution(centers, angles):
    if not centers:
        return centers, angles
    # Recenter by centroid
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    centers = [(x - mx, y - my) for x, y in centers]
    # Wrap angles
    angles = [((a + math.pi) % TAU) - math.pi for a in angles]
    return centers, angles


def final_cleanup(centers, angles):
    centers, angles = normalize_solution(centers, angles)
    if has_overlap(centers, angles):
        centers = repair(centers, angles)
    # Slight tighten by a uniform shrink while maintaining validity
    lo, hi = 0.85, 1.0
    if not has_overlap([(x * lo, y * lo) for x, y in centers], angles):
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            test = [(x * mid, y * mid) for x, y in centers]
            if has_overlap(test, angles):
                lo = mid
            else:
                hi = mid
        centers = [(x * hi, y * hi) for x, y in centers]
    return centers, angles


# ---------------- Main pack ----------------

def pack(n):
    """Return a valid packing of n unit regular pentagons."""
    if n <= 0:
        return [], [], 0.0

    best = None
    best_s = float("inf")

    candidates = generate_candidates(n)

    # Evaluate and locally optimize each candidate
    for centers, angles in candidates:
        centers, angles = normalize_solution(centers, angles)

        # First make sure it's valid enough to optimize from
        if has_overlap(centers, angles):
            centers = repair(centers, angles)

        # Multi-stage optimization
        centers, angles = optimize_configuration(centers, angles, max_iter=600)
        centers, angles = final_cleanup(centers, angles)

        s = enclosing_side(centers, angles)
        if s < best_s and all_inside(centers, angles, s) and not has_overlap(centers, angles):
            best_s = s
            best = (centers, angles, s)

    # Fallback: if optimization failed, use a conservative grid
    if best is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        pitch_x = WIDTH + 0.01
        pitch_y = HEIGHT + 0.01
        centers, angles = [], []
        for k in range(n):
            i, j = k % cols, k // cols
            x = (i - (cols - 1) / 2.0) * pitch_x
            y = (j - (rows - 1) / 2.0) * pitch_y
            centers.append((x, y))
            angles.append(math.pi / 2.0 if (i + j) % 2 == 0 else -math.pi / 2.0)
        centers = repair(centers, angles)
        centers, angles = final_cleanup(centers, angles)
        best = (centers, angles, enclosing_side(centers, angles))

    centers, angles, s = best

    # One final conservative validation pass; if needed, slightly dilate.
    if has_overlap(centers, angles):
        centers = repair(centers, angles)
        s = enclosing_side(centers, angles)

    if not all_inside(centers, angles, s):
        s = enclosing_side(centers, angles)

    return centers, angles, s
