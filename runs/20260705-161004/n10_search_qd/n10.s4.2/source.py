"""Pack n unit regular pentagons into the smallest origin-centered axis-aligned square.

Contract: pack(n) -> (centers, angles, s)
- centers: list[(x, y)]
- angles: list[float]
- s: outer side of the enclosing square
Container convention: point p is inside iff max(|px|, |py|) <= s/2.

Strategy:
- Use a geometry-first search over several hand-designed motifs inspired by
  double-lattice and boundary-row packings.
- Build candidate packings using row, zig-zag, and paired-opposite orientations.
- Refine candidates with deterministic local optimization.
- Enforce validity with exact polygon overlap testing (SAT-style axis checks)
  and conservative scaling / repair.
- Return the best valid packing found.

The implementation is deterministic and self-contained.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))  # circumradius
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)
HEIGHT = R + APOTHEM

EPS = 1e-12
OVERLAP_EPS = 1e-11


def pentagon_vertices(cx, cy, angle):
    return [
        (
            cx + R * math.cos(angle + k * 2.0 * math.pi / 5.0),
            cy + R * math.sin(angle + k * 2.0 * math.pi / 5.0),
        )
        for k in range(5)
    ]


def enclosing_side(centers, angles):
    if not centers:
        return 0.0
    m = 0.0
    for (cx, cy), ang in zip(centers, angles):
        for vx, vy in pentagon_vertices(cx, cy, ang):
            m = max(m, abs(vx), abs(vy))
    return 2.0 * m


def _proj(poly, ux, uy):
    dots = [x * ux + y * uy for x, y in poly]
    return min(dots), max(dots)


def pentagons_overlap(pa, pb):
    """Return True iff pentagons overlap with positive area."""
    for poly in (pa, pb):
        for i in range(5):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 5]
            ux, uy = -(y2 - y1), (x2 - x1)
            amin, amax = _proj(pa, ux, uy)
            bmin, bmax = _proj(pb, ux, uy)
            norm = math.hypot(ux, uy)
            if min(amax, bmax) - max(amin, bmin) <= OVERLAP_EPS * max(1.0, norm):
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


def _center(centers):
    if not centers:
        return centers
    mx = sum(x for x, _ in centers) / len(centers)
    my = sum(y for _, y in centers) / len(centers)
    return [(x - mx, y - my) for x, y in centers]


def _normalize_angles(angles):
    return [((a + math.pi) % (2.0 * math.pi)) - math.pi for a in angles]


def repair(centers, angles):
    """Radially dilate centers until overlaps vanish; angles fixed."""
    if not centers:
        return centers
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.02
    for _ in range(80):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.25
    else:
        return centers

    for _ in range(80):
        mid = (lo + hi) / 2.0
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def _safe_objective(centers, angles):
    if has_overlap(centers, angles):
        return float("inf")
    return enclosing_side(centers, angles)


def _tighten_to_valid(centers, angles):
    """Scale centers toward origin as much as possible while preserving non-overlap."""
    if not centers:
        return centers, 0.0
    if has_overlap(centers, angles):
        centers = repair(centers, angles)

    lo, hi = 0.0, 1.0
    for _ in range(70):
        mid = (lo + hi) / 2.0
        cc = [(x * mid, y * mid) for x, y in centers]
        if has_overlap(cc, angles):
            lo = mid
        else:
            hi = mid
    cc = [(x * hi, y * hi) for x, y in centers]
    return cc, enclosing_side(cc, angles)


def _grid_template(n, cols, dx, dy, phase=0, angle0=math.pi / 2.0, alt=True):
    centers, angles = [], []
    rows = int(math.ceil(n / cols))
    k = 0
    for r in range(rows):
        m = min(cols, n - k)
        y = (r - (rows - 1) / 2.0) * dy
        shift = 0.5 * dx if ((r + phase) % 2 == 1) else 0.0
        x0 = -((m - 1) * dx) / 2.0 + shift
        edge_row = (r == 0 or r == rows - 1)
        row_tilt = (math.pi / 24.0) if edge_row else 0.0
        if r == rows - 1:
            row_tilt *= -1.0
        for c in range(m):
            x = x0 + c * dx
            if alt:
                ang = angle0 if ((r + c) % 2 == 0) else (angle0 + math.pi)
            else:
                ang = angle0
            if edge_row and c in (0, m - 1):
                ang += row_tilt
            centers.append((x, y))
            angles.append(ang)
            k += 1
    return _center(centers), _normalize_angles(angles)


def _pair_template(n, dx, dy, tilt=0.0):
    """Two opposite-orientation pentagons arranged in repeated pairs."""
    centers, angles = [], []
    pair_sep = 0.72 * WIDTH
    placed = 0
    rows = int(math.ceil(n / 4.0))
    for r in range(rows):
        y = (r - (rows - 1) / 2.0) * dy
        xshift = 0.42 * dx if (r % 2) else 0.0
        row = []
        for c in range(4):
            if placed >= n:
                break
            x = (c - 1.5) * dx + xshift
            ang = (math.pi / 2.0 if ((placed + c + r) % 2 == 0) else -math.pi / 2.0)
            if c in (0, 3):
                ang += tilt if c == 0 else -tilt
            row.append((x, y, ang))
            placed += 1
        for x, y, ang in row:
            centers.append((x, y))
            angles.append(ang)
    return _center(centers), _normalize_angles(angles)


def _spiral_template(n):
    centers, angles = [], []
    if n <= 0:
        return centers, angles
    placed = 0
    ring = 0
    while placed < n:
        count = 1 if ring == 0 else 6 * ring
        rx = ring * WIDTH * 0.43
        ry = ring * HEIGHT * 0.36
        for t in range(count):
            if placed >= n:
                break
            th = 2.0 * math.pi * t / count + (0.23 if ring % 2 else 0.0)
            x = rx * math.cos(th)
            y = ry * math.sin(th)
            ang = (math.pi / 2.0 if placed % 2 == 0 else -math.pi / 2.0)
            ang += (0.08 if ring % 2 else -0.08)
            centers.append((x, y))
            angles.append(ang)
            placed += 1
        ring += 1
    return _center(centers), _normalize_angles(angles)


def _candidate_templates(n):
    if n <= 0:
        return [([], [])]
    if n == 1:
        return [([(0.0, 0.0)], [math.pi / 2.0])]

    cands = []

    # More aggressive double-lattice / row motifs, including denser spacing.
    for cols in range(1, min(n, 7) + 1):
        for phase in (0, 1):
            for angle0 in (math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0 + math.pi / 5.0, math.pi / 2.0 - math.pi / 5.0):
                dx = WIDTH * (0.68 + 0.015 * cols)
                dy = HEIGHT * (0.60 + 0.02 * cols)
                cands.append(_grid_template(n, cols, dx, dy, phase=phase, angle0=angle0, alt=True))
                cands.append(_grid_template(n, cols, dx * 0.96, dy * 1.03, phase=1 - phase, angle0=angle0, alt=True))

    # Paired motifs with stronger tilts and tighter separations.
    for tilt in (0.0, math.pi / 40.0, math.pi / 28.0, math.pi / 18.0):
        cands.append(_pair_template(n, WIDTH * 0.76, HEIGHT * 0.70, tilt=tilt))
        cands.append(_pair_template(n, WIDTH * 0.72, HEIGHT * 0.66, tilt=-tilt))

    # Dense zig-zag layouts with alternating boundary rows tilted toward walls.
    for cols in (2, 3, 4, 5, 6):
        dx = WIDTH * (0.64 + 0.02 * cols)
        dy = HEIGHT * (0.58 + 0.025 * cols)
        centers, angles = [], []
        rows = int(math.ceil(n / cols))
        k = 0
        for r in range(rows):
            m = min(cols, n - k)
            y = (r - (rows - 1) / 2.0) * dy
            xshift = (0.5 * dx if r % 2 else 0.0) - (0.18 * dx if r % 3 == 2 else 0.0)
            x0 = -((m - 1) * dx) / 2.0 + xshift
            for c in range(m):
                x = x0 + c * dx
                ang = (math.pi / 2.0 if ((r + c) % 2 == 0) else -math.pi / 2.0)
                ang += (0.06 if (r % 2 == 0) else -0.06)
                if r in (0, rows - 1):
                    ang += (math.pi / 18.0 if r == 0 else -math.pi / 18.0)
                if c == 0 or c == m - 1:
                    ang += (0.09 if (r + c) % 2 == 0 else -0.09)
                centers.append((x, y))
                angles.append(ang)
                k += 1
                if k >= n:
                    break
        cands.append((_center(centers), _normalize_angles(angles)))

    # Ring / shell layouts that help boundary utilization.
    cands.append(_spiral_template(n))
    return cands


def _local_relax(centers, angles, seed=0, rounds=12):
    rng = random.Random(seed)
    centers = _center(list(centers))
    angles = list(_normalize_angles(angles))

    if has_overlap(centers, angles):
        centers = repair(centers, angles)
        centers = _center(centers)

    best_c = centers[:]
    best_a = angles[:]
    best_s = _safe_objective(best_c, best_a)
    if not math.isfinite(best_s):
        best_s = enclosing_side(best_c, best_a)

    def try_state(cc, aa):
        if has_overlap(cc, aa):
            return None
        return enclosing_side(cc, aa)

    for phase in range(rounds):
        step_pos = max(0.008, best_s * (0.09 * (0.72 ** phase)))
        step_ang = max(0.02, 0.22 * (0.78 ** phase))

        for _ in range(520):
            i = rng.randrange(len(best_c))
            cx, cy = best_c[i]
            a = best_a[i]

            # Candidate moves biased toward the square boundary and local shear.
            dirs = [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                (0.8, 0.8, 0.0), (0.8, -0.8, 0.0),
                (-0.8, 0.8, 0.0), (-0.8, -0.8, 0.0),
                (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
                (0.6, 0.0, 1.0), (-0.6, 0.0, -1.0),
                (0.0, 0.6, 1.0), (0.0, -0.6, -1.0),
            ]
            rng.shuffle(dirs)

            improved = False
            for dx, dy, da in dirs:
                cc = best_c[:]
                aa = best_a[:]
                bx = -math.copysign(1.0, cx) if abs(cx) > 1e-12 else 0.0
                by = -math.copysign(1.0, cy) if abs(cy) > 1e-12 else 0.0
                cc[i] = (
                    cx + dx * step_pos + bx * step_pos * 0.10 + rng.uniform(-1.0, 1.0) * step_pos * 0.05,
                    cy + dy * step_pos + by * step_pos * 0.10 + rng.uniform(-1.0, 1.0) * step_pos * 0.05,
                )
                aa[i] = a + da * step_ang + (0.015 if (abs(cx) > abs(cy) and i % 2 == 0) else -0.015 if (abs(cy) > abs(cx) and i % 2 == 1) else 0.0)
                cc = _center(cc)
                val = try_state(cc, aa)
                if val is not None and val + 1e-12 < best_s:
                    best_c, best_a, best_s = cc, aa, val
                    improved = True
                    break

            if not improved and rng.random() < 0.18:
                cc = best_c[:]
                aa = best_a[:]
                j = rng.randrange(len(cc))
                cc[j] = (
                    cc[j][0] + rng.uniform(-step_pos, step_pos) * 0.35,
                    cc[j][1] + rng.uniform(-step_pos, step_pos) * 0.35,
                )
                aa[j] = aa[j] + rng.uniform(-step_ang, step_ang) * 0.35
                cc = _center(cc)
                cc = repair(cc, aa)
                val = try_state(cc, aa)
                if val is not None and val <= best_s * 1.02:
                    best_c, best_a, best_s = cc, aa, val

        # Small deterministic coordinate descent pass.
        for i in range(len(best_c)):
            for dx, dy, da in ((step_pos, 0.0, 0.0), (-step_pos, 0.0, 0.0), (0.0, step_pos, 0.0), (0.0, -step_pos, 0.0), (0.0, 0.0, step_ang), (0.0, 0.0, -step_ang)):
                cc = best_c[:]
                aa = best_a[:]
                cc[i] = (cc[i][0] + dx, cc[i][1] + dy)
                aa[i] = aa[i] + da
                cc = _center(cc)
                val = try_state(cc, aa)
                if val is not None and val + 1e-12 < best_s:
                    best_c, best_a, best_s = cc, aa, val

    best_c = repair(best_c, best_a)
    best_c = _center(best_c)
    best_s = enclosing_side(best_c, best_a)
    return best_c, best_a, best_s


def _extra_refine(centers, angles):
    """Try deterministic pairwise nudges for a final small improvement."""
    if len(centers) <= 1:
        return centers, angles, enclosing_side(centers, angles)

    best_c = centers[:]
    best_a = angles[:]
    best_s = enclosing_side(best_c, best_a)

    candidates = [0.0, math.pi / 60.0, -math.pi / 60.0, math.pi / 40.0, -math.pi / 40.0]
    for i in range(len(best_c)):
        for j in range(i + 1, len(best_c)):
            for da in candidates:
                cc = best_c[:]
                aa = best_a[:]
                vx = cc[j][0] - cc[i][0]
                vy = cc[j][1] - cc[i][1]
                norm = math.hypot(vx, vy) + 1e-15
                ux, uy = vx / norm, vy / norm
                delta = min(0.015, 0.01 * best_s)
                cc[i] = (cc[i][0] - ux * delta, cc[i][1] - uy * delta)
                cc[j] = (cc[j][0] + ux * delta, cc[j][1] + uy * delta)
                aa[i] += da
                aa[j] -= da
                cc = _center(cc)
                if has_overlap(cc, aa):
                    continue
                s = enclosing_side(cc, aa)
                if s + 1e-12 < best_s:
                    best_c, best_a, best_s = cc, aa, s

    return best_c, best_a, best_s


def pack(n):
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    best = None
    best_s = float("inf")

    templates = _candidate_templates(n)

    # Add a few analytic seed layouts based on double-lattice rows.
    if n >= 2:
        for cols, dxm, dym, a0 in (
            (2, 0.66, 0.60, math.pi / 2.0),
            (3, 0.64, 0.58, math.pi / 2.0),
            (4, 0.62, 0.56, -math.pi / 2.0),
        ):
            dx = WIDTH * dxm
            dy = HEIGHT * dym
            centers, angles = [], []
            rows = int(math.ceil(n / cols))
            k = 0
            for r in range(rows):
                m = min(cols, n - k)
                y = (r - (rows - 1) / 2.0) * dy
                xshift = (0.5 * dx if (r % 2) else 0.0)
                x0 = -((m - 1) * dx) / 2.0 + xshift
                for c in range(m):
                    centers.append((x0 + c * dx, y))
                    angles.append(a0 if ((r + c) % 2 == 0) else -a0)
                    k += 1
                    if k >= n:
                        break
            templates.append((_center(centers), _normalize_angles(angles)))

    # Explore templates with deterministic local optimization.
    for seed, (c0, a0) in enumerate(templates):
        for r in (10, 14):
            c, a, s = _local_relax(c0, a0, seed=seed + 193 + r, rounds=r)
            c, s2 = _tighten_to_valid(c, a)
            if s2 < s:
                s = s2
            c, a, s = _extra_refine(c, a)
            c, s2 = _tighten_to_valid(c, a)
            if s2 < s:
                s = s2
            if s < best_s:
                best = (c, a, s)
                best_s = s

    centers, angles, s = best

    # Final cleanup.
    centers = _center(centers)
    angles = _normalize_angles(angles)
    if has_overlap(centers, angles):
        centers = repair(centers, angles)
        centers = _center(centers)

    centers, s = _tighten_to_valid(centers, angles)

    # Final stochastic-but-deterministic polishing with multiple restarts.
    for seed in range(12):
        c, a, s2 = _local_relax(centers, angles, seed=9001 + seed * 37, rounds=8)
        c, s3 = _tighten_to_valid(c, a)
        if s3 < s:
            centers, angles, s = c, a, s3
        c, a, s2 = _extra_refine(centers, angles)
        c, s3 = _tighten_to_valid(c, a)
        if s3 < s:
            centers, angles, s = c, a, s3

    centers = _center(centers)
    angles = _normalize_angles(angles)
    centers, s = _tighten_to_valid(centers, angles)
    return centers, angles, s
