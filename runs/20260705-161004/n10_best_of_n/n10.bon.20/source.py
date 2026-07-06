"""Improved seed program: pack n unit regular pentagons into a square,
minimizing side s.

Contract: pack(n) -> (centers, angles, s). Container = origin-centered
axis-aligned square, side s: point p inside iff max(|px|, |py|) <= s/2.
"""

import math
import random

SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))       # circumradius ~ 0.8507
APOTHEM = SIDE / (2.0 * math.tan(math.pi / 5.0))  # ~ 0.6882
WIDTH = 2.0 * R * math.sin(2.0 * math.pi / 5.0)   # widest extent ~ 1.618
HEIGHT = R + APOTHEM                              # point-up bounding height ~ 1.539
TAU = 2.0 * math.pi

OVERLAP_EPS = 1e-9


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


def pentagons_overlap(pa, pb):
    # Separating-axis theorem with a small tolerance.
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


def repair(centers, angles):
    """Dilation repair: scale centers about origin until all overlaps vanish."""
    if not centers:
        return centers
    if not has_overlap(centers, angles):
        return centers

    def dilated(lam):
        return [(x * lam, y * lam) for x, y in centers]

    lo, hi = 1.0, 1.1
    for _ in range(60):
        if not has_overlap(dilated(hi), angles):
            break
        lo, hi = hi, hi * 1.25
    else:
        return centers

    for _ in range(60):
        mid = (lo + hi) * 0.5
        if has_overlap(dilated(mid), angles):
            lo = mid
        else:
            hi = mid
    return dilated(hi)


def _base_angles():
    # Opposite-orientation pairing encourages denser local motifs.
    return [math.pi / 2.0, -math.pi / 2.0]


def _pack_grid(n, cols, rows, pitch_x, pitch_y, xoff=0.0, yoff=0.0,
               angle_mode="pair", jitter=0.0):
    centers, angles = [], []
    for k in range(n):
        i, j = k % cols, k // cols
        x = (i - (cols - 1) / 2.0) * pitch_x + xoff
        y = (j - (rows - 1) / 2.0) * pitch_y + yoff
        if jitter:
            x += jitter * math.sin(1.7 * (i + 1) + 2.3 * (j + 1))
            y += jitter * math.cos(2.1 * (i + 1) + 1.9 * (j + 1))
        centers.append((x, y))

        if angle_mode == "pair":
            angles.append(math.pi / 2.0 if ((i + j) & 1) == 0 else -math.pi / 2.0)
        elif angle_mode == "stripe":
            angles.append(math.pi / 2.0 if (j & 1) == 0 else -math.pi / 2.0)
        elif angle_mode == "column":
            angles.append(math.pi / 2.0 if (i & 1) == 0 else -math.pi / 2.0)
        elif angle_mode == "tilt":
            angles.append(math.pi / 2.0 + 0.14 * math.sin(0.9 * i + 1.1 * j))
        else:
            angles.append(math.pi / 2.0)

    return centers, angles


def _score(centers, angles):
    return enclosing_side(centers, angles)


def _local_refine(centers, angles, steps=1200):
    """Simple stochastic coordinate/angle search on the container side."""
    if not centers:
        return centers, angles

    rng = random.Random(1234567)
    best_c = [list(p) for p in centers]
    best_a = list(angles)
    best_s = _score(best_c, best_a)

    def valid(cand_c, cand_a):
        return not has_overlap(cand_c, cand_a)

    temp0 = 0.06 * best_s if best_s > 0 else 0.05
    for t in range(steps):
        i = rng.randrange(len(best_c))
        c2 = [p[:] for p in best_c]
        a2 = best_a[:]

        temp = temp0 * (1.0 - t / max(1, steps - 1))
        dx = rng.uniform(-1.0, 1.0) * temp
        dy = rng.uniform(-1.0, 1.0) * temp
        da = rng.uniform(-1.0, 1.0) * (0.35 * temp + 0.02)

        c2[i][0] += dx
        c2[i][1] += dy
        a2[i] = (a2[i] + da) % TAU

        # Small collective nudges sometimes help.
        if rng.random() < 0.15:
            mx = sum(x for x, _ in c2) / len(c2)
            my = sum(y for _, y in c2) / len(c2)
            shrink = rng.uniform(0.985, 1.0)
            for k in range(len(c2)):
                c2[k][0] = mx + (c2[k][0] - mx) * shrink
                c2[k][1] = my + (c2[k][1] - my) * shrink

        if not valid([tuple(p) for p in c2], a2):
            continue

        s = _score([tuple(p) for p in c2], a2)
        if s + 1e-12 < best_s:
            best_s = s
            best_c = c2
            best_a = a2

    return [tuple(p) for p in best_c], best_a


def _candidate_packings(n):
    if n <= 0:
        return []

    # Try several near-square and rectangular layouts with mixed orientations.
    out = []
    base = int(math.ceil(math.sqrt(n)))
    shapes = []
    for cols in range(max(1, base - 2), base + 3):
        rows = int(math.ceil(n / cols))
        shapes.append((cols, rows))
    # Some extra elongated candidates for awkward counts.
    shapes.extend([
        (max(1, base - 1), int(math.ceil(n / max(1, base - 1)))),
        (base + 1, int(math.ceil(n / (base + 1)))),
    ])

    shapes = list(dict.fromkeys(shapes))

    for cols, rows in shapes:
        # Baseline spacing based on pentagon extents, then tighten.
        pitch_x0 = WIDTH * 0.90
        pitch_y0 = HEIGHT * 0.88

        for mode in ("pair", "stripe", "column", "tilt"):
            for sx in (0.96, 0.92, 0.88, 0.84):
                for sy in (0.96, 0.92, 0.88, 0.84):
                    pitch_x = pitch_x0 * sx
                    pitch_y = pitch_y0 * sy
                    jitter = 0.0 if mode != "tilt" else 0.02
                    centers, angles = _pack_grid(
                        n, cols, rows, pitch_x, pitch_y,
                        angle_mode=mode, jitter=jitter
                    )
                    centers = repair(centers, angles)
                    out.append((centers, angles, _score(centers, angles)))

    return out


def pack(n):
    """Return a valid packing of n unit regular pentagons."""
    if n <= 0:
        return [], [], 0.0
    if n == 1:
        centers = [(0.0, 0.0)]
        angles = [math.pi / 2.0]
        return centers, angles, enclosing_side(centers, angles)

    best = None
    best_s = float("inf")

    for centers, angles, s in _candidate_packings(n):
        if s < best_s - 1e-12:
            best = (centers, angles, s)
            best_s = s

    centers, angles, s = best
    centers, angles = _local_refine(centers, angles, steps=1600 if n >= 8 else 1000)
    centers = repair(centers, angles)
    s = enclosing_side(centers, angles)

    return centers, angles, s
