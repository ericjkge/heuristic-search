"""Hard-verifier gate: import a candidate, call pack(n), validate feasibility.

Problem: pack n unit regular pentagons (side 1) into the smallest axis-aligned,
origin-centered SQUARE of side s. A point p is inside iff max(|px|, |py|) <= s/2.

Checks: exactly n pentagons, finite values, s > 0, no pairwise overlap (SAT), all
inside the square of the *reported* s. Reported s is trusted as the raw score:
understating it fails containment, overstating it only hurts the score.

The search loop runs this file as a subprocess (`python evaluate.py <prog> <n>
<out_dir>`) so a crashing/hanging candidate only kills that one process. Writes
metrics.json (+ geometry.npz when pack() returns). Geometry here is independent
of initial.py on purpose -- we never trust the candidate's own math.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

TOL = 1e-6  # touching is feasible; only penetration > TOL counts as overlap
SIDE = 1.0
R = SIDE / (2.0 * math.sin(math.pi / 5.0))  # circumradius of a unit pentagon ~ 0.8507


def pentagon_vertices(cx, cy, angle):
    return [
        (cx + R * math.cos(angle + k * 2.0 * math.pi / 5.0),
         cy + R * math.sin(angle + k * 2.0 * math.pi / 5.0))
        for k in range(5)
    ]


def _poly_axes(poly):
    axes = []
    m = len(poly)
    for i in range(m):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % m]
        nx, ny = -(y2 - y1), (x2 - x1)
        length = math.hypot(nx, ny)
        if length > 0:
            axes.append((nx / length, ny / length))
    return axes


def _overlap(pa, pb):
    """Separating-axis test: True iff convex polygons penetrate by more than TOL."""
    for ax, ay in _poly_axes(pa) + _poly_axes(pb):
        amin = min(x * ax + y * ay for x, y in pa)
        amax = max(x * ax + y * ay for x, y in pa)
        bmin = min(x * ax + y * ay for x, y in pb)
        bmax = max(x * ax + y * ay for x, y in pb)
        if min(amax, bmax) - max(amin, bmin) <= TOL:
            return False
    return True


def validate(centers, angles, s, n):
    """Return (feasible, reason)."""
    if len(centers) != n or len(angles) != n:
        return False, f"expected {n} pentagons, got {len(centers)} centers / {len(angles)} angles"
    flat = [c for xy in centers for c in xy] + list(angles) + [s]
    if not all(math.isfinite(v) for v in flat):
        return False, "non-finite value in output"
    if s <= 0:
        return False, f"s must be > 0, got {s}"

    polys = [pentagon_vertices(cx, cy, a) for (cx, cy), a in zip(centers, angles)]

    half = s / 2.0
    for idx, poly in enumerate(polys):
        for vx, vy in poly:
            if abs(vx) > half + TOL or abs(vy) > half + TOL:
                return False, f"pentagon {idx} exits the square (side {s:.6f})"

    for i in range(n):
        for j in range(i + 1, n):
            if _overlap(polys[i], polys[j]):
                return False, f"pentagons {i} and {j} overlap"

    return True, "ok"


def run_program(program_path, n):
    """Import the candidate module and call pack(n); return (parsed_or_None, reason)."""
    try:
        spec = importlib.util.spec_from_file_location("candidate", str(program_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        centers, angles, s = module.pack(n)
        centers = [(float(x), float(y)) for x, y in centers]
        angles = [float(a) for a in angles]
        s = float(s)
    except Exception as e:
        return None, f"program error: {e!r}"
    return (centers, angles, s), "ok"


def evaluate(program_path, n, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    parsed, reason = run_program(program_path, n)
    if parsed is None:
        metrics = {"feasible": False, "raw_s": None, "reason": reason, "n": n}
    else:
        centers, angles, s = parsed
        feasible, reason = validate(centers, angles, s, n)
        metrics = {
            "feasible": feasible,
            "raw_s": s if feasible else None,
            "reason": reason,
            "n": n,
        }
        np.savez(
            out / "geometry.npz",
            centers=np.array(centers, dtype=float),
            angles=np.array(angles, dtype=float),
            s=float(s),
        )

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        print(json.dumps(evaluate(sys.argv[1], int(sys.argv[2]), sys.argv[3]), indent=2))
    else:  # smoke: baseline must pass, broken inputs must fail
        here = Path(__file__).parent
        for n in (6, 10):
            print(f"n={n}:", evaluate(here / "initial.py", n, f"/tmp/pentpack_smoke/n{n}"))
        print("overlap  ->", validate([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], 4.0, 2))
        print("tiny s   ->", validate([(0.0, 0.0)], [0.0], 0.001, 1))
        print("count    ->", validate([(0.0, 0.0)], [0.0], 4.0, 2))
