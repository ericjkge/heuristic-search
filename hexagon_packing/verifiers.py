"""Soft verifiers: eval single-expression verify_codes against a packing.

Namespace for a verify_code: centers (n,2), angles (n,), vertices (n,6,2), x, y,
s, n, np, math (+ builtins). Scores are dense [0,1]; constraint-only -- a verifier
measures a property, it never suggests a fix.
"""

import math

import numpy as np

_HEX_ANGLES = np.arange(6) * (math.pi / 3.0)


def geometry_vars(centers, angles, s):
    """Build the eval namespace variables from a packing."""
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    angles = np.asarray(angles, dtype=float).reshape(-1)
    ang = angles[:, None] + _HEX_ANGLES[None, :]  # (n, 6)
    vertices = np.stack(
        [centers[:, 0, None] + np.cos(ang), centers[:, 1, None] + np.sin(ang)],
        axis=-1,
    )
    return {
        "centers": centers,
        "angles": angles,
        "vertices": vertices,
        "x": centers[:, 0],
        "y": centers[:, 1],
        "s": float(s),
        "n": int(centers.shape[0]),
    }


def try_eval(code, eval_vars):
    """Return (ok, value_or_error) -- lets callers tell a crash from a real 0.0."""
    try:
        return True, eval(code, {"np": np, "math": math, **eval_vars})
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def verify_one(code, eval_vars):
    """Dense score in [0,1]: bool -> {0,1}; floats clipped; errors/NaN -> 0.0."""
    ok, result = try_eval(code, eval_vars)
    if not ok:
        return 0.0
    if isinstance(result, bool):
        return 1.0 if result else 0.0
    try:
        v = float(result)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def score_vector(codes, eval_vars):
    """Phi(x): score of each verify_code on this packing."""
    return [verify_one(c, eval_vars) for c in codes]


if __name__ == "__main__":
    import initial

    centers, angles, s = initial.pack(11)
    ev = geometry_vars(centers, angles, s)
    codes = [
        "min(1.0, n / s**2)",                       # density
        "np.mean(np.linalg.norm(centers, axis=1) < 1.5)",
        "min(1.0, undefined_name / n)",             # broken -> 0.0
        "bool(np.max(np.abs(x)) < s)",              # bool -> 1.0
    ]
    print("vertices:", ev["vertices"].shape, " scores:",
          [round(v, 3) for v in score_vector(codes, ev)])
    print("try_eval(broken):", try_eval(codes[2], ev))
