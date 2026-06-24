"""AIME answer extraction and validation. Modular: HMMT support can be added here."""

from __future__ import annotations

import re

# Priority 1: \boxed{N}
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
# Priority 2: boxed{N} (no backslash — sometimes LLMs omit it)
_BOXED_NO_SLASH_RE = re.compile(r"(?<!\\)boxed\{([^}]+)\}")
# Priority 3: "the answer is N" or "Answer: N" (case-insensitive)
_ANSWER_IS_RE = re.compile(r"(?:the\s+answer\s+is|answer\s*:)\s*(\d+)", re.IGNORECASE)


def _to_int(s: str) -> int | None:
    """Return integer value of s, or None if not a valid integer."""
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return None


def _is_valid_aime_int(n: int) -> bool:
    return 0 <= n <= 999


def extract_aime_answer(text: str) -> str | None:
    """
    Parse the first valid AIME answer (integer in [0, 999]) from solution text.

    Priority: \\boxed{N} > boxed{N} > "the answer is N" / "Answer: N"

    Returns the answer string as found (preserving any leading zeros),
    or None if no valid integer found or if multiple conflicting values found.
    """
    # Try \boxed{...} first
    boxed_matches = _BOXED_RE.findall(text)
    if boxed_matches:
        return _resolve_matches(boxed_matches)

    # Try boxed{...} (no backslash)
    no_slash_matches = _BOXED_NO_SLASH_RE.findall(text)
    if no_slash_matches:
        return _resolve_matches(no_slash_matches)

    # Try natural language
    nl_matches = _ANSWER_IS_RE.findall(text)
    if nl_matches:
        return _resolve_matches(nl_matches)

    return None


def _resolve_matches(raw_matches: list[str]) -> str | None:
    """
    Given a list of raw match strings, validate each as an AIME integer.
    If all valid matches agree on the same integer value, return the first match string.
    If they conflict (different integer values), return None.
    If none are valid AIME integers, return None.
    """
    valid: list[tuple[str, int]] = []
    for raw in raw_matches:
        n = _to_int(raw)
        if n is not None and _is_valid_aime_int(n):
            valid.append((raw, n))

    if not valid:
        return None

    values = {n for _, n in valid}
    if len(values) > 1:
        # Conflicting answers
        return None

    return valid[0][0]


def validate_aime_answer(answer: str | None) -> bool:
    """True iff answer is a string representation of an integer in [0, 999]."""
    if answer is None:
        return False
    n = _to_int(answer)
    return n is not None and _is_valid_aime_int(n)


def answers_match(predicted: str | None, ground_truth: str) -> bool:
    """
    Numeric equality comparison: "007" == "7" == "7.0" -> True.
    Returns False if predicted is None.
    """
    if predicted is None:
        return False
    p = _to_int(predicted)
    g = _to_int(ground_truth)
    if p is None or g is None:
        return False
    return p == g
