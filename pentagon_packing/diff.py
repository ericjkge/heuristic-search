"""Apply BES-style SEARCH/REPLACE diffs to program source.

    <<<<<<< SEARCH
    <exact text to find>
    =======
    <replacement>
    >>>>>>> REPLACE

Each SEARCH must match the source verbatim (first occurrence replaced). Malformed
or non-matching diffs RAISE here; search.py catches and resamples with the error
in-conversation (BES max_patch_attempts pattern).
"""

from typing import List, Tuple


def _is_search(line):
    s = line.strip()
    return s.startswith("<<<<<<<") and "SEARCH" in s


def _is_sep(line):
    s = line.strip()
    return len(s) >= 3 and set(s) == {"="}


def _is_replace(line):
    s = line.strip()
    return s.startswith(">>>>>>>") and "REPLACE" in s


def parse_diff(diff_text: str) -> List[Tuple[str, str]]:
    """Parse into (search, replace) pairs; prose outside blocks is ignored."""
    blocks, search, replace, mode = [], [], [], None
    for line in diff_text.splitlines():
        if _is_search(line):
            if mode is not None:
                raise ValueError("nested/unterminated SEARCH block in diff")
            mode, search, replace = "search", [], []
        elif _is_sep(line) and mode == "search":
            mode = "replace"
        elif _is_replace(line) and mode == "replace":
            blocks.append(("\n".join(search), "\n".join(replace)))
            mode = None
        elif mode == "search":
            search.append(line)
        elif mode == "replace":
            replace.append(line)
    if mode is not None:
        raise ValueError("unterminated SEARCH/REPLACE block in diff")
    return blocks


def apply_diff(source: str, diff_text: str) -> str:
    blocks = parse_diff(diff_text)
    if not blocks:
        raise ValueError("no SEARCH/REPLACE blocks found in diff")
    result = source
    for i, (search, replace) in enumerate(blocks):
        if search == "":
            raise ValueError(f"block {i}: empty SEARCH text")
        idx = result.find(search)
        if idx == -1:
            preview = search if len(search) <= 300 else search[:300] + " ..."
            raise ValueError(f"block {i}: SEARCH text not found in source:\n{preview}")
        result = result[:idx] + replace + result[idx + len(search):]
    return result


if __name__ == "__main__":
    src = "def pack(n):\n    angles = [0.0] * n\n    return centers, angles, s\n"
    good = ("<<<<<<< SEARCH\n    angles = [0.0] * n\n=======\n"
            "    angles = [i * 0.1 for i in range(n)]\n>>>>>>> REPLACE\n")
    print(apply_diff(src, good))
    for name, bad in [
        ("no blocks", "just prose"),
        ("no match", "<<<<<<< SEARCH\nnot in source\n=======\nx\n>>>>>>> REPLACE\n"),
        ("unterminated", "<<<<<<< SEARCH\nfoo\n=======\nbar\n"),
    ]:
        try:
            apply_diff(src, bad)
            print(f"{name}: NO RAISE (bug)")
        except ValueError as e:
            print(f"{name}: raised -> {str(e).splitlines()[0]}")
