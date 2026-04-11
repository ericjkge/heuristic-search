"""miniF2F theorem proving utilities: parsing, Lean compilation check."""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from utils.logging import get_logger

logger = get_logger(__name__)

LAKE = Path.home() / ".elan" / "bin" / "lake"
DEFAULT_WORKSPACE = Path(__file__).resolve().parent.parent / "lean_workspace"


def _extract_output_tag(text: str) -> Optional[str]:
    """Extract content between the last <OUTPUT> and </OUTPUT> tags."""
    matches = re.findall(r"<OUTPUT>(.*?)</OUTPUT>", text, re.DOTALL)
    return matches[-1] if matches else None


def parse_proof(text: str) -> Optional[str]:
    """Parse proof body from <OUTPUT> tags. Returns None on failure."""
    tagged = _extract_output_tag(text)
    if tagged is None:
        return None
    proof = tagged.strip()
    return proof if proof else None


def check_proof(
    header: str,
    formal_statement: str,
    proof_body: str,
    workspace: Path = DEFAULT_WORKSPACE,
    timeout: int = 600,
) -> bool:
    """Check if a proof compiles. Returns True if no errors."""
    code = f"{header}\n{formal_statement}\n{proof_body}"

    with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [str(LAKE), "env", "lean", tmp_path],
            capture_output=True,
            text=True,
            cwd=str(workspace),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Lean timed out after %ds", timeout)
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if "sorry" in proof_body:
        logger.debug("Proof contains sorry")
        return False
    if result.returncode != 0:
        logger.debug("Lean errors:\n%s", result.stdout + result.stderr)
        return False
    return True
