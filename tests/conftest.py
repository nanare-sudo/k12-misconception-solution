"""Paths for the tests: repository root (kaggle_core), src/ and tools/; flat imports as on the GPU machines."""
import sys
from pathlib import Path

COMP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMP_ROOT
for p in (REPO_ROOT, COMP_ROOT / "src", COMP_ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
