"""Label space of the competition: 75 taxonomy codes plus ``NONE``.

Facts (data page and our own count in train.csv):

* ``root_code`` is one of the 75 codes from ``taxonomy_v1.csv`` or ``NONE``.
* Codes are CASE-SENSITIVE: ``Dec-01`` and ``Dec-02`` are written in mixed case. The metric
  ignores unknown codes, so a ``DEC-01`` would waste a slot.
* "Every code that is an answer in the test set has at least one training example": codes
  without a training example (``NOT-02``, ``PRO-03``) can never be correct on the test set.

The class order is fixed (``NONE`` first, then the taxonomy in file order) so that the OOF and
test arrays of all runs share the same column meaning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NONE = "NONE"


def load_classes(taxonomy_path: str | Path) -> list[str]:
    """Fixed class list: ``NONE`` + all taxonomy codes in file order (76 classes)."""
    tax = pd.read_csv(taxonomy_path)
    codes = [str(c) for c in tax["id"]]
    if len(set(codes)) != len(codes):
        raise ValueError("Taxonomy contains duplicate codes.")
    if NONE in codes:
        raise ValueError("Taxonomy unexpectedly contains the code NONE.")
    return [NONE] + codes


def encode(labels, classes: list[str]) -> np.ndarray:
    """Labels -> class indices. Raises on an unknown label (including wrong capitalisation)."""
    index = {c: i for i, c in enumerate(classes)}
    try:
        return np.array([index[str(x)] for x in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown label {exc.args[0]!r} — spelled exactly as in the taxonomy?") from exc


def seen_mask(train_labels, classes: list[str]) -> np.ndarray:
    """Boolean mask over the classes: True if the class occurs in the training data.

    According to the host, classes with False cannot be the correct answer on the test set;
    their probability is set to 0 before the top-3 selection.
    """
    seen = set(str(x) for x in train_labels)
    return np.array([c in seen for c in classes], dtype=bool)
