"""OOF/test/meta artifacts: write, collect, deduplicate.

Every run writes artifacts, even with a mediocre score: OOF predictions are the
currency of blending, so a meta-blend over earlier runs is the default rather than
the exception. Naming convention: oof_<name>.npy / test_<name>.npy /
meta_<name>.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _array_hash(arr: np.ndarray) -> str:
    """Stable content hash of an array (used for deduplication)."""
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes()
    ).hexdigest()[:16]


def save_oof(
    name: str,
    oof: np.ndarray,
    test: np.ndarray,
    meta: dict,
    out_dir: str | Path = "artifacts",
) -> dict[str, Path]:
    """Write oof_<name>.npy, test_<name>.npy and meta_<name>.json.

    Args:
        name: model name, e.g. "lgbm_seed42_fs1".
        oof: OOF predictions (n, K) or (n,).
        test: test predictions (n_test, K) or (n_test,).
        meta: metadata — should contain at least folds_hash, model_type, cv_score
            and feature_set (not enforced, but `load_all_artifacts` checks
            folds_hash compatibility).
        out_dir: target directory (created if missing).

    Returns:
        dict with the written paths (oof, test, meta).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = np.asarray(oof, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)

    paths = {
        "oof": out_dir / f"oof_{name}.npy",
        "test": out_dir / f"test_{name}.npy",
        "meta": out_dir / f"meta_{name}.json",
    }
    np.save(paths["oof"], oof)
    np.save(paths["test"], test)
    full_meta = {
        "name": name,
        "oof_shape": list(oof.shape),
        "test_shape": list(test.shape),
        "oof_hash": _array_hash(oof),
        **meta,
    }
    paths["meta"].write_text(
        json.dumps(full_meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return paths


def load_all_artifacts(
    paths: list[str | Path] | str | Path,
    verbose: bool = True,
) -> dict[str, dict]:
    """Collect artifacts recursively, deduplicate them and check compatibility.

    Searches all given paths recursively for meta_*.json and loads the matching
    oof_/test_ arrays. Duplicates (same OOF hash) are taken only once. Artifacts
    whose OOF shape or folds_hash differs from the majority are skipped with a
    clear message.

    Returns:
        dict name -> {"oof": ndarray, "test": ndarray, "meta": dict}.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    raw: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
    for root in paths:
        root = Path(root)
        if not root.exists():
            if verbose:
                print(f"   [artifacts] path missing, skipped: {root}")
            continue
        for meta_path in sorted(root.rglob("meta_*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if verbose:
                    print(f"   [artifacts] {meta_path}: corrupt meta file ({exc}), skipped.")
                continue
            name = meta.get("name", meta_path.stem.removeprefix("meta_"))
            oof_path = meta_path.with_name(f"oof_{name}.npy")
            test_path = meta_path.with_name(f"test_{name}.npy")
            if not oof_path.exists() or not test_path.exists():
                if verbose:
                    print(
                        f"   [artifacts] {name}: oof/test file missing next to "
                        f"{meta_path}, skipped."
                    )
                continue
            raw.append((name, np.load(oof_path), np.load(test_path), meta))

    if not raw:
        return {}

    # Deduplicate by OOF content hash, ALWAYS computed on the loaded array.
    # meta["oof_hash"] is not trusted: a stale or copied hash in the meta file
    # would otherwise silently displace a non-duplicate.
    seen_hashes: set[str] = set()
    deduped: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
    for name, oof, test, meta in raw:
        h = _array_hash(oof)
        meta_hash = meta.get("oof_hash")
        if meta_hash and meta_hash != h and verbose:
            print(
                f"   [artifacts] {name}: meta.oof_hash ({meta_hash}) does not "
                f"match the array content ({h}) — stale meta? The content wins."
            )
        if h in seen_hashes:
            if verbose:
                print(f"   [artifacts] {name}: duplicate (hash {h}), skipped.")
            continue
        seen_hashes.add(h)
        deduped.append((name, oof, test, meta))

    # Majority reference for shape and folds_hash; deterministic tie-break
    # (sorted candidates), otherwise two runs over the same folder could load
    # different model subsets (str hash randomisation).
    shapes = [oof.shape for _, oof, _, _ in deduped]
    ref_shape = max(sorted(set(shapes), key=str), key=shapes.count)
    fold_hashes = [
        m.get("folds_hash") for _, _, _, m in deduped if m.get("folds_hash")
    ]
    ref_folds = (
        max(sorted(set(fold_hashes)), key=fold_hashes.count)
        if fold_hashes
        else None
    )

    result: dict[str, dict] = {}
    for name, oof, test, meta in deduped:
        if oof.shape != ref_shape:
            if verbose:
                print(
                    f"   [artifacts] {name}: OOF shape {oof.shape} != reference "
                    f"{ref_shape}, skipped."
                )
            continue
        fh = meta.get("folds_hash")
        if ref_folds is not None and fh is not None and fh != ref_folds:
            if verbose:
                print(
                    f"   [artifacts] {name}: folds_hash {fh} != reference "
                    f"{ref_folds} — different folds, unusable for the blend, "
                    "skipped."
                )
            continue
        key = name
        i = 2
        while key in result:  # name collision with different content
            key = f"{name}__{i}"
            i += 1
        result[key] = {"oof": oof, "test": test, "meta": meta}

    if verbose:
        print(
            f"   [artifacts] {len(result)} artifacts loaded "
            f"({len(raw) - len(result)} skipped/deduplicated)."
        )
    return result
