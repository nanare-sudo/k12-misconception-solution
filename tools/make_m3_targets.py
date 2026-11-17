#!/usr/bin/env python3
"""Training targets of M3 (the fine-tuned 27B reasoner, job kind proto_sft; src/vlm/sft.py), frozen per fold.

    python -I tools/make_m3_targets.py --full            # folds 0-4 -> data/derived/sft/m3_fold<k>_targets.jsonl (+ .meta.json),
                                                         #   and all 1,500 rows (full fit) -> m3_full_targets.jsonl (+ .meta.json)
    python -I tools/make_m3_targets.py --full --check    # writes nothing: rebuilds the six target files in memory and compares them
                                                         #   with the files present and with targets_sha256 of the adapter recipes

For fold k only rows of the OTHER folds (the full fit: all rows). One record per row:
  notes       thinking text of the first pass of the protocol channel (data/derived/proto_notes/train_pr27.jsonl, as exported;
              truncation happens only at training time, in the model's tokenisation, `sft_notes_max_tokens`)
  notes_loss  true where the protocol channel puts the gold code at the top — source: the stored list
              artifacts/eda/private/m3_targets/sft_manifest.json (field "pr27_ok" per fold); this tool recomputes the list from
              artifacts/oof_pr27.npy (stable sort over the classes that can be a root) and stops on any difference
  finding     full explanation of the annotators' root finding (train.csv, `misconceptions`, root = true), "none" for NONE
  code        gold code
No reasoning texts of the other channels are used as targets: they were written with same-question examples from all folds.

The target files contain training data and row positions; they live under data/derived/ (not part of this repository) and are
delivered with the private inputs. Only counts, token lengths and checksums are printed. Token lengths in the meta files use the
tokenizer of Qwen3.5-0.8B (`--tokenizer`; the same vocabulary as the 27B is assumed; the training measures them again with the
real tokenizer); image tokens are computed from the header of the training photo and of the question diagram (one token per
32x32 pixels after resizing), without decoding. The meta files are not byte-reproducible (they record the creation time); the
target files are: an existing target file is never overwritten — a rebuild must be byte-identical, otherwise the tool stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import labels  # noqa: E402
from vlm import common as C  # noqa: E402
from vlm import proto as PR  # noqa: E402
from vlm import sft as S  # noqa: E402

IMPOSSIBLE = ("NOT-02", "PRO-03")  # never a root (the taxonomy codes that only occur as minor findings)
MANIFEST = COMP / "artifacts" / "eda" / "private" / "m3_targets" / "sft_manifest.json"
MANIFEST_SHA = "ebc6be96761bc53df619de3a9ba06f8e702e7f29d27fa0c44a4eec4fdf76a977"
# adapter recipes that pin the target file they were trained on (field targets_sha256 of sft_recipe.json)
RECIPES = {0: "m3_f034_rp/m3_f034_rp/m3_train_f0", 1: "m3_pilot_rp/m3_pilot_rp/m3_train_f1", 2: "m3_pilot_rp/m3_pilot_rp/m3_train_f2",
           3: "m3_f034_rp/m3_f034_rp/m3_train_f3", 4: "m3_f034_rp/m3_f034_rp/m3_train_f4", None: "m3_fullte_rp/m3_fullte_rp/m3_train_full"}


def quant(a) -> dict:
    a = np.asarray(a)
    return {str(p): int(np.percentile(a, p)) for p in (5, 25, 50, 75, 90, 95, 100)} if len(a) else {}


def pr27_ok(classes: list[str], y: np.ndarray, have_notes: np.ndarray) -> np.ndarray:
    """Rows whose protocol notes lead to gold: notes present and top code of oof_pr27 (stable sort, only classes that can be a
    root) = gold."""
    P = np.load(COMP / "artifacts" / "oof_pr27.npy")
    allowed = ~np.isin(np.array(classes), IMPOSSIBLE)
    top = np.argsort(-np.where(allowed[None, :], P, -np.inf), axis=1, kind="stable")[:, 0]
    return have_notes & (top == y)


def image_tokens(root: Path, row, max_side: int = 1024, q_side: int = 448) -> int:
    """Image tokens from the file header (one token per 32x32 pixels after resizing), photo and diagram — without decoding."""
    from PIL import Image

    n = 0
    for split, name, side in (("train", row["image_file"], max_side), ("questions", row["question_image_file"], q_side)):
        if not name:
            continue
        with Image.open(C.image_path(root, split, name)) as im:
            w, h = im.size
        s = min(1.0, side / max(w, h))
        n += int(np.ceil(w * s / 32) * np.ceil(h * s / 32))
    return n


def target_lines(rows, folds, ok, notes, findings, codes) -> str:
    """The target file as text (one JSON record per line, in row order)."""
    return "".join(json.dumps({"table": "train", "row": int(i), "fold": int(folds[i]), "notes_loss": bool(ok[i]), "notes_state": notes[int(i)][1],
                               "notes": notes[int(i)][0], "finding": findings[i], "code": codes[i]}, ensure_ascii=False) + "\n" for i in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--full", action="store_true", help="also the full fit on all training rows")
    ap.add_argument("--notes", default="derived/proto_notes/train_pr27.jsonl", help="notes file relative to the data folder")
    ap.add_argument("--tokenizer", default=str(COMP / "data" / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--out", default="derived/sft", help="output folder relative to the data folder")
    ap.add_argument("--check", action="store_true", help="write nothing: compare the rebuilt target files with the files present and the recipes")
    a = ap.parse_args()

    root = C.find_data_root()
    train = C.load_table(root, "train")
    folds = C.load_folds(len(train))
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes)
    n = len(y)

    notes_file = root / a.notes
    notes_meta = json.loads(notes_file.with_suffix(".meta.json").read_text())
    if notes_meta.get("sha256") != S.sha256_file(notes_file) or notes_meta.get("table") != "train":
        raise SystemExit(f"{a.notes}: checksum or table does not match the meta file")
    notes: dict[int, tuple[str, str]] = {}
    for line in notes_file.read_text().split("\n"):  # not splitlines(): U+2028 and similar characters occur raw in the texts
        if line.strip():
            r = json.loads(line)
            notes[int(r["row"])] = (str(r["text"]), str(r.get("state", "")))
    have = np.array([i in notes and bool(notes[i][0].strip()) for i in range(n)])

    ok = pr27_ok(classes, y, have)
    if S.sha256_file(MANIFEST) != MANIFEST_SHA:
        raise SystemExit(f"{MANIFEST.relative_to(COMP)}: checksum differs from the stored list")
    manifest = json.loads(MANIFEST.read_text())
    for k in range(5):  # provenance: the stored list must follow exactly from oof_pr27 and the notes
        if set(manifest[str(k)]["pr27_ok"]) != set(np.where(ok & (folds != k))[0].tolist()):
            raise SystemExit(f"stored list, fold {k}: pr27_ok differs from the recomputation from oof_pr27 — nothing written")
    findings = [S.finding_text(train.iloc[i]["root_code"], train.iloc[i]["misconceptions"]) for i in range(n)]
    codes = train["root_code"].astype(str).tolist()
    out_dir = root / a.out
    plans = [(k, folds != k, f"m3_fold{k}_targets") for k in a.folds] + ([(None, np.ones(n, bool), "m3_full_targets")] if a.full else [])

    if a.check:
        bad = 0
        for k, use, name in plans:
            got = hashlib.sha256(target_lines(np.where(use)[0], folds, ok, notes, findings, codes).encode("utf-8")).hexdigest()
            rf = COMP / "artifacts" / "runs" / RECIPES[k] / "sft_recipe.json" if k in RECIPES else None
            recipe = json.loads(rf.read_text()) if rf is not None and rf.is_file() else {}
            want = recipe.get("targets_sha256") if recipe.get("targets") == f"{name}.jsonl" else None
            path = out_dir / f"{name}.jsonl"
            have_file = S.sha256_file(path) if path.is_file() else None
            ok_r = want is None or got == want
            ok_f = have_file is None or got == have_file
            bad += not (ok_r and ok_f)
            print(f"{name}: rebuilt SHA-256 {got[:16]}…; adapter recipe {RECIPES.get(k)}: "
                  f"{'missing' if want is None else ('equal' if ok_r else 'DIFFERS')}; file present: "
                  f"{'no' if have_file is None else ('equal' if ok_f else 'DIFFERS')}")
        return 4 if bad else 0

    from transformers import AutoTokenizer

    tax = C.load_taxonomy(root)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ntok = np.array([len(tok(notes[i][0], add_special_tokens=False)["input_ids"]) if have[i] else 0 for i in range(n)])
    ftok = np.array([len(tok(" " + findings[i], add_special_tokens=False)["input_ids"]) for i in range(n)])
    head = len(tok(S.system_text(tax), add_special_tokens=False)["input_ids"])
    user = np.array([sum(len(tok(v, add_special_tokens=False)["input_ids"]) for kind, v in PR.user_parts(train.iloc[i], None, None) if kind == "text")
                     for i in range(n)])
    imgs = np.array([image_tokens(root, train.iloc[i]) for i in range(n)])
    seq = head + user + imgs + np.minimum(ntok, 2500) + ftok + 24  # role markers, closing, "Finding:", "Root:", code, end

    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {"notes": {"file": a.notes, "sha256": notes_meta["sha256"]},
               "manifest": {"file": str(MANIFEST.relative_to(COMP)), "sha256": S.sha256_file(MANIFEST)},
               "oof_pr27_sha256": S.sha256_file(COMP / "artifacts" / "oof_pr27.npy"), "train_ids_sha256": C.ID_SHA256["train"],
               "folds_hash": C.FOLDS_HASH, "tokenizer": Path(a.tokenizer).name}
    for k, use, name in plans:
        rows = np.where(use)[0]
        path = out_dir / f"{name}.jsonl"
        text = target_lines(rows, folds, ok, notes, findings, codes)
        if path.is_file():
            if path.read_text(encoding="utf-8") != text:
                raise SystemExit(f"{path.name} exists with different content — not overwritten")
            print(f"{name}: present and byte-identical to the rebuild; meta file left unchanged")
            continue
        path.write_text(text, encoding="utf-8")
        err = y[rows] != 0
        loss = ok[rows]
        meta = {"name": name, "fold": k, "table": "train", "rows": int(len(rows)), "notes_loss_rows": int(loss.sum()),
                "without_notes_loss": int((~loss).sum()), "share_none": float(np.mean(~err)),
                "groups": {"NONE": {"rows": int((~err).sum()), "notes_loss": int((~err & loss).sum())},
                           "error": {"rows": int(err.sum()), "notes_loss": int((err & loss).sum())}},
                "notes_tokens_all": quant(ntok[rows]), "notes_tokens_loss": quant(ntok[rows][loss]),
                "notes_share_over": {str(c): float(np.mean(ntok[rows] > c)) for c in (1500, 1800, 2000, 2500)},
                "finding_tokens_error": quant(ftok[rows][err]), "head_tokens": head,
                "sequence_tokens_estimate": quant(seq[rows]), "after_head_estimate": quant(seq[rows] - head),
                "sha256": S.sha256_file(path), "sources": sources, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{name}: {meta['rows']} rows (fold {k} held out), notes with loss {meta['notes_loss_rows']}, without {meta['without_notes_loss']}, "
              f"share NONE {meta['share_none']:.3f}; notes tokens median {meta['notes_tokens_all']['50']}, above 1,800: "
              f"{meta['notes_share_over']['1800']:.3f}; finding tokens median {meta['finding_tokens_error'].get('50')}; "
              f"sequence estimate median {meta['sequence_tokens_estimate']['50']} (after the header {meta['after_head_estimate']['50']}); "
              f"SHA-256 {meta['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
