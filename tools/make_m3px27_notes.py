#!/usr/bin/env python3
"""Channel m3px27: notes of the FIRST pr27 pass (own Qwen3.8-27B without training in the teacher protocol, first draw) in the
format of the frozen M3 notes — input of the M3 ranking (probe proto_rank in sft mode) behind exactly the notes behind which the
untrained 27B computed the channel px27 (pr27xm = ½(pr27 + px27)).

    python -I tools/make_m3px27_notes.py                 # training rows -> data/derived/sft_m3px27/m3px27_train_notes.jsonl
    python -I tools/make_m3px27_notes.py --table test    # test rows     -> data/derived/sft_m3px27te/m3px27_test_notes.jsonl
    python -I tools/make_m3px27_notes.py --table both

SOURCES (the runs of the first pr27 pass, job proto_gen; exactly one first pass per row):
  training: artifacts/runs/pr27_rp/pr27_rp/{proto_a,proto_b}/proto_raw.jsonl (860 + 640 rows, table train)
  test:     artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_raw.jsonl (3,498 rows, table test; frozen)
  Model according to summary.json of every source: Qwen/Qwen3.8-27B @ 1d4bf0f2… (otherwise abort).
NOTE per row = vlm/proto.rank_notes (the THINKING of the first pass without the answer text, without thinking markers, without
whitespace at the edges, at most the last 16,000 characters) — the same path as tools/export_proto_notes.py, from which px27 read its
notes.
EQUALITY WITH px27 (mandatory, otherwise abort without writing): every note is character-identical with the note of the same row in
the frozen export file behind which px27 computed (training data/derived/proto_notes/train_pr27.jsonl, SHA-256 d615b707…, run
px27_rp/rank_tr with proto_from = this file; test data/derived/proto_notes_test/test_pr27.jsonl, SHA-256 1b459a40…, run
n1_rp/rank_test), and carries the same label-free state. The checksums of the export files are pinned here.
TRAINING DISTRIBUTION (training only, report only): the same notes are the field notes of the M3 training targets
(data/derived/sft/m3_full_targets.jsonl and m3_fold<k>_targets.jsonl, frozen inputs); counted is how many training-target notes are
character-identical. In training the runner cut them to 2,500 tokens (sft_notes_max_tokens); the ranking reads them uncut, like px27 —
the number of notes above 2,500 tokens is recorded in the meta file.

FORMAT as in tools/make_q397m3_notes.py and vlm/sft.generate (read by vlm/sft.load_notes):
    {"table", "row", "state": "open" | "empty", "text": note, "finding": null, "pr27_state": <recorded label, see RECORDED_STATE>}
finding = null: M3 writes its finding in the ranking ITSELF (runner._sft_finish, greedy up to "Root:", finish_tokens) and then scores
all codes exactly behind "NOTE\\n</think>\\n\\nFinding: FINDING\\nRoot:" (vlm/sft.rank_prefix). pr27_state is label-free and serves
the evaluation only (as tools/export_proto_notes.note_state): complete = the first pass ended regularly, draft = stopped, but the note
already names "root_code", open = stopped before the verdict. The field stores the recorded label of the state (RECORDED_STATE).

OUTPUT: only into the upload folder (it carries exactly the note file and its meta file; it is mounted on the remote machine as
data/derived/sft_m3px27, and the run reads sft_from: derived/sft_m3px27/m3px27_train_notes.jsonl; test side accordingly sft_m3px27te).
Reproducible byte for byte (no timestamps). Meta file: sources with checksums, model, equality with px27, states, characters and
tokens as quantiles (tokenizer of Qwen3.8-27B from the Hugging Face cache, otherwise Qwen3.5-0.8B; empty = without), notes above
2,500 tokens, SHA-256 of the file. Test notes are never inspected: the tool reads them by machine, compares and counts; only counts,
quantiles and checksums are printed. Exit 4 = check failed, nothing written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

from vlm import proto as PR  # noqa: E402

MARKS = re.compile(r"</?think>|<\|[^|<>\n]{1,40}\|>")  # as in tools/export_proto_notes.py / make_q397m3_notes.py
NOTE_CHARS = 16000  # as in export_proto_notes (--note-chars), with which px27 and the M3 training targets were made
TOKEN_CAP = 2500  # note limit of the M3 training (sft_notes_max_tokens) — statistic only, nothing is truncated
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCES = {"train": ("runs/pr27_rp/proto_a/proto", "runs/pr27_rp/proto_b/proto"), "test": ("runs/pr27te_rp/proto_te/proto",)}
# export files behind which px27 computed (px27_rp/rank_tr: proto_from = train_pr27.jsonl; n1_rp/rank_test: test_pr27.jsonl)
PX27_NOTES = {"train": ("derived/proto_notes/train_pr27.jsonl", "d615b707778e1e7b685ff4a9b28c0b2b756b46aba7ce149c295f99527dea388d"),
              "test": ("derived/proto_notes_test/test_pr27.jsonl", "1b459a40278b814922bfedbff9867028f9e0e56e39ca54b6c7b471c42a833f3a")}
PX27_RUNS = {"train": "runs/px27_rp/rank_tr/bench", "test": "runs/n1_rp/rank_test/test"}
OUT = {"train": Path("derived") / "sft_m3px27" / "m3px27_train_notes.jsonl", "test": Path("derived") / "sft_m3px27te" / "m3px27_test_notes.jsonl"}
NAMES = {"train": "m3px27_train_notes", "test": "m3px27_test_notes"}
M3_TARGETS = ("m3_full_targets",) + tuple(f"m3_fold{k}_targets" for k in range(5))  # training targets of the M3 adapters (field notes)
HF_27B = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.8-27B" / "snapshots" / REVISION
TOKENIZER_FALLBACK = COMP / "data" / "models" / "Qwen3.5-0.8B"
QUANTS = (5, 25, 50, 75, 90, 95, 100)
# State of a note (field pr27_state) -> label recorded in the note files. The labels are the ones the frozen note files (and the px27
# export files they are compared with) were written with (short German words); the note files are model inputs pinned by checksum, so
# the labels are kept as recorded (as in tools/export_proto_notes.RECORDED_STATE).
RECORDED_STATE = {"complete": "fertig", "draft": "entwurf", "open": "offen"}


class ConvertError(RuntimeError):
    """Check failed: nothing written."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    """Records of a file (split at \\n only: U+2028 and similar stand raw in the texts)."""
    out = []
    for k, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n")):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            raise ConvertError(f"{Path(path).name}: line {k + 1} is not JSON (interrupted write?)") from None
    return out


def note_state(rec: dict, text: str) -> str:
    """Label-free state as in tools/export_proto_notes.note_state, returned as its recorded label (RECORDED_STATE)."""
    if rec.get("end") == "stop":
        return RECORDED_STATE["complete"]
    return RECORDED_STATE["draft"] if "root_code" in text else RECORDED_STATE["open"]


def quantiles(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    return {str(p): int(np.percentile(a, p)) for p in QUANTS} if len(a) else {}


def run_dir(spec: str, art: Path) -> tuple[Path, str]:
    bits = str(spec).split("/")
    if len(bits) != 4 or bits[0] != "runs" or not all(bits[1:]):
        raise ConvertError(f"{spec}: expected runs/<run>/<job>/<tag>")
    _, run, job, tag = bits
    return Path(art) / "runs" / run / run / job, tag


def load_sources(specs, art: Path, table: str, n: int, expect_model: str | None = MODEL) -> tuple[dict[int, dict], list[dict]]:
    """First passes of all sources: table according to the meta file, model (and revision) according to summary.json, exactly one
    first pass per row, no row in two sources; together EXACTLY all n rows. -> ({row: record}, details per source)."""
    firsts: dict[int, dict] = {}
    info = []
    for spec in specs:
        d, tag = run_dir(spec, art)
        raw, meta_f, summ_f = d / f"{tag}_raw.jsonl", d / f"{tag}_meta.json", d / "summary.json"
        if not raw.is_file() or not meta_f.is_file():
            raise ConvertError(f"{spec}: {raw.name} or {meta_f.name} missing (run not fetched)")
        meta = json.loads(meta_f.read_text())
        if meta.get("table") != table:
            raise ConvertError(f"{spec}: output for table {meta.get('table')!r}, required {table}")
        model = (json.loads(summ_f.read_text()) if summ_f.is_file() else {}).get("model") or {}
        if expect_model is not None and (model.get("repo") != expect_model or model.get("revision") != REVISION):
            raise ConvertError(f"{spec}: model {model.get('repo')!r}@{str(model.get('revision'))[:8]} instead of {expect_model}@{REVISION[:8]} (summary.json)")
        got = 0
        for r in read_jsonl(raw):
            if r.get("table", table) != table:
                raise ConvertError(f"{spec}: record of another table")
            if r.get("phase") != "first":
                continue
            i = r.get("row")
            if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= int(i) < n:
                raise ConvertError(f"{spec}: record with an invalid row position (table {table}, {n} rows)")
            if int(i) in firsts:
                raise ConvertError(f"{spec}: a row has more than one first pass (also across sources)")
            firsts[int(i)] = r
            got += 1
        if int(meta.get("rows", -1)) != got:
            raise ConvertError(f"{spec}: {got} first passes, the meta file names {meta.get('rows')} rows")
        info.append({"source": spec, "rows": got, "source_sha256": sha256_file(raw), "meta_sha256": sha256_file(meta_f),
                     "model": model.get("repo"), "revision": model.get("revision"), "failed": meta.get("failed")})
    missing = n - len(firsts)
    if missing:
        raise ConvertError(f"{missing} of {n} rows without a first pass — sources incomplete")
    return firsts, info


def build_rows(firsts: dict[int, dict], n: int, table: str, note_chars: int = NOTE_CHARS) -> tuple[list[dict], dict]:
    """Note file in the M3 format for all n rows (table order). -> (records, statistics)."""
    rows, cut, dropped = [], 0, 0
    for i in range(n):
        rec = firsts[i]
        note = PR.rank_notes(rec, note_chars)
        full = str(rec.get("thinking") or "").replace("<think>", "").replace("</think>", "").strip()
        cut += int(len(full) > int(note_chars))
        dropped += int(bool(str(rec.get("text") or "").strip()))
        rows.append({"table": table, "row": i, "state": "open" if note else "empty", "text": note, "finding": None,
                     "pr27_state": note_state(rec, note)})
    marked = sum(bool(MARKS.search(r["text"])) for r in rows)
    if marked:
        raise ConvertError(f"{marked} notes contain a marker of the chat template (<think>, </think>, <|…|>) — nothing written")
    stats = {"rows": n, "with_record": n, "missing": 0, "states": dict(sorted(Counter(r["state"] for r in rows).items())),
             "pr27_states": dict(sorted(Counter(r["pr27_state"] for r in rows).items())),
             "ends_first": dict(sorted(Counter(str(firsts[i].get("end", "")) for i in range(n)).items())),
             "notes_cut_chars": cut, "answer_text_dropped": dropped, "chars": quantiles(len(r["text"]) for r in rows)}
    return rows, stats


def check_px27(rows: list[dict], data_root: Path, table: str, want_sha: str | None = None) -> dict:
    """Equality with the export file behind which px27 computed: checksum pinned, the same rows, every note character-identical,
    the same state. Violated -> ConvertError."""
    rel, sha_fixed = PX27_NOTES[table]
    want = sha_fixed if want_sha is None else want_sha
    f = Path(data_root) / rel
    if not f.is_file():
        raise ConvertError(f"{rel} missing — the px27 notes are the reference (tools/export_proto_notes.py)")
    got = sha256_file(f)
    if got != want:
        raise ConvertError(f"{rel}: checksum {got[:16]}… instead of {want[:16]}… — not the file behind which px27 computed")
    exp: dict[int, tuple[str, str]] = {}
    for r in read_jsonl(f):
        if r.get("table") != table:
            raise ConvertError(f"{rel}: record of another table")
        i = int(r["row"])
        if i in exp:
            raise ConvertError(f"{rel}: a row twice")
        exp[i] = (str(r.get("text") or ""), str(r.get("state") or ""))
    mine = {r["row"]: (r["text"], r["pr27_state"]) for r in rows}
    if sorted(exp) != sorted(mine):
        raise ConvertError(f"{rel}: {len(exp)} rows against {len(mine)} — another set of rows than px27")
    diff_text = sum(exp[i][0] != mine[i][0] for i in mine)
    diff_state = sum(exp[i][1] != mine[i][1] for i in mine)
    if diff_text or diff_state:
        raise ConvertError(f"{rel}: {diff_text} notes and {diff_state} states differ from the px27 notes — nothing written")
    return {"file": rel, "sha256": got, "rows": len(exp), "identical_notes": len(mine), "identical_states": len(mine), "px27_run": PX27_RUNS[table]}


def check_m3_targets(rows: list[dict], data_root: Path) -> dict:
    """Report only (training): are the training-target notes of the M3 adapters character-identical with these notes?"""
    mine = {r["row"]: r["text"] for r in rows}
    out = {}
    for name in M3_TARGETS:
        f = Path(data_root) / "derived" / "sft" / f"{name}.jsonl"
        if not f.is_file():
            out[name] = {"missing": True}
            continue
        recs = read_jsonl(f)
        same = sum(str(r.get("notes") or "") == mine.get(int(r["row"])) for r in recs if r.get("table") == "train")
        out[name] = {"rows": len(recs), "identical_notes": int(same), "sha256": sha256_file(f),
                     "notes_loss_rows": int(sum(bool(r.get("notes_loss")) for r in recs))}
    return out


def load_tokenizer(spec: str | None):
    """Tokenizer: path, '' = none, None = Qwen3.8-27B from the HF cache (revision of the run), otherwise Qwen3.5-0.8B."""
    if spec == "":
        return None, None
    from transformers import AutoTokenizer

    path = Path(spec) if spec else (HF_27B if (HF_27B / "tokenizer.json").is_file() else TOKENIZER_FALLBACK)
    name = "Qwen3.8-27B@" + REVISION[:8] if path == HF_27B else path.name
    return AutoTokenizer.from_pretrained(str(path)), name


def token_counts(texts: list[str], tok) -> list[int]:
    out = []
    for s in range(0, len(texts), 128):
        enc = tok(texts[s:s + 128], add_special_tokens=False)
        out.extend(len(ids) for ids in enc["input_ids"])
    return out


def token_stats(texts: list[str], tok) -> dict | None:
    if tok is None:
        return None
    t = token_counts(texts, tok)
    return {"quantiles": quantiles(t), "mean": float(np.mean(t)), f"over_{TOKEN_CAP}": int(sum(x > TOKEN_CAP for x in t)), "rows": len(t)}


def body_of(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def write_outputs(rows: list[dict], meta: dict, out: Path) -> dict:
    """Write the note file + meta file into the upload folder (the folder may only contain these two files)."""
    body = body_of(rows)
    meta = {**meta, "sha256": sha256_bytes(body.encode("utf-8"))}
    allowed = {out.name, out.with_suffix(".meta.json").name}
    if out.parent.is_dir():
        extra = sorted(p.name for p in out.parent.iterdir() if p.name not in allowed)
        if extra:
            raise ConvertError(f"upload folder {out.parent} contains foreign entries {extra} — it may only carry the note file")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if sha256_file(out) != meta["sha256"]:
        raise ConvertError("written file differs from the checksum")
    return meta


def table_len(data_root: Path, table: str) -> int:
    """Number of rows of the table (an aggregate only; no row is inspected)."""
    import pandas as pd

    return int(len(pd.read_csv(Path(data_root) / f"{table}.csv", usecols=["trace_id"], dtype=str, keep_default_na=False)))


def convert(table: str, data_root: Path, art: Path, tok, tok_name, note_chars: int = NOTE_CHARS, px27_sha: str | None = None,
            expect_model: str | None = MODEL, out: Path | None = None) -> tuple[dict, Path]:
    n = table_len(data_root, table)
    firsts, src = load_sources(SOURCES[table], art, table, n, expect_model)
    rows, stats = build_rows(firsts, n, table, note_chars)
    px = check_px27(rows, data_root, table, px27_sha)
    meta = {"name": NAMES[table], "table": table, "source": "+".join(SOURCES[table]), "sources": src, "model": MODEL, "revision": REVISION,
            "note_chars": note_chars, **stats, "px27_notes": px,
            "tokens": token_stats([r["text"] for r in rows], tok), "tokenizer": tok_name, "token_cap_m3_training": TOKEN_CAP,
            "format": "vlm/sft.load_notes: table, row, state, text, finding (null: M3 writes the finding itself), pr27_state",
            "created_by": "tools/make_m3px27_notes.py",
            "purpose": "input of the M3 ranking (sft_from) behind the notes of the first pr27 pass — the same notes as px27"}
    if len(src) == 1:  # test side: fields as read by the note guard of tools/make_submission.py (_notes_file_check, rank_only_notes_guard)
        meta.update(source_sha256=src[0]["source_sha256"], meta_sha256=src[0]["meta_sha256"])
    if table == "train":
        meta["m3_targets"] = check_m3_targets(rows, data_root)
    target = out if out is not None else Path(data_root) / OUT[table]
    return write_outputs(rows, meta, target), target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", choices=("train", "test", "both"), default="train")
    ap.add_argument("--tokenizer", default=None, help="path; empty = without token lengths; default Qwen3.8-27B from the HF cache")
    ap.add_argument("--data-root", help="data root (default: vlm.common.find_data_root)")
    ap.add_argument("--artifacts", default=str(COMP / "artifacts"))
    ap.add_argument("--px27-sha", help="expected checksum of the px27 export file (mode `full` after new protocol answers; docs/VERIFICATION.md §5.2)")
    ap.add_argument("--expect-model", default=MODEL, help="'' = do not check the model (code checks only)")
    a = ap.parse_args(argv)

    from vlm import common as C

    data_root = Path(a.data_root) if a.data_root else C.find_data_root()
    art = Path(a.artifacts)
    tables = ("train", "test") if a.table == "both" else (a.table,)
    try:
        tok, tok_name = load_tokenizer(a.tokenizer)
        done = []
        for table in tables:  # check and write per table; an abort writes nothing for that table
            meta, out = convert(table, data_root, art, tok, tok_name, px27_sha=a.px27_sha, expect_model=a.expect_model or None)
            done.append((table, meta, out))
    except ConvertError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 4
    for table, meta, out in done:
        tk = (meta.get("tokens") or {})
        print(f"written: {out} (+ .meta.json)")
        print(f"  table {table}: {meta['rows']} rows, with a first pass {meta['with_record']}, states {meta['states']}, "
              f"pr27 states {meta['pr27_states']}, end of the first pass {meta['ends_first']}")
        print(f"  characters {meta['chars']}; tokens ({meta['tokenizer']}) {tk.get('quantiles', '—')}, above {TOKEN_CAP}: "
              f"{tk.get(f'over_{TOKEN_CAP}', '—')}; truncated (characters) {meta['notes_cut_chars']}, answer text dropped {meta['answer_text_dropped']}")
        px = meta["px27_notes"]
        print(f"  equal to the px27 notes ({px['file']}, SHA-256 {px['sha256'][:16]}…): {px['identical_notes']}/{px['rows']} notes and states")
        for name, rec in (meta.get("m3_targets") or {}).items():
            print(f"  M3 training targets {name}: " + ("missing" if rec.get("missing") else f"{rec['identical_notes']}/{rec['rows']} notes character-identical "
                                                         f"(with loss {rec['notes_loss_rows']})"))
        print(f"  sources: " + "; ".join(f"{s['source']} ({s['rows']} rows, SHA-256 {s['source_sha256'][:16]}…)" for s in meta["sources"]))
        print(f"  SHA-256 {meta['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
