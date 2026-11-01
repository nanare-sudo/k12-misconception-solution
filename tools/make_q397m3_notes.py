#!/usr/bin/env python3
""""The 397B finds, M3 names": notes of Qwen3.5-397B in the format of the frozen M3 notes — input of the M3 ranking (probe
proto_rank in sft mode, `sft_from: derived/sft/q397m3_<table>_notes.jsonl`).

    python -I tools/make_q397m3_notes.py                                            # training rows (Tinker measurement)
    python -I tools/make_q397m3_notes.py --test-from runs/q397te_rp/proto_te/proto  # test rows (own compute path)

SOURCES
  training: data/derived/open_teacher/qwen35-397b/raw.jsonl (tools/open_teacher.py, phase first, exactly one record per
            row). The measurement is the same as that of the channel q397: labels.jsonl next to it must carry the checksum
            recorded for the q397 decision (eda/c110_q397_deploy.TINKER_SHA256), otherwise abort.
  test:     artifacts/runs/<run>/<run>/<job>/<tag>_raw.jsonl of the proto_gen job (vlm/proto.generate) with
            <tag>_meta.json (table test) and summary.json (model Qwen/Qwen3.5-397B-A17B-FP8; --expect-model).
            Test notes are never inspected: this tool reads them, checks them by machine and only prints counts,
            quantiles and checksums.

NOTES per row = the THINKING of the first pass, exactly as for the pr27 notes that M3 was trained on
(tools/export_proto_notes.py -> vlm/proto.rank_notes): without the answer text (the JSON names the codes of the 397B — the
ranking behind it would merely copy it), without thinking markers, without surrounding whitespace, at most the last
16,000 characters. Tinker records that stopped at the token limit carry the raw text under `text` (thinking empty):
tools/open_teacher.tinker_split splits at the end of the thinking block, an answer text started after it is dropped — just
as the own compute path separates thinking and answer while writing. There is no further cleaning. If a marker of the chat
template (`<think>`, `</think>`, `<|…|>`) still appears in a note afterwards, the tool aborts (it would become a real
control token during scoring).

FORMAT as in vlm/sft.generate and tools/freeze_m3_texts.py (<tag>_notes.jsonl, read by vlm/sft.load_notes):
    {"table", "row", "state": "open" | "empty", "text": notes, "finding": null, "q397_state": <recorded label, see RECORDED_STATE>}
finding = null: M3 writes its finding in the ranking ITSELF (runner._sft_finish, greedy up to "Root:", finish_tokens)
and then scores all codes exactly behind "NOTES\\n</think>\\n\\nFinding: FINDING\\nRoot:" (vlm/sft.rank_prefix).
q397_state (label-free, evaluation only; as in tools/export_proto_notes.note_state): complete = first pass ended regularly,
draft = stopped, but the note already names "root_code", open = stopped before the verdict; missing = no record (only with
--allow-missing; then empty text and state "empty": M3 decides there without a 397B note). The field stores the recorded label
of the state (RECORDED_STATE).

OUTPUT: data/derived/sft/q397m3_<table>_notes.jsonl (+ .meta.json with source, checksums, statistics, SHA-256 of the file) and a
copy of both files in the upload folder — data/derived/sft_q397m3/ (training) or data/derived/sft_q397m3te/ (test); every
upload folder contains only its one note file and its meta file (it is mounted as data/derived/sft on the remote machine).
The file is byte-identically reproducible (no timestamps).
Statistics (aggregates only): rows, states, characters and tokens (tokenizer Qwen3.5-0.8B, assumed to share the vocabulary of
the 27B; empty = without tokens) as quantiles, share above 2,500 tokens (note limit in the M3 training), truncated notes,
dropped answer texts. Training additionally: token lengths of M3's own notes (draws d1, d3) and of the pr27 notes (notes of
the M3 training targets) and — if the fidelity check of the own compute path exists locally — the notes of the own compute
path against Tinker on the same training rows (transfer OOF -> test side).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

MARKS = re.compile(r"</?think>|<\|[^|<>\n]{1,40}\|>")  # as in tools/export_proto_notes.py
NOTE_CHARS = 16000  # as in export_proto_notes (--note-chars) and the finishing pass of the protocol
TOKEN_CAP = 2500  # note limit of the M3 training (sft_notes_max_tokens) — statistic only, nothing is truncated
TINKER_DIR = Path("derived") / "open_teacher" / "qwen35-397b"  # relative to the data root
TINKER_LABELS_SHA256 = "6935614248a22da250054dd14b180fc5875832ddba7afc49c4b3da0337503280"  # = eda/c110_q397_deploy.TINKER_SHA256
TEST_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
PARITY_DEFAULT = "runs/q397p_rp/proto_p/proto"  # fidelity check of the own compute path: 64 training rows
UPLOAD = {"train": Path("derived") / "sft_q397m3", "test": Path("derived") / "sft_q397m3te"}
NAMES = {"train": "q397m3_train_notes", "test": "q397m3_test_notes"}
# M3's own notes for the same training rows (only for comparing lengths): draw -> outputs of the sft_gen jobs
M3_NOTES = {
    "d1": [("m3_pilot_rp", "m3_gen_f1"), ("m3_pilot_rp", "m3_gen_f2"), ("m3_f034_rp", "m3_gen_f0"), ("m3_f034_rp", "m3_gen_f3"),
           ("m3_f034_rp", "m3_gen_f4")],
    "d3": [("m3d3_oof_rp", f"m3d3_gen_f{k}") for k in range(5)],
}
PR27_NOTES = Path("derived") / "proto_notes" / "train_pr27.jsonl"  # notes of the M3 training targets (tools/make_sft_targets.py)
QUANTS = (5, 25, 50, 75, 90, 95, 100)
# State of a note (field q397_state) -> label recorded in the note files. The labels are the ones the note files were written
# with (short German words); the note files are model inputs pinned by checksum, so the labels are kept as recorded.
RECORDED_STATE = {"complete": "fertig", "draft": "entwurf", "open": "offen", "missing": "fehlt"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OT = _load("q397m3_open_teacher", COMP / "tools" / "open_teacher.py")  # tinker_split: the same split as used for the Tinker measurement (not copied)


class ConvertError(RuntimeError):
    """Check failed: nothing written."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    """Records of an output file (split at \\n only: U+2028 and similar stand raw in the texts)."""
    out = []
    for k, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n")):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            raise ConvertError(f"{Path(path).name}: line {k + 1} is not JSON (interrupted write?)") from None
    return out


def end_of(rec: dict) -> str:
    """End of the first pass: field `end` (own compute path) or `stop` (Tinker)."""
    return str(rec.get("end", rec.get("stop", "")) or "")


def thinking_of(rec: dict, kind: str) -> str:
    """Thinking of a first pass. tinker: open_teacher.tinker_split (raw text under `text` if the template could not split at the
    token limit); own: field `thinking` (vlm/proto.generate splits while writing)."""
    if kind == "tinker":
        return OT.tinker_split(rec)[0]
    if kind == "own":
        return str(rec.get("thinking") or "")
    raise ValueError(f"unknown source kind {kind!r}")


def answer_of(rec: dict, kind: str) -> str:
    """Answer text of a first pass (NOT taken over; only counted)."""
    return OT.tinker_split(rec)[1] if kind == "tinker" else str(rec.get("text") or "")


def note_of(thinking: str, note_chars: int = NOTE_CHARS) -> tuple[str, bool]:
    """Note in the format of the pr27 notes (vlm/proto.rank_notes): thinking markers removed, edges trimmed, last note_chars
    characters. -> (note, truncated)."""
    full = str(thinking or "").replace("<think>", "").replace("</think>", "").strip()
    note = PR.rank_notes({"phase": "first", "thinking": thinking}, note_chars)
    return note, len(full) > int(note_chars)


def q397_state(rec: dict | None, note: str) -> str:
    """Label-free state of the note as in tools/export_proto_notes.note_state (complete / draft / open; missing without a
    record), returned as its recorded label (RECORDED_STATE)."""
    if rec is None:
        return RECORDED_STATE["missing"]
    if end_of(rec) == "stop":
        return RECORDED_STATE["complete"]
    return RECORDED_STATE["draft"] if "root_code" in note else RECORDED_STATE["open"]


def first_records(records: list[dict], n: int, table: str) -> dict[int, dict]:
    """First passes per table position; positions integer in [0, n), at most one per row, field table (if present)."""
    out: dict[int, dict] = {}
    for r in records:
        if r.get("table", table) != table:
            raise ConvertError(f"record of table {r.get('table')!r} in a source of table {table}")
        if r.get("phase") != "first":
            continue
        i = r.get("row")
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= int(i) < n:
            raise ConvertError(f"record with an invalid row position (table {table}, {n} rows)")
        if int(i) in out:
            raise ConvertError("a row has more than one first pass")
        out[int(i)] = r
    return out


def build_rows(firsts: dict[int, dict], n: int, table: str, kind: str, note_chars: int = NOTE_CHARS,
               allow_missing: bool = False) -> tuple[list[dict], dict]:
    """Note file in the M3 format for all n rows of the table. -> (records in table order, statistics)."""
    missing = [i for i in range(n) if i not in firsts]
    if missing and not allow_missing:
        raise ConvertError(f"{len(missing)} of {n} rows without a first pass — run incomplete (deliberately: --allow-missing)")
    rows, marked, cut, dropped, closed_text = [], 0, 0, 0, 0
    for i in range(n):
        rec = firsts.get(i)
        if rec is None:
            note, was_cut = "", False
        else:
            note, was_cut = note_of(thinking_of(rec, kind), note_chars)
            dropped += int(bool(answer_of(rec, kind).strip()))
            closed_text += int(kind == "tinker" and not str(rec.get("thinking") or "") and "</think>" in str(rec.get("text") or ""))
        marked += int(bool(MARKS.search(note)))
        cut += int(was_cut)
        rows.append({"table": table, "row": i, "state": "open" if note else "empty", "text": note, "finding": None,
                     "q397_state": q397_state(rec, note)})
    if marked:
        raise ConvertError(f"{marked} notes still contain a marker of the chat template (<think>, </think>, <|…|>) after cleaning — nothing written")
    stats = {"rows": n, "with_record": n - len(missing), "missing": len(missing),
             "states": dict(sorted(Counter(r["state"] for r in rows).items())),
             "q397_states": dict(sorted(Counter(r["q397_state"] for r in rows).items())),
             "ends_first": dict(sorted(Counter(end_of(r) for r in firsts.values()).items())),
             "notes_cut_chars": cut, "answer_text_dropped": dropped, "tinker_closed_in_text": closed_text,
             "chars": quantiles([len(r["text"]) for r in rows])}
    return rows, stats


def quantiles(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    return {str(p): int(np.percentile(a, p)) for p in QUANTS} if len(a) else {}


def body_of(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def token_counts(texts: list[str], tok) -> list[int]:
    if tok is None:
        return []
    out = []
    for s in range(0, len(texts), 256):
        enc = tok(texts[s:s + 256], add_special_tokens=False)
        out.extend(len(ids) for ids in enc["input_ids"])
    return out


def token_stats(texts: list[str], tok) -> dict | None:
    t = token_counts(texts, tok)
    if not t:
        return None
    return {"quantiles": quantiles(t), "mean": float(np.mean(t)), f"over_{TOKEN_CAP}": int(sum(x > TOKEN_CAP for x in t)), "rows": len(t)}


# ------------------------------------------------------------------------------------------------ sources

def load_tinker(data_root: Path, n: int, check_labels: bool = True) -> tuple[dict[int, dict], dict]:
    d = Path(data_root) / TINKER_DIR
    raw, lab = d / "raw.jsonl", d / "labels.jsonl"
    if not raw.is_file():
        raise ConvertError(f"{raw} missing")
    info = {"source_kind": "tinker", "source": str(TINKER_DIR / "raw.jsonl"), "source_sha256": sha256_file(raw),
            "model": "Qwen/Qwen3.5-397B-A17B (Tinker-API, tools/open_teacher.py)"}
    if check_labels:
        got = sha256_file(lab) if lab.is_file() else None
        if got != TINKER_LABELS_SHA256:
            raise ConvertError(f"{lab.name}: checksum {str(got)[:16]}… instead of {TINKER_LABELS_SHA256[:16]}… — not the measurement of the channel q397")
        info["labels_sha256"] = got
    return first_records(read_jsonl(raw), n, "train"), info


def run_dir(spec: str, art: Path) -> tuple[Path, str, str, str]:
    bits = str(spec).split("/")
    if len(bits) != 4 or bits[0] != "runs" or not all(bits[1:]):
        raise ConvertError(f"{spec}: expected runs/<run>/<job>/<tag>")
    _, run, job, tag = bits
    return Path(art) / "runs" / run / run / job, run, job, tag


def load_own(spec: str, art: Path, table: str, n: int, expect_model: str | None, full_table: bool = True) -> tuple[dict[int, dict], dict]:
    """proto_gen job of the own compute path: table according to the meta file, model according to summary.json (expect_model
    None = do not check); full_table: the job must have computed ALL n rows of the table (test side; the fidelity check is a
    subset)."""
    d, run, job, tag = run_dir(spec, art)
    raw, meta_f = d / f"{tag}_raw.jsonl", d / f"{tag}_meta.json"
    if not raw.is_file() or not meta_f.is_file():
        raise ConvertError(f"{spec}: {raw.name} or {meta_f.name} missing (run not finished or not fetched)")
    meta = json.loads(meta_f.read_text())
    if meta.get("table") != table:
        raise ConvertError(f"{spec}: output for table {meta.get('table')!r}, required {table}")
    if full_table and int(meta.get("rows", -1)) != n:
        raise ConvertError(f"{spec}: the job computed {meta.get('rows')} rows, table {table} has {n}")
    summ_f = d / "summary.json"
    model = ((json.loads(summ_f.read_text()) if summ_f.is_file() else {}).get("model") or {}).get("repo")
    if expect_model is not None and model != expect_model:
        raise ConvertError(f"{spec}: model {model!r} instead of {expect_model} (summary.json)")
    info = {"source_kind": "own", "source": spec, "source_sha256": sha256_file(raw), "meta_sha256": sha256_file(meta_f), "model": model,
            "run_answered": meta.get("answered"), "run_failed": meta.get("failed")}
    return first_records(read_jsonl(raw), n, table), info


# ------------------------------------------------------------------------------------------------ comparisons (training only)

def notes_texts(path: Path) -> dict[int, str]:
    out = {}
    for r in read_jsonl(path):
        if r.get("table") != "train":
            raise ConvertError(f"{path.name}: no training rows")
        out[int(r["row"])] = str(r.get("text") or "")
    return out


def compare_lengths(rows: list[dict], data_root: Path, art: Path, tok) -> dict:
    """Lengths of the q397 notes against M3's own notes (d1, d3) and the pr27 notes of the M3 training targets — on the same rows."""
    mine = {r["row"]: r["text"] for r in rows}
    out = {}
    sources = {name: [Path(art) / "runs" / run / run / job / "gen_notes.jsonl" for run, job in parts] for name, parts in M3_NOTES.items()}
    sources["pr27"] = [Path(data_root) / PR27_NOTES]
    for name, files in sources.items():
        miss = [str(f.relative_to(COMP)) if f.is_relative_to(COMP) else str(f) for f in files if not f.is_file()]
        if miss:
            out[name] = {"missing": miss}
            continue
        texts: dict[int, str] = {}
        for f in files:
            got = notes_texts(f)
            if set(got) & set(texts):
                raise ConvertError(f"{name}: duplicate rows in {f.name}")
            texts.update(got)
        common = sorted(set(texts) & set(mine))
        rec = {"rows": len(common), "chars": quantiles([len(texts[i]) for i in common]),
               "chars_q397": quantiles([len(mine[i]) for i in common])}
        ts = token_stats([texts[i] for i in common], tok)
        if ts is not None:
            rec["tokens"] = ts
            rec["tokens_q397"] = token_stats([mine[i] for i in common], tok)
        out[name] = rec
    return out


def parity(rows: list[dict], spec: str, art: Path, n: int, tok) -> dict | None:
    """Fidelity check (training rows in the own compute path) against the Tinker notes of the same rows: how well does the note
    that M3 reads on the training rows (Tinker) transfer to the test side (own compute path)?"""
    d, _, _, tag = run_dir(spec, art)
    if not (d / f"{tag}_raw.jsonl").is_file():
        return None
    firsts, info = load_own(spec, art, "train", n, None, full_table=False)
    own = {i: note_of(thinking_of(r, "own"))[0] for i, r in firsts.items()}
    tin = {r["row"]: r["text"] for r in rows}
    common = sorted(own)
    rec = {"source": spec, "source_sha256": info["source_sha256"], "rows": len(common),
           "identical_notes": int(sum(own[i] == tin[i] for i in common)),
           "same_state": int(sum(q397_state(firsts[i], own[i]) == rows[i]["q397_state"] for i in common)),
           "chars_own": quantiles([len(own[i]) for i in common]), "chars_tinker": quantiles([len(tin[i]) for i in common])}
    if tok is not None:
        a, b = token_counts([own[i] for i in common], tok), token_counts([tin[i] for i in common], tok)
        rec.update(tokens_own_median=float(np.median(a)), tokens_tinker_median=float(np.median(b)),
                   tokens_abs_diff_median=float(np.median(np.abs(np.asarray(a) - np.asarray(b)))))
    return rec


# ------------------------------------------------------------------------------------------------ writing

def write_outputs(rows: list[dict], meta: dict, out: Path, upload: Path | None) -> dict:
    """Write the note file + meta file; copy into the upload folder (which may only contain these two files)."""
    body = body_of(rows)
    meta = {**meta, "sha256": sha256_bytes(body.encode("utf-8"))}
    meta_text = json.dumps(meta, indent=1, ensure_ascii=False) + "\n"
    if upload is not None:
        allowed = {out.name, out.with_suffix(".meta.json").name}
        if upload.is_dir():
            extra = sorted(p.name for p in upload.iterdir() if p.name not in allowed)
            if extra:
                raise ConvertError(f"upload folder {upload} contains foreign entries {extra} — it may only carry the note file")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    out.with_suffix(".meta.json").write_text(meta_text, encoding="utf-8")
    if upload is not None:
        upload.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, upload / out.name)
        shutil.copyfile(out.with_suffix(".meta.json"), upload / out.with_suffix(".meta.json").name)
        if sha256_file(upload / out.name) != meta["sha256"]:
            raise ConvertError("copy in the upload folder differs")
    return meta


def table_len(data_root: Path, table: str) -> int:
    """Number of rows of the table (an aggregate only; no row is inspected)."""
    import pandas as pd

    return int(len(pd.read_csv(Path(data_root) / f"{table}.csv", usecols=["trace_id"], dtype=str, keep_default_na=False)))


def resolve(path: str | None, base: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else base / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test-from", help="runs/<run>/<job>/<tag> of the proto_gen job of the test side (otherwise: training rows from Tinker)")
    ap.add_argument("--expect-model", default=TEST_MODEL, help="model according to summary.json of the test side ('' = do not check; only for code checks)")
    ap.add_argument("--allow-missing", action="store_true", help="write rows without a first pass with an empty note (state empty)")
    ap.add_argument("--note-chars", type=int, default=NOTE_CHARS)
    ap.add_argument("--tokenizer", default=str(COMP / "data" / "models" / "Qwen3.5-0.8B"), help="empty = without token lengths")
    ap.add_argument("--parity", default=PARITY_DEFAULT, help="fidelity check (training rows, own compute path) against Tinker; empty = off")
    ap.add_argument("--out", help="output (relative to the data root or absolute); default derived/sft/q397m3_<table>_notes.jsonl")
    ap.add_argument("--upload", help="upload folder (relative to the data root or absolute); default derived/sft_q397m3[te]/")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--data-root", help="data root (default: vlm.common.find_data_root)")
    ap.add_argument("--artifacts", default=str(COMP / "artifacts"))
    a = ap.parse_args(argv)

    from vlm import common as C

    data_root = Path(a.data_root) if a.data_root else C.find_data_root()
    art = Path(a.artifacts)
    table = "test" if a.test_from else "train"
    try:
        n = table_len(data_root, table)
        tok = None
        if a.tokenizer:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(a.tokenizer)
        if table == "train":
            firsts, info = load_tinker(data_root, n)
        else:
            firsts, info = load_own(a.test_from, art, "test", n, a.expect_model or None)
        rows, stats = build_rows(firsts, n, table, info["source_kind"], a.note_chars, a.allow_missing)
        meta = {"name": NAMES[table], "table": table, **info, "note_chars": a.note_chars, **stats,
                "tokens": token_stats([r["text"] for r in rows], tok), "tokenizer": "Qwen3.5-0.8B" if tok is not None else None,
                "format": "vlm/sft.load_notes: table, row, state, text, finding (null: M3 writes the finding itself), q397_state",
                "created_by": "tools/make_q397m3_notes.py", "purpose": "input of the M3 ranking (sft_from) behind the 397B notes"}
        if table == "train":
            meta["compare_m3"] = compare_lengths(rows, data_root, art, tok)
            meta["parity_own_vs_tinker"] = parity(rows, a.parity, art, n, tok) if a.parity else None
        out = resolve(a.out, data_root) or data_root / "derived" / "sft" / f"{NAMES[table]}.jsonl"
        upload = None if a.no_upload else (resolve(a.upload, data_root) or data_root / UPLOAD[table])
        meta = write_outputs(rows, meta, out, upload)
    except ConvertError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 4
    tk = (meta.get("tokens") or {}).get("quantiles", {})
    print(f"written: {out} (+ .meta.json){'' if upload is None else f'; copy in {upload}/'}")
    print(f"  table {table}: {meta['rows']} rows, with record {meta['with_record']}, states {meta['states']}, 397B states {meta['q397_states']}")
    print(f"  characters {meta['chars']}; tokens {tk or '—'} (above {TOKEN_CAP}: {(meta.get('tokens') or {}).get(f'over_{TOKEN_CAP}', '—')}); "
          f"truncated {meta['notes_cut_chars']}, answer text dropped {meta['answer_text_dropped']}")
    if table == "train":
        for name, rec in meta["compare_m3"].items():
            if "missing" in rec:
                print(f"  comparison {name}: missing ({len(rec['missing'])} files)")
                continue
            t = (rec.get("tokens") or {}).get("quantiles", {})
            tq = (rec.get("tokens_q397") or {}).get("quantiles", {})
            print(f"  comparison {name} ({rec['rows']} rows): tokens median {t.get('50', '—')} / p90 {t.get('90', '—')} / max {t.get('100', '—')} "
                  f"against q397 {tq.get('50', '—')} / {tq.get('90', '—')} / {tq.get('100', '—')}")
        par = meta.get("parity_own_vs_tinker")
        if par:
            print(f"  fidelity check own compute path against Tinker ({par['rows']} training rows): identical notes {par['identical_notes']}, "
                  f"same state {par['same_state']}, tokens median {par.get('tokens_own_median', '—')} against {par.get('tokens_tinker_median', '—')}")
    print(f"  SHA-256 {meta['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
