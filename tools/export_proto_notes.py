#!/usr/bin/env python3
"""Merge the notes of the protocol channel (job proto_gen) into ONE file — the input of the probe proto_rank (`proto_from`).

    python tools/export_proto_notes.py --name train_pr27 --from pr27_rp/proto_a/proto --from pr27_rp/proto_b/proto

Each source is `<run>/<job>/<tag>` and means artifacts/runs/<run>/<run>/<job>/<tag>_raw.jsonl. Per row the REASONING of the first pass
counts (vlm/proto.rank_notes) — without the answer text, which already names its codes. All sources must concern the same table (meta
file), and no row may occur twice. Written are data/derived/proto_notes/<name>.jsonl (not versioned; {"table": …, "row": table position,
"state": …, "text": notes}) and next to it a meta file with sources, number of rows and SHA-256. `state` is label-free and serves the
evaluation only: complete (the first pass ended regularly — the verdict is in the notes), draft (cut off, the notes already name
`root_code`), open (cut off before the verdict — only there the ranking has to decide by itself). The files store these states with the
labels recorded when the frozen note files were produced (RECORDED_STATE); other labels would produce other files.

If a note contains a marker of the chat template (`<think>`, `</think>`, `<|…|>`), the export stops: such a marker would become a real
control token during scoring — the count replaces a look into the texts.

Only counts are printed — the texts themselves never appear. Notes of test rows are row-level derivatives: they are written only with
--allow-test, into a SEPARATE folder (data/derived/proto_notes_test/), so that a run on training rows does not carry them along; they are
never looked at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP / "src")]

from vlm import proto as PR  # noqa: E402

MARKS = re.compile(r"</?think>|<\|[^|<>\n]{1,40}\|>")
# note states: name -> label recorded in the frozen note files (data/derived/proto_notes*/); the order is the order of the meta file
RECORDED_STATE = {"complete": "fertig", "draft": "entwurf", "open": "offen"}


def note_state(rec: dict, text: str) -> str:
    """State of the note, without label (see the module docstring), as its recorded label (RECORDED_STATE)."""
    if rec.get("end") == "stop":
        return RECORDED_STATE["complete"]
    return RECORDED_STATE["draft"] if "root_code" in text else RECORDED_STATE["open"]


def collect(sources: list[str], runs_dir: Path, note_chars: int) -> tuple[str, dict[int, str], list[dict], dict[int, str]]:
    """-> (table, {table position: notes}, details per source, {table position: state}).
    Rows without reasoning are left out."""
    rows: dict[int, str] = {}
    states: dict[int, str] = {}
    tables, used = set(), []
    for spec in sources:
        run, job, tag = spec.split("/")
        d = runs_dir / run / run / job
        meta = json.loads((d / f"{tag}_meta.json").read_text())
        tables.add(str(meta.get("table")))
        got, empty = {}, 0
        for line in (d / f"{tag}_raw.jsonl").read_text().split("\n"):  # not splitlines(): U+2028 and similar characters stand raw in the texts
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("phase") != "first":
                continue
            if int(rec["row"]) in got:
                raise SystemExit(f"{spec}: a row has more than one first pass")
            text = PR.rank_notes(rec, note_chars)
            if text:
                got[int(rec["row"])] = text
                states[int(rec["row"])] = note_state(rec, text)
            else:
                empty += 1
        dup = set(got) & set(rows)
        if dup:
            raise SystemExit(f"{spec}: {len(dup)} rows already occur in an earlier source")
        rows.update(got)
        used.append({"source": spec, "rows": len(got), "without_notes": empty})
    if len(tables) != 1:
        raise SystemExit(f"sources concern different tables: {sorted(tables)}")
    return tables.pop(), rows, used, states


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--from", dest="sources", action="append", required=True, help="<run>/<job>/<tag>")
    ap.add_argument("--note-chars", type=int, default=16000, help="per row at most this many characters from the end of the notes (as in the completion of the protocol)")
    ap.add_argument("--allow-test", action="store_true", help="write notes of test rows (only for the deployment of the channel; never look at them)")
    a = ap.parse_args()

    table, rows, used, states = collect(a.sources, COMP / "artifacts" / "runs", a.note_chars)
    if table != "train" and not a.allow_test:
        raise SystemExit(f"the sources concern the table {table} — only with --allow-test (row-level derivatives)")
    marked = sum(bool(MARKS.search(t)) for t in rows.values())
    if marked:
        raise SystemExit(f"{marked} notes contain a marker of the chat template (<think>, </think>, <|…|>) — not exported")
    for u in used:
        print(f"  {u['source']}: {u['rows']} rows with notes, {u['without_notes']} without (table {table})")
    out = COMP / "data" / "derived" / ("proto_notes" if table == "train" else "proto_notes_test") / f"{a.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for i in sorted(rows):
            fh.write(json.dumps({"table": table, "row": int(i), "state": states[i], "text": rows[i]}, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    chars = sorted(len(t) for t in rows.values())
    by_state = {s: sum(v == s for v in states.values()) for s in RECORDED_STATE.values()}
    out.with_suffix(".meta.json").write_text(json.dumps({"name": a.name, "table": table, "rows": len(rows), "sources": used, "note_chars": a.note_chars,
                                                         "chars_median": chars[len(chars) // 2] if chars else 0, "states": by_state, "sha256": sha}, indent=1))
    print(f"written: {out.relative_to(COMP)} — {len(rows)} rows (table {table}), median characters {chars[len(chars) // 2] if chars else 0}, "
          f"states {by_state}, SHA-256 {sha[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
