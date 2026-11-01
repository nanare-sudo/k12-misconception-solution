#!/usr/bin/env python3
"""Freeze M3 texts: notes + finding per row, exactly as the ranking scored them.

An sft_gen job delivers notes per row and — if the answer was finished — the finding (`<tag>_notes.jsonl`). If the finding
is missing, the ranking job generates it before scoring (`_sft_finish`, greedy up to "Root:") and stores it as
"[finding generated] …" in `<tag>_proto_rank.jsonl`. This tool combines both into ONE file in the format of
`<tag>_notes.jsonl` in which every row carries its finding. If a second machine scores the rows with
`sft_from: <this file>`, it scores exactly the same texts — the deterministic part of the channel thus becomes repeatable
without generating again. Rows whose generated finding was empty stay without a finding (the second machine then generates
it itself; counted in the meta file).

    python tools/freeze_m3_texts.py --gen runs/m3_pilot_rp/m3_pilot_rp/m3_gen_f1/gen \\
        --rank runs/m3_pilot_rp/m3_pilot_rp/m3_rank_f1/bench --out derived/sft/m3_f1_frozen_notes.jsonl

Paths relative to artifacts/ (runs) and data/ (output). Only counts and checksums are printed, never contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
GENERATED = "[finding generated] "


def freeze(gen_notes: Path, rank_texts: Path) -> tuple[list[dict], dict]:
    gen = [json.loads(line) for line in gen_notes.read_text().splitlines() if line.strip()]
    tables = {str(r["table"]) for r in gen}
    if len(tables) != 1:
        raise ValueError(f"exactly one table expected, found {sorted(tables)}")
    made: dict[int, str] = {}
    for line in rank_texts.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            t = str(r.get("text") or "")
            if t.startswith(GENERATED):
                made[int(r["i"])] = t[len(GENERATED):]
    out, counts = [], {"rows": 0, "finding_from_gen": 0, "finding_from_rank": 0, "still_open": 0, "rank_without_gen_row": 0}
    seen = set()
    for r in gen:
        i = int(r["row"])
        if i in seen:
            raise ValueError("a table position appears twice in the generation")
        seen.add(i)
        f = r.get("finding")
        f = str(f) if f is not None and str(f).strip() else None
        if f is not None:
            counts["finding_from_gen"] += 1
            state = r.get("state", "complete")
        elif made.get(i, "").strip():
            f = made[i]
            counts["finding_from_rank"] += 1
            state = "finding_from_rank"
        else:
            counts["still_open"] += 1
            state = r.get("state", "open")
        rec = {"table": r["table"], "row": i, "text": r.get("text") or "", "finding": f, "state": state}
        if r.get("examples_sha256") is not None:
            rec["examples_sha256"] = r["examples_sha256"]
        out.append(rec)
    counts["rows"] = len(out)
    counts["rank_without_gen_row"] = len(set(made) - seen)
    if counts["rank_without_gen_row"]:
        raise ValueError(f"{counts['rank_without_gen_row']} generated findings without a row in the generation — wrong pairing?")
    return out, {"table": tables.pop(), **counts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gen", required=True, help="<run>/<job>/<tag> of the sft_gen job (under artifacts/)")
    ap.add_argument("--rank", required=True, help="<run>/<job>/<tag> of the ranking job (under artifacts/)")
    ap.add_argument("--out", required=True, help="output relative to data/, e.g. derived/sft/m3_f1_frozen_notes.jsonl")
    a = ap.parse_args()
    gen = COMP / "artifacts" / f"{a.gen}_notes.jsonl"
    rank = COMP / "artifacts" / f"{a.rank}_proto_rank.jsonl"
    rows, meta = freeze(gen, rank)
    out = COMP / "data" / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out.write_text(body, encoding="utf-8")
    meta.update(source_gen=a.gen, source_rank=a.rank, sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                gen_sha256=hashlib.sha256(gen.read_bytes()).hexdigest(), rank_sha256=hashlib.sha256(rank.read_bytes()).hexdigest())
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if not k.endswith("sha256")} | {"sha256": meta["sha256"][:16]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
