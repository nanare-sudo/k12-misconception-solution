#!/usr/bin/env python3
"""Re-record the checksums of new test-side protocol answers in the files that pin them.

    python -I tools/repin_test_side.py              # the same as --check
    python -I tools/repin_test_side.py --check      # report every pinned checksum; write nothing
    python -I tools/repin_test_side.py --write      # back up the changed files, then write the new checksums

Submissions A and B pin three test-side inputs by SHA-256: the protocol answers of the untrained 27B (run pr27te_rp),
for A also the answers of the 397B (run q397te_rp), and the protocol notes exported from the 27B answers. Mode `full` of
tools/gpu_inference.sh writes these files anew, and the two make_submission calls that write A and B then stop at the
first pin. This tool re-records exactly the fields listed in FIELDS below. It refits nothing and changes no other field.

--check prints one line per field: "unchanged" or "new <first 16 hex digits>". It exits with 0 only if every field is
unchanged. --write first copies each file it changes to artifacts/repin_backup/<UTC time>/, then writes the file in the
same JSON layout as before. The tool prints checksums only, never file contents.

Exit: 0 all unchanged (--check) or written (--write); 1 a field is new (--check); 3 a file is missing; 4 a file does
not have the expected fields or JSON layout (nothing is written); 5 a file cannot be written.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

PR27_RAW = "artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_raw.jsonl"
PR27_LABELS = "artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_labels.jsonl"
Q397_RAW = "artifacts/runs/q397te_rp/q397te_rp/proto_te/proto_raw.jsonl"
Q397_LABELS = "artifacts/runs/q397te_rp/q397te_rp/proto_te/proto_labels.jsonl"
PX27_NOTES = "derived/proto_notes_test/test_pr27.jsonl"  # relative to the data folder (data/)
EXTRA = (PR27_RAW, PR27_LABELS, Q397_RAW, Q397_LABELS)

# (file under artifacts/, path of the field in the JSON file, source file whose SHA-256 the field records)
FIELDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("stage2_primary_m3x4.json", ("mech27", "test", "sha256"), PR27_RAW),  # submission B
    ("stage2_primary_m3x5h.json", ("mech27", "test", "sha256"), PR27_RAW),  # submission A
    *(("stage2_primary_m3x5h.json", ("extra_columns", "test", "files_sha256", f), f) for f in EXTRA),
    ("stage2_primary_cols.json", ("mech27", "test", "sha256"), PR27_RAW),  # intermediate form between B and A
    *(("stage2_primary_cols.json", ("extra_columns", "test", "files_sha256", f), f) for f in EXTRA),
    ("meta_m3px27xm.json", ("rank_mean", "notes", "test", "px27_sha256"), PX27_NOTES),  # M3 ranking behind the protocol notes
)
# Fields next to the pinned ones that must name the expected file; otherwise the file has another layout.
COMPANION = {("mech27", "test", "sha256"): (("mech27", "test", "file"), PR27_RAW),
             ("rank_mean", "notes", "test", "px27_sha256"): (("rank_mean", "notes", "test", "px27_file"), PX27_NOTES)}
# JSON layouts of the writers of these files; --write keeps the one that reproduces the file byte for byte.
LAYOUTS = ({"indent": 1, "ensure_ascii": False, "end": "\n"}, {"indent": 2, "ensure_ascii": False, "end": ""},
           {"indent": 1, "ensure_ascii": True, "end": "\n"}, {"indent": 2, "ensure_ascii": True, "end": ""},
           {"indent": 1, "ensure_ascii": False, "end": ""}, {"indent": 2, "ensure_ascii": False, "end": "\n"})


class Problem(Exception):
    def __init__(self, code: int, text: str):
        super().__init__(text)
        self.code = code


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get(d: dict, path: tuple[str, ...]):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            raise KeyError(".".join(path))
        d = d[k]
    return d


def put(d: dict, path: tuple[str, ...], value) -> None:
    for k in path[:-1]:
        d = d[k]
    if path[-1] not in d:
        raise KeyError(".".join(path))
    d[path[-1]] = value


def dump(d: dict, layout: dict) -> bytes:
    return (json.dumps(d, indent=layout["indent"], ensure_ascii=layout["ensure_ascii"]) + layout["end"]).encode("utf-8")


def layout_of(raw: bytes, d: dict) -> dict | None:
    return next((lay for lay in LAYOUTS if dump(d, lay) == raw), None)


def flatten(x, prefix: tuple = ()) -> dict:
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flatten(v, prefix + (k,)))
        return out or {prefix: {}}
    return {prefix: x}


def label(path: tuple[str, ...]) -> str:
    if path[:3] == ("extra_columns", "test", "files_sha256"):
        bits = path[3].split("/")
        return f"extra_columns.test.files_sha256[{bits[2]}/{bits[-1]}]"
    return ".".join(path)


def source_file(rel: str, root: Path, data_root: Path) -> Path:
    return root / rel if rel.startswith("artifacts/") else data_root / rel


def survey(art: Path, data_root: Path) -> tuple[list[dict], dict[str, tuple[bytes, dict]]]:
    """Read every deployment file and every source file -> (one record per field, {file: (bytes, parsed JSON)})."""
    root = art.parent
    files: dict[str, tuple[bytes, dict]] = {}
    for name in dict.fromkeys(f for f, _, _ in FIELDS):
        p = art / name
        if not p.is_file():
            raise Problem(3, f"artifacts/{name}: missing")
        raw = p.read_bytes()
        files[name] = (raw, json.loads(raw))
    for name, (_, d) in files.items():
        if any(f == name and path[0] == "extra_columns" for f, path, _ in FIELDS):
            got = sorted(get(d, ("extra_columns", "test", "files_sha256")))
            if got != sorted(EXTRA):
                raise Problem(4, f"artifacts/{name}: extra_columns.test.files_sha256 names {len(got)} other files than expected")
    records = []
    for name, path, rel in FIELDS:
        d = files[name][1]
        try:
            pinned = get(d, path)
            if path in COMPANION and get(d, COMPANION[path][0]) != COMPANION[path][1]:
                raise Problem(4, f"artifacts/{name}: {'.'.join(COMPANION[path][0])} does not name {COMPANION[path][1]}")
        except KeyError as exc:
            raise Problem(4, f"artifacts/{name}: field {exc.args[0]} missing") from None
        src = source_file(rel, root, data_root)
        if not src.is_file():
            raise Problem(3, f"{rel}: missing (source of artifacts/{name} {label(path)})")
        records.append({"file": name, "path": path, "source": rel, "pinned": str(pinned), "now": sha256_file(src)})
    return records, files


def report(records: list[dict]) -> int:
    w_file = max(len(r["file"]) for r in records)
    w_field = max(len(label(r["path"])) for r in records)
    for r in records:
        where = f"{r['file'].ljust(w_file)}  {label(r['path']).ljust(w_field)}"
        if r["pinned"] == r["now"]:
            print(f"  {where}  unchanged  {r['now'][:16]}")
        else:
            print(f"  {where}  new {r['now'][:16]}  (pinned {r['pinned'][:16]})")
    new = sum(r["pinned"] != r["now"] for r in records)
    print(f"{len(records)} fields: {len(records) - new} unchanged, {new} new")
    return new


def write(records: list[dict], files: dict[str, tuple[bytes, dict]], art: Path) -> list[str]:
    """Write the changed files (backup first); -> names of the written files."""
    todo: dict[str, list[dict]] = {}
    for r in records:
        if r["pinned"] != r["now"]:
            todo.setdefault(r["file"], []).append(r)
    plans = []
    for name, recs in todo.items():
        raw, d = files[name]
        lay = layout_of(raw, d)
        if lay is None:
            raise Problem(4, f"artifacts/{name}: JSON layout not recognised; nothing written")
        new = copy.deepcopy(d)
        for r in recs:
            put(new, r["path"], r["now"])
        before, after = flatten(d), flatten(new)
        changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        if changed != {r["path"] for r in recs}:
            raise Problem(4, f"artifacts/{name}: the change would touch other fields; nothing written")
        plans.append((name, raw, dump(new, lay)))
    if not plans:
        return []
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = art / "repin_backup" / stamp
    k = 1
    while backup.exists():
        k += 1
        backup = art / "repin_backup" / f"{stamp}_{k}"
    backup.mkdir(parents=True)
    for name, raw, _ in plans:
        (backup / name).write_bytes(raw)
    for name, _, body in plans:
        tmp = art / f".{name}.repin_tmp"
        tmp.write_bytes(body)
        os.replace(tmp, art / name)
    print(f"backup of the old files: {backup.relative_to(art.parent)}/")
    return [name for name, _, _ in plans]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report every pinned checksum and write nothing (default)")
    mode.add_argument("--write", action="store_true", help="back up the changed files, then write the new checksums")
    ap.add_argument("--artifacts", default=str(COMP / "artifacts"), help="artifacts folder (default: artifacts/ of this repository)")
    ap.add_argument("--data-root", help="data folder (default: as vlm.common.find_data_root, normally data/)")
    a = ap.parse_args(argv)
    art = Path(a.artifacts)
    if a.data_root:
        data_root = Path(a.data_root)
    else:
        from vlm import common as C

        data_root = C.find_data_root()
    try:
        records, files = survey(art, data_root)
        new = report(records)
        if not a.write:
            return 1 if new else 0
        written = write(records, files, art)
        for name in written:
            print(f"written: artifacts/{name}")
        if not written:
            print("nothing to write")
        after = survey(art, data_root)[0]
        if any(r["pinned"] != r["now"] for r in after):
            raise Problem(4, "a field is still new after writing")
        print(f"check after writing: all {len(after)} fields unchanged")
        return 0
    except Problem as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.code
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
