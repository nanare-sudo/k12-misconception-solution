#!/usr/bin/env python3
"""Rotation list of the test side of channel v1r: choice repro|legacy and checks for the build scripts of the final lines.

Channel v1r (fine-tuned 9B) reads rotated test photos turned upright: for the rows of a rotation list make_submission takes the
measurement of the T4 rotation run rot_test (`--test "v1r=v1rot_t4/rot_test/test@<list>" --fill v1r=v1`), for all other rows v1.

  repro   data/derived/orient_repro/test_c80_rows.npy (344 rows) — the stored orientation network (tools/orient_net.py,
          artifacts/orient/orient_cnn.pt); every row lies in the existing rot_test run, with the same rotation. Default of the
          build scripts (ORIENT unset or repro).
  legacy  data/derived/orient/test_c80_rows.npy (350 rows) — the first version of the orientation network, whose weights were not
          stored; the earlier submissions with v1r were built from it. Only for rebuilding those historical files.

The build scripts choose the list via ORIENT=repro|legacy (tools/builds/orient_list.sh; the same constants as here, the test
tests/test_orient_list.py keeps both equal). This tool checks the chosen list and compares files as aggregates:

    python -I tools/orient_list.py spec [--mode repro|legacy]
    python -I tools/orient_list.py check --mode repro [--spec <v1r source>] [--json <file>]
    python -I tools/orient_list.py verify-sub --mode repro <sub_….json>
    python -I tools/orient_list.py compare <reference.csv> <new.csv> [--label <text>] [--json <file>]

check       path and checksum of the list of the mode (repro rejects the first-version list, also under another path or name),
            shape (unique, ascending table positions), JSON list = row list (keys = identifiers of exactly these rows), meta file,
            in repro the provenance from artifacts/orient/orient_cnn.json (lists_at_training.test_c80, weights file), coverage by
            rot_test (all rows computed) and the same rotation as the list rot_test was computed with (field
            recipe.image.rotate_from of its meta file, fixed checksum).
verify-sub  source of v1r in the written sub_*.json: exactly one draw rot_test with the row list of the mode and its row count, all
            other test rows from v1.
compare     number and share of test rows with another top code or other first three (ordered) against a reference file, split
            into the rows whose membership in the rotation list differs between legacy and repro, and all other rows.

Only counts, shares and checksums are printed — no row positions, no identifiers, no codes.
Exit: 0 passed, 2 usage, 3 input missing, 4 check failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COMP = Path(__file__).resolve().parent.parent
V1R_RUN = "v1rot_t4/rot_test/test"
MODES = {
    "legacy": {"rows": "derived/orient/test_c80_rows.npy", "rows_sha256": "f0d10dd2179cc36d08fc6d2eaee734f3a9f3a5cc894274dceea5c198ce8bd91f",
               "json": "derived/orient/test_c80.json", "json_sha256": "11b310cc87a783d9fe30b823dc0f14d2985032817c66389b778bcd08aee34dda",
               "n_rows": 350},
    "repro": {"rows": "derived/orient_repro/test_c80_rows.npy", "rows_sha256": "aa6515c8bf17bf593c5c90aa90d9240e9992d84f650c5445a37a5f015dbbff1d",
              "json": "derived/orient_repro/test_c80.json", "json_sha256": "2937d66d8c42c763ec5ba0b1c848585306ca72f8a63f0992a334d7117e4ae462",
              "n_rows": 344},
}
# rotation list rot_test turned the photos upright with (configs/vlm/v1rot_t4.yaml, field recipe.image.rotate_from of the meta file)
RUN_ROTATE_SHA256 = {"derived/orient/test_c50.json": "37d7f496ed76da6420f6dfe6f0fc7a848db4d9fe3d275faf93f0150a62af3f88"}
WEIGHTS_DESC = "artifacts/orient/orient_cnn.json"
WEIGHTS_SOURCE = "artifacts/orient/orient_cnn.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_mode(mode: str | None = None, env: dict | None = None) -> str:
    """Mode as in tools/builds/orient_list.sh: explicit, otherwise ORIENT of the environment, otherwise repro."""
    m = mode or (os.environ if env is None else env).get("ORIENT") or "repro"
    if m not in MODES:
        raise ValueError(f"ORIENT={m!r} — allowed are {sorted(MODES)}")
    return m


def v1r_spec(mode: str, modes: dict = MODES, run: str = V1R_RUN) -> str:
    return f"v1r={run}@{modes[mode]['rows']}"


def data_root() -> Path:
    """Data root as in make_submission (vlm.common.find_data_root: K12_DATA, Kaggle, <repository>/data)."""
    sys.path[:0] = [str(COMP), str(COMP / "src")]
    from vlm import common as C  # noqa: E402

    return C.find_data_root()


def test_ids(root: Path) -> list[str]:
    """Identifiers of the test rows in table order (the same order as vlm.common.load_table) — only for internal matching."""
    return pd.read_csv(Path(root) / "test.csv", dtype=str, keep_default_na=False, usecols=["trace_id"])["trace_id"].tolist()


def _meta_path(json_path: Path) -> Path:
    return json_path.with_name(json_path.name[: -len(".json")] + ".meta.json")


def check(mode: str, root: Path, comp: Path = COMP, spec: str | None = None, modes: dict = MODES, run: str = V1R_RUN,
          rotate_sha: dict = RUN_ROTATE_SHA256) -> tuple[list[str], dict]:
    """Checks of the list of the mode (see the module docstring). -> (problems, statistics); no problems = passed."""
    problems: list[str] = []
    info: dict = {"mode": mode}
    if mode not in modes:
        return [f"mode {mode!r} unknown (allowed {sorted(modes)})"], info
    m, legacy = modes[mode], modes["legacy"]
    if spec is not None:
        want = v1r_spec(mode, modes, run)
        rows_in_spec = spec.partition("@")[2]
        info["spec"] = spec
        if mode == "repro" and (rows_in_spec == legacy["rows"] or rows_in_spec.startswith("derived/orient/")):
            problems.append(f"v1r source reads the first-version rotation list ({rows_in_spec}) — rejected in the repro variant")
        if spec != want:
            problems.append(f"v1r source {spec!r} is not the one of mode {mode} ({want!r})")
    f_rows = Path(root) / m["rows"]
    if not f_rows.is_file():
        return problems + [f"row list {m['rows']} missing"], info
    sha = sha256_file(f_rows)
    info["rows_file"], info["rows_sha256"] = m["rows"], sha
    if mode == "repro" and sha == legacy["rows_sha256"]:
        problems.append(f"{m['rows']} has the checksum of the first-version rotation list — rejected in the repro variant")
    if sha != m["rows_sha256"]:
        problems.append(f"{m['rows']}: checksum {sha[:16]}… instead of {m['rows_sha256'][:16]}…")
    ids = test_ids(root)
    n_test = len(ids)
    rows = np.load(f_rows)
    info["n_test"], info["rows"] = n_test, int(rows.size)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        return problems + [f"row list: shape {rows.shape}, type {rows.dtype} instead of 1-D integer"], info
    if rows.size and (rows.min() < 0 or rows.max() >= n_test):
        problems.append("row list: position outside the test table")
    if np.unique(rows).size != rows.size or not np.all(np.diff(rows) > 0):
        problems.append("row list: positions not unique or not ascending")
    if rows.size != m["n_rows"]:
        problems.append(f"row list: {rows.size} rows instead of {m['n_rows']}")
    # JSON list (identifier -> quarter turns counter-clockwise to turn the photo upright) and meta file
    f_json = Path(root) / m["json"]
    mapping: dict = {}
    if not f_json.is_file():
        problems.append(f"JSON list {m['json']} missing")
    else:
        jsha = sha256_file(f_json)
        info["json_sha256"] = jsha
        if jsha != m["json_sha256"]:
            problems.append(f"{m['json']}: checksum {jsha[:16]}… instead of {m['json_sha256'][:16]}…")
        mapping = json.loads(f_json.read_text())
        pos = {t: i for i, t in enumerate(ids)}
        unknown = sum(1 for k in mapping if k not in pos)
        key_rows = sorted(pos[k] for k in mapping if k in pos)
        if unknown or key_rows != [int(r) for r in rows]:
            problems.append(f"JSON list and row list cover different rows ({len(mapping)} keys, {unknown} unknown)")
        turns = list(mapping.values())
        if any(t not in (1, 2, 3) for t in turns):
            problems.append("JSON list: rotation outside 1–3")
        info["turns_ccw"] = {str(k): int(sum(1 for t in turns if t == k)) for k in (1, 2, 3)}
        f_meta = _meta_path(f_json)
        if not f_meta.is_file():
            problems.append(f"meta file {f_meta.name} missing")
        else:
            meta = json.loads(f_meta.read_text())
            if meta.get("table") != "test" or int(meta.get("rows", -1)) != rows.size or int(meta.get("of", -1)) != n_test \
                    or meta.get("sha256") != jsha or meta.get("turns_ccw") != info["turns_ccw"]:
                problems.append(f"meta file {f_meta.name} does not match the list (table, rows, checksum or rotations)")
            info["meta_source"] = meta.get("source")
            if mode == "repro" and meta.get("source") != WEIGHTS_SOURCE:
                problems.append(f"meta file names {meta.get('source')!r} as source instead of {WEIGHTS_SOURCE}")
    # provenance of the repro list: description of the stored orientation network
    if mode == "repro":
        f_desc = Path(comp) / WEIGHTS_DESC
        if not f_desc.is_file():
            problems.append(f"{WEIGHTS_DESC} missing")
        else:
            desc = json.loads(f_desc.read_text())
            lt = (desc.get("lists_at_training") or {}).get("test_c80") or {}
            if lt.get("sha256_rows") != sha or lt.get("sha256_json") != info.get("json_sha256") or int(lt.get("rows", -1)) != rows.size:
                problems.append("the list is not the one of the stored orientation network (orient_cnn.json, lists_at_training.test_c80)")
            w = desc.get("weights") or {}
            f_w = Path(comp) / WEIGHTS_SOURCE
            if f_w.is_file():
                wsha = sha256_file(f_w)
                info["weights_sha256"] = wsha
                if wsha != w.get("weights_sha256"):
                    problems.append("orient_cnn.pt does not match the checksum in orient_cnn.json")
            else:
                info["weights_sha256"] = None  # weights not local (gitignored): provenance only via the description
    # coverage by the T4 rotation run and the same rotation
    r, job, tag = run.split("/")
    d = Path(comp) / "artifacts" / "runs" / r / r / job
    f_meta_run, f_idx, f_done = d / f"{tag}_meta.json", d / f"{tag}_idx.npy", d / f"{tag}_diag.done.npy"
    if not (f_meta_run.is_file() and f_idx.is_file()):
        problems.append(f"rotation run {run} missing")
    else:
        rmeta = json.loads(f_meta_run.read_text())
        idx = np.load(f_idx)
        if rmeta.get("table") != "test" or int(rmeta.get("n", -1)) != idx.size:
            problems.append(f"rotation run {run}: table {rmeta.get('table')!r}, {rmeta.get('n')} rows")
        if f_done.is_file() and not np.load(f_done).all():
            problems.append(f"rotation run {run}: measurement incomplete")
        in_run = int(np.isin(rows, idx).sum())
        info["run_rows"], info["rows_in_run"] = int(idx.size), in_run
        if in_run != rows.size:
            problems.append(f"{rows.size - in_run} rows of the list were not computed by {run}")
        rot_from = ((rmeta.get("recipe") or {}).get("image") or {}).get("rotate_from")
        info["run_rotate_from"] = rot_from
        if not rot_from:
            problems.append(f"rotation run {run}: meta file names no rotation list (recipe.image.rotate_from)")
        elif mapping:
            f_rot = Path(root) / rot_from
            if not f_rot.is_file():
                problems.append(f"rotation list of the run {rot_from} missing")
            else:
                rsha = sha256_file(f_rot)
                if rotate_sha.get(rot_from) != rsha:
                    problems.append(f"rotation list of the run {rot_from}: checksum {rsha[:16]}… is not the recorded one")
                run_turns = json.loads(f_rot.read_text())
                same = sum(1 for k, v in mapping.items() if run_turns.get(k) == v)
                info["rows_same_turn_as_run"] = int(same)
                if same != len(mapping):
                    problems.append(f"{len(mapping) - same} rows of the list were computed by {run} with another rotation or none")
    # comparison with the first-version list (counts only)
    f_old = Path(root) / legacy["rows"]
    if mode != "legacy" and f_old.is_file():
        a, b = set(np.load(f_old).tolist()), set(rows.tolist())
        info["vs_legacy"] = {"both": len(a & b), "only_legacy": len(a - b), "only_this": len(b - a), "changed_rows": len(a ^ b)}
    return problems, info


def verify_sub(mode: str, sub_json: Path, modes: dict = MODES, run: str = V1R_RUN) -> tuple[list[str], dict]:
    """Source of v1r in the JSON file written by make_submission (see the module docstring). -> (problems, statistics)."""
    problems: list[str] = []
    if mode not in modes:
        return [f"mode {mode!r} unknown"], {}
    m = modes[mode]
    rec = json.loads(Path(sub_json).read_text())
    src = (rec.get("sources") or {}).get("v1r")
    n_test = int((rec.get("check") or {}).get("n_rows", -1))
    info = {"mode": mode, "n_test": n_test}
    if not isinstance(src, list):
        return [f"{Path(sub_json).name}: no source for v1r"], info
    draws = [e for e in src if "draw" in e]
    fills = [e for e in src if "fill" in e]
    if len(draws) != 1:
        problems.append(f"v1r: {len(draws)} draws instead of one")
    else:
        dr = draws[0]
        info.update(rows_file=dr.get("rows_file"), rows_from_run=dr.get("rows"))
        if dr.get("draw") != run:
            problems.append(f"v1r: draw {dr.get('draw')!r} instead of {run}")
        if mode == "repro" and dr.get("rows_file") == modes["legacy"]["rows"]:
            problems.append("v1r: row list is the first-version rotation list — rejected in the repro variant")
        if dr.get("rows_file") != m["rows"] or dr.get("rows") != m["n_rows"]:
            problems.append(f"v1r: row list {dr.get('rows_file')!r} with {dr.get('rows')} rows instead of {m['rows']} with {m['n_rows']}")
    if len(fills) != 1 or fills[0].get("fill") != "v1" or fills[0].get("rows") != n_test - m["n_rows"]:
        problems.append(f"v1r: remainder not completely from v1 ({fills})")
    else:
        info["rows_from_v1"] = fills[0]["rows"]
    return problems, info


def _read_sub(path: Path) -> pd.Series:
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("trace_id")["root_code"]


def compare(ref_csv: Path, new_csv: Path, root: Path, modes: dict = MODES) -> dict:
    """Aggregates of the comparison of two submission files (see the module docstring); rows are matched internally only, never printed."""
    a = _read_sub(ref_csv)
    b = _read_sub(new_csv)
    if set(a.index) != set(b.index) or len(a) != len(b):
        raise ValueError("the files cover different test rows")
    b = b.loc[a.index]
    d1 = (a.str.split().str[0] != b.str.split().str[0]).to_numpy()
    d3 = (a != b).to_numpy()
    dset = (a.str.split().map(frozenset) != b.str.split().map(frozenset)).to_numpy()
    ids = test_ids(root)
    lists = {k: set(np.load(Path(root) / v["rows"]).tolist()) for k, v in modes.items()}
    only_legacy = {ids[i] for i in lists["legacy"] - lists["repro"]}
    only_repro = {ids[i] for i in lists["repro"] - lists["legacy"]}
    in_old = a.index.isin(list(only_legacy))
    in_new = a.index.isin(list(only_repro))
    changed = in_old | in_new
    n = len(a)

    def part(mask: np.ndarray) -> dict:
        return {"rows": int(mask.sum()), "other_top1": int((d1 & mask).sum()), "other_top3_ordered": int((d3 & mask).sum()),
                "other_top3_set": int((dset & mask).sum())}

    return {"rows": n, "other_top1": int(d1.sum()), "share_other_top1": float(d1.mean()), "other_top3_ordered": int(d3.sum()),
            "share_other_top3_ordered": float(d3.mean()), "other_top3_set": int(dset.sum()), "share_other_top3_set": float(dset.mean()),
            "orient_changed": part(changed), "orient_only_legacy": part(in_old), "orient_only_repro": part(in_new),
            "elsewhere": part(~changed)}


def _write_json(path: str | None, obj: dict) -> None:
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("spec", help="source of v1r for make_submission (--test)")
    p.add_argument("--mode", choices=sorted(MODES))
    p = sub.add_parser("check", help="check the list of the mode (repro: first-version list rejected, coverage, rotation, provenance)")
    p.add_argument("--mode", choices=sorted(MODES))
    p.add_argument("--spec", help="v1r source of the build script (checked against the mode)")
    p.add_argument("--json", help="write the statistics here (aggregates only)")
    p = sub.add_parser("verify-sub", help="check the source of v1r in the written sub_*.json")
    p.add_argument("--mode", choices=sorted(MODES))
    p.add_argument("sub_json")
    p = sub.add_parser("compare", help="compare two submission files as aggregates")
    p.add_argument("ref_csv")
    p.add_argument("new_csv")
    p.add_argument("--label", default="reference file")
    p.add_argument("--json", help="write the statistics here (aggregates only)")
    a = ap.parse_args(argv)
    if a.cmd == "spec":
        print(v1r_spec(select_mode(a.mode)))
        return 0
    root = data_root()
    if a.cmd == "check":
        mode = select_mode(a.mode)
        problems, info = check(mode, root, spec=a.spec)
        _write_json(a.json, {**info, "problems": problems, "ok": not problems})
        vs = info.get("vs_legacy") or {}
        print(f"  rotation list {mode}: {info.get('rows')} rows ({info.get('rows_file')}, SHA-256 {str(info.get('rows_sha256'))[:16]}…), "
              f"rotations {info.get('turns_ccw')}; computed in run {V1R_RUN} {info.get('rows_in_run')}/{info.get('rows')}, same rotation "
              f"{info.get('rows_same_turn_as_run')}/{info.get('rows')} (run turned by {info.get('run_rotate_from')})"
              + (f"; against the first-version list: in both {vs.get('both')}, only first version {vs.get('only_legacy')}, only this list {vs.get('only_this')} "
                 f"-> v1r input differs on {vs.get('changed_rows')} test rows" if vs else ""))
        for pr in problems:
            print(f"  ABORT rotation list: {pr}", file=sys.stderr)
        print(f"  rotation list {mode}: {'PASSED' if not problems else 'FAILED'}")
        return 4 if problems else 0
    if a.cmd == "verify-sub":
        mode = select_mode(a.mode)
        if not Path(a.sub_json).is_file():
            print(f"ABORT: {a.sub_json} missing", file=sys.stderr)
            return 3
        problems, info = verify_sub(mode, Path(a.sub_json))
        print(f"  source of v1r in {Path(a.sub_json).name}: {info.get('rows_from_run')} test rows from {V1R_RUN} via {info.get('rows_file')}, "
              f"{info.get('rows_from_v1')} from v1 -> {'PASSED' if not problems else 'FAILED'} (ORIENT={mode})")
        for pr in problems:
            print(f"  ABORT source of v1r: {pr}", file=sys.stderr)
        return 4 if problems else 0
    if a.cmd == "compare":
        res = compare(Path(a.ref_csv), Path(a.new_csv), root)
        _write_json(a.json, {"ref": a.ref_csv, "new": a.new_csv, "label": a.label, **res})
        oc, el = res["orient_changed"], res["elsewhere"]
        print(f"  against {a.label}: other top code in {res['share_other_top1']:.4f} of the rows ({res['other_top1']}), other first three "
              f"(ordered) in {res['share_other_top3_ordered']:.4f} ({res['other_top3_ordered']}), as a set {res['other_top3_set']}; "
              f"of these on the {oc['rows']} rows with a changed rotation list: top code {oc['other_top1']}, first three {oc['other_top3_ordered']} "
              f"(only first version {res['orient_only_legacy']['other_top1']}/{res['orient_only_legacy']['rows']}, only released list "
              f"{res['orient_only_repro']['other_top1']}/{res['orient_only_repro']['rows']} with another top code); other rows: top code "
              f"{el['other_top1']}, first three {el['other_top3_ordered']}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
