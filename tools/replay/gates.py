#!/usr/bin/env python3
"""Gates of the reasoning channels: regenerate the row lists under data/derived/think_rows/ deterministically from the
channel outputs.

    python -I tools/replay/gates.py check [--table train test] [--line all|safe|best] [--sources a40|t4] [--json <file>]
    python -I tools/replay/gates.py write --out <folder> [--table …] [--line …] [--sources …] [--json <file>]

Step S4 of the rerun. The frozen lists were originally produced partly with a development gate script (wide gate only)
and partly with ad-hoc scripts, both not included. This tool states the rules and recomputes them against the frozen lists:
training side bit-identical (bytes of the .npy files), test side via SHA-256 and count (only counts, checksums and the
NUMBER of differing rows are printed, never positions).

RULES (q = fast channel zs27q of the primary line, t = zs27t of the strictly inductive line; P(NONE) and blends as below):
    wide_q        P_q(NONE) < 0.5                                    train.npy / test.npy           reasoning channel, draw g
    wide_t        P_t(NONE) < 0.5                                    train_t / test_t               inductive line
    union         wide_q ∪ wide_t                                    train_union / test_union       both lines
    tight_q       wide_q ∩ top of the blend v1+zs27q < 0.6           train_tight_q / test_tight_q   narrow gate, primary line
    tight_t       wide_t ∩ top of the blend v1+zs27t < 0.6           train_tight_t / test_tight_t   narrow gate, inductive line
    tight_union   tight_q ∪ tight_t                                  …_tight_union                  zs27v_n2_rp
    band_q        not wide_q and P_q(NONE) < 0.95                    train_band / test_band_q       extended gate (band)
    band_t        not wide_t and P_t(NONE) < 0.95                    train_band_t / test_band_t     extended gate (band)
    band_union    band_q ∪ band_t                                    …_band_union                   draw n
    tb_q          tight_q ∪ band_q   (draw n in channel think18nb)   train_tb_q / test_tb_q         primary line
    tb_t          tight_t ∪ band_t   (channel think18sb)             train_tb_t / test_tb_t         inductive line
    n3            tight_union ∪ band_union (run zs27v_n3_rp)         test_n3                        draw n, test side
                  training: the same without D                       train_n3                       draw n, training side
    tight_union_todo   tight_union without D                         train_tight_union_todo         zs27v_n2_rp
    union_todo         union without D                               train_union_todo               zs27v_n2_rp
    rot_v50       wide_q ∩ derived/orient/train_v50_rows.npy         train_rot_v50                  control
D = wide_q ∩ folds {1, 2}: the rows that already had a reasoning text without the question block (run zs27v_n_rp,
configs/vlm/zs27v_n_rp.yaml). The blends are log-linear blends without prior: log p = log-softmax(θ_v1 · log P_v1
+ θ_x · log P_x), P clipped at 1e-9, θ fitted on LogLoss (eda/a2_probe_stack.fit_mix). Training: cross-fitted per fold
(θ from the other folds), bit-identical to artifacts/oof_v1_zs27t.npy for v1+zs27t; test: θ from all 1,500 rows,
bit-identical to artifacts/stack_v1_zs27t.json for v1+zs27t (for v1+zs27q no file existed; θ is computed the same way).

INPUTS. Training: artifacts/oof_{v1,zs27q,zs27t}.npy, configs/folds.parquet, train.csv (labels for θ), for rot_v50
derived/orient/train_v50_rows.npy. Test (`--sources a40`, the versions the frozen lists come from):
v1 = runs v1_test_rp/v1_test/test, zs27q = zs27q_test0_rp/zs27q_test0/test + zs27q_test1_rp/zs27q_test1/test,
zs27t = zs27t_all_rp/zs27t_test/test; probe diag, P = vlm.engine.probs_from_logp(lp, absent = classes without a training
example), only rows with the done marker, as in the original gate script. `--sources t4`: v1 and zs27t from the offline
T4 versions used by the submitted files (v1te_a/b_t4, px27_t4zt0/1; zs27q exists only from the A40), only to measure
how many rows would move with different numerics; the frozen lists do NOT come from them.
`--src channel=<run>/<job>/<tag>[+…]` replaces a source, `--runs-dir`/`--artifacts` point to the folders of a rerun.

LINES. `--line safe` produces only the lists of the strictly inductive line (wide_t, tight_t, band_t, tb_t) and opens only
v1 and zs27t, no channel with a teacher block (zs27q carries the teacher labels pool_a+pool_b in its question block).
`--line best` produces wide_q, tight_q, band_q, tb_q. `--line all` produces all lists (unions and n3 need both channels).

`write` writes <name>.npy (int64, ascending, np.save) into a NEW folder, never into the frozen one
(data/derived/think_rows) and never over an existing file with different content. `check` compares with the frozen
folder (or `--frozen`); exit code 1 if a list differs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parents[2]
for _p in (str(COMP), str(COMP / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WIDE_T = 0.5     # P(NONE) of the fast channel below 0.5 -> generate a reasoning text
BAND_HI = 0.95   # outside the wide gate, P(NONE) below 0.95
TIGHT_T = 0.6    # top of the blend without reasoning channel below 0.6
EPS = 1e-9       # clipping before the logarithm, as in the original blends
DONE_FOLDS = (1, 2)  # wide_q on folds 1 and 2 already had a reasoning text without the block (zs27v_n_rp)
N_FOLDS = 5

TEST_SOURCES = {
    "a40": {"v1": "v1_test_rp/v1_test/test",
            "zs27q": "zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test",
            "zs27t": "zs27t_all_rp/zs27t_test/test"},
    "t4": {"v1": "v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test",
           "zs27q": "zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test",  # only the A40 version exists
           "zs27t": "px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test"},
}
TRAIN_OOF = {"v1": "oof_v1.npy", "zs27q": "oof_zs27q.npy", "zs27t": "oof_zs27t.npy"}
STORED_MIX = {"zs27t": ("oof_v1_zs27t.npy", "stack_v1_zs27t.json")}  # stored blends for cross-checking
V50_ROWS = "derived/orient/train_v50_rows.npy"

# key, file name training, file name test (None = does not exist), lines, rule
LISTS: list[tuple[str, str, str | None, tuple[str, ...], str]] = [
    ("wide_q", "train", "test", ("best", "all"), "P_zs27q(NONE) < 0.5"),
    ("wide_t", "train_t", "test_t", ("safe", "all"), "P_zs27t(NONE) < 0.5"),
    ("union", "train_union", "test_union", ("all",), "wide_q ∪ wide_t"),
    ("tight_q", "train_tight_q", "test_tight_q", ("best", "all"), "wide_q ∩ top(v1+zs27q) < 0.6"),
    ("tight_t", "train_tight_t", "test_tight_t", ("safe", "all"), "wide_t ∩ top(v1+zs27t) < 0.6"),
    ("tight_union", "train_tight_union", "test_tight_union", ("all",), "tight_q ∪ tight_t"),
    ("band_q", "train_band", "test_band_q", ("best", "all"), "not wide_q, P_zs27q(NONE) < 0.95"),
    ("band_t", "train_band_t", "test_band_t", ("safe", "all"), "not wide_t, P_zs27t(NONE) < 0.95"),
    ("band_union", "train_band_union", "test_band_union", ("all",), "band_q ∪ band_t"),
    ("tb_q", "train_tb_q", "test_tb_q", ("best", "all"), "tight_q ∪ band_q"),
    ("tb_t", "train_tb_t", "test_tb_t", ("safe", "all"), "tight_t ∪ band_t"),
    ("n3", "train_n3", "test_n3", ("all",), "tight_union ∪ band_union (training: without D)"),
    ("tight_union_todo", "train_tight_union_todo", None, ("all",), "tight_union without D"),
    ("union_todo", "train_union_todo", None, ("all",), "union without D"),
    ("rot_v50", "train_rot_v50", None, ("all",), "wide_q ∩ derived/orient/train_v50_rows.npy"),
]
LINE_CHANNELS = {"safe": ("v1", "zs27t"), "best": ("v1", "zs27q"), "all": ("v1", "zs27q", "zs27t")}


# ----------------------------------------------------------------------------------------------- rules (pure functions)
def gate_masks(p_none_q: np.ndarray | None = None, p_none_t: np.ndarray | None = None, top_q: np.ndarray | None = None,
               top_t: np.ndarray | None = None, done: np.ndarray | None = None, v50: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """All gates that can be formed from the given inputs -> {key: bool mask}.

    p_none_*: P(NONE) of the fast channel per row; top_*: top of the blend v1+fast channel per row; done: rows D that
    already have a reasoning text (training only; n3 and *_todo without them); v50: mask of the orientation list
    (training only, rot_v50)."""
    out: dict[str, np.ndarray] = {}
    for key, p, top in (("q", p_none_q, top_q), ("t", p_none_t, top_t)):
        if p is None:
            continue
        p = np.asarray(p, dtype=float)
        wide = p < WIDE_T
        out[f"wide_{key}"] = wide
        out[f"band_{key}"] = (p < BAND_HI) & ~wide
        if top is not None:
            if np.shape(top) != np.shape(p):
                raise ValueError(f"top_{key} and p_none_{key} have different lengths")
            out[f"tight_{key}"] = wide & (np.asarray(top, dtype=float) < TIGHT_T)
            out[f"tb_{key}"] = out[f"tight_{key}"] | out[f"band_{key}"]
    if "wide_q" in out and "wide_t" in out:
        out["union"] = out["wide_q"] | out["wide_t"]
        out["band_union"] = out["band_q"] | out["band_t"]
        if "tight_q" in out and "tight_t" in out:
            out["tight_union"] = out["tight_q"] | out["tight_t"]
            n3 = out["tight_union"] | out["band_union"]
            if done is not None:
                d = np.asarray(done, dtype=bool)
                out["n3"] = n3 & ~d
                out["tight_union_todo"] = out["tight_union"] & ~d
                out["union_todo"] = out["union"] & ~d
            else:
                out["n3"] = n3
    if v50 is not None and "wide_q" in out:
        out["rot_v50"] = out["wide_q"] & np.asarray(v50, dtype=bool)
    return out


def done_rows(wide_q: np.ndarray, folds: np.ndarray, done_folds: tuple[int, ...] = DONE_FOLDS) -> np.ndarray:
    """D: wide gate of the primary line on the folds that already had reasoning texts without the block."""
    return np.asarray(wide_q, dtype=bool) & np.isin(np.asarray(folds), list(done_folds))


def positions(mask: np.ndarray) -> np.ndarray:
    """Mask -> ascending table positions (int64), the format of the frozen lists."""
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)


def npy_bytes(pos: np.ndarray) -> bytes:
    """Bytes of a list as np.save writes them (for the bit-identical comparison without a file)."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(pos, dtype=np.int64))
    return buf.getvalue()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def list_names(table: str, line: str) -> list[tuple[str, str, str]]:
    """(key, file name without .npy, rule) of the lists of a table and line."""
    out = []
    for key, tr, te, lines, rule in LISTS:
        name = tr if table == "train" else te
        if name is not None and line in lines:
            out.append((key, name, rule))
    return out


# ----------------------------------------------------------------------------------------------- blends
def _a2():
    spec = importlib.util.spec_from_file_location("gates_a2", COMP / "eda" / "a2_probe_stack.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _logs(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(P, EPS, 1))


def crossfit_mixture(chans: list[np.ndarray], y: np.ndarray, folds: np.ndarray, a2=None) -> np.ndarray:
    """OOF blend without prior, as in the original gate computation: per fold θ from the other folds (a2.fit_mix), applied with a2.apply_mix."""
    a2 = a2 or _a2()
    logs = [_logs(c) for c in chans]
    P = np.zeros_like(logs[0])
    for k in sorted(set(np.asarray(folds).tolist())):
        tr, va = folds != k, folds == k
        th = a2.fit_mix([c[tr] for c in logs], None, y[tr])
        P[va] = a2.apply_mix(th, [c[va] for c in logs], None)
    return P


def full_theta(chans: list[np.ndarray], y: np.ndarray, a2=None) -> np.ndarray:
    """θ from all training rows (for the test side), as in the original gate computation."""
    a2 = a2 or _a2()
    return a2.fit_mix([_logs(c) for c in chans], None, y)


def apply_theta(theta: np.ndarray, chans: list[np.ndarray], a2=None) -> np.ndarray:
    a2 = a2 or _a2()
    return a2.apply_mix(np.asarray(theta, dtype=float), [_logs(c) for c in chans], None)


# ----------------------------------------------------------------------------------------------- inputs
def load_test_channel(spec: str, n_test: int, absent: np.ndarray, runs_dir: Path) -> np.ndarray:
    """Probabilities of a test channel from run outputs (probe diag), as in the original gate script; every test row
    covered exactly once, otherwise abort."""
    from vlm.engine import probs_from_logp

    P = np.full((n_test, len(absent)), np.nan)
    seen = np.zeros(n_test, dtype=np.int64)
    for shard in spec.split("+"):
        run, job, tag = shard.split("/")[:3]
        d = runs_dir / run / run / job
        meta = json.loads((d / f"{tag}_meta.json").read_text())
        if meta.get("table") != "test":
            raise SystemExit(f"{shard}: no output for test rows (table={meta.get('table')!r})")
        idx = np.load(d / f"{tag}_idx.npy")
        lp = np.load(d / f"{tag}_diag.npy")
        done = d / f"{tag}_diag.done.npy"
        if done.is_file():
            keep = np.load(done)
            idx, lp = idx[keep], lp[keep]
        P[idx] = probs_from_logp(lp, absent=absent)
        seen[idx] += 1
    if not (seen == 1).all():
        raise SystemExit(f"{spec}: coverage incomplete: {int((seen == 0).sum())} test rows missing, {int((seen > 1).sum())} twice")
    return P


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(table: str, line: str, sources: str = "a40", src_over: dict[str, str] | None = None, runs_dir: Path | None = None,
          art_dir: Path | None = None, data_root: Path | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Masks of all lists of a table and line from the inputs; -> (masks, report without row contents)."""
    import labels
    from vlm import common as C

    runs_dir = runs_dir or COMP / "artifacts" / "runs"
    art_dir = art_dir or COMP / "artifacts"
    root = data_root or C.find_data_root()
    train = C.load_table(root, "train")
    folds = C.load_folds(len(train))
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes)
    absent = ~labels.seen_mask(train["root_code"], classes)
    need = LINE_CHANNELS[line]
    a2 = _a2()
    oof = {k: np.load(art_dir / TRAIN_OOF[k]) for k in need}
    rep: dict = {"table": table, "line": line, "inputs": {f"artifacts/{TRAIN_OOF[k]}": _file_sha(art_dir / TRAIN_OOF[k])[:16] for k in need},
                 "checks": {}}
    fast = [k for k in ("zs27q", "zs27t") if k in need]
    if table == "train":
        p_none = {k: oof[k][:, 0] for k in fast}
        top = {}
        for k in fast:
            M = crossfit_mixture([oof["v1"], oof[k]], y, folds, a2)
            top[k] = M.max(1)
            stored = STORED_MIX.get(k)
            if stored and (art_dir / stored[0]).is_file():
                rep["checks"][f"oof_v1_{k} cross-fitted again == artifacts/{stored[0]}"] = bool(np.array_equal(M, np.load(art_dir / stored[0])))
        wide_q = p_none["zs27q"] < WIDE_T if "zs27q" in p_none else None
        done = done_rows(wide_q, folds) if (wide_q is not None and line == "all") else None
        v50 = None
        if line == "all":
            v = np.zeros(len(train), dtype=bool)
            v[np.load(root / V50_ROWS)] = True
            v50 = v
            rep["inputs"][f"data/{V50_ROWS}"] = _file_sha(root / V50_ROWS)[:16]
        masks = gate_masks(p_none.get("zs27q"), p_none.get("zs27t"), top.get("zs27q"), top.get("zs27t"), done=done, v50=v50)
    else:
        test = C.load_table(root, "test")
        srcs = {**TEST_SOURCES[sources], **(src_over or {})}
        P = {k: load_test_channel(srcs[k], len(test), absent, runs_dir) for k in need}
        for k in need:
            rep["inputs"][f"runs:{k}"] = srcs[k]
        top, theta = {}, {}
        for k in fast:
            th = full_theta([oof["v1"], oof[k]], y, a2)
            theta[k] = [float(v) for v in th]
            stored = STORED_MIX.get(k)
            if stored and (art_dir / stored[1]).is_file():
                th_st = np.asarray(json.loads((art_dir / stored[1]).read_text())["theta"], dtype=float)
                rep["checks"][f"θ v1+{k} from all rows == artifacts/{stored[1]}"] = bool(np.array_equal(th, th_st))
            top[k] = apply_theta(th, [P["v1"], P[k]], a2).max(1)
        rep["theta"] = theta
        p_none = {k: P[k][:, 0] for k in fast}
        masks = gate_masks(p_none.get("zs27q"), p_none.get("zs27t"), top.get("zs27q"), top.get("zs27t"))
    return masks, rep


# ----------------------------------------------------------------------------------------------- comparison and writing
def compare(masks: dict[str, np.ndarray], table: str, line: str, frozen_dir: Path) -> list[dict]:
    """Per list: rows, SHA-256 new/frozen, bit-identical; number of differing rows (for training also per direction)."""
    out = []
    for key, name, rule in list_names(table, line):
        if key not in masks:
            raise SystemExit(f"{name}: rule {key} cannot be formed from the inputs")
        m = masks[key]
        b = npy_bytes(positions(m))
        rec = {"list": f"{name}.npy", "rule": rule, "rows": int(m.sum()), "sha256": sha256(b)}
        f = frozen_dir / f"{name}.npy"
        if f.is_file():
            fb = f.read_bytes()
            s = np.zeros(len(m), dtype=bool)
            s[np.load(f)] = True
            rec.update(frozen_rows=int(s.sum()), frozen_sha256=sha256(fb), equal=b == fb, differing_rows=int((s ^ m).sum()))
            if table == "train":
                rec.update(only_frozen=int((s & ~m).sum()), only_new=int((m & ~s).sum()))
        else:
            rec.update(frozen_sha256=None, equal=None)
        out.append(rec)
    return out


def write_lists(masks: dict[str, np.ndarray], table: str, line: str, out_dir: Path, frozen_dir: Path) -> list[str]:
    """Lists as <name>.npy into a separate folder; never into the frozen one, never over another existing file."""
    out_dir = Path(out_dir)
    if out_dir.resolve() == Path(frozen_dir).resolve():
        raise SystemExit(f"--out {out_dir}: this is the frozen folder; nothing is written there")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, name, _ in list_names(table, line):
        b = npy_bytes(positions(masks[key]))
        f = out_dir / f"{name}.npy"
        if f.is_file():
            if f.read_bytes() != b:
                raise SystemExit(f"{f}: exists with different content; not overwritten")
            continue
        f.write_bytes(b)
        written.append(str(f))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["check", "write"])
    ap.add_argument("--table", nargs="+", default=["train", "test"], choices=["train", "test"])
    ap.add_argument("--line", default="all", choices=sorted(LINE_CHANNELS))
    ap.add_argument("--sources", default="a40", choices=sorted(TEST_SOURCES), help="test sources of the fast channels (a40 = state of the frozen lists)")
    ap.add_argument("--src", action="append", default=[], help="<channel>=<run>/<job>/<tag>[+…]: replace a test source (v1, zs27q, zs27t)")
    ap.add_argument("--runs-dir", type=Path, default=None, help="folder of the run outputs (default artifacts/runs)")
    ap.add_argument("--artifacts", type=Path, default=None, help="folder of the OOF artifacts (default artifacts)")
    ap.add_argument("--data", type=Path, default=None, help="data root (default: vlm.common.find_data_root)")
    ap.add_argument("--frozen", type=Path, default=None, help="comparison folder (default <data>/derived/think_rows)")
    ap.add_argument("--out", type=Path, help="write: target folder (new)")
    ap.add_argument("--json", type=Path, help="report (counts and checksums only)")
    a = ap.parse_args(argv)
    src_over = dict(s.split("=", 1) for s in a.src)
    bad = set(src_over) - {"v1", "zs27q", "zs27t"}
    if bad:
        raise SystemExit(f"--src: unknown channels {sorted(bad)}")
    if a.mode == "write" and a.out is None:
        raise SystemExit("write requires --out")
    from vlm import common as C

    root = a.data or C.find_data_root()
    frozen = a.frozen or root / "derived" / "think_rows"
    report = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": a.mode, "line": a.line, "sources": a.sources,
              "src_override": src_over, "frozen_dir": str(frozen), "tables": {}}
    all_equal = True
    for table in a.table:
        masks, rep = build(table, a.line, a.sources, src_over, a.runs_dir, a.artifacts, root)
        rows = compare(masks, table, a.line, frozen)
        rep["lists"] = rows
        if a.mode == "write":
            rep["written"] = write_lists(masks, table, a.line, a.out, frozen)
        report["tables"][table] = rep
        print(f"== {table} (line {a.line}, sources {a.sources if table == 'test' else 'OOF'}): " +
              "; ".join(f"{k}: {v}" for k, v in rep["checks"].items()))
        for r in rows:
            eq = {True: "BIT-IDENTICAL", False: "DIFFERS", None: "no frozen list"}[r["equal"]]
            extra = ""
            if r["equal"] is not None:
                extra = f" | frozen {r['frozen_rows']} ({r['frozen_sha256'][:16]}) | differing rows {r['differing_rows']}"
                if table == "train":
                    extra += f" (only frozen {r['only_frozen']}, only new {r['only_new']})"
            print(f"  {r['list']:<28} {r['rows']:5d} rows ({r['sha256'][:16]}) {eq}{extra}")
            all_equal &= r["equal"] is not False
    report["all_equal"] = all_equal
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        print(f"report: {a.json}")
    print("ALL LISTS BIT-IDENTICAL" if all_equal else "AT LEAST ONE LIST DIFFERS")
    return 0 if all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
