"""Deployment path of the intermediate build primary_cols (prefix c151) = primary submission B (eda/c135_primary_m3x4.py,
tools/builds/primary_m3x4.sh) with extra columns at the end of the second stage — variants (--cols) q (the 397B group of the bundle test), qf,
qfo, qfop (NINE sources, the bundle of the second round of the bundle test), qfp (q + fac27_rule + psy_adm_sec without oth_novelty, whose task
count violates the test guard; the deployed variant and the default) — production code src/stage2_sources.py. Blend UNCHANGED. Build path
tools/builds/primary_cols.sh (variant qfp).

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c151_primary_cols.py test-sources                            # test sources: completeness, parser cross-check, note checksums
    python -I eda/c151_primary_cols.py fit (--save | --verify) [--base m3x4] [--cols qfp]   # deployment file (names: out_name)
    python -I eda/c151_primary_cols.py copy [--base …] [--cols …]              # COPY NULL POINT of the training side (columns := 0) -> private copy deployment file
    python -I eda/c151_primary_cols.py measure [--base …] [--cols …] [--seeds orig,4000-4019]  # measurement per split: base, columns, placebo
    python -I eda/c151_primary_cols.py null --source F --out G [--base …]      # null model (weight 0) for the deployment file
    python -I eda/c151_primary_cols.py test-guard --report R                   # test guard from the report of a make_submission run
    python -I eda/c151_primary_cols.py controls --null N --dir D --name STEM -- <make_submission sources>   # positive controls of the test guard

SOURCES (src/stage2_sources.SOURCE_ORDER; definitions exactly as in the measurement): six protocol columns of the sources pr27 (own
computation path) and q397 (training Tinker, test own computation path q397te_rp with the parser of the Tinker measurement), fac27_rule (pr27
final_answer_correct × {minor, ANS-01}), oth_novelty (novelty statistics of the blend channels × OTH-00), psy_adm_sec (section of the
candidate seen as gold within the topic). Label-dependent are k_nov_single and psy_adm_sec: labels only from the reference rows (other folds;
training blocks without the outer fold; test rows all 1,500).
CHAIN (`fit`): context of the base (primary_m3x4: c135.m3px27_context + c135.swap_ctx with m3x4_full and stack_primary_m3x4.json),
c121.prepare (standard cross-fit, β "select" per outer fold, nested forms). (1) REGRESSION in the process: deployment features
main_guard_i1_mm and main model = models.main of the deployment file of the base (checksum of the features, w, mu, sd, gate, in-sample).
(2) Nested recomputation in F, oL, M: MM (c124.run_mm) and MM + columns (run_x: as c124.run_mm, the columns at the end of every block as in
the frame of the measurement). (3) Deployment features = MM + columns (reference per fold: the other folds) = evaluated OOF features of the
form M (null point (e)). (4) Main model (gate with guard, λ 0.1, K 5) -> deployment file = fields of the base, models.main replaced, fallback
unchanged, new: extra_columns (sources, column names, blend channels, files with checksums, matrix checksums, reference and rule of the test
guard) and cols_build. Only contents that are equal without private files.
COPY NULL POINT (`copy`): the same path with columns := 0 -> rows = MM bit-identical, main model = models.main of the base (w, mu, sd of the
first 64 equal; column weights 0), all other fields equal; the copy deployment file carries the real field extra_columns (the test side
computes the real columns, weight 0) -> build path: CSV and test matrix byte-identical to the base.
MEASUREMENT (`measure`): per split (original split or c121.split_folds) MM, MM + columns and a placebo (values permuted within the row across
the candidates), without gate, in F and M; over the splits ½(F+M), F, M against MM and real − placebo (aggregates only, training rows).
TEST GUARD: src/stage2_sources.GUARD_RULE (pre-registered); make_submission computes it on every run with the field extra_columns and writes no
file if it is violated. `test-guard` reads the report (--stage2-report), exit 5 if violated; `controls` runs make_submission for every
applicable positive control (--stage2-xcontrol) and requires each of them to fire the guard (otherwise exit 6).
Only aggregates and checksums are printed. Exit 3 = input missing; 4 = checksum/recomputation/null point; 5 = test guard violated;
6 = positive control does not fire.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import stage2_sources as XS  # noqa: E402

ART = COMP / "artifacts"
PRIV0 = ART / "eda" / "private" / "primary_cols"
M3X4, M3X4_OOF_SHA = "m3x4_full", "96ad8725bbaa9d274376a0bb68549abc4051878171ebb572f7707962600399a6"
STACK_M3X4_SHA = ""  # stack_primary_m3x4.json (eda/c135_primary_m3x4.py stack); empty = first build of this repository: printed, then pinned
BASES = {
    "m3x4": {"label": "primary_m3x4", "stem": "primary_m3x4", "stage2": "stage2_primary_m3x4.json",
             "stage2_sha": "", "stack_sha": STACK_M3X4_SHA,
             "out": "primary_cols", "tag": "cols", "mid": "m3px27xm", "build_field": "m3x4_build", "units": [7592, 7542], "candidate": "primary_cols"},
}
DEFAULT_COLS = "qfp"
FORMS3 = ("fold", "pre", "set")
UNITS_ROWS = 9000
DECISION = "bundle test of the extra columns (expected-value rule)"
KNOWLEDGE_NOTE = ("labels and task count only from the reference rows — training rows: the other folds (training blocks of the nested "
                  "recomputation also without the outer fold); test rows: all 1,500 training rows (as the task knowledge of the second stage); "
                  "key question_id|topic_name (stage2.task_keys), topic topic_name")

_MODS: dict = {}


class TechnicalError(RuntimeError):
    """Checksum, recomputation or null point violated: exit 4."""


class Missing(TechnicalError):
    """Input missing: exit 3."""


class GuardFail(RuntimeError):
    """Test guard violated: exit 5."""


class ControlFail(RuntimeError):
    """Positive control does not fire: exit 6."""


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def c135():
    """eda/c135_primary_m3x4.py — loaded once per process under one module name (one primary_m3px27 context per process; eda/c166 uses the
    same module through this function)."""
    if "c135" not in _MODS:
        _MODS["c135"] = _load("c151_c135", COMP / "eda" / "c135_primary_m3x4.py")
    return _MODS["c135"]


def c133():
    return c135().c133()


def c118():
    return c133().c118()


def c121():
    return c133().c121()


def c124():
    return c133().c124()


def S2():
    return c124().S2


def say(*a) -> None:
    print(*a, flush=True)


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(COMP))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    return XS.sha256_file(Path(path))


def sha256_array(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()


def to_json(obj) -> str:
    return c133().to_json(obj)


def units_of(rows: np.ndarray) -> int:
    u = np.rint(6.0 * np.asarray(rows, dtype=np.float64))
    if not np.allclose(u / 6.0, rows, rtol=0.0, atol=1e-9):
        raise TechnicalError("row values are not multiples of 1/6")
    return int(u.sum())


def exit_code_of(exc: BaseException) -> int | None:
    names = {k.__name__ for k in type(exc).__mro__}
    if "GuardFail" in names:
        return 5
    if "ControlFail" in names:
        return 6
    if "SourceError" in names:
        return 4
    return 3 if "Missing" in names else (4 if "TechnicalError" in names else None)


def out_name(base: str, cols: str) -> str:
    """Stem of the outputs: primary_cols for the deployed variant qfp, otherwise with the suffix of the variant."""
    b = BASES[base]["out"]
    return b if cols == DEFAULT_COLS else f"{b}_{cols}"


def tag_of(base: str, cols: str) -> str:
    t = BASES[base]["tag"]
    return t if cols == DEFAULT_COLS else f"{t}{cols}"


def priv(base: str, cols: str) -> Path:
    """Private report directory of a variant: artifacts/eda/private/<stem> (primary_cols for qfp)."""
    return PRIV0.parent / out_name(base, cols)


def write_or_verify(path: Path, text: str, save: bool) -> str:
    """--save: writes what is missing, requires the same content for existing files (never overwrites a differing file); --verify:
    requires the file."""
    if save:
        if path.is_file() and path.read_text() != text:
            raise TechnicalError(f"{rel(path)} exists with different content — not overwritten (delete it deliberately, then rebuild)")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            path.write_text(text)
    else:
        if not path.is_file():
            raise Missing(f"{rel(path)} missing — build it with --save first")
        if path.read_text() != text:
            raise TechnicalError(f"{rel(path)} is not the recomputation")
    return sha256_file(path)


def check_pin(path: Path, want: str, what: str) -> None:
    """Checksum pin of an input: a set pin is enforced; an empty pin (first build of this repository, the file is written by its own code)
    prints the checksum instead — pin it here, then the second build must give the same file."""
    got = sha256_file(path)
    if not want:
        say(f"FIRST BUILD — checksum {got} {rel(path)}")
    elif got != want:
        raise TechnicalError(f"{rel(path)} is not {what} (checksum)")


# ------------------------------------------------------------------------------------------------ context of the base and columns

def m3px27() -> dict:
    if "ctx_m3px27" not in _MODS:
        _MODS["ctx_m3px27"] = c135().m3px27_context()
    return _MODS["ctx_m3px27"]


def base_files_check(base: str) -> dict:
    b = BASES[base]
    f, st = ART / b["stage2"], ART / f"stack_{b['stem']}.json"
    for p, want in ((f, b["stage2_sha"]), (st, b["stack_sha"])):
        if not p.is_file():
            raise Missing(f"{rel(p)} missing — run the build path of the base ({b['label']}) first")
        check_pin(p, want, f"the file of the base {b['label']}")
    return json.loads(f.read_text())


def base_context(base: str) -> dict:
    """Training context of the base: primary_m3x4 = c135.m3px27_context + c135.swap_ctx (m3x4_full, stack_primary_m3x4.json). Pinned
    inputs."""
    B = c135()
    f = ART / f"oof_{M3X4}.npy"
    if not f.is_file() or sha256_file(f) != M3X4_OOF_SHA:
        raise TechnicalError(f"{rel(f)} missing or not the M3 channel of primary_m3x4 (checksum)")
    check_pin(ART / "stack_primary_m3x4.json", STACK_M3X4_SHA, "the blend file of primary_m3x4")
    ctx = B.swap_ctx(m3px27(), M3X4, np.load(f), "primary_m3x4")
    if list(ctx["spec"]["names"]) != ["v1r", "zs27q", "think18nb", "m3px27xm", M3X4, "q397m3"]:
        raise TechnicalError(f"recipe channels {ctx['spec']['names']} are not those of primary_m3x4")
    return ctx


def taxonomy(ctx: dict):
    import pandas as pd

    return pd.read_csv(ctx["root"] / "competition_data" / "taxonomy_v1.csv", dtype=str, keep_default_na=False)


def train_inputs(ctx: dict, sources: list[str]) -> tuple[dict, dict, list[str]]:
    """Inputs of the columns for the 1,500 training rows: protocol answers (src/stage2_sources.TRAIN_SPEC), pr27 final_answer_correct, blend
    channels BEFORE the floor in the order of the blend (ctx mats = OOF artifacts) and the draws d1–d3 of the second stage, table, taxonomy.
    -> (inputs, details of the files, minor codes)."""
    keys = [str(k) for k in ctx["train"]["correct_answer"].tolist()]
    try:
        P, info = XS.load_side(XS.TRAIN_SPEC, "train", COMP, ctx["root"], ctx["classes"], keys)
    except FileNotFoundError as e:
        raise Missing(str(e)) from None
    tax = taxonomy(ctx)
    minor = XS.minor_vector(ctx["classes"], dict(zip(tax["id"], tax["default_severity"])))
    if not np.array_equal(minor, np.asarray(ctx["minor"], np.float64)):
        raise TechnicalError("minor mask of the columns != minor of the second stage (answer_match.minor_mask)")
    codes = [c for c, v in zip(ctx["classes"], minor) if v]
    st = ctx["static"]
    chans = [ctx["mats"][nm] for nm in ctx["spec"]["names"]]
    inp = {"classes": list(ctx["classes"]), "P": P, "minor": minor, "row_keys": ctx["key"], "ref_keys": ctx["key"], "ref_y": ctx["y"],
           "fac": XS.fac_rows([COMP / r for r in XS.TRAIN_SPEC["pr27"]["raw"]], ctx["n"]),
           "sev_minor": XS.sev_minor_raw(ctx["classes"], tax["id"].tolist(), tax["default_severity"].tolist()), "chans": chans,
           "draws": list(ctx["draws"]), "seen": ctx["seen"], "row_topics": ctx["topic_s"], "ref_topics": ctx["topic_s"],
           "sections": XS.class_sections(ctx["classes"], tax["id"].tolist(), tax["section"].tolist())}
    if "oth_novelty" in sources:  # top counts = those of the second stage (Static)
        F5 = XS.novelty_rows(chans, inp["draws"], ctx["seen"], np.isin(np.array(ctx["classes"]), XS.GENERAL))
        if not (np.array_equal(F5[:, 3], np.asarray(st.ch_ntop, float)) and np.array_equal(F5[:, 4], np.asarray(st.d_ntop, float))):
            raise TechnicalError("novelty statistics: number of distinct channel/draw top codes != stage2.Static of the chain")
    return inp, info, codes


# ------------------------------------------------------------------------------------------------ nested recomputation with columns

def run_x(ctx: dict, std: dict, nested: dict, form: str, cols, gate: bool = True) -> dict:
    """As c124.run_mm ("real"; NONE guard also in the gate) with the columns `cols` at the end of every feature block: training block j of the
    outer fold k with reference = folds except {j, k}, evaluated rows of fold k with reference = folds ≠ k (as the frame of the measurement).
    gate=False: without gate (the rows "forced" do not depend on it; abstention = forced)."""
    C, S = c118(), S2()
    y, folds, n, Kc = ctx["y"], np.asarray(ctx["folds"]), ctx["n"], S.K_CAND
    forced, abst, base = np.zeros(n), np.zeros(n), np.zeros(n)
    changed, changed_forced, changed3 = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    top_all, cand_all, protected = np.zeros((n, Kc), np.int64), np.zeros((n, Kc), np.int64), np.zeros(n, bool)
    choice, X_eval, names = {}, {}, None
    for k in range(5):
        forms_T = nested[k]
        E = np.flatnonzero(folds == k)
        XT, candT, rowsT = [], [], []
        for jf in [j for j in range(5) if j != k]:
            Rj = np.flatnonzero(folds == jf)
            X, cand, _ = C.block(ctx, Rj, forms_T[form][Rj], forms_T["pre"][Rj], S.MM_SET, exclude=(k,))
            XT.append(np.concatenate([X, cols.cols(Rj, cand, np.flatnonzero((folds != jf) & (folds != k)))], 1))
            candT.append(cand)
            rowsT.append(Rj)
        XE, candE, names = C.block(ctx, E, std[form][E], std["pre"][E], S.MM_SET)
        XE = np.concatenate([XE, cols.cols(E, candE, np.flatnonzero(folds != k))], 1)
        X_eval[k] = XE
        labT = [S.labels_of(c, y[r]) for c, r in zip(candT, rowsT)]
        g = S.gate([(XT[b], labT[b], candT[b], y[rowsT[b]]) for b in range(len(XT))], protect=True) if gate else {"abstain": False}
        model = S.fit_clogit(np.concatenate(XT), np.concatenate(labT), Kc, S.LAMBDA)
        top0 = S.reorder(candE, S.scores(model, XE, Kc))
        topE = S.protect_none(top0, candE)
        protected[E] = (top0 != topE).any(1)
        rr, bb = S.map_rows(topE, y[E]), S.map_rows(candE, y[E])
        forced[E], abst[E], base[E] = rr, (bb if g["abstain"] else rr), bb
        changed_forced[E] = topE[:, 0] != candE[:, 0]
        changed3[E] = (topE[:, :3] != candE[:, :3]).any(1)
        changed[E] = changed_forced[E] & (not g["abstain"])
        top_all[E], cand_all[E] = topE, candE
        choice[k] = g
    return {"forced": forced, "abstain": abst, "base": base, "changed_top": changed, "changed_top_forced": changed_forced,
            "changed_top3_forced": changed3, "choice": choice, "features": list(names) + list(cols.features), "X_eval": X_eval, "top": top_all,
            "cand": cand_all, "protected": protected}


ROW_KEYS = ("forced", "abstain", "top", "cand", "protected")


def same_rows(a: dict, b: dict) -> bool:
    return bool(all(np.array_equal(a[k], b[k]) for k in ROW_KEYS))


def entry(r: dict, ref: dict, rec: np.ndarray, ctx: dict, draws_q) -> dict:
    C = c118()
    d = r["forced"] - ref["forced"]
    return {"map_forced": float(r["forced"].mean()), "units_forced": units_of(r["forced"]), "map_abstain": float(r["abstain"].mean()),
            "abstain_folds": int(sum(bool(c["abstain"]) for c in r["choice"].values())),
            "changed_top_share": float(r["changed_top"].mean()), "changed_top_share_forced": float(r["changed_top_forced"].mean()),
            "changed_top3_share_forced": float(r["changed_top3_forced"].mean()), "rows_protected": int(r["protected"].sum()),
            "vs_mm": C.boot(d, ctx, draws_q), "vs_recipe_beta": C.boot(r["forced"] - rec, ctx, draws_q),
            "rows_other_top1_vs_mm": int((r["top"][:, 0] != ref["top"][:, 0]).sum()),
            "rows_other_top3_vs_mm": int((r["top"][:, :3] != ref["top"][:, :3]).any(1).sum())}


def deploy_extra(ctx: dict, cand: np.ndarray, cols, guard: bool = False) -> np.ndarray:
    """Columns of the deployment form in table order: rows of fold k with reference = folds ≠ k (= the evaluated rows of the recomputation).
    guard=True: test form of the reference of the test guard (sources with guard_ref_all, i.e. the label-free task count, with reference =
    all training rows)."""
    folds, n, Kc = np.asarray(ctx["folds"]), ctx["n"], S2().K_CAND
    E = np.zeros((n * Kc, len(cols.features)))
    for k in range(5):
        rows = np.flatnonzero(folds == k)
        pos = (rows[:, None] * Kc + np.arange(Kc)[None, :]).reshape(-1)
        ref = np.flatnonzero(folds != k)
        E[pos] = cols.guard_cols(rows, cand[rows], ref, np.arange(n)) if guard else cols.cols(rows, cand[rows], ref)
    return E


def fit_main(ctx: dict, X: np.ndarray, cand: np.ndarray) -> tuple[dict, dict, dict]:
    """Main model as c133.model_main: gate with guard over the five folds, conditional logit (λ 0.1, K 5) or weight 0, NONE guard."""
    C, S = c118(), S2()
    y, Kc = ctx["y"], S.K_CAND
    lab = S.labels_of(cand, y)
    g = S.gate(C.fold_blocks(ctx, X, cand, lab), protect=True)
    model = S.zero_model(X.shape[1]) if g["abstain"] else S.fit_clogit(X, lab, Kc, S.LAMBDA)
    top0 = S.reorder(cand, S.scores(model, X, Kc))
    top = S.protect_none(top0, cand)
    ins = {"map_base": float(S.map_rows(cand, y).mean()), "map_reranked": float(S.map_rows(top, y).mean()),
           "changed_top_share": float((top[:, 0] != cand[:, 0]).mean()), "rows_protected": int((top0 != top).any(1).sum())}
    return model, g, ins


def main_entry(names: list[str], feats_x: list[str], model: dict, g: dict, ins: dict, oof: dict, X: np.ndarray, X_mm: np.ndarray) -> dict:
    """models.main of the deployment file (fields as c133.model_main; comparison against MM)."""
    S = S2()
    return {"features": list(names) + list(feats_x), "feature_set": S.MM_SET, "protect_none": True, "extra_columns": list(feats_x),
            "w": [float(v) for v in model["w"]], "mu": [float(v) for v in model["mu"]], "sd": [float(v) for v in model["sd"]], "gate": g,
            "abstain": g["abstain"], "oof_changed_top_share": float(oof["set"]["changed_top_share_forced"]),
            "oof_changed_top_share_with_abstention": float(oof["set"]["changed_top_share"]),
            "oof_changed_top3_share": float(oof["set"]["changed_top3_share_forced"]),
            "oof_vs_mm": {f: oof[f]["vs_mm"]["delta"] for f in FORMS3}, "oof_vs_recipe_beta": {f: oof[f]["vs_recipe_beta"]["delta"] for f in FORMS3},
            "features_sha256": sha256_array(X), "features_sha256_mm": sha256_array(X_mm), "insample": ins}


def files_sha(spec: dict, root: Path) -> dict:
    out = {}
    for f in XS.source_files(spec):
        p = XS._resolve(f, COMP, root)
        if not p.is_file():
            raise Missing(f"{f} missing")
        out[f] = sha256_file(p)
    return out


def extra_field(ctx: dict, cols, info: dict, minor_codes: list[str], guard_ref: dict, mats_sha: dict) -> dict:
    """Field extra_columns of the deployment file: sources and column names, definitions, blend channels, files with checksums (training:
    provenance; test: required by make_submission), checksums of the training matrices, minor codes, reference and rule of the test guard.
    Without timestamps and without private files."""
    return {"decision": DECISION, "module": "src/stage2_sources.py", "version": XS.VERSION, "variant": XS.variant_of(cols.names),
            "names": list(cols.names), "features": list(cols.features), "definitions": {nm: XS.DEFINITIONS[nm] for nm in cols.names},
            "channels": list(ctx["spec"]["names"]),
            "train": {"spec": XS.TRAIN_SPEC, "files_sha256": info["files"], "matrices_sha256": mats_sha, "rows": int(ctx["n"]),
                      "sources": {s: {k: v for k, v in d.items() if v is not None} for s, d in info["sources"].items()}},
            "test": {"spec": XS.TEST_SPEC, "files_sha256": files_sha(XS.TEST_SPEC, ctx["root"])},
            "knowledge": KNOWLEDGE_NOTE, "minor_codes": minor_codes, "guard_reference": guard_ref, "guard_rule": XS.GUARD_RULE,
            "controls": {c: XS.CONTROL_TARGET[c] for c in XS.CONTROLS if XS.CONTROL_NEEDS[c] in cols.names}}


# ------------------------------------------------------------------------------------------------ chain: recomputation, regression, deployment file

def chain(base: str, cols_v: str, zero: bool = False, log=say) -> dict:
    """Common path of fit and copy. zero=True: columns := 0 (copy null point)."""
    t0 = time.time()
    C, S, M121, M124 = c118(), S2(), c121(), c124()
    b = BASES[base]
    sources = list(XS.VARIANTS[cols_v])
    e0 = base_files_check(base)
    ctx = base_context(base)
    inp, info, codes = train_inputs(ctx, sources)
    cols_real = XS.build_columns(sources, inp)
    cols = cols_real.zero() if zero else cols_real
    log(f"   context {b['label']} and inputs loaded ({time.time() - t0:.0f} s); variant {cols_v}: {len(sources)} sources, {len(cols.features)} columns"
        f"{' (copy: := 0)' if zero else ''}")
    std, betas, nested = M121.prepare(ctx)
    forms, betas2 = C.standard_forms(ctx, "select")
    if [float(v) for v in betas] != [float(v) for v in betas2] or not all(np.array_equal(forms[f], std[f]) for f in FORMS3):
        raise TechnicalError("standard cross-fit recomputed != that of the nested recomputation")
    df = (e0.get("m3px27_build") or {}).get("deploy_forms") or {}
    if [float(v) for v in betas] != [float(v) for v in df.get("beta_per_fold") or []]:
        raise TechnicalError(f"β per fold {betas} != deployment file of the base {df.get('beta_per_fold')}")
    # (1) regression in the process: main model MM of the base
    X_mm, cand, names = C.deploy_features(ctx, forms, S.MM_SET)
    m0 = e0["models"]["main"]
    if names != m0["features"] or sha256_array(X_mm) != m0["features_sha256"]:
        raise TechnicalError("deployment features MM != models.main of the base (names or checksum)")
    mod_mm, g_mm, ins_mm = fit_main(ctx, X_mm, cand)
    reg = {"w": [float(v) for v in mod_mm["w"]] == m0["w"], "mu": [float(v) for v in mod_mm["mu"]] == m0["mu"],
           "sd": [float(v) for v in mod_mm["sd"]] == m0["sd"], "gate": g_mm == m0["gate"], "insample": ins_mm == m0["insample"]}
    if not all(reg.values()):
        raise TechnicalError(f"regression: main model MM in the process != models.main of the base {b['stage2']} ({[k for k, v in reg.items() if not v]})")
    log(f"   regression in the process: main model main_guard_i1_mm = models.main of {b['stage2']} (features, w, mu, sd, gate, in-sample) "
        f"({time.time() - t0:.0f} s)")
    # (2) nested recomputation: MM and MM + columns
    y, seen = ctx["y"], ctx["seen"]
    draws_q = C.C107.question_draws(ctx["q_code"])
    rec = {f: C.C107.rows_map(y, std[f], seen) for f in FORMS3}
    r_mm, r_x, oof = {}, {}, {}
    for f in FORMS3:
        r_mm[f] = M124.run_mm(ctx, std, nested, f, "real")
        r_x[f] = run_x(ctx, std, nested, f, cols)
        oof[f] = entry(r_x[f], r_mm[f], rec[f], ctx, draws_q)
        log(f"   [{f}] MM {units_of(r_mm[f]['forced'])}/9000, MM + columns {oof[f]['units_forced']}/9000 (Δ {oof[f]['vs_mm']['units']:+d} sixths, "
            f"other top code {oof[f]['rows_other_top1_vs_mm']}) ({time.time() - t0:.0f} s)")
    u_mm = [units_of(r_mm["fold"]["forced"]), units_of(r_mm["set"]["forced"])]
    want_mm = ((e0.get(b["build_field"]) or {}).get("oof_rows") or {}).get("units")
    if u_mm != list(b["units"]) or (want_mm is not None and list(want_mm) != u_mm):
        raise TechnicalError(f"MM sixths {u_mm} != base {b['units']} / deployment file {want_mm}")
    null_i = {f: same_rows(r_x[f], r_mm[f]) for f in FORMS3} if zero else None
    if zero and not all(null_i.values()):
        raise TechnicalError(f"copy null point (i) violated: columns := 0 does not hit MM bit-identically ({null_i})")
    # (3) deployment form with columns, null point (e)
    E_dep = deploy_extra(ctx, cand, cols)
    X = np.concatenate([X_mm, E_dep], 1)
    if sha256_array(X) != sha256_array(C.eval_rows_in_table_order(ctx, r_x["set"]["X_eval"])):
        raise TechnicalError("null point (e) violated: deployment features (MM + columns) != evaluated OOF features of the form M")
    # (4) main model; reference of the test guard (always the real columns; task count in test form)
    model, g, ins = fit_main(ctx, X, cand)
    main = main_entry(names, cols.features, model, g, ins, oof, X, X_mm)
    E_guard = deploy_extra(ctx, cand, cols_real, guard=True)
    guard_ref = XS.guard_stats(cols_real, E_guard, cand, inp, np.arange(ctx["n"]), np.arange(ctx["n"]))
    mats = XS.matrices_sha256(cols_real, ctx["n"], ctx["K"])
    return {"base": base, "cols": cols_v, "sources": sources, "ctx": ctx, "e0": e0, "betas": [float(v) for v in betas], "main": main, "model": model,
            "gate": g, "insample": ins, "oof": oof, "r_mm": r_mm, "r_x": r_x, "null_i": null_i, "u_mm": u_mm, "guard_ref": guard_ref, "mats_sha": mats,
            "info": info, "minor_codes": codes, "X": X, "cand": cand, "cols_obj": cols_real, "inp": inp, "seconds": round(time.time() - t0, 1),
            "reg": reg}


def cols_field(res: dict, kind: str) -> dict:
    b = BASES[res["base"]]
    o = res["oof"]
    return {"decision": DECISION, "kind": kind, "candidate": b["candidate"], "variant": res["cols"],
            "base": {"label": b["label"], "file": f"artifacts/{b['stage2']}", "sha256": sha256_file(ART / b["stage2"]),
                     "stack": f"artifacts/stack_{b['stem']}.json", "stack_sha256": sha256_file(ART / f"stack_{b['stem']}.json")},
            "feature_set": "main_guard_i1_mm (64, unchanged) + extra columns extra_columns.features at the end (src/stage2_sources.py), NONE guard",
            "regression_base_main": {"equal": True, "fields": sorted(res["reg"])},
            "oof_units": {f: {"mm": units_of(res["r_mm"][f]["forced"]), "with_columns": o[f]["units_forced"]} for f in FORMS3},
            "oof_vs_mm": {f: {k: o[f]["vs_mm"][k] for k in ("delta", "se_q", "z", "units", "fold_units", "folds_pos")} for f in FORMS3},
            "null_point_e": True, "null_point_i": None if res["null_i"] is None else {"zero_equals_mm": res["null_i"]},
            "beta_outer": res["betas"],
            "test_side": "make_submission --stage2 <deployment file>: columns of the test rows from extra_columns (checksums required), test guard per "
                         "column and source (extra_columns.guard_rule), abort without a file if violated",
            "copy_null_point": "eda/c151_primary_cols.py copy: the same path with columns := 0 hits the base (rows, main model, fields; test side "
                               "byte-identical in the build path)",
            "rule": "bundle test: over 20 untouched splits mean > 0 in M and ½(F+M), real − placebo > 0 (expected-value rule)"}


def compose(res: dict, kind: str) -> dict:
    e0 = res["e0"]
    out = {}
    for k, v in e0.items():
        out[k] = {"main": res["main"], "fallback": v["fallback"]} if k == "models" else v
    out["extra_columns"] = extra_field(res["ctx"], res["cols_obj"], res["info"], res["minor_codes"], res["guard_ref"], res["mats_sha"])
    out["cols_build"] = cols_field(res, kind)
    return out


def cmd_fit(base: str, cols_v: str, save: bool) -> int:
    res = chain(base, cols_v)
    out = compose(res, "deploy")
    u_x = [res["oof"]["fold"]["units_forced"], res["oof"]["set"]["units_forced"]]
    name = out_name(base, cols_v)
    p = priv(base, cols_v)
    sha = write_or_verify(ART / f"stage2_{name}.json", to_json(out), save)
    rep = {"kind": f"{BASES[base]['candidate']} ({cols_v}): recomputation of the chain (MM, MM + columns), regression, null point (e)",
           "out": f"artifacts/stage2_{name}.json", "out_sha256": sha, "oof": res["oof"], "u_mm": res["u_mm"], "u_with_columns": u_x,
           "betas": res["betas"], "gate": res["gate"], "insample": res["insample"], "guard_reference": res["guard_ref"], "regression": res["reg"]}
    if save:  # private report (aggregates only) and row values: companion files, not part of the pins
        p.mkdir(parents=True, exist_ok=True)
        (p / f"c151_fit_{tag_of(base, cols_v)}.json").write_text(to_json(rep))
        rows = {f"{nm}_{f}_{k}": d[f][k] for nm, d in (("mm", res["r_mm"]), ("x", res["r_x"])) for f in FORMS3 for k in ROW_KEYS}
        np.savez_compressed(p / f"c151_rows_{tag_of(base, cols_v)}.npz", **rows)
    m = out["models"]["main"]
    o = res["oof"]
    say(f"stage2_{name}.json: main_guard_i1_mm + {len(m['extra_columns'])} columns from {len(res['sources'])} sources ({len(m['features'])} features, "
        f"guard on; gate {m['gate']['inner']:.5f} against {m['gate']['inner_base']:.5f} -> {'ABSTAIN' if m['abstain'] else 'apply'}; OOF new top code "
        f"{m['oof_changed_top_share']:.4f}; in-sample {m['insample']['map_base']:.5f} → {m['insample']['map_reranked']:.5f}); fallback unchanged "
        f"({len(out['models']['fallback']['features'])})")
    say(f"   nested against MM: F {o['fold']['vs_mm']['units']:+d}, oL {o['pre']['vs_mm']['units']:+d}, M {o['set']['vs_mm']['units']:+d} sixths "
        f"(½(F+M) {float(Fraction(o['fold']['vs_mm']['units'] + o['set']['vs_mm']['units'], 2 * UNITS_ROWS)):+.5f}); MM {res['u_mm']}, with columns {u_x}")
    say(f"   {'stored' if save else 'CHECKED'}: artifacts/stage2_{name}.json (SHA-256 {sha}) ({res['seconds']:.0f} s)")
    return 0


# ------------------------------------------------------------------------------------------------ copy null point of the training side

def normalized(d: dict) -> dict:
    """Comparison form: fields of the base; models.main cut down to the features of the base, fields without a counterpart removed."""
    out = copy.deepcopy({k: v for k, v in d.items() if k not in ("extra_columns", "cols_build")})
    m = out["models"]["main"]
    k64 = len([f for f in m["features"] if f not in (m.get("extra_columns") or [])])
    for key in ("features", "w", "mu", "sd"):
        m[key] = m[key][:k64]
    if "features_sha256_mm" in m:
        m["features_sha256"] = m.pop("features_sha256_mm")
    for key in ("extra_columns", "oof_vs_mm", "oof_vs_i1"):
        m.pop(key, None)
    return out


def cmd_copy(base: str, cols_v: str) -> int:
    res = chain(base, cols_v, zero=True)
    out = compose(res, "copy")
    m = out["models"]["main"]
    nx = len(m["extra_columns"])
    k = len(m["features"]) - nx
    extra_ok = m["w"][k:] == [0.0] * nx and m["mu"][k:] == [0.0] * nx and m["sd"][k:] == [1e-9] * nx
    vs0 = all(v == 0.0 for v in m["oof_vs_mm"].values())
    d = c133().diff_keys(normalized(out), normalized(res["e0"]))
    p = priv(base, cols_v)
    f_copy = p / "copy" / f"stage2_{tag_of(base, cols_v)}k.json"
    f_copy.parent.mkdir(parents=True, exist_ok=True)
    f_copy.write_text(to_json(out))
    rep = {"kind": f"{BASES[base]['candidate']} ({cols_v}): copy null point of the training side (columns := 0)", "base": BASES[base]["stage2"],
           "null_point_i": res["null_i"], "field_diff": d, "extra_weights_zero": extra_ok, "oof_vs_mm_zero": vs0, "copy_file": rel(f_copy),
           "copy_sha256": sha256_file(f_copy)}
    (p / f"c151_copy_{tag_of(base, cols_v)}.json").write_text(to_json(rep))
    if d or not extra_ok or not vs0:
        raise TechnicalError(f"copy null point of the training side violated: fields {d[:8]}; column weights 0 {extra_ok}; Δ against MM 0 {vs0}")
    say(f"copy null point (columns := 0, variant {cols_v}): rows F/oL/M = MM bit-identical {res['null_i']}; deployment file field-equal to "
        f"{BASES[base]['stage2']} (main model: w, mu, sd of the first {k} equal, {nx} column weights 0, standardisation 0/1e-9; gate, in-sample, OOF "
        f"shares, fallback, all other fields equal) -> copy deployment file {rel(f_copy)} ({res['seconds']:.0f} s)")
    return 0


# ------------------------------------------------------------------------------------------------ splits and cross-checks of the test sources

def parse_seeds(text: str) -> list:
    out = []
    for t in str(text).split(","):
        t = t.strip()
        if t == "orig":
            out.append(None)
        elif "-" in t:
            a, z = t.split("-")
            out.extend(range(int(a), int(z) + 1))
        elif t:
            out.append(int(t))
    return out


def split_folds(y: np.ndarray, seed) -> np.ndarray:
    """Original split (artifacts/folds.npy) or c121.split_folds(y, seed) — the same function as in the bundle test."""
    if seed is None:
        return np.load(ART / "folds.npy")
    return _load("c151_c121_split", COMP / "eda" / "c121_stage2_i1.py").split_folds(y, int(seed))


def cmd_test_sources() -> int:
    """TEST SOURCES (aggregates and checksums only; no row): completeness (exactly one first pass per row), meta (table test, 3,498 rows,
    model), parser cross-check against proto_labels.jsonl of the same run, notes = frozen test notes of the channels of primary_m3x4 (pr27:
    data/derived/proto_notes_test/test_pr27.jsonl, q397: data/derived/sft/q397m3_test_notes.jsonl — checksum of the newly formed file against
    the meta file), number of records per pass."""
    import labels
    from vlm import common as VC

    t0 = time.time()
    root = VC.find_data_root()
    test = VC.load_table(root, "test")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    keys = [str(k) for k in test["correct_answer"].tolist()]
    P, info = XS.load_side(XS.TEST_SPEC, "test", COMP, root, classes, keys)
    n = len(test)
    rows_pr, _ = XS.read_own([COMP / r for r in XS.TEST_SPEC["pr27"]["raw"]], n, classes, "json", "pr27")
    notes = {i: r["notes"] for i, r in enumerate(rows_pr) if r["notes"]}
    body = XS.notes_body(notes, {i: rows_pr[i]["state"] for i in notes}, "test")
    m_pr = json.loads((root / "derived/proto_notes_test/test_pr27.meta.json").read_text())
    rows_q, _ = XS.read_own([COMP / r for r in XS.TEST_SPEC["q397"]["raw"]], n, classes, "tinker", "q397")
    body_q = "".join(json.dumps({"table": "test", "row": i, "state": "open" if r["notes"] else "empty", "text": r["notes"], "finding": None,
                                 "q397_state": r["state"]}, ensure_ascii=False) + "\n" for i, r in enumerate(rows_q))
    m_q = json.loads((root / "derived/sft/q397m3_test_notes.meta.json").read_text())
    res = {"files_sha256": info["files"], "sources": info["sources"],
           "pr27_notes_vs_frozen": {"rows": len(notes), "sha256_equal": hashlib.sha256(body.encode("utf-8")).hexdigest() == m_pr["sha256"]},
           "q397_notes_vs_frozen": {"rows": n, "sha256_equal": hashlib.sha256(body_q.encode("utf-8")).hexdigest() == m_q["sha256"]}}
    PRIV0.mkdir(parents=True, exist_ok=True)
    (PRIV0 / "c151_test_sources.json").write_text(to_json(res))
    ok = res["pr27_notes_vs_frozen"]["sha256_equal"] and res["q397_notes_vs_frozen"]["sha256_equal"]
    say(f"test sources: {', '.join(f'{s} {d['kind']} ({d['records']} records, {d['phases']}; parser cross-check {d['labels_check']})' for s, d in info['sources'].items())}")
    say(f"   notes = frozen test notes: pr27 {res['pr27_notes_vs_frozen']}, q397 {res['q397_notes_vs_frozen']} -> {'PASSED' if ok else 'VIOLATED'} "
        f"({time.time() - t0:.0f} s)")
    if not ok:
        raise TechnicalError("test notes of the new path != frozen test notes")
    return 0


# ------------------------------------------------------------------------------------------------ measurement per split (training rows)

class _Perm:
    """Placebo of the bundle test: values permuted within the row across the candidates (as in the frame of the measurement)."""

    def __init__(self, base, seed: int, n: int, Kc: int):
        self.base, self.features = base, base.features
        self.P = np.stack([np.random.default_rng([int(seed), i]).permutation(Kc) for i in range(n)])

    def cols(self, rows, cand, ref):
        X = self.base.cols(rows, cand, ref)
        R, Kc = cand.shape
        return np.take_along_axis(X.reshape(R, Kc, -1), self.P[np.asarray(rows)][:, :, None], 1).reshape(R * Kc, -1)


def cmd_measure(base: str, cols_v: str, seeds: list) -> int:
    """Measurement of the variant on the base: per split MM, MM + columns and the placebo (values permuted within the row, seed 99500 + split
    seed as in the bundle test); ½(F+M), F, M against MM (aggregates only; training rows)."""
    t0 = time.time()
    M121 = c121()
    sources = list(XS.VARIANTS[cols_v])
    ctx0 = base_context(base)
    inp, _, _ = train_inputs(ctx0, sources)
    cols = XS.build_columns(sources, inp)
    recs = {}
    for seed in seeds:
        folds = split_folds(ctx0["y"], seed)
        ctx = dict(ctx0)
        ctx["folds"] = folds
        std, _betas, nested = M121.prepare(ctx)
        pl = _Perm(cols, 99500 + (0 if seed is None else int(seed)), ctx["n"], S2().K_CAND)
        u = {}
        for nm, c in (("base", cols.zero()), ("real", cols), ("placebo", pl)):
            u[nm] = [units_of(run_x(ctx, std, nested, f, c, gate=False)["forced"]) for f in ("fold", "set")]
        recs["orig" if seed is None else str(seed)] = u
        d = [u["real"][i] - u["base"][i] for i in range(2)]
        say(f"[{'orig' if seed is None else seed}] {cols_v} on {BASES[base]['label']}: base {u['base']}, with columns {u['real']} (Δ F {d[0]:+d}, "
            f"M {d[1]:+d}), placebo {u['placebo']} ({time.time() - t0:.0f} s)")
    sp = [v for k, v in recs.items() if k != "orig"]
    summ = None
    if sp:
        h = np.array([(v["real"][0] - v["base"][0] + v["real"][1] - v["base"][1]) / (2 * UNITS_ROWS) for v in sp])
        M = np.array([(v["real"][1] - v["base"][1]) / UNITS_ROWS for v in sp])
        F = np.array([(v["real"][0] - v["base"][0]) / UNITS_ROWS for v in sp])
        hp = np.array([(v["placebo"][0] - v["base"][0] + v["placebo"][1] - v["base"][1]) / (2 * UNITS_ROWS) for v in sp])
        se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None  # noqa: E731
        summ = {"n_splits": len(sp), "half_mean": float(h.mean()), "half_se": se(h), "share_pos": float((h > 0).mean()), "F_mean": float(F.mean()),
                "M_mean": float(M.mean()), "M_se": se(M), "placebo_half_mean": float(hp.mean()), "real_minus_placebo": float((h - hp).mean()),
                "real_minus_placebo_se": se(h - hp)}
        say(f"   {len(sp)} splits: ½(F+M) {summ['half_mean']:+.5f} ± {summ['half_se']:.5f} (> 0 in {summ['share_pos']:.0%}), F {summ['F_mean']:+.5f}, "
            f"M {summ['M_mean']:+.5f} ± {summ['M_se']:.5f}; placebo {summ['placebo_half_mean']:+.5f}; real − placebo {summ['real_minus_placebo']:+.5f} ± "
            f"{summ['real_minus_placebo_se']:.5f}")
    PRIV0.mkdir(parents=True, exist_ok=True)
    (PRIV0 / f"c151_measure_{tag_of(base, cols_v)}.json").write_text(to_json({"seeds": recs, "summary": summ}))
    return 0


# ------------------------------------------------------------------------------------------------ null model, test guard, positive controls

def cmd_null(base: str, source: Path, out: Path) -> int:
    """Null model (weight 0) for the deployment file: c118.null_stage2 + mech27, draws_first and extra_columns of the source (the test side
    computes the columns and the test guard; score 0 everywhere -> base order)."""
    ref = json.loads(Path(source).read_text())
    res = c118().null_stage2(BASES[base]["stem"], Path(source))
    res["mech27"] = ref["mech27"]
    if "draws_first" in ref:
        res = {k2: v2 for k, v in res.items() for k2, v2 in ([(k, v), ("draws_first", ref["draws_first"])] if k == "draws_channel" else [(k, v)])}
    if "extra_columns" in ref:
        res["extra_columns"] = ref["extra_columns"]
    Path(out).write_text(to_json(res))
    say(f"null model for {rel(Path(source))} -> {out}")
    return 0


def cmd_test_guard(report: Path) -> int:
    """Test guard from the report of a make_submission run (--stage2-report, field extra_columns): prints scope, statistic, training and test
    value (aggregates only); exit 5 if a statistic lies outside."""
    rep = json.loads(Path(report).read_text())
    x = rep.get("extra_columns")
    if not x:
        raise TechnicalError(f"{report}: no field extra_columns (run without extra columns?)")
    if x.get("control"):
        raise TechnicalError(f"{report}: report of a control run ({x['control']}) — test guard only on the real run")
    g = x["guard"]
    for c in g["checks"]:
        say(f"  {c['scope']:26s} {c['stat']:26s} training {c['train']:+.5f}  test {c['test']:+.5f}  z {c['z']:+7.2f}  "
            f"{'ok (' + str(c['passed_by']) + ')' if c['ok'] else 'VIOLATED'}" + (f"  floor {c['floor']:g}" if c.get("floor") is not None else ""))
    if not g["ok"]:
        for c in g["failed"]:
            say(f"VIOLATED: {c['scope']} {c['stat']}: training {c['train']:.5f}, test {c['test']:.5f}, z {c['z']:+.2f}"
                + (f", floor {c['floor']:g}" if c.get("floor") is not None else ""))
        raise GuardFail(f"test guard violated in {len(g['failed'])} statistics")
    say(f"-> test guard PASSED ({len(g['checks'])} statistics, rule {g['rule']}: band ±{g['rel']:g} relative or |z| ≤ {g['z_max']:g}; "
        f"floors {g.get('floors')})")
    return 0


def cmd_controls(null: Path, out_dir: Path, name: str, ms_args: list[str]) -> int:
    """Positive controls: make_submission with the null model and --stage2-xcontrol <c> (in memory only; never a file) for every control whose
    source is in the deployment file — each must fire the test guard (code 5 and a report with violated statistics). Exit 6 otherwise."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = (json.loads(Path(null).read_text()).get("extra_columns") or {}).get("names") or []
    res, bad = {}, []
    for c in [c for c in XS.CONTROLS if XS.CONTROL_NEEDS[c] in names]:
        rep = out_dir / f"guards_ctl_{c}.json"
        csv = out_dir / f"sub_ctl_{c}.csv"
        for f in (rep, csv):
            if f.is_file():
                f.unlink()
        suffix = f"_x3ctl_{c}"
        cmd = [sys.executable, "-I", str(COMP / "tools" / "make_submission.py"), "--name", name, *ms_args, "--out", str(csv), "--suffix", suffix,
               "--stage2", str(null), "--stage2-report", str(rep), "--stage2-xcontrol", c]
        p = subprocess.run(cmd, cwd=str(COMP), capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        tm = ART / f"test_{name}{suffix}.npy"
        wrote = csv.is_file() or tm.is_file()
        for f in (tm, csv, csv.with_suffix(".json")):
            if f.is_file():
                f.unlink()
        if p.returncode not in (5, 6) or not rep.is_file() or wrote:
            tail = " | ".join(line[:200] for line in (p.stdout + p.stderr).splitlines()[-3:])
            raise TechnicalError(f"control run {c}: code {p.returncode}, report {'present' if rep.is_file() else 'MISSING'}, file written {wrote} — {tail}")
        g = json.loads(rep.read_text())["extra_columns"]["guard"]
        fired = p.returncode == 5 and not g["ok"]
        res[c] = {"target": XS.CONTROL_TARGET[c], "exit": p.returncode, "fired": bool(fired), "failed": g["failed"]}
        say(f"  control {c} ({XS.CONTROL_TARGET[c]}): {'fires' if fired else 'DOES NOT FIRE'} — "
            + (", ".join(f"{x['scope']} {x['stat']} ({x['train']:.4f} -> {x['test']:.4f})" for x in g["failed"][:6]) or "no violated statistic"))
        if not fired:
            bad.append(c)
    (out_dir / "controls.json").write_text(to_json(res))
    if bad:
        raise ControlFail(f"positive controls without effect: {bad}")
    say(f"-> positive controls PASSED: all {len(res)} applicable controls fire the test guard")
    return 0


# ------------------------------------------------------------------------------------------------ entry point

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ms_args: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, ms_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test-sources")
    for nm in ("fit", "copy", "measure", "null"):
        p = sub.add_parser(nm)
        p.add_argument("--base", choices=sorted(BASES), default="m3x4")
        p.add_argument("--cols", choices=sorted(XS.VARIANTS), default=DEFAULT_COLS)
        if nm == "fit":
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument("--save", action="store_true")
            g.add_argument("--verify", action="store_true")
        if nm == "measure":
            p.add_argument("--seeds", default="orig,4000,4001")
        if nm == "null":
            p.add_argument("--source", required=True)
            p.add_argument("--out", required=True)
    p = sub.add_parser("test-guard")
    p.add_argument("--report", required=True)
    p = sub.add_parser("controls")
    p.add_argument("--null", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "test-sources":
            return cmd_test_sources()
        if a.cmd == "fit":
            return cmd_fit(a.base, a.cols, a.save)
        if a.cmd == "copy":
            return cmd_copy(a.base, a.cols)
        if a.cmd == "measure":
            return cmd_measure(a.base, a.cols, parse_seeds(a.seeds))
        if a.cmd == "null":
            return cmd_null(a.base, Path(a.source), Path(a.out))
        if a.cmd == "test-guard":
            return cmd_test_guard(Path(a.report))
        return cmd_controls(Path(a.null), Path(a.dir), a.name, ms_args)
    except Exception as exc:  # noqa: BLE001 — exit code per exception kind of the loaded modules
        code = exit_code_of(exc)
        if code is None:
            raise
        label = {3: "NOT YET EVALUABLE", 4: "ABORT (technical)", 5: "TEST GUARD VIOLATED", 6: "POSITIVE CONTROL WITHOUT EFFECT"}[code]
        print(f"{label}: {exc}", file=sys.stderr)
        return code


if __name__ == "__main__":
    sys.exit(main())
