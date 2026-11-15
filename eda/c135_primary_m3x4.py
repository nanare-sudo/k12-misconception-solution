"""PRIMARY CANDIDATE primary_m3x4 = primary_m3px27 with a fourth M3 draw d4 (channel m3x3_full -> m3x4_full = mean of the four draws
d1..d4 in a fixed order; second-stage features still from d1–d3). Evaluation under the pre-registered four-draw rule in the primary_m3px27
recipe, and deployment path (build script tools/builds/primary_m3x4.sh).

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    python -I eda/c135_primary_m3x4.py bank --seeds orig,1000,1001        # fair bank per split (R3_j, R4_j, copies) -> private, per seed
    python -I eda/c135_primary_m3x4.py final --seeds 2000,2001            # final check (R3_j, R4_j; NOT used for selection) -> private
    python -I eda/c135_primary_m3x4.py placebo --seeds 0,1,2              # d4 rows permuted within each fold (original split) -> private
    python -I eda/c135_primary_m3x4.py rule                               # verdict of the four-draw rule (1)–(2) from the stored seeds
    python -I eda/c135_primary_m3x4.py oof --save                         # artifacts/oof_m3x4_full.npy + meta (writes what is missing, else requires equality)
    python -I eda/c135_primary_m3x4.py stack --save                       # artifacts/stack_primary_m3x4.json (blend)
    python -I eda/c135_primary_m3x4.py fit --save                         # artifacts/stage2_primary_m3x4.json (chain main form -> … -> final form in the primary_m3px27 recipe)
    python -I eda/c135_primary_m3x4.py copy [--keep]                      # COPY NULL POINT 'd4 removed' -> blend and chain = primary_m3px27
    python -I eda/c135_primary_m3x4.py clean-copy                         # remove the temporary copy files under artifacts/
    python -I eda/c135_primary_m3x4.py test-guard [--rank SPEC]           # technical checks of the d4 test side (aggregates only; exit 0 / 3 missing / 5 violated)
    python -I eda/c135_primary_m3x4.py null --source F --out G            # null model (weight 0) for the deployment file, with mech27 and draws_first

DEPLOYMENT PATH (build script tools/builds/primary_m3x4.sh): channel m3x4_full = a9 mean of the four OOF draws d1, d2, d3 (meta_m3x3_full.json)
and d4 (row list = derived/m3x3_full_rows.npy, all 1,500 rows; the first three = m3x3_full bit-identical); blend = stack_primary_m3px27.json
with m3x4_full instead of m3x3_full (θ on log loss from all 1,500 rows, β rule, rank_only with four draws); second stage = chain of c133
(oof_chain, compose) in the primary_m3px27 context with the swapped channel, draws_channel m3x4_full and draws_first 3 (features d1–d3;
make_submission.stage2_draws_first); the MM rows F and M of the chain = the evaluation rows R4_0 of the bank (bit-identical, recomputed in
process; the stored sixths of the bank if present). COPY NULL POINT 'd4 removed' (rule part (3), weight 0): channel m3x4k = m3x3_full under
the new path -> blend numbers = primary_m3px27, all chain rows = primary_m3px27 chain (c133_oof_rows.npz), files a–e = those of primary_m3px27
except provenance, channel names and draws_first; test side in the build script.
TEST GUARD d4: 3,498/3,498 exact at the ranking files (not status.json), settings of ranking and generation equal to those of the OOF draw d4,
NONE at the top within the band of the d1–d3 test sides [min − 0.02, max + 0.02] (fixed before fetching: d1–d3 test 0.3456–0.3468) and
within the generic band 0.30–0.38, training regime (full fit m3_fullte_rp/m3_train_full, recipe = fold adapters of d4); positive control
with the d3 test ranking: only the seed of the generation differs (3 instead of 4).

RECIPE primary_m3px27 (decides): c124.load_context(s032_stage2) + c133.chain_ctx (mean channel m3px27xm = ½(pr27 + m3px27) in place of
pr27xm, blend file stack_primary_m3px27.json); nested as in c132/c133: c121.prepare (standard cross-fit, θ on log loss and β "select" per
outer fold, nested forms) and c124.run_mm "real" (second stage main_guard_i1_mm with NONE guard and mech27) in the forms F (per fold) and M
(one set); value of one computation ½(F+M) = (u_F + u_M) / 18,000 exactly (u = sixths). NULL POINT: R3_0 (triple d1d2d3) on the original
split hits the rows of the expected-value rule computation of primary_m3px27 (c132_m3px27_eval_rows.npz, primary_final_m3px27_fold/_set)
BIT FOR BIT, otherwise abort.
DRAW BANK: d1, d2, d3 (recipe, meta_m3x3_full.json) and d4 = runs/m3d4_oof_rp/m3d4_rank_f{0..4}/bench (sampled like d3 with another seed;
all 1,500 rows exact), notes per draw c118.train_note_texts (fold fidelity).
FAIR BANK (rule part (1)): the four draws are exchangeable up to their seed. A comparison "four draws against the stored three" would
credit the fourth draw with any luck or bad luck of the stored triple d1d2d3; the fair bank therefore lets every triple of the four draws
play the role of "the three draws" and compares, in the same frame, four against three averaged over the triples. Per split and triple
T_j ∈ {d1d2d3, d4d2d3, d1d4d3, d1d2d4}:
  R3_j     channel m3x3_full := mean of the three draws of T_j (order of T_j); draw and note features from T_j;
  R4_j     channel := mean of ALL FOUR draws d1..d4 (fixed order); draw and note features from T_j (this is how the test side is deployed:
           channel from 4, features from d1–d3) — the blend (prepare) is the same for all j and is computed once;
  K_{j,k}  copy: channel := mean (T_j, then T_j[k] once more), features from T_j (one draw of the triple duplicated as fourth member).
  G = mean_j R4_j − mean_j R3_j;  G_copy = mean_{j,k} K_{j,k} − mean_j R3_j (copy null point = mean of the twelve duplicates);
  strict (only the exchangeable pair d3/d4, triples d1d2d3 and d1d2d4): G_ex, G_copy_ex over j ∈ {0, 3};
  old form (report only): R4_0 − R3_0; its copy null point in the same frame mean_k K_{0,k} − R3_0.
SPLITS: c121.split_folds (StratifiedKFold(5, shuffle, seed) by root_code); decision on the 20 splits 1000 … 1019 (as c132); final check on
  2000 … 2019 (only R3_j/R4_j, never used for selection). Channel OOFs fixed (not refitted per split).
PLACEBO (report only): d4 := d4 with its rows permuted within each fold (artifacts/folds.npy; np.random.default_rng(seed), seeds 0 … 19),
  channel mean (d1, d2, d3, d4_perm), features from d1–d3; Δ_P = R4P_0 − R3_0 on the original split.
PRE-REGISTERED FOUR-DRAW RULE: deploy (primary slot, expected-value rule) only if, in the primary_m3px27 recipe (m3x3_full -> m3x4_full =
  mean of the four draws in the fixed order d1..d4; blend and second stage nested anew), ALL hold: (1) the fair bank evaluation (G = mean
  over the triples of R4 − R3 in the same frame) over ≥ 10 meta-splits is > 0 and above the copy null point (one draw of the triple
  duplicated as fourth member); (2) original ½(F+M) m3x4 against m3x3 above the copy null point in the same frame; (3) copy null point of
  the test side (m3x4 with d4 := copy of d3, or weight 0) -> primary_m3px27 byte-identical, regressions byte-identical; (4) offline rebuild
  byte-identical.
READING (fixed before the run): (1) holds if over EXACTLY the 20 splits 1000 … 1019 mean G > 0 AND mean G > mean G_copy (exact in
  sixths); (2) holds if on the original split R4_0 − R3_0 > mean_k K_{0,k} − R3_0 (exact); "above" = strictly greater. (3)/(4) are checked
  by the build script (tools/builds/primary_m3x4.sh) and the offline rebuild. Reported, not deciding: G on the original split, the strict
  reading d3/d4, the old form, the share of splits > 0, the placebo, the final check 2000 … 2019.
Output: aggregates only (means, counts, checksums) under artifacts/eda/private/primary_m3x4/; no row, no identifier; no test row.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

ART = COMP / "artifacts"
PRIV = ART / "eda" / "private" / "primary_m3x4"
MEASURE_VERSION = "c135-bank-1"
M3, M3X4 = "m3x3_full", "m3x4_full"
MID, STEM_M3PX27, STEM_REF = "m3px27xm", "primary_m3px27", "s032_stage2"
D4_SPEC = "+".join(f"runs/m3d4_oof_rp/m3d4_rank_f{k}/bench" for k in range(5))
TRIPLES = ((0, 1, 2), (3, 1, 2), (0, 3, 2), (0, 1, 3))  # indices into [d1, d2, d3, d4]
EXCHANGEABLE = (0, 3)  # d1d2d3 and d1d2d4: only d3 <-> d4 (same settings, other seed)
RULE_SEEDS = tuple(range(1000, 1020))
FINAL_SEEDS = tuple(range(2000, 2020))
PLACEBO_SEEDS = tuple(range(20))
UNITS_ROWS = 9000  # 1,500 rows × 6 sixths
C132_ROWS = ART / "eda" / "private" / "m3px27_eval" / "c132_m3px27_eval_rows.npz"
C132_ROWS_SHA = "cf4af62a74cd3d315d7a0e0c3eefe8f3b8151f9db3d36cd4d2855ee40b5f0fc5"  # pinned in tools/builds/primary_m3px27.sh
MID_OOF_SHA = "47e128eed7865deec8a5818cdbd514ed29246dec79a56f7da59d86ed3897cfe8"
STACK_M3PX27_SHA = "ebdd585e4650922d59972b3a84eb85c44af4f2dc469dd723ef2100ab7d797a3c"
RULE_TEXT = ("four-draw rule (1) fair bank over the 20 splits 1000…1019: mean G > 0 and > mean G_copy; (2) original R4_0 − R3_0 > "
             "mean_k K_{0,k} − R3_0; (3) copy null point of the test side byte-identical to primary_m3px27 and regressions byte-identical "
             "(build script); (4) offline rebuild byte-identical")

_MODS: dict = {}


class TechnicalError(RuntimeError):
    """Checksum, recomputation or null point violated: exit 4."""


class Missing(TechnicalError):
    """Input missing: exit 3."""


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def c133():
    if "c133" not in _MODS:
        _MODS["c133"] = _load("c135_c133", COMP / "eda" / "c133_primary_m3px27.py")
    return _MODS["c133"]


def c124():
    return c133().c124()


def c121():
    return c133().c121()


def c118():
    return c133().c118()


def say(*a) -> None:
    print(*a, flush=True)


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(COMP))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def npy_sha256(P: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.asarray(P, dtype=np.float64))
    return hashlib.sha256(buf.getvalue()).hexdigest()


def units_of(rows: np.ndarray) -> int:
    """Sum of the row values (MAP@3 per row) in sixths, exact (otherwise TechnicalError)."""
    u = np.rint(6.0 * np.asarray(rows, dtype=np.float64))
    if not np.allclose(u / 6.0, rows, rtol=0.0, atol=1e-9):
        raise TechnicalError("row values are not multiples of 1/6")
    return int(u.sum())


def half(u: tuple[int, int] | list[int]) -> Fraction:
    """½(F+M) of one computation from (u_F, u_M), exact."""
    return Fraction(int(u[0]) + int(u[1]), 2 * UNITS_ROWS)


# ------------------------------------------------------------------------------------------------ context primary_m3px27 and draw bank

def m3px27_context() -> dict:
    """Recipe primary_m3px27 of the training side (c124.load_context + c133.chain_ctx) with pinned inputs."""
    X = c133()
    f = ART / f"oof_{MID}.npy"
    if not f.is_file():
        raise Missing(f"{rel(f)} missing — run bash tools/builds/primary_m3px27.sh first")
    if sha256_file(f) != MID_OOF_SHA:
        raise TechnicalError(f"{rel(f)} is not the mean channel of primary_m3px27 (checksum)")
    sf = ART / f"stack_{STEM_M3PX27}.json"
    if not sf.is_file() or sha256_file(sf) != STACK_M3PX27_SHA:
        raise TechnicalError(f"{rel(sf)} missing or not the blend file of primary_m3px27 (checksum)")
    ctx0 = X.c124().load_context(STEM_REF)
    ctx = X.chain_ctx(ctx0, MID, np.load(f), STEM_M3PX27)
    if list(ctx["spec"]["names"]) != ["v1r", "zs27q", "think18nb", MID, M3, "q397m3"]:
        raise TechnicalError(f"recipe channels {ctx['spec']['names']} are not those of primary_m3px27")
    return ctx


def load_d4(ctx: dict) -> tuple[np.ndarray, list[str], dict]:
    """d4: exact ranking on all 1,500 rows (a9.load_draw) and notes per row (c118.train_note_texts, fold fidelity)."""
    C = c118()
    P4, h4, info = C.A9.load_draw(D4_SPEC, ctx["n"], ctx["seen"])
    if not h4.all():
        raise TechnicalError(f"d4 ({D4_SPEC}) covers {int(h4.sum())} instead of {ctx['n']} rows")
    T4 = C.train_note_texts(D4_SPEC, ctx["folds"])
    if len(T4) != ctx["n"]:
        raise TechnicalError("notes of d4 not on all rows")
    return P4, T4, info


def bank_inputs(ctx: dict, P4: np.ndarray, T4: list[str]) -> dict:
    """Checksums of the inputs of a measurement (for merging the seeds)."""
    return {"version": MEASURE_VERSION, "mid": MID_OOF_SHA, "stack_primary_m3px27": STACK_M3PX27_SHA, "m3x3_full": npy_sha256(ctx["mats"][M3]),
            "draws": [npy_sha256(P) for P in ctx["draws"]], "d4": npy_sha256(P4),
            "texts": [hashlib.sha256("\x1e".join(t).encode("utf-8")).hexdigest() for t in list(ctx["note_texts"]) + [T4]],
            "folds": sha256_file(ART / "folds.npy"), "inputs_primary_final": c118().inputs_sha256(ctx)}


def mean_draws(Ps: list[np.ndarray], n: int, K: int) -> np.ndarray:
    """Mean of the draws in the given order (a9.mean_of_draws, all rows)."""
    return c118().A9.mean_of_draws([(P, np.ones(n, bool)) for P in Ps], n, K)[0]


class Bank:
    """Variants of the primary_m3px27 context: channel m3x3_full and draw/note features of the second stage from a triple of the bank.
    T4 = None: only triples without d4 (features d1–d3; the deployment path does not need the notes of d4)."""

    def __init__(self, ctx: dict, P4: np.ndarray, T4: list[str] | None):
        self.ctx, self.P4, self.T4 = ctx, P4, T4
        self.allP = list(ctx["draws"]) + [P4]
        self.allT = list(ctx["note_texts"]) + [T4]
        self._notes: dict = {}
        if not np.array_equal(mean_draws(list(ctx["draws"]), ctx["n"], ctx["K"]), ctx["mats"][M3]):
            raise TechnicalError("mean of the three recipe draws != m3x3_full of the context")

    def notes(self, triple: tuple) -> dict:
        if tuple(triple) == (0, 1, 2):
            return self.ctx["notes"]
        if self.T4 is None and 3 in triple:
            raise TechnicalError("a triple with d4 requires the notes of d4 (bank without T4)")
        if triple not in self._notes:
            self._notes[triple] = c124().S2.notes_features([self.allT[i] for i in triple], self.ctx["classes"])
        return self._notes[triple]

    def channel(self, members: list[int] | tuple) -> np.ndarray:
        return mean_draws([self.allP[i] for i in members], self.ctx["n"], self.ctx["K"])

    def variant(self, triple: tuple, channel: np.ndarray, folds: np.ndarray) -> dict:
        S2 = c124().S2
        ctx = self.ctx
        draws = [self.allP[i] for i in triple]
        mats = dict(ctx["mats"])
        mats[M3] = np.asarray(channel, dtype=np.float64)
        notes = self.notes(tuple(triple))
        st = S2.Static(ctx["spec"]["names"], mats, draws, ctx["grade"], ctx["seen"], ctx["classes"], notes)
        S2.attach_mech(st, ctx["mech27"], ctx["minor"])
        c = dict(ctx)
        c.update(mats=mats, draws=draws, note_texts=[self.allT[i] for i in triple], notes=notes, static=st, folds=np.asarray(folds))
        return c


def load_d4_matrix(ctx: dict) -> tuple[np.ndarray, dict]:
    """Only the ranking of d4 (a9.load_draw: shards complete and exact, settings equal per fold) — without notes."""
    P4, h4, info = c118().A9.load_draw(D4_SPEC, ctx["n"], ctx["seen"])
    if not h4.all():
        raise TechnicalError(f"d4 ({D4_SPEC}) covers {int(h4.sum())} instead of {ctx['n']} rows")
    return P4, info


def run_units(c: dict, prep=None) -> tuple[tuple[int, int], dict, list]:
    """Recipe primary_m3px27 nested: (u_F, u_M), rows (F, M), β per outer fold. prep = (std, betas, nested) of c121.prepare reused (only for
    the same channels, recipe and split)."""
    std, betas, nested = c121().prepare(c) if prep is None else prep
    rF = c124().run_mm(c, std, nested, "fold", "real")
    rM = c124().run_mm(c, std, nested, "set", "real")
    return (units_of(rF["forced"]), units_of(rM["forced"])), {"fold": rF["forced"], "set": rM["forced"]}, [float(b) for b in betas]


def folds_of(ctx: dict, seed) -> np.ndarray:
    return np.asarray(ctx["folds"]) if seed is None else c121().split_folds(ctx["y"], int(seed))


def check_null(rows: dict) -> dict:
    """R3_0 on the original split = rows of the expected-value rule computation of primary_m3px27 (c132), bit-identical."""
    if not C132_ROWS.is_file():
        raise Missing(f"{rel(C132_ROWS)} missing")
    if sha256_file(C132_ROWS) != C132_ROWS_SHA:
        raise TechnicalError(f"{rel(C132_ROWS)}: checksum differs")
    z = np.load(C132_ROWS)
    ok = bool(np.array_equal(rows["fold"], z["primary_final_m3px27_fold"]) and np.array_equal(rows["set"], z["primary_final_m3px27_set"]))
    if not ok:
        raise TechnicalError("null point violated: R3_0 (d1d2d3) on the original split does not hit the rows of the expected-value rule "
                             "computation of primary_m3px27 bit for bit")
    return {"file": rel(C132_ROWS), "sha256": C132_ROWS_SHA, "bit_identical": True,
            "units": [units_of(z["primary_final_m3px27_fold"]), units_of(z["primary_final_m3px27_set"])]}


def seed_label(seed) -> str:
    return "orig" if seed is None else str(int(seed))


def parse_seeds(text: str) -> list:
    out = []
    for t in str(text).split(","):
        t = t.strip()
        if not t:
            continue
        out.append(None if t == "orig" else int(t))
    return out


def out_file(kind: str, seed) -> Path:
    return PRIV / kind / f"{seed_label(seed)}.json"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def measure_split(bank: Bank, seed, with_copies: bool, inputs: dict, log=say) -> dict:
    """One split of the fair bank: R3_j (4 blends), R4_j (one blend, four feature sets), copies K_{j,k} (12 blends)."""
    t0 = time.time()
    ctx = bank.ctx
    folds = folds_of(ctx, seed)
    rec: dict = {"seed": seed, "kind": "bank" if with_copies else "final", "inputs": inputs, "R3": [], "R4": [], "copy": {}, "betas": {}}
    for j, T in enumerate(TRIPLES):
        u, rows, betas = run_units(bank.variant(T, bank.channel(T), folds))
        if seed is None and j == 0:
            rec["null_point"] = check_null(rows)
        rec["R3"].append(list(u))
        rec["betas"][f"R3_{j}"] = betas
    ch4 = bank.channel((0, 1, 2, 3))
    prep = c121().prepare(bank.variant(TRIPLES[0], ch4, folds))
    rec["betas"]["R4"] = [float(b) for b in prep[1]]
    for j, T in enumerate(TRIPLES):
        u, _, _ = run_units(bank.variant(T, ch4, folds), prep=prep)
        rec["R4"].append(list(u))
    if with_copies:
        for j, T in enumerate(TRIPLES):
            for k in range(3):
                u, _, betas = run_units(bank.variant(T, bank.channel(list(T) + [T[k]]), folds))
                rec["copy"][f"{j}_{k}"] = list(u)
                rec["betas"][f"K_{j}_{k}"] = betas
    rec.update(summarize_split(rec))
    rec["seconds"] = round(time.time() - t0, 1)
    msg = (f"[{seed_label(seed)}] R3 {[round(float(half(u)), 5) for u in rec['R3']]} R4 {[round(float(half(u)), 5) for u in rec['R4']]} | "
           f"G {float(rec['G']):+.5f}")
    if with_copies:
        msg += f" copy {float(rec['G_copy']):+.5f} | G_ex {float(rec['G_ex']):+.5f} copy_ex {float(rec['G_copy_ex']):+.5f}"
    log(msg + f" | old form {float(rec['old_form']):+.5f} ({rec['seconds']:.0f} s)")
    return rec


def summarize_split(rec: dict) -> dict:
    """Statistics of one split as fractions (exact) — G, G_copy, strict d3/d4, old form and its copy null point."""
    R3 = [half(u) for u in rec["R3"]]
    R4 = [half(u) for u in rec["R4"]]
    out = {"R3_half": R3, "R4_half": R4, "G": sum(R4) / 4 - sum(R3) / 4,
           "G_ex": sum(R4[j] for j in EXCHANGEABLE) / 2 - sum(R3[j] for j in EXCHANGEABLE) / 2, "old_form": R4[0] - R3[0],
           "G_F": Fraction(sum(u[0] for u in rec["R4"]) - sum(u[0] for u in rec["R3"]), 4 * UNITS_ROWS),
           "G_M": Fraction(sum(u[1] for u in rec["R4"]) - sum(u[1] for u in rec["R3"]), 4 * UNITS_ROWS)}
    if rec.get("copy"):
        K = {key: half(u) for key, u in rec["copy"].items()}
        out["G_copy"] = sum(K.values()) / 12 - sum(R3) / 4
        out["G_copy_ex"] = sum(K[f"{j}_{k}"] for j in EXCHANGEABLE for k in range(3)) / 6 - sum(R3[j] for j in EXCHANGEABLE) / 2
        out["old_form_copy"] = sum(K[f"0_{k}"] for k in range(3)) / 3 - R3[0]
    return out


def placebo_perm(folds: np.ndarray, seed: int) -> np.ndarray:
    """Row permutation of d4 within each fold: target row r gets row perm[r] of the same fold (default_rng(seed), folds in order)."""
    rng = np.random.default_rng(int(seed))
    perm = np.arange(len(folds))
    for k in range(int(folds.max()) + 1):
        idx = np.flatnonzero(folds == k)
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def measure_placebo(bank: Bank, seed: int, inputs: dict, log=say) -> dict:
    t0 = time.time()
    ctx = bank.ctx
    folds = np.asarray(ctx["folds"])
    perm = placebo_perm(folds, seed)
    P4p = bank.P4[perm]
    ch = mean_draws(list(ctx["draws"]) + [P4p], ctx["n"], ctx["K"])
    u, _, betas = run_units(bank.variant(TRIPLES[0], ch, folds))
    rec = {"seed": int(seed), "kind": "placebo", "inputs": inputs, "u": list(u), "half": half(u), "betas": betas,
           "rows_moved": int((perm != np.arange(len(perm))).sum()), "seconds": round(time.time() - t0, 1)}
    log(f"[placebo {seed}] R4P_0 {float(rec['half']):.5f} ({rec['rows_moved']} rows permuted; {rec['seconds']:.0f} s)")
    return rec


def cmd_measure(kind: str, seeds: list, force: bool) -> int:
    t0 = time.time()
    ctx = m3px27_context()
    P4, T4, _ = load_d4(ctx)
    bank = Bank(ctx, P4, T4)
    inputs = bank_inputs(ctx, P4, T4)
    say(f"primary_m3px27 context and d4 loaded ({time.time() - t0:.0f} s); channels {ctx['spec']['names']}")
    for s in seeds:
        f = out_file(kind, s)
        if f.is_file() and not force:
            old = json.loads(f.read_text())
            if old.get("inputs") == json.loads(json.dumps(inputs, default=str)):
                say(f"[{seed_label(s)}] present ({rel(f)}) — skipped")
                continue
        if kind == "placebo":
            if s is None:
                raise SystemExit("placebo: seeds are integers")
            rec = measure_placebo(bank, int(s), inputs)
        else:
            if kind == "final" and (s is None or int(s) in RULE_SEEDS):
                raise SystemExit("final check only on new splits (not orig, not 1000…1019)")
            rec = measure_split(bank, s, kind == "bank", inputs)
        write_json(f, rec)
    say(f"done ({time.time() - t0:.0f} s)")
    return 0


# ------------------------------------------------------------------------------------------------ verdict

def load_records(kind: str, seeds) -> tuple[dict, list]:
    recs, missing = {}, []
    for s in seeds:
        f = out_file(kind, s)
        if not f.is_file():
            missing.append(seed_label(s))
            continue
        recs[seed_label(s)] = json.loads(f.read_text())
    return recs, missing


def frac(v) -> Fraction:
    return Fraction(v) if not isinstance(v, Fraction) else v


def stats(vals: list[Fraction]) -> dict:
    a = np.array([float(v) for v in vals])
    return {"n": len(vals), "mean": (sum(vals) / len(vals)) if vals else None, "se": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None,
            "sd": float(a.std(ddof=1)) if len(a) > 1 else None, "share_pos": float(np.mean(a > 0)) if len(a) else None,
            "min": float(a.min()) if len(a) else None, "max": float(a.max()) if len(a) else None}


def decide(orig: dict, splits: list[dict], n_splits: int = len(RULE_SEEDS)) -> dict:
    """Pre-registered four-draw rule (1)–(2) in the reading of the module docstring. orig / splits: statistics per split (fractions)."""
    if len(splits) != n_splits:
        raise ValueError(f"requires exactly {n_splits} splits, {len(splits)} given")
    G = [frac(r["G"]) for r in splits]
    Gc = [frac(r["G_copy"]) for r in splits]
    mG, mGc = sum(G) / n_splits, sum(Gc) / n_splits
    d_orig, c_orig = frac(orig["old_form"]), frac(orig["old_form_copy"])
    checks = {"(1) mean G > 0": bool(mG > 0), "(1) mean G > mean G_copy": bool(mG > mGc),
              "(2) original m3x4 − m3x3 > copy null point": bool(d_orig > c_orig)}
    reasons = []
    if not checks["(1) mean G > 0"]:
        reasons.append(f"(1) mean G over {n_splits} splits {float(mG):+.5f} not > 0")
    if not checks["(1) mean G > mean G_copy"]:
        reasons.append(f"(1) mean G {float(mG):+.5f} not above the copy null point {float(mGc):+.5f}")
    if not checks["(2) original m3x4 − m3x3 > copy null point"]:
        reasons.append(f"(2) original {float(d_orig):+.5f} not above the copy null point {float(c_orig):+.5f}")
    return {"verdict": "DEPLOY" if all(checks.values()) else "NO DEPLOY", "checks": checks, "reasons": reasons, "rule": RULE_TEXT,
            "values": {"splits_mean_G": mG, "splits_mean_G_copy": mGc, "orig_m3x4_vs_m3x3": d_orig, "orig_copy": c_orig}}


def cmd_rule(allow_partial: bool) -> int:
    recs, missing = load_records("bank", [None, *RULE_SEEDS])
    if missing and not allow_partial:
        say(f"NOT YET EVALUABLE: splits missing: {', '.join(missing)}")
        return 3
    inputs = {json.dumps(r["inputs"], sort_keys=True) for r in recs.values()}
    if len(inputs) != 1:
        raise TechnicalError("seeds computed with different inputs — they cannot be merged")
    for r in recs.values():  # fractions recomputed from the units (not from the JSON string)
        r.update(summarize_split(r))
    orig = recs.get("orig")
    sp = [recs[str(s)] for s in RULE_SEEDS if str(s) in recs]
    res = {"kind": "verdict of the four-draw rule (c135 rule)", "version": MEASURE_VERSION, "seeds": [s for s in recs], "missing": missing,
           "orig": None if orig is None else {k: orig[k] for k in ("G", "G_copy", "G_ex", "G_copy_ex", "old_form", "old_form_copy", "G_F", "G_M",
                                                                   "R3_half", "R4_half")},
           "splits": {k: stats([frac(r[k]) for r in sp]) for k in ("G", "G_copy", "G_ex", "G_copy_ex", "old_form", "old_form_copy", "G_F", "G_M")},
           "G_minus_copy": stats([frac(r["G"]) - frac(r["G_copy"]) for r in sp]),
           "G_ex_minus_copy_ex": stats([frac(r["G_ex"]) - frac(r["G_copy_ex"]) for r in sp])}
    if orig is not None:
        res["null_point"] = orig.get("null_point")
    fin, fmiss = load_records("final", FINAL_SEEDS)
    if fin:
        for r in fin.values():
            r.update(summarize_split(r))
        res["final_check"] = {"seeds": list(fin), "missing": fmiss, "G": stats([frac(r["G"]) for r in fin.values()]),
                              "G_ex": stats([frac(r["G_ex"]) for r in fin.values()]), "old_form": stats([frac(r["old_form"]) for r in fin.values()]),
                              "note": "final check (seeds 2000…2019, never used for selection): fair G = expectation of the submission; "
                                      "old form = channel from 4 against 3 on the same (favourable) OOF draws, biased against d4"}
    pla, pmiss = load_records("placebo", PLACEBO_SEEDS)
    if pla and orig is not None:
        base = half(orig["R3"][0])
        dp = [frac(r["half"]) - base for r in pla.values()]
        d0 = frac(orig["old_form"])
        res["placebo"] = {"seeds": list(pla), "missing": pmiss, "delta": stats(dp), "orig_delta": d0,
                          "share_placebo_ge_orig": float(np.mean([v >= d0 for v in dp])),
                          "orig_minus_placebo_mean": d0 - sum(dp) / len(dp)}
    if orig is not None and len(sp) == len(RULE_SEEDS):
        res["decision"] = decide(orig, sp)
    else:
        res["decision"] = {"verdict": "OPEN (incomplete)", "missing": missing}
    PRIV.mkdir(parents=True, exist_ok=True)
    out = PRIV / "c135_rule.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, Fraction) else str(o)) + "\n")
    for line in rule_lines(res):
        say(line)
    say(f"-> {rel(out)}")
    return 0


def rule_lines(res: dict) -> list[str]:
    f = lambda v: "–" if v is None else f"{float(v):+.5f}"  # noqa: E731
    s = res["splits"]
    lines = []
    if res.get("orig"):
        o = res["orig"]
        lines.append(f"original split: G {f(o['G'])} (F {f(o['G_F'])}, M {f(o['G_M'])}), copy {f(o['G_copy'])}; strict d3/d4 G_ex {f(o['G_ex'])}, "
                     f"copy {f(o['G_copy_ex'])}; m3x4 against m3x3 (deployment form) {f(o['old_form'])}, copy null point {f(o['old_form_copy'])}")
    lines.append(f"{s['G']['n']} splits 1000…: G {f(s['G']['mean'])} ± {s['G']['se'] or 0:.5f} ({s['G']['share_pos'] or 0:.0%} > 0); copy "
                 f"{f(s['G_copy']['mean'])} ± {s['G_copy']['se'] or 0:.5f}; G − copy {f(res['G_minus_copy']['mean'])} ({res['G_minus_copy']['share_pos'] or 0:.0%} > 0); "
                 f"strict G_ex {f(s['G_ex']['mean'])} ± {s['G_ex']['se'] or 0:.5f} ({s['G_ex']['share_pos'] or 0:.0%} > 0), copy_ex {f(s['G_copy_ex']['mean'])}; "
                 f"old form {f(s['old_form']['mean'])} ± {s['old_form']['se'] or 0:.5f}, its copy null point {f(s['old_form_copy']['mean'])}")
    if res.get("placebo"):
        p = res["placebo"]
        lines.append(f"placebo ({p['delta']['n']} seeds, d4 permuted within each fold): Δ {f(p['delta']['mean'])} ± SD {p['delta']['sd'] or 0:.5f} "
                     f"[{p['delta']['min']:+.5f} … {p['delta']['max']:+.5f}]; original {f(p['orig_delta'])}; share placebo ≥ original {p['share_placebo_ge_orig']:.2f}")
    if res.get("final_check"):
        q = res["final_check"]
        lines.append(f"final check ({q['G']['n']} splits 2000…): fair G {f(q['G']['mean'])} ± {q['G']['se'] or 0:.5f} ({q['G']['share_pos'] or 0:.0%} > 0), "
                     f"strict {f(q['G_ex']['mean'])}; old form {f(q['old_form']['mean'])} ± {q['old_form']['se'] or 0:.5f}")
    d = res["decision"]
    lines.append(f"VERDICT of the four-draw rule (1)–(2): {d['verdict']}" + (f" — {'; '.join(d['reasons'])}" if d.get("reasons") else "")
                 + (f" (missing: {', '.join(d['missing'])})" if d.get("missing") else ""))
    return lines


# ------------------------------------------------------------------------------------------------ deployment path: artifact m3x4_full

def light() -> dict:
    """Labels, classes, folds (without the recipe)."""
    return c133().light()


def four_specs() -> list[str]:
    """OOF draws d1, d2, d3 (meta_m3x3_full.json, order of the file) and d4."""
    meta = json.loads((ART / f"meta_{M3}.json").read_text())
    specs = [str(d["source"]) for d in (meta.get("rank") or {}).get("draws") or []]
    if len(specs) != 3 or ",".join(specs) != str(meta["rank"]["source"]):
        raise TechnicalError(f"meta_{M3}.json: not three draws in rank.draws / rank.source")
    if D4_SPEC in specs:
        raise TechnicalError("d4 is already among the draws of m3x3_full")
    return specs + [D4_SPEC]


def build_m3x4(lc: dict) -> tuple[np.ndarray, dict]:
    """Channel m3x4_full = mean of the four draws d1..d4 (eda/a9_m3_draws_oof.build in this order) and its meta fields as in
    eda/a9_m3_draws_oof.py main; row list = the existing derived/m3x3_full_rows.npy (all 1,500 rows carry all four draws — same checksum;
    the file is not written). Checks: the first three draws are those of m3x3_full (mean bit-identical to oof_m3x3_full)."""
    import metric
    from kaggle_core.folds import folds_hash

    A9 = c118().A9
    n, K, y, seen, folds, root = lc["n"], lc["K"], lc["y"], lc["seen"], lc["folds"], lc["root"]
    specs = four_specs()
    P, cnt, draws, infos = A9.build(specs, n, seen)
    rows_all = np.flatnonzero(cnt == len(specs)).astype(np.int64)
    if len(rows_all) != n:
        raise TechnicalError(f"{n - len(rows_all)} training rows without all four draws")
    m3 = np.load(ART / f"oof_{M3}.npy")
    if not np.array_equal(A9.mean_of_draws(draws[:3], n, K)[0], m3):
        raise TechnicalError(f"mean of the first three draws != oof_{M3}.npy (bit-identical required)")
    meta3 = json.loads((ART / f"meta_{M3}.json").read_text())
    rk3 = meta3["rank"]
    rows_file = str(rk3["rows_all_draws_file"])
    f_rows = Path(root) / rows_file
    if not f_rows.is_file() or A9.rows_sha256(np.load(f_rows)) != A9.rows_sha256(rows_all) or rk3["rows_all_draws_sha256"] != A9.rows_sha256(rows_all):
        raise TechnicalError(f"row list {rows_file} missing or not exactly the rows with all four draws")
    for j, (Pj, hj) in enumerate(draws):
        infos[j]["cv_alone_own_rows"] = float(metric.map3_rows_from_proba(y[hj], Pj[hj]).mean())
        infos[j]["none_top_own_rows"] = float(np.mean(Pj[hj].argmax(1) == 0))
    r_all = metric.map3_rows_from_proba(y, P)
    on_all = {"rows": int(len(rows_all)), "mean": float(r_all[rows_all].mean()),
              **{f"draw{j + 1}": float(metric.map3_rows_from_proba(y[rows_all], Pj[rows_all]).mean()) for j, (Pj, _) in enumerate(draws)}}
    checks = {"m3x3_full": {"first_three_mean_bit_identical": True, "oof_sha256": sha256_file(ART / f"oof_{M3}.npy"),
                            "meta_sha256": sha256_file(ART / f"meta_{M3}.json")},
              "rows_file": {"file": rows_file, "reused_from": M3, "sha256": A9.rows_sha256(rows_all)}}
    meta = {"folds_hash": folds_hash(folds), "model_type": "vlm-sft-rank", "sources": specs, "cv_score": float(r_all.mean()),
            "cv_sigma": float(metric.bootstrap_sigma(r_all)),
            "feature_set": "image+task, without task knowledge (trained model, ranking behind its own text)",
            "rows_without_answer": 0, "neutral_rows": None, "rows_neutral": 0, "rank": A9.rank_record(specs, cnt, infos, rows_file, rows_all),
            "cv_rows_all_draws": on_all, "checks": checks,
            "cv_note": "four draws on all rows (d1, d2, d3 as m3x3_full, d4 sampled like d3 with another seed); deployment under the four-draw "
                       "rule (eda/c135_primary_m3x4.py oof): channel from four draws, second-stage features from d1–d3 (field draws_first of the "
                       "deployment file)"}
    return P, meta


def cmd_oof(save: bool) -> int:
    lc = light()
    P, meta = build_m3x4(lc)
    info = c133().save_or_verify(M3X4, P, meta, lc["K"], save)
    rk = meta["rank"]
    say(f"{M3X4}: mean of {len(rk['draws'])} draws on {rk['rows_all_draws']} rows — MAP@3 alone {meta['cv_score']:.4f} ± {meta['cv_sigma']:.4f} "
        f"(per draw {[round(meta['cv_rows_all_draws'][f'draw{j + 1}'], 4) for j in range(4)]}); first three = {M3} bit-identical; row list "
        f"{rk['rows_all_draws_file']} reused; {'stored' if save else 'CHECKED'}: {info['oof']} ({info['oof_sha256'][:16]}…), "
        f"{info['meta']} ({info['meta_sha256'][:16]}…)")
    return 0


# ------------------------------------------------------------------------------------------------ blend

STEM, STEM_COPY = "primary_m3x4", "primary_m3x4_copy"
CH_COPY = "m3x4k"  # copy null point 'd4 removed': channel = m3x3_full bit-identical (three draws) under a new name
STAGE2 = ART / "stage2_primary_m3x4.json"
STAGE2_M3PX27 = ART / "stage2_primary_m3px27.json"
STAGE2_M3PX27_SHA = "9321d597b175d672fe5b31bf284ce984fbce149cb2e2cd8941a31aa8070d2500"
CHAIN_M3PX27 = ART / "eda" / "private" / "primary_m3px27" / "chain"
ROWS_M3PX27 = ART / "eda" / "private" / "primary_m3px27" / "c133_oof_rows.npz"
DEFINITION4 = ("mean of the probabilities of the four M3 draws d1, d2, d3 (as m3x3_full) and d4 (sampled like d3, seed 4) in a fixed "
               "order (eda/a9_m3_draws_oof.mean_of_draws); OOF per fold the adapter without this fold, test rows M3 of the full fit; before the "
               "floor; floor in the blend 0.05; second-stage features still from d1–d3 (draws_first) — four-draw rule")


def channel_meta_copy() -> dict:
    """Meta fields of the copy channel m3x4k: those of meta_m3x3_full.json (without name/oof_shape/test_shape/oof_hash) and a copy mark."""
    m = json.loads((ART / f"meta_{M3}.json").read_text())
    out = {k: v for k, v in m.items() if k not in ("name", "oof_shape", "test_shape", "oof_hash")}
    out["copy_null_point"] = {"of": M3, "note": "copy null point 'd4 removed' (eda/c135_primary_m3x4.py copy): the path of primary_m3x4 with "
                                                "the three draws d1–d3 — temporary, never to be submitted"}
    return out


def build_stack(name: str, P: np.ndarray) -> tuple[dict, dict]:
    """Blend file of primary_m3px27 with `name` (matrix P, artifact oof_/meta_<name>) in place of m3x3_full — procedure as
    c133.build_stack: θ of ALL channels on log loss from all 1,500 rows (c110.fit_all, reasoning factor), β by arith_beta_rule
    (c110.choose_beta), field cv = cross-fit per fold with the last step (c107.cross_fit), check make_submission.oof_mixture."""
    X = c133().c110()
    C107 = X.C107
    f, mf = ART / f"oof_{name}.npy", ART / f"meta_{name}.json"
    if not f.is_file() or not mf.is_file() or not np.array_equal(np.load(f), P):
        raise Missing(f"{rel(f)} / {rel(mf)} missing or not the matrix of the channel — run oof --save or copy first")
    meta_n = json.loads(mf.read_text())
    ref_file = ART / f"stack_{STEM_M3PX27}.json"
    if sha256_file(ref_file) != STACK_M3PX27_SHA:
        raise TechnicalError(f"{rel(ref_file)} is not the blend file of primary_m3px27")
    ref = json.loads(ref_file.read_text())
    ctx = X.setup()
    y, folds, seen, n, K = ctx["y"], ctx["folds"], ctx["seen"], ctx["n"], ctx["K"]
    mats, sha16, metas = X.load_mats(ctx, M3)
    Pq, sha_q, _, _ = X.load_replacement(ctx, "q397m3")
    mats["q397m3"], sha16["q397m3"] = Pq, sha_q
    fm = ART / f"oof_{MID}.npy"
    if sha256_file(fm) != MID_OOF_SHA:
        raise TechnicalError(f"{rel(fm)} is not the mean channel of primary_m3px27")
    mats[MID], sha16[MID] = np.load(fm), MID_OOF_SHA[:16]
    mats[name], sha16[name] = P, sha256_file(f)[:16]
    _, spec_b = X.line_specs("best", M3)
    spec_m3px27 = X.replaced_spec(X.replaced_spec(spec_b, X.CH, "q397m3"), "pr27xm", MID)  # recipe primary_m3px27
    if spec_m3px27["names"] != list(ref["channels"]):
        raise TechnicalError(f"recipe {spec_m3px27['names']} != channels of {ref_file.name}")
    spec_n = X.replaced_spec(spec_m3px27, M3, name)
    th, th0, L = X.fit_all(mats, spec_n, y, seen)
    beta, lls = X.choose_beta(th, L, y, seen)
    spec_s = {**spec_n, "beta": beta}
    draws_q = C107.question_draws(ctx["q_code"])
    fit_s = C107.cross_fit(y, folds, seen, mats, spec_s)
    fit_02 = C107.cross_fit(y, folds, seen, mats, spec_n)
    fit_ref = C107.cross_fit(y, folds, seen, mats, {**spec_m3px27, "beta": float(ref["arith_beta"])})
    r_s, r_02, r_ref = (C107.rows_map(y, fx["Q"], seen) for fx in (fit_s, fit_02, fit_ref))
    ll_s, ll_02, ll_ref = (C107.rows_ll(y, fx["Q"]) for fx in (fit_s, fit_02, fit_ref))
    if int(C107.units(r_ref).sum()) != int(round(float(ref["cv"]) * 6 * n)):
        raise TechnicalError(f"cross-fit of the primary_m3px27 blend {C107.units(r_ref).sum()}/{6 * n} does not hit the field cv of {ref_file.name}")
    cmp_ref, cmp_02 = C107.compare(r_s, r_ref, ctx, draws_q, ll_s, ll_ref), C107.compare(r_s, r_02, ctx, draws_q, ll_s, ll_02)
    pre_s, set_s = C107.rows_map(y, fit_s["P"], seen), C107.rows_map(y, C107.set_align(fit_s["P"], y, seen), seen)
    c = {"map3": float(r_s.mean()), "units": int(C107.units(r_s).sum()), "logloss": float(ll_s.mean()), "map3_pre_last_step": float(pre_s.mean()),
         "map3_one_set": float(set_s.mean()), "per_fold": [float(r_s[folds == k].mean()) for k in range(C107.N_FOLDS)]}
    d, d0 = X.summarize_cross(cmp_ref), X.summarize_cross(cmp_02)
    ren = lambda k: name if k == M3 else k  # noqa: E731
    if X.rank_only_entry(M3, metas[M3], sha16[M3], n) != ref["rank_only"][M3]:
        raise TechnicalError(f"rank_only entry {M3} differs from {ref_file.name}")
    if (ref["rank_only"].get("q397m3") or {}).get("oof_sha256") != sha_q:
        raise TechnicalError(f"rank_only entry q397m3 of {ref_file.name} names a different artifact")
    stack = copy.deepcopy(ref)
    stack["channels"] = list(spec_n["names"])
    stack["theta"] = [float(v) for v in th]
    stack["floor"] = {ren(k): float(v) for k, v in ref["floor"].items()}
    stack["base"] = {"channels": list(spec_n["names"]), "theta": [float(v) for v in th0]}
    stack["rank_only"] = {ren(k): (X.rank_only_entry(name, meta_n, sha16[name], n) if k == M3 else v) for k, v in ref["rank_only"].items()}
    stack["arith_beta"] = float(beta)
    stack["cv"] = c["map3"]
    stack["cv_note"] = (f"cross-fit per fold on 1,500 rows WITH the last step at the arithmetic share β {beta:g} (weights per fold from the other "
                        f"four, eda/c107_q397_full.py cross_fit; β by arith_beta_rule from all 1,500 rows) with {name} instead of {M3}; primary_m3px27 "
                        f"blend {float(ref['cv']):.7f}, Δ {d['delta']:+.5f} ± {d['se_q']:.5f} over tasks (z {d['z_q']:+.2f}); at β 0.2 {float(r_02.mean()):.7f}; "
                        "base OOFs not fully nested in the meta fold; decision on the channel and the second stage: four-draw rule (c135 rule)")
    stack["cross"] = {"map3": c["map3"], "logloss": c["logloss"], "per_fold": c["per_fold"], "map3_pre_last_step": c["map3_pre_last_step"],
                      "map3_one_set": c["map3_one_set"], "primary_m3px27": float(ref["cv"]), "beta_0_2": float(r_02.mean()),
                      "delta_vs_primary_m3px27": {k: d[k] for k in ("delta", "se_q", "z_q", "folds_gt0", "ll_gain", "ll_z_q")}}
    stack["oof_sha256"] = {nm: sha16[nm] for nm in spec_n["names"]}
    stack["replaces"] = {"channel": M3, "by": name, "definition": DEFINITION4 if name == M3X4 else f"copy of {M3} (copy null point 'd4 removed')",
                         "stack_primary_m3px27": ref_file.name, "stack_primary_m3px27_sha256": STACK_M3PX27_SHA, "decision": "four-draw rule (c135 rule)",
                         "previous": ref.get("replaces")}
    stack["note"] = (f"eda/c135_primary_m3x4.py stack: recipe primary_m3px27 ({ref_file.name}) with {name} instead of {M3} (same position, floor "
                     f"0.05); weights of ALL channels on log loss from all {n} rows (c110.fit_all, then {spec_n['think']} × {spec_n['factor']:g}); β by "
                     f"arith_beta_rule; build script tools/builds/primary_m3x4.sh")
    allr = np.arange(n)
    P_ms = X.MS.oof_mixture(stack, seen, ART, n_rows=n)
    Lm = C107.logs_of(mats, spec_s["names"], spec_s["floors"], seen, True)
    P_mask, P_way = X.MS.mix_channels(stack, Lm, None, seen), X.MS.mix_channels(stack, L, None, seen)
    d_mask, d_way = float(np.abs(P_ms - P_mask).max()), float(np.abs(P_ms - P_way).max())
    if d_mask != 0.0 or d_way > 1e-9:
        raise TechnicalError(f"check oof_mixture: deviation {d_mask:.1e} / {d_way:.1e}")
    Q_in = C107.apply_last(P_way, X.C21.fit_weights(P_way, C107.target_of(y, allr, K)), seen)
    stack["insample"] = {"map3": float(C107.rows_map(y, Q_in, seen).mean()), "map3_pre_last_step": float(C107.rows_map(y, P_way, seen).mean()),
                         "logloss": float(C107.rows_ll(y, Q_in).mean())}
    rule = dict(ref["arith_beta_rule"])
    rule.update({"grid": list(X.BETA_GRID), "logloss_pre_last_step": lls, "chosen": float(beta), "selected": True,
                 "stack_beta_0_2": None, "stack_beta_0_2_sha256": None,
                 "cross_beta_0_2": {"map3": float(r_02.mean()), "units": int(C107.units(r_02).sum()), "logloss": float(ll_02.mean())},
                 "delta_vs_beta_0_2": {k: d0[k] for k in ("delta", "se_q", "z_q", "folds_gt0", "ll_gain", "ll_z_q")}})
    stack["arith_beta_rule"] = rule
    if not C107.sane(Q_in, seen) or not C107.sane(P_ms, seen):
        raise TechnicalError("predictions not normalised or with mass on classes without a training example")
    rep = {"names": spec_n["names"], "beta": float(beta), "cross": c, "delta_vs_primary_m3px27": d, "delta_vs_beta_0_2": d0,
           "theta": dict(zip(spec_n["names"], map(float, th))), "insample": stack["insample"],
           "crosscheck_oof_mixture": {"max_abs_mask_first": d_mask, "max_abs_test_way": d_way}}
    return stack, rep


def stack_numbers_equal(a: dict, b: dict, name: str) -> list[str]:
    """Copy null point of the blend: numbers of a (channel `name`) = those of b (primary_m3px27, channel m3x3_full) — θ, base θ, floors, β,
    cv, cross-fit, in-sample and β-rule numbers, rank_only (except for the name), rows and last step."""
    ren = lambda k: M3 if k == name else k  # noqa: E731
    bad = [k for k in ("theta", "arith_beta", "cv") if a[k] != b[k]]
    if a["base"]["theta"] != b["base"]["theta"] or [ren(c) for c in a["channels"]] != b["channels"]:
        bad.append("channels/base.theta")
    if {ren(k): v for k, v in a["floor"].items()} != b["floor"] or a["scaled"] != b["scaled"]:
        bad.append("floor/scaled")
    if {ren(k): v for k, v in a["rank_only"].items()} != b["rank_only"]:
        bad.append("rank_only")
    for k in ("map3", "logloss", "per_fold", "map3_pre_last_step", "map3_one_set"):
        if a["cross"][k] != b["cross"][k]:
            bad.append(f"cross.{k}")
    if a["insample"] != b["insample"]:
        bad.append("insample")
    for k in ("logloss_pre_last_step", "chosen", "grid"):
        if a["arith_beta_rule"][k] != b["arith_beta_rule"][k]:
            bad.append(f"arith_beta_rule.{k}")
    for k in ("rows", "folds", "fit_rows", "fit_rows_sha256", "none", "probes", "last_step"):
        if a[k] != b[k]:
            bad.append(k)
    return bad


def cmd_stack(save: bool) -> int:
    f = ART / f"oof_{M3X4}.npy"
    if not f.is_file():
        raise Missing(f"{rel(f)} missing — run oof --save first")
    stack, rep = build_stack(M3X4, np.load(f))
    sha = c133().write_or_verify_text(ART / f"stack_{STEM}.json", c133().to_json(stack), save)
    d = rep["delta_vs_primary_m3px27"]
    say(f"stack_{STEM}.json: channels {stack['channels']}; β {rep['beta']:g}; θ " + " · ".join(f"{k} {v:.4f}" for k, v in rep["theta"].items()))
    say(f"   cross-fit {rep['cross']['map3']:.5f} ({rep['cross']['units']}/9000) against the primary_m3px27 blend Δ {d['delta']:+.5f} ± {d['se_q']:.5f} "
        f"(z {d['z_q']:+.2f}); without the last step {rep['cross']['map3_pre_last_step']:.5f}, one set {rep['cross']['map3_one_set']:.5f}; oof_mixture check "
        f"{rep['crosscheck_oof_mixture']['max_abs_mask_first']:.1e}/{rep['crosscheck_oof_mixture']['max_abs_test_way']:.1e}; "
        f"{'stored' if save else 'CHECKED'} (SHA-256 {sha})")
    return 0


# ------------------------------------------------------------------------------------------------ chain of the second stage

def swap_ctx(ctx_m3px27: dict, name: str, P: np.ndarray, stem: str) -> dict:
    """primary_m3px27 context with `name` (matrix P) in place of m3x3_full and the blend file stack_<stem>.json; draws and notes of the
    second stage stay d1–d3 (as c133.chain_ctx for the mean channel)."""
    C, S2 = c118(), c124().S2
    stack_file = ART / f"stack_{stem}.json"
    if not stack_file.is_file():
        raise Missing(f"{rel(stack_file)} missing — run stack --save (or copy) first")
    stack = json.loads(stack_file.read_text())
    spec = C.C115.replaced(ctx_m3px27["spec"], M3, name)
    floors = {k: float(v) for k, v in spec["floors"].items() if v}
    if stack.get("channels") != list(spec["names"]) or {k: float(v) for k, v in (stack.get("floor") or {}).items()} != floors:
        raise TechnicalError(f"{stack_file.name}: channels/floors {stack.get('channels')} / {stack.get('floor')} do not match the recipe {spec['names']}")
    if (stack.get("scaled") or {}) != {spec["think"]: float(spec["factor"])}:
        raise TechnicalError(f"{stack_file.name}: reasoning factor {stack.get('scaled')}")
    if not np.array_equal(np.load(ART / f"oof_{name}.npy"), P):
        raise TechnicalError(f"artifacts/oof_{name}.npy is not the matrix of the channel")
    ctx = dict(ctx_m3px27)
    ctx["mats"] = {**ctx_m3px27["mats"], name: np.asarray(P, dtype=np.float64)}
    ctx.update(spec=spec, stack=stack, stack_file=stack_file, stack_stem=stem, beta=float(stack.get("arith_beta") or 0.0))
    st = S2.Static(spec["names"], ctx["mats"], ctx["draws"], ctx["grade"], ctx["seen"], ctx["classes"], ctx["notes"])
    S2.attach_mech(st, ctx["mech27"], ctx["minor"])
    ctx["static"] = st
    return ctx


class _DrawsChannelAs:
    """During the chain build (c133.compose -> c118.build_stage2): draws_channel = `name` and right after it draws_first = 3 (channel from
    several draws, second-stage features from the first three = d1–d3; tools/make_submission.stage2_draws_first). The fields pass from step a
    into b–e; the provenance checksums of the chain are computed on the texts with the fields. Afterwards c118.build_stage2 is restored."""

    def __init__(self, name: str):
        self.name, self.orig = name, None

    def __enter__(self):
        C = c118()
        self.orig = C.build_stage2
        orig, name, k = self.orig, self.name, len(c124().S2.DRAWS)

        def wrapped(*a, **kw):
            out, rep = orig(*a, **kw)
            if out.get("draws_channel") != M3 or "draws_first" in out:
                raise TechnicalError(f"c118.build_stage2: draws_channel {out.get('draws_channel')!r} instead of {M3}")
            new = {}
            for key, v in out.items():
                new[key] = name if key == "draws_channel" else v
                if key == "draws_channel":
                    new["draws_first"] = k
            return new, rep

        C.build_stage2 = wrapped
        return self

    def __exit__(self, *exc):
        c118().build_stage2 = self.orig
        return False


def oof_rows_check(rows: dict, ctx_m3px27: dict) -> dict:
    """DEPLOY: the MM row values F and M of the chain (channel m3x4_full) = those of the evaluation (bank: triple d1d2d3, channel from four
    draws under the name m3x3_full, nested anew) BIT FOR BIT; if the stored original split of the bank is present, also its sixths R4_0."""
    P4, _ = load_d4_matrix(ctx_m3px27)
    bank = Bank(ctx_m3px27, P4, None)
    u, brows, betas = run_units(bank.variant(TRIPLES[0], bank.channel((0, 1, 2, 3)), np.asarray(ctx_m3px27["folds"])))
    ok = bool(np.array_equal(rows["mm_fold_forced"], brows["fold"]) and np.array_equal(rows["mm_set_forced"], brows["set"]))
    if not ok:
        raise TechnicalError("MM rows of the chain != evaluation rows of the bank (R4_0) — not bit-identical")
    out = {"label": "bank R4_0 (c135 bank, recomputed)", "bit_identical": True, "units": list(u), "betas": betas}
    f = out_file("bank", None)  # private evaluation: abort guard only, does NOT go into the deployment file (missing in the offline package)
    if f.is_file():
        rec = json.loads(f.read_text())
        if list(rec["R4"][0]) != list(u):
            raise TechnicalError(f"sixths of the chain {list(u)} != stored evaluation {rec['R4'][0]} ({rel(f)})")
        say(f"   MM rows of the chain = evaluation R4_0 bit-identical; sixths {list(u)} = stored bank ({rel(f)})")
    else:
        say(f"   MM rows of the chain = evaluation R4_0 bit-identical (recomputed); {rel(f)} not in the tree — comparison with the stored bank skipped")
    return out


def copy_rows_check(rows: dict) -> dict:
    """COPY: all row values of the chain = those of the primary_m3px27 chain (c133 fit, c133_oof_rows.npz) bit-identical."""
    if not ROWS_M3PX27.is_file():
        raise Missing(f"{rel(ROWS_M3PX27)} missing — run bash tools/builds/primary_m3px27.sh (c133 fit --save) first")
    z = np.load(ROWS_M3PX27)
    bad = [k for k in z.files if k not in rows or not np.array_equal(rows[k], z[k])]
    if bad or set(rows) != set(z.files):
        raise TechnicalError(f"copy null point: row values of the chain != primary_m3px27 ({bad[:5]}; keys {sorted(set(rows) ^ set(z.files))[:5]})")
    return {"label": "primary_m3px27 chain (c133_oof_rows.npz)", "keys": len(z.files), "bit_identical": True}


def m3x4_field(kind: str, name: str, check: dict, res: dict) -> dict:
    return {"decision": "four-draw rule (c135 rule)", "kind": kind, "channel": {"slot": M3, "by": name},
            "definition": DEFINITION4 if kind == "deploy" else f"copy of {M3} ('d4 removed')", "draws_first": len(c124().S2.DRAWS),
            "test_side": f"make_submission --rank-only {name}=<d1>,<d2>,<d3>,<d4> --stage2 (draws_first: features from the first three)",
            "oof_rows": check, "beta_outer": res["beta_outer"], "null_point_i": res["null_point_i"], "rule": RULE_TEXT,
            "copy_null_point": "eda/c135_primary_m3x4.py copy: the same path with the three draws d1–d3 (d4 removed) hits primary_m3px27 (blend, "
                               "chain, test side byte-identical)"}


def run_chain(name: str, P: np.ndarray, stem: str, tag: str, kind: str, log=say) -> tuple[list, dict, dict, Path, dict]:
    X = c133()
    ctx_m3px27 = m3px27_context()
    ctx = swap_ctx(ctx_m3px27, name, P, stem)
    res, rows, std = X.oof_chain(ctx, log=log)
    check = oof_rows_check(rows, ctx_m3px27) if kind == "deploy" else copy_rows_check(rows)
    res["reproduction"] = {"kind": "evaluation rows R4_0 bit-identical" if kind == "deploy" else "primary_m3px27 chain bit-identical", **check}
    m3px27_src = json.loads(STAGE2_M3PX27.read_text()).get("m3px27_build", {}).get("rank_artifact")
    s2b = X.m3px27_build_field(res, None, m3px27_src, "deploy")
    out_dir = PRIV / ("chain" if kind == "deploy" else "copy")
    oof_file = PRIV / f"c135_oof_{tag}.json"
    with _DrawsChannelAs(name):
        steps = X.compose(ctx, res, std, oof_file, out_dir, tag, s2b)
    steps[-1][2]["m3x4_build"] = m3x4_field(kind, name, check, res)
    return steps, res, rows, oof_file, ctx


def cmd_fit(save: bool) -> int:
    t0 = time.time()
    X = c133()
    if sha256_file(STAGE2_M3PX27) != STAGE2_M3PX27_SHA:
        raise TechnicalError(f"{rel(STAGE2_M3PX27)} is not the deployment file of primary_m3px27")
    f = ART / f"oof_{M3X4}.npy"
    if not f.is_file():
        raise Missing(f"{rel(f)} missing — run oof --save first")
    steps, res, rows, oof_file, ctx = run_chain(M3X4, np.load(f), STEM, "m3x4", "deploy")
    texts = [(s, (STAGE2 if p is None else p), X.to_json(d)) for s, p, d in steps]
    X.write_or_verify_text(oof_file, X.to_json(res), save)
    shas = {s: X.write_or_verify_text(p, t, save) for s, p, t in texts}
    e = steps[-1][2]
    m, fb, df = e["models"]["main"], e["models"]["fallback"], e["m3px27_build"]["deploy_forms"]
    say(f"stage2_{STEM}.json: main_guard_i1_mm ({len(m['features'])} features, guard on; gate {m['gate']['inner']:.5f} vs. {m['gate']['inner_base']:.5f} -> "
        f"{'ABSTAIN' if m['abstain'] else 'apply'}; OOF new top code {m['oof_changed_top_share']:.4f}; in-sample {m['insample']['map_base']:.5f} → "
        f"{m['insample']['map_reranked']:.5f}); fallback {len(fb['features'])} features; β {e['arith_beta']:g}; draws_channel {e['draws_channel']} "
        f"(draws_first {e['draws_first']}); MM rows = evaluation R4_0 bit-identical")
    say(f"   chain nested: MM F {res['summary']['mm']['fold']:.6f} M {res['summary']['mm']['set']:.6f}; deployment form standard cross-fit, β per fold "
        f"{df['beta_per_fold']} (blend file {df['beta_stack']:g})")
    sens, srep = X.sensitivity_fixed(ctx, e)
    sens["probe_note"] = ("SENSITIVITY (eda/c135_primary_m3x4.py fit): main model on c118.deploy_forms (fixed β of the blend file) instead of "
                          "the standard cross-fit of the reproduction; report only (aggregates), never to be submitted")
    X.write_or_verify_text(PRIV / "sens" / "stage2_primary_m3x4_fixbeta.json", X.to_json(sens), save)
    X.write_or_verify_text(PRIV / "sens" / "c135_sens_fixbeta.json", X.to_json(srep), save)
    say(f"   sensitivity (fixed β, report only): max |Δw| {srep['max_abs_dw']:.4f}, correlation of the weights {srep['corr_w']:.4f}")
    say(f"   chain a–e {'stored' if save else 'CHECKED'}: " + "; ".join(f"{s} {sha[:16]}…" for s, sha in shas.items()) + f" ({time.time() - t0:.0f} s)")
    return 0


PROVENANCE_TOP = ("stack", "stack_sha256", "inputs_sha256", "oof_reproduction", "derived_from", "m3px27_build", "m3x4_build")
PROVENANCE_BUILD = ("oof", "splits", "rule_oof", "fallback")


def normalized(d: dict, name: str) -> dict:
    """Comparison form for the copy null point: channel name -> m3x3_full (channels, features, draws_channel), draws_first and provenance
    fields removed."""
    out = copy.deepcopy({k: v for k, v in d.items() if k not in PROVENANCE_TOP and k != "draws_first"})
    ren = lambda s: s.replace(f"{name}_", f"{M3}_", 1) if isinstance(s, str) and s.startswith(f"{name}_") else (M3 if s == name else s)  # noqa: E731
    if "channels" in out:
        out["channels"] = [ren(c) for c in out["channels"]]
    if "draws_channel" in out:
        out["draws_channel"] = ren(out["draws_channel"])
    for m in (out.get("models") or {}).values():
        m["features"] = [ren(f) for f in m["features"]]
    for b in ("guard_build", "i1_build", "mm_build"):
        if b in out:
            out[b] = {k: v for k, v in out[b].items() if k not in PROVENANCE_BUILD}
    return out


COPY_FILES = (f"oof_{CH_COPY}.npy", f"test_{CH_COPY}.npy", f"meta_{CH_COPY}.json", f"stack_{STEM_COPY}.json")
COPY_STAGE2 = PRIV / "copy" / "stage2_m3x4k_e.json"


def clean_copy() -> list[str]:
    gone = []
    for f in COPY_FILES:
        p = ART / f
        if p.is_file():
            p.unlink()
            gone.append(f)
    return gone


def cmd_copy(keep: bool) -> int:
    """COPY NULL POINT 'd4 removed' of the training side: channel m3x4k = m3x3_full (three draws) bit-identical; blend (numbers =
    stack_primary_m3px27.json), chain (rows = primary_m3px27 chain, files a–e = those of primary_m3px27 except provenance, channel names and
    draws_first). --keep: the copy files under artifacts/ stay for the copy run of the test side (build script)."""
    t0 = time.time()
    X = c133()
    lc = light()
    clean_copy()
    try:
        P = np.load(ART / f"oof_{M3}.npy")
        X.save_or_verify(CH_COPY, P, channel_meta_copy(), lc["K"], True)
        stack, _ = build_stack(CH_COPY, P)
        ref = json.loads((ART / f"stack_{STEM_M3PX27}.json").read_text())
        bad = stack_numbers_equal(stack, ref, CH_COPY)
        if bad:
            raise TechnicalError(f"copy null point of the blend violated: {bad}")
        (ART / f"stack_{STEM_COPY}.json").write_text(X.to_json(stack))
        say(f"copy null point (blend): stack_{STEM_COPY}.json hits θ, base θ, floors, rank_only, β, cv, cross-fit, in-sample and β-rule numbers "
            f"of stack_{STEM_M3PX27}.json exactly")
        steps, res, rows, oof_file, _ = run_chain(CH_COPY, P, STEM_COPY, "m3x4k", "copy")
        PRIV.mkdir(parents=True, exist_ok=True)
        oof_file.write_text(X.to_json(res))
        refs = {s: (CHAIN_M3PX27 / f"stage2_primary_m3px27_{s}.json") for s in "abcd"}
        refs["e"] = STAGE2_M3PX27
        comp = {}
        for s, p, d in steps:
            if not refs[s].is_file():
                raise Missing(f"{rel(refs[s])} missing — run bash tools/builds/primary_m3px27.sh first")
            dk = X.diff_keys(normalized(d, CH_COPY), normalized(json.loads(refs[s].read_text()), M3))
            comp[s] = {"reference": rel(refs[s]), "equal": not dk, "diff": dk}
            path = COPY_STAGE2 if p is None else p
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(X.to_json(d))
        bad = {s: v["diff"] for s, v in comp.items() if not v["equal"]}
        rep = {"kind": "copy null point 'd4 removed' of the training side", "stack": f"stack_{STEM_COPY}.json", "chain": comp,
               "rows": res["reproduction"], "seconds": round(time.time() - t0, 1)}
        (PRIV / "c135_copy.json").write_text(X.to_json(rep))
        if bad:
            raise TechnicalError(f"copy null point of the chain violated: {bad}")
        say(f"copy null point (chain): row values = primary_m3px27 chain bit-identical; files a–e = those of primary_m3px27 (except provenance, "
            f"channel names, draws_first) -> deployment file of the copy {rel(COPY_STAGE2)} ({rep['seconds']} s)")
    except Exception:
        if not keep:
            clean_copy()
        raise
    if not keep:
        say(f"   temporary copy files removed: {', '.join(clean_copy())}")
    else:
        say(f"   copy files for the copy run of the test side lie under artifacts/: {', '.join(COPY_FILES)} (then clean-copy)")
    return 0


# ------------------------------------------------------------------------------------------------ test side d4

D4_TEST = "runs/m3d4te_rp/m3d4_test_rank/test"
TEST_RANKS3 = ("runs/m3_fullte_rp/m3_test_rank/test", "runs/m3d2te_rp/m3d2_test_rank/test", "runs/m3d3te_rp/m3d3_test_rank/test")
FULL_TRAIN = ("m3_fullte_rp", "m3_train_full")
NONE_BAND_GENERIC = (0.30, 0.38)  # make_submission --rank-none-band (default), the band every test draw d1–d3 passes
NONE_BAND_WIDTH = 0.02  # rule "NONE at the top within the band of the d1–d3 test sides": [min, max] of the d1–d3 test sides ± 0.02 (fixed
# before fetching: d1–d3 test 0.3456–0.3468; spread of the four OOF draws 0.3253–0.3367, i.e. ≈ 0.011 between draws)
SAME_TOP_WIDTH = 0.05  # report only: same top code d4 against d1–d3 on the test side against the range of the d1–d3 pairs ± 0.05


def sft_log_lines(logs: list[Path], n_test: int) -> int:
    """Number of log lines "sft_from: notes for <n_test> rows of table test, …" (log line of vlm/runner.py). The recorded worker logs use
    an earlier, German wording of the same line; both wordings are accepted (pattern c133.LOG_LINE)."""
    n = 0
    for f in logs:
        for line in f.read_text(errors="replace").splitlines():
            m = c133().LOG_LINE.search(line)
            g = [v for v in m.groups() if v is not None] if m else []
            n += int(bool(g) and int(g[0]) == n_test and g[1] == "test")
    return n


def test_guard(spec: str) -> tuple[int, list[str], dict]:
    """Technical checks of the d4 test ranking (aggregates only): complete and exact (3,498/3,498, checked at the ranking files, not at
    status.json), table test, settings of ranking AND generation equal to those of the OOF draw d4 (make_submission.rank_only_draw_differences),
    NONE at the top within the band of the d1–d3 test sides and within the generic band, training regime (test model = merge of the full fit
    m3_fullte_rp/m3_train_full as for d1–d3; recipe equal to the fold adapters of d4). Report: same top code against d1–d3, generation
    (answered/complete)."""
    import metric
    from vlm import common as VC

    X = c133()
    lc = light()
    root, seen, n_train = lc["root"], lc["seen"], lc["n"]
    n_test = len(VC.load_table(root, "test"))
    M = X.ms()
    missing = M.c95.rank_files_missing(spec)
    if missing:
        return 3, [f"test ranking {spec}: MISSING ({', '.join(missing)})"], {}
    A9, A8, C108 = X.c132().c115().A9, X.c132().c115().A8, X.c132().c115().C108
    run, job, tag = A9.parse_source(spec)
    problems: list[str] = []
    t_meta = A9.read_json(A9.run_dir(run) / job / f"{tag}_meta.json") or {}
    if t_meta.get("table") != "test":
        problems.append(f"test ranking computed table {t_meta.get('table')!r}")
    R, h, st = A8.load_rank(spec, n_test, seen, table="test", allow_partial=True)
    hr = h & M.c95.exact_rank_rows(spec, n_test, seen)
    exact = int(hr.sum())
    complete = bool(st["rows_probe"] == st["rows_done"] == st["rows_exact"] == exact == n_test)
    if not complete:
        problems.append(f"test ranking not completely exact: probe {st['rows_probe']}, computed {st['rows_done']}, exact {st['rows_exact']}, "
                        f"exact ranking {exact} of {n_test}")
    partial: list[str] = []  # statistics that are only judged on the COMPLETE test side (partial states compute the longest prompts first)
    if not (np.isfinite(st["mass_median"]) and st["mass_median"] >= 0.9):
        partial.append(f"median of the candidate mass {st['mass_median']:.3f} below 0.9 (make_submission --rank-min-mass)")
    none_top = M.c95.top_none_share(R, hr)
    ref_none, ref_top = [], []
    for s in TEST_RANKS3:
        Rj, hj, _ = A8.load_rank(s, n_test, seen, table="test", allow_partial=True)
        hj = hj & M.c95.exact_rank_rows(s, n_test, seen)
        ref_none.append(M.c95.top_none_share(Rj, hj))
        ref_top.append((metric.top3_indices(Rj, seen)[:, 0], hj))
    lo, hi = min(ref_none) - NONE_BAND_WIDTH, max(ref_none) + NONE_BAND_WIDTH
    if not (lo <= none_top <= hi):
        partial.append(f"NONE at the top {none_top:.4f} outside the band of the d1–d3 test sides {lo:.4f}–{hi:.4f}")
    if not (NONE_BAND_GENERIC[0] <= none_top <= NONE_BAND_GENERIC[1]):
        partial.append(f"NONE at the top {none_top:.4f} outside {NONE_BAND_GENERIC[0]:.2f}–{NONE_BAND_GENERIC[1]:.2f}")
    if complete:  # on a partial state mass and NONE share are not meaningful (only reported, not judged)
        problems += partial
    diff = M.rank_only_draw_differences(spec, D4_SPEC)
    if diff:
        problems.append(f"settings of the test ranking/generation differ from those of the OOF draw d4: {diff}")
    # training regime: test model = merge of the full fit; recipe equal to the fold adapters of d4
    recs = C108.fold_check(D4_SPEC, lc["folds"], {})
    problems += [f"OOF d4 {r['run']}/{r['job']}: {p}" for r in recs for p in r["problems"] + r["unknown"]]
    fold_recipes = {}
    for r in recs:
        if r.get("train"):
            r2, j2 = str(r["train"]).split("/", 1)
            fold_recipes[r["train"]] = A9.read_json(A9.run_dir(r2) / j2 / "sft_recipe.json") or {}
    if len(fold_recipes) != 5:
        problems.append(f"recipes of the fold adapters of d4: {len(fold_recipes)} instead of 5")
    model = str((t_meta.get("recipe") or {}).get("model"))
    rec, where, p = X.adapter_recipe(run, model)
    problems += [f"Test {run}/{job}: {x}" for x in p]
    if where.get("train") != "/".join(FULL_TRAIN):
        problems.append(f"test model from {where.get('train')} instead of {'/'.join(FULL_TRAIN)} (full fit)")
    if rec is not None:
        if rec.get("fold") is not None:
            problems.append(f"test model holds out fold {rec.get('fold')} — not a full fit")
        if rec.get("teacher_derived") or rec.get("pool"):
            problems.append("test model with teacher targets (not gold-only)")
        if int(rec.get("rows_used") or -1) != int(n_train):
            problems.append(f"test model learned {rec.get('rows_used')} instead of {n_train} training rows")
        if rec.get("steps_done") != rec.get("steps_total"):
            problems.append(f"training of the test model not finished ({rec.get('steps_done')}/{rec.get('steps_total')})")
        for nm, fr in fold_recipes.items():
            dd = {k: (rec.get(k), fr.get(k)) for k in X.RECIPE_SAME if rec.get(k) != fr.get(k)}
            if dd:
                problems.append(f"test model fitted differently from the fold adapter {nm}: {dd}")
    # report: same top code against d1–d3 (test side) and of the pairs d1–d3 among each other; generation
    t4 = metric.top3_indices(R, seen)[:, 0]
    same = []
    for (tj, hj) in ref_top:
        b = hr & hj
        same.append(float(np.mean(t4[b] == tj[b])) if b.any() else float("nan"))
    pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            b = ref_top[i][1] & ref_top[j][1]
            pairs.append(float(np.mean(ref_top[i][0][b] == ref_top[j][0][b])))
    same_ok = bool(all(min(pairs) - SAME_TOP_WIDTH <= v <= max(pairs) + SAME_TOP_WIDTH for v in same))
    src = str((t_meta.get("sft") or {}).get("sft_from") or "")
    gen = {}
    if "/" in src:
        gj, gt = src.split("/", 1)
        g = A9.read_json(A9.run_dir(run) / gj / f"{gt}_meta.json") or {}
        stt = g.get("states") or {}
        gen = {"rows": g.get("rows"), "answered": g.get("answered"), "failed": g.get("failed"), "complete": stt.get("complete"),
               "share_complete": (float(stt.get("complete", 0)) / float(g["answered"])) if g.get("answered") else None, "max_tokens": g.get("max_tokens")}
    logs = sorted(A9.run_dir(run).glob("worker*.log"))
    log_lines = sft_log_lines(logs, n_test)
    info = {"spec": spec, "coverage": {"rows_probe": st["rows_probe"], "rows_done": st["rows_done"], "rows_exact": st["rows_exact"], "exact_rows": exact,
                                       "rows_table": n_test, "complete": complete, "mass_median": st["mass_median"], "mass_p05": st["mass_p05"]},
            "none_top": {"test_d4": none_top, "test_d1_d3": ref_none, "band": [lo, hi], "generic_band": list(NONE_BAND_GENERIC)},
            "settings_equal": not diff,
            "training_regime": {"test_model": where, "test_recipe": {k: (rec or {}).get(k) for k in ("task", "fold", "rows_used", "steps_done", "steps_total",
                                                                                                       "taxonomy", "notes_max_tokens", "pool", "teacher_derived")},
                                "fold_adapters": [{k: r.get(k) for k in ("job", "train", "train_fold", "gold_only", "adapter_matches_merge")} for r in recs]},
            "same_top_vs_d1_d3": same, "same_top_pairs_d1_d3": pairs, "same_top_in_band": same_ok, "generation": gen, "log_lines": log_lines}
    lines = [f"test ranking {spec}: {exact}/{n_test} exact (probe {st['rows_probe']}, computed {st['rows_done']}), mass median {st['mass_median']:.3f}; "
             f"NONE at the top {none_top:.4f} (d1–d3 test {', '.join(f'{v:.4f}' for v in ref_none)}; band {lo:.4f}–{hi:.4f}); settings against the "
             f"OOF draw d4 {'equal' if not diff else 'DIFFERENT'}",
             f"training regime: test model {where.get('model')} <- {where.get('train')} (fold {(rec or {}).get('fold')}, {(rec or {}).get('rows_used')} rows, "
             f"{(rec or {}).get('steps_done')}/{(rec or {}).get('steps_total')} steps); fold adapters of d4 {len(fold_recipes)}",
             f"report: same top code d4 against d1/d2/d3 {', '.join(f'{v:.3f}' for v in same)} (pairs d1–d3 {', '.join(f'{v:.3f}' for v in pairs)}; "
             f"{'within the' if same_ok else 'OUTSIDE the'} band ± {SAME_TOP_WIDTH:g}); generation {gen.get('answered')}/{gen.get('rows')} answered, "
             f"failures {gen.get('failed')}, complete {gen.get('share_complete') if gen.get('share_complete') is None else round(gen['share_complete'], 3)}; "
             f"log lines sft_from {log_lines}"]
    lines += [f"PROBLEM: {x}" for x in problems]
    if not complete:
        lines += [f"(partial state, not judged: {x})" for x in partial]
        if len(problems) == 1:  # only completeness missing: run not finished or not completely fetched (exit 3, no verdict)
            return 3, lines, info
    return (5 if problems else 0), lines, info


def cmd_test_guard(spec: str) -> int:
    code, lines, info = test_guard(spec)
    for line in lines:
        say(line)
    PRIV.mkdir(parents=True, exist_ok=True)
    out = PRIV / ("c135_test_guard.json" if spec == D4_TEST else f"c135_test_guard_{''.join(ch if ch.isalnum() else '_' for ch in spec)}.json")
    out.write_text(c133().to_json({"code": code, "lines": lines, **info}))
    say(f"-> {'PASSED' if code == 0 else ('MISSING/INCOMPLETE' if code == 3 else 'VIOLATED')} (exit {code}); {rel(out)}")
    return code


def cmd_null(source: Path, out: Path, stem: str) -> int:
    """Null model (weight 0) for the deployment file: c118.null_stage2 + mech27 and draws_first of the source (at the same position as
    there)."""
    X = c133()
    ref = json.loads(Path(source).read_text())
    res = X.c118().null_stage2(stem, Path(source))
    res["mech27"] = ref["mech27"]
    if "draws_first" in ref:
        res = {k2: v2 for k, v in res.items() for k2, v2 in ([(k, v), ("draws_first", ref["draws_first"])] if k == "draws_channel" else [(k, v)])}
    Path(out).write_text(X.to_json(res))
    say(f"null model for {rel(Path(source))} -> {out}")
    return 0


# ------------------------------------------------------------------------------------------------ command line

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("bank", "final", "placebo"):
        p = sub.add_parser(nm)
        p.add_argument("--seeds", required=True)
        p.add_argument("--force", action="store_true")
    r = sub.add_parser("rule")
    r.add_argument("--partial", action="store_true", help="intermediate state (no verdict while splits are missing)")
    for nm in ("oof", "stack", "fit"):
        p = sub.add_parser(nm)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--save", action="store_true", help="writes what is missing, requires the same content for existing files (never overwrites)")
        g.add_argument("--verify", action="store_true", help="check only (requires all files)")
    c = sub.add_parser("copy")
    c.add_argument("--keep", action="store_true")
    sub.add_parser("clean-copy")
    t = sub.add_parser("test-guard")
    t.add_argument("--rank", default=D4_TEST)
    nl = sub.add_parser("null")
    nl.add_argument("--source", default=str(STAGE2))
    nl.add_argument("--out", required=True)
    nl.add_argument("--stack", default=STEM)
    a = ap.parse_args(argv)
    try:
        if a.cmd in ("bank", "final", "placebo"):
            return cmd_measure(a.cmd, parse_seeds(a.seeds), a.force)
        if a.cmd == "rule":
            return cmd_rule(a.partial)
        if a.cmd == "oof":
            return cmd_oof(a.save)
        if a.cmd == "stack":
            return cmd_stack(a.save)
        if a.cmd == "fit":
            return cmd_fit(a.save)
        if a.cmd == "copy":
            return cmd_copy(a.keep)
        if a.cmd == "clean-copy":
            say(f"removed: {', '.join(clean_copy()) or 'nothing'}")
            return 0
        if a.cmd == "test-guard":
            return cmd_test_guard(a.rank)
        return cmd_null(Path(a.source), Path(a.out), a.stack)
    except Missing as exc:
        print(f"NOT YET EVALUABLE: {exc}", file=sys.stderr)
        return 3
    except TechnicalError as exc:
        print(f"ABORT (technical): {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
