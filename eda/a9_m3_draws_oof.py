"""M3 as the MEAN OF SEVERAL DRAWS: OOF artifact from the exact rankings of several draws of the same trained model (probe
proto_rank in sft mode) — per training row the arithmetic mean of the probabilities over the draws that cover the row —,
plus the list of the rows with ALL draws (fit rows of the blend, eda/c99_m3_deploy.py --rows) and a meta file for c99 and
tools/make_submission.py --rank-only <channel>=<test draw 1>,<test draw 2>.

    python -I eda/a9_m3_draws_oof.py <name> --draw <source> --draw <source> [...] [--rows-out derived/<name>_rows.npy]
                                    [--ref m3p5] [--check-c98d]
    python -I eda/a9_m3_draws_oof.py --verify <name>          # recompute the artifact from the sources in its meta file (writes nothing)
    python -I eda/a9_m3_draws_oof.py --gen-status runs/<run>/<job>/<tag> [--table test] [--min-complete 0.5] [--max-failed 0.05]

DRAW (--draw; the order is that of the test draws in make_submission): an exact ranking of the training rows
`runs/<run>/<job>/<tag>` (bench output or combined output with combined_from) or several SHARDS of the same draw joined
with `+` — this way the path later picks up the second draws of folds 0/3/4 without a code change, e.g.
`--draw runs/m3long_comb/m3l_rank/bench+runs/<run>/<job_f0>/bench+…`. Every shard must be COMPLETE and EXACT
(a8.load_rank without partial state, completeness mask required); shards and fold jobs must not repeat, neither within a
draw nor between draws; the settings of the generation (<tag>_generation.json behind sft.sft_from: max_tokens,
temperature, system text, image, backend) and of the ranking (beams, image, finish, number format, sft_examples) must be
identical and known within a draw.
CONVERSION: per draw a8.load_rank (vlm.engine.probs_from_logp with absent = classes without a training example — the same
as a8 --rank), then per row Σ P_j / number of draws of the row. Rows with ONE draw carry its
distribution bit-identically (a8 --rank), rows with two draws bit-identically 0.5·(P_1 + P_2) (addition is commutative,
division by 2 is exact). Every training row needs at least one draw (no neutral remainder).
ROW LIST (--rows-out, relative to the data folder; default derived/<name>_rows.npy): sorted table positions (int64) of the
rows with ALL draws — with two draws "at least two". Checksum and path are stored in the meta file; c99 and
make_submission require that the blend weights were fitted on these rows only (all draws exist on the test rows).
META (kaggle_core.artifacts.save_oof): rank.source = sources of the draws joined with commas (the same notation as
make_submission --rank-only), rank.draws = per draw source, shards, fold jobs (run, job, model, ranking, generation),
settings (rank_recipe, generation, max_tokens) and statistics; rank.rows_by_draws; rank.draws_per_row (number of draws per
training row); rank.rows_all_draws(_file, _sha256). An existing artifact of the same name that does not come from this
script (without rank.draws, e.g. m3p5) is never overwritten.
CHECKS before writing (abort on deviation):
  --ref <channel> the draw with the same source as meta_<channel>.json (rank.source) is bit-identical to oof_<channel>.npy
                  on all its rows, and the artifact is bit-identical to oof_<channel>.npy on the rows with ONE draw;
  --check-c98d    on the rows of the two-draw evaluation bit-identical to the mean as computed there (script not included)
                  (c98b.load_rank, probs_from_logp, 0.5·(new + pilot)); MAP@3 of the mean and of the pilot draw are reported.
--verify <name>: recomputes the artifact from the sources in its meta file (bit-identical) and checks the row list; exit 0 / 4.
--gen-status: state of the generation behind a ranking (sft.sft_from -> <tag>_meta.json; aggregates ONLY, also for test
runs): rows = table, answered ≥ (1 − max_failed)·n, failures ≤ max_failed·n, share of complete answers ≥ --min-complete
(tipping thresholds fixed in advance: 0.3 for the first draw, 0.5 for the longer draws). Exit 0 = complete, 3 = not.
Only training rows in the artifact (table train of every source is checked); output only aggregates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import labels  # noqa: E402
import metric  # noqa: E402


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A8 = load("a9_a8", COMP / "eda" / "a8_proto_oof.py")
# definition of the channel, recorded in meta_<name>.json (field rank)
DEFINITION = ("mean of the probabilities of several draws: per draw the exact distribution over all codes (probe "
              "proto_rank, a8.load_rank / vlm.engine.probs_from_logp with absent = classes without a training example), per row "
              "Σ P_j / number of draws of the row; floor only in the blend (eda/c99_m3_deploy.py: 0.05)")
GEN_FREE_KEYS = ("table", "model_dir", "rows_sha256")  # free between runs: table, path of the merged model, row set
C98D_SOURCES = (("m3long_comb", "m3l_rank"), ("m3_pilot_rp", "m3_rank"))  # (new draw, pilot) of the two-draw evaluation


def rows_sha256(rows) -> str:
    """Checksum of a row list as in eda/c99_m3_deploy.rows_sha256 (int64 little-endian, full length)."""
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).astype("<i8").tobytes()).hexdigest()


def run_dir(run: str) -> Path:
    return A8.COMP / "artifacts" / "runs" / run / run


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def parse_source(src: str) -> tuple[str, str, str]:
    bits = str(src).split("/")
    if len(bits) != 4 or bits[0] != "runs" or not all(bits[1:]):
        raise SystemExit(f"{src}: expected runs/<run>/<job>/<tag>")
    return bits[1], bits[2], bits[3]


def generation(run: str, rank_meta: dict) -> dict | None:
    """Settings of the generation behind a ranking: sft.sft_from = <job>/<tag> in the same run -> <tag>_generation.json,
    field recipe without the free fields. None = cannot be determined."""
    src = (rank_meta.get("sft") or {}).get("sft_from")
    if not src or "/" not in str(src):
        return None
    job, tag = str(src).split("/", 1)
    rec = read_json(run_dir(run) / job / f"{tag}_generation.json")
    if rec is None or not isinstance(rec.get("recipe"), dict):
        return None
    return {k: v for k, v in rec["recipe"].items() if k not in GEN_FREE_KEYS}


def rank_settings(meta: dict) -> dict:
    """Ranking settings of a fold job as in tools/make_submission.rank_only_recipe_differences (without model, fold, note
    source; missing fields = None) plus sft_examples (M3k)."""
    out = {f"recipe.{k}": v for k, v in (meta.get("recipe") or {}).items() if k not in ("model", "fold", "proto_from")}
    out.update({f"sft.{k}": v for k, v in (meta.get("sft") or {}).items() if k != "sft_from"})
    out.update({k: meta.get(k) for k in ("probes", "dtype", "quant", "adapter", "sft_examples")})
    return {k: v for k, v in out.items() if v is not None}


def constituents(shard: str) -> list[dict]:
    """Fold jobs behind a ranking (combined_from resolved): run, job, model, settings of the ranking and of the generation."""
    run, job, tag = parse_source(shard)
    meta = read_json(run_dir(run) / job / f"{tag}_meta.json")
    if meta is None:
        raise SystemExit(f"{shard}: {tag}_meta.json missing (run not finished or not fetched)")
    parts = [(run, job, meta)]
    if meta.get("combined_from"):
        parts = []
        for s in meta["combined_from"]:
            r, j = s.get("run", run), s["job"]
            m = read_json(run_dir(r) / j / f"{tag}_meta.json")
            if m is None:
                raise SystemExit(f"{shard}: meta file of fold job {r}/{j} missing")
            parts.append((r, j, m))
    return [{"run": r, "job": j, "model": (m.get("recipe") or {}).get("model"), "rank": rank_settings(m), "generation": generation(r, m)}
            for r, j, m in parts]


def load_draw(spec: str, n: int, seen_cls: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """One draw from one or more shards (joined with +), each complete and exact, without overlap.
    -> (P [n, K] — rows without this draw 0, rows of the draw [n], statistics)."""
    P, have = np.zeros((n, len(seen_cls))), np.zeros(n, bool)
    shards = []
    for shard in str(spec).split("+"):
        parse_source(shard)
        R, h, st = A8.load_rank(shard, n, seen_cls, table="train", allow_partial=False)
        if not st["completion_manifest"]:
            raise SystemExit(f"{shard}: completeness mask (proto_rank.done.npy) missing — not a run that can support a decision")
        if (have & h).any():
            raise SystemExit(f"{spec}: {int((have & h).sum())} rows covered twice (shards of the same draw overlap)")
        P[h], have[h] = R[h], True
        shards.append({"source": shard, **st, "constituents": constituents(shard)})
    cons = [c for s in shards for c in s["constituents"]]
    gens = [c["generation"] for c in cons]
    if any(g is None for g in gens):
        miss = [f"{c['run']}/{c['job']}" for c in cons if c["generation"] is None]
        raise SystemExit(f"{spec}: generation settings cannot be determined for {', '.join(miss)} (<tag>_generation.json behind sft.sft_from missing)")
    gens = [{k: v for k, v in g.items() if v is not None} for g in gens]  # missing field = None (as in make_submission)
    if len({json.dumps(g, sort_keys=True) for g in gens}) != 1:
        raise SystemExit(f"{spec}: the fold jobs of this draw were generated differently "
                         f"({sorted({json.dumps(g, sort_keys=True) for g in gens})}) — a draw is ONE form")
    ranks = [c["rank"] for c in cons]
    if len({json.dumps(r, sort_keys=True, default=str) for r in ranks}) != 1:
        raise SystemExit(f"{spec}: the fold jobs of this draw were computed differently "
                         f"({sorted({json.dumps(r, sort_keys=True, default=str) for r in ranks})}) — a draw is ONE form")
    info = {"source": str(spec), "shards": shards, "rows": int(have.sum()), "rows_sha256": rows_sha256(np.flatnonzero(have)),
            "rank_recipe": ranks[0], "generation": gens[0], "max_tokens": (gens[0].get("settings") or {}).get("max_tokens")}
    return P, have, info


def mean_of_draws(draws: list[tuple[np.ndarray, np.ndarray]], n: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row Σ P_j / number of draws of the row (over the draws that cover it); rows without a draw 0.
    -> (P, number of draws per row)."""
    S, cnt = np.zeros((n, K)), np.zeros(n, dtype=np.int64)
    for P, h in draws:
        S[h] += P[h]
        cnt[h] += 1
    out = np.zeros((n, K))
    have = cnt > 0
    out[have] = S[have] / cnt[have, None]
    return out, cnt


def build(specs: list[str], n: int, seen_cls: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], list[dict]]:
    """Load and average all draws. -> (P, number of draws per row, [(P_j, rows_j)], statistics per draw)."""
    specs = [str(s) for s in specs]
    if len(specs) < 2:
        raise SystemExit("at least two draws (--draw twice) — a single draw is built by eda/a8_proto_oof.py --rank")
    if any("," in s for s in specs):
        raise SystemExit("one draw per --draw (shards with +, never with a comma)")
    loaded = [load_draw(s, n, seen_cls) for s in specs]
    shard_src = [sh["source"] for _, _, info in loaded for sh in info["shards"]]
    jobs = [(c["run"], c["job"]) for _, _, info in loaded for sh in info["shards"] for c in sh["constituents"]]
    if len(set(shard_src)) != len(shard_src) or len(set(jobs)) != len(jobs):
        raise SystemExit("the same ranking or the same fold job appears more than once in the draws — a draw would be averaged with itself")
    draws = [(P, h) for P, h, _ in loaded]
    P, cnt = mean_of_draws(draws, n, len(seen_cls))
    if (cnt == 0).any():
        raise SystemExit(f"{int((cnt == 0).sum())} training rows without any draw — every row needs at least one (no neutral remainder)")
    return P, cnt, draws, [info for _, _, info in loaded]


def rank_record(specs: list[str], cnt: np.ndarray, infos: list[dict], rows_file: str, rows_all: np.ndarray) -> dict:
    """Field rank of the meta file (notation as in a8 --rank, extended by the draws)."""
    return {"source": ",".join(specs), "rows": int((cnt > 0).sum()), "rows_named_only": 0, "partial": False, "draws": infos,
            "rows_by_draws": {str(k): int((cnt == k).sum()) for k in sorted(set(cnt.tolist()))},
            "draws_per_row": [int(c) for c in cnt],  # number of draws per training row (table position)
            "rows_all_draws": int(len(rows_all)), "rows_all_draws_file": rows_file, "rows_all_draws_sha256": rows_sha256(rows_all),
            "definition": DEFINITION, "function": "eda/a9_m3_draws_oof.py: build() / mean_of_draws()",
            "note": "fit the blend weights only on the rows with ALL draws (eda/c99_m3_deploy.py --rows <rows_all_draws_file>)"}


def context() -> dict:
    from vlm import common as C

    root = C.find_data_root()
    train = C.load_table(root, "train")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    return {"root": root, "folds": C.load_folds(len(train)), "classes": classes, "y": labels.encode(train["root_code"], classes),
            "seen": labels.seen_mask(train["root_code"], classes), "n": len(train)}


def check_ref(ref: str, P: np.ndarray, cnt: np.ndarray, draws: list[tuple[np.ndarray, np.ndarray]], specs: list[str], art: Path) -> dict:
    """--ref: the draw with the same source as meta_<ref>.json is bit-identical to oof_<ref>.npy; rows with one draw bit-identical."""
    meta = read_json(art / f"meta_{ref}.json")
    if meta is None or not (meta.get("rank") or {}).get("source"):
        raise SystemExit(f"--ref {ref}: meta_{ref}.json missing or names no ranking")
    R = np.load(art / f"oof_{ref}.npy")
    if R.shape != P.shape:
        raise SystemExit(f"--ref {ref}: shape {R.shape} instead of {P.shape}")
    src = str(meta["rank"]["source"])
    j = specs.index(src) if src in specs else None
    out = {"ref": ref, "ref_source": src, "draw": None if j is None else j + 1}
    if j is not None:
        Pj, hj = draws[j]
        out["draw_rows"] = int(hj.sum())
        out["draw_bit_identical"] = bool(np.array_equal(Pj[hj], R[hj]))
    single = cnt == 1
    out["rows_single"] = int(single.sum())
    out["single_bit_identical"] = bool(np.array_equal(P[single], R[single]))
    return out


def check_c98d(P: np.ndarray, cnt: np.ndarray, y: np.ndarray, seen: np.ndarray) -> dict:
    """--check-c98d: recompute the mean on the rows of the two-draw evaluation with its computation (development script,
    not included; bit-identical) and report MAP@3 of the mean and of the pilot draw."""
    B = load("a9_c98b", COMP / "eda" / "c98b_pool_eval.py")
    new, pil = B.load_rank(*C98D_SOURCES[0]), B.load_rank(*C98D_SOURCES[1])
    rows = np.array(sorted(set(new) & set(pil)), dtype=np.int64)
    absent = ~seen
    pn = B.probs_from_logp(np.stack([new[int(i)] for i in rows]), absent=absent)
    pp = B.probs_from_logp(np.stack([pil[int(i)] for i in rows]), absent=absent)
    pm = 0.5 * (pn + pp)
    out = {"rows": int(len(rows)), "all_have_two_draws": bool((cnt[rows] == 2).all()), "bit_identical": bool(np.array_equal(P[rows], pm)),
           "max_abs": float(np.abs(P[rows] - pm).max()), "map3_mean": float(metric.map3_rows_from_proba(y[rows], P[rows]).mean()),
           "map3_pilot": float(metric.map3_rows_from_proba(y[rows], pp).mean())}
    return out


def verify(name: str) -> int:
    """--verify: recompute the artifact from the sources in its meta file (bit-identical) and check the row list; writes nothing."""
    ctx = context()
    art = COMP / "artifacts"
    meta = read_json(art / f"meta_{name}.json")
    rk = (meta or {}).get("rank") or {}
    if not rk.get("draws"):
        print(f"--verify {name}: meta_{name}.json missing or not produced by eda/a9_m3_draws_oof.py (rank.draws)")
        return 4
    specs = str(rk["source"]).split(",")
    P, cnt, _, _ = build(specs, ctx["n"], ctx["seen"])
    stored = np.load(art / f"oof_{name}.npy")
    same = bool(stored.shape == P.shape and np.array_equal(stored, P))
    rows_all = np.flatnonzero(cnt == len(specs)).astype(np.int64)
    f = ctx["root"] / rk["rows_all_draws_file"]
    rows_ok = f.is_file() and rows_sha256(np.load(f)) == rk["rows_all_draws_sha256"] == rows_sha256(rows_all)
    sha = hashlib.sha256((art / f"oof_{name}.npy").read_bytes()).hexdigest()
    print(f"--verify {name}: {len(specs)} draws recomputed from the meta file -> oof_{name}.npy {'BIT-IDENTICAL' if same else 'DEVIATION'} "
          f"(SHA-256 {sha[:16]}…); row list {rk['rows_all_draws_file']} ({len(rows_all)} rows with all draws) "
          f"{'matches' if rows_ok else 'DOES NOT MATCH (missing, different checksum or different rows)'}")
    return 0 if same and rows_ok else 4


def gen_status(spec: str, table: str, min_complete: float, max_failed: float) -> int:
    """--gen-status: state of the generation behind a ranking (aggregates only). 0 = complete, 3 = not."""
    from vlm import common as C

    run, job, tag = parse_source(spec)
    meta = read_json(run_dir(run) / job / f"{tag}_meta.json")
    if meta is None:
        print(f"{spec}: MISSING ({tag}_meta.json)")
        return 3
    if meta.get("table") != table:
        print(f"{spec}: table {meta.get('table')!r} instead of {table!r}")
        return 3
    src = str((meta.get("sft") or {}).get("sft_from") or "")
    if "/" not in src:
        print(f"{spec}: no generation given (sft.sft_from missing)")
        return 3
    gjob, gtag = src.split("/", 1)
    g = read_json(run_dir(run) / gjob / f"{gtag}_meta.json")
    if g is None:
        print(f"{spec}: generation {run}/{gjob}: {gtag}_meta.json MISSING")
        return 3
    n = len(C.load_table(C.find_data_root(), table))
    rows, answered, failed = int(g.get("rows", -1)), int(g.get("answered", -1)), int(g.get("failed", 0))
    states = g.get("states") or {}
    complete = int(states.get("complete", 0))
    share = complete / answered if answered > 0 else 0.0
    ok = (g.get("table") == table and rows == n and answered >= (1.0 - max_failed) * n and failed <= max_failed * n and share >= min_complete)
    print(f"{spec}: generation {run}/{gjob} (table {table}, {n} rows): {rows} rows, answered {answered}, failures {failed} "
          f"(at most {max_failed:.0%}); complete answers {complete} ({share:.3f}; required ≥ {min_complete:g}), open {states.get('open')}, "
          f"empty {states.get('empty')}; max_tokens {g.get('max_tokens')} -> {'COMPLETE' if ok else 'INCOMPLETE'}")
    return 0 if ok else 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", nargs="?", help="name of the artifact (artifacts/oof_<name>.npy, meta_<name>.json)")
    ap.add_argument("--draw", action="append", default=[], help="one draw: runs/<run>/<job>/<tag>[+runs/… more shards]; at least twice")
    ap.add_argument("--rows-out", help="row list of the rows with all draws (relative to the data folder; default derived/<name>_rows.npy)")
    ap.add_argument("--ref", help="channel whose artifact must carry the draw with its source bit-identically (e.g. m3p5)")
    ap.add_argument("--check-c98d", action="store_true", help="check the mean on the rows of the two-draw evaluation against its computation (development script, not included)")
    ap.add_argument("--verify", metavar="NAME", help="only recompute: artifact from the sources in its meta file (writes nothing)")
    ap.add_argument("--gen-status", metavar="RANK", help="only report the state of the generation behind a ranking (aggregates)")
    ap.add_argument("--table", default="test", choices=["train", "test"], help="table for --gen-status")
    ap.add_argument("--min-complete", type=float, default=0.0, help="--gen-status: smallest share of complete answers")
    ap.add_argument("--max-failed", type=float, default=0.05, help="--gen-status: largest share of failed requests")
    a = ap.parse_args()
    if a.gen_status:
        return gen_status(a.gen_status, a.table, a.min_complete, a.max_failed)
    if a.verify:
        return verify(a.verify)
    if not a.name or not a.name.replace("_", "").isalnum():
        raise SystemExit("artifact name missing or containing characters other than letters, digits and _")
    art = COMP / "artifacts"
    old = read_json(art / f"meta_{a.name}.json")
    if old is not None and not (old.get("rank") or {}).get("draws"):
        raise SystemExit(f"artifacts/meta_{a.name}.json belongs to a different artifact (not from this script) — choose another name")
    ctx = context()
    y, seen, n, folds = ctx["y"], ctx["seen"], ctx["n"], ctx["folds"]
    specs = [str(s) for s in a.draw]
    P, cnt, draws, infos = build(specs, n, seen)
    rows_all = np.flatnonzero(cnt == len(specs)).astype(np.int64)
    if len(rows_all) == 0:
        raise SystemExit("no row carries all draws — no fit rows")
    rows_file = a.rows_out or f"derived/{a.name}_rows.npy"
    for j, (Pj, hj) in enumerate(draws):
        infos[j]["cv_alone_own_rows"] = float(metric.map3_rows_from_proba(y[hj], Pj[hj]).mean())
        infos[j]["none_top_own_rows"] = float(np.mean(Pj[hj].argmax(1) == 0))
    r_all = metric.map3_rows_from_proba(y, P)
    on_all = {"rows": int(len(rows_all)), "mean": float(r_all[rows_all].mean()),
              **{f"draw{j + 1}": float(metric.map3_rows_from_proba(y[rows_all], Pj[rows_all]).mean()) for j, (Pj, _) in enumerate(draws)}}
    print(f"{a.name}: mean of {len(specs)} draws; rows by number of draws {dict(sorted((int(k), int((cnt == k).sum())) for k in set(cnt.tolist())))}")
    for j, info in enumerate(infos):
        print(f"  draw {j + 1}: {info['source']} — {info['rows']} rows, fold jobs "
              + ", ".join(f"{c['run']}/{c['job']}" for sh in info["shards"] for c in sh["constituents"])
              + f"; max_tokens {info['max_tokens']}; alone on its rows MAP@3 {info['cv_alone_own_rows']:.4f}, NONE at the top {info['none_top_own_rows']:.3f}")
    print(f"  on the {len(rows_all)} rows with all draws: MEAN {on_all['mean']:.4f}; "
          + ", ".join(f"draw {j + 1} alone {on_all[f'draw{j + 1}']:.4f}" for j in range(len(draws))))
    checks: dict = {}
    if a.ref:
        c = check_ref(a.ref, P, cnt, draws, specs, art)
        checks["ref"] = c
        print(f"  check --ref {a.ref}: draw {c['draw'] or '–'} (source as in meta_{a.ref}.json: {c['ref_source']}) "
              + (f"on its {c['draw_rows']} rows {'BIT-IDENTICAL' if c['draw_bit_identical'] else 'DEVIATION'} to oof_{a.ref}.npy; " if c["draw"] else "missing among the draws; ")
              + f"artifact on the {c['rows_single']} rows with one draw {'BIT-IDENTICAL' if c['single_bit_identical'] else 'DEVIATION'}")
        if not c["single_bit_identical"] or not c.get("draw_bit_identical", False):
            raise SystemExit(f"--ref {a.ref}: not bit-identical — nothing written")
    if a.check_c98d:
        c = check_c98d(P, cnt, y, seen)
        checks["c98d"] = c
        print(f"  check --check-c98d: {c['rows']} rows of the two-draw evaluation, all with two draws: {c['all_have_two_draws']}; mean "
              f"{'BIT-IDENTICAL' if c['bit_identical'] else 'DEVIATION'} to the c98d computation (largest deviation {c['max_abs']:.1e}); MAP@3 mean "
              f"{c['map3_mean']:.6f} / pilot {c['map3_pilot']:.6f}")
        if not (c["all_have_two_draws"] and c["bit_identical"]):
            raise SystemExit("--check-c98d: mean of the two-draw evaluation not reproduced — nothing written")
    from kaggle_core import artifacts as KA
    from kaggle_core.folds import folds_hash

    sigma = float(metric.bootstrap_sigma(r_all))
    meta = {"folds_hash": folds_hash(folds), "model_type": "vlm-sft-rank", "sources": specs, "cv_score": float(r_all.mean()), "cv_sigma": sigma,
            "feature_set": "image+task, without task knowledge (trained model, ranking behind its own text)",
            "rows_without_answer": 0, "neutral_rows": None, "rows_neutral": 0,
            "rank": rank_record(specs, cnt, infos, rows_file, rows_all), "cv_rows_all_draws": on_all, "checks": checks,
            "cv_note": "cv_score over all rows mixes rows with one and with all draws; comparison on the same rows: cv_rows_all_draws"}
    out_rows = ctx["root"] / rows_file
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    before = rows_sha256(np.load(out_rows)) if out_rows.is_file() else None
    np.save(out_rows, rows_all)
    KA.save_oof(a.name, P, np.zeros((0, len(ctx["classes"]))), meta, art)
    sha = hashlib.sha256((art / f"oof_{a.name}.npy").read_bytes()).hexdigest()
    print(f"  MAP@3 over all {n} rows {r_all.mean():.4f} ± {sigma:.4f} (mixed: rows with one and with all draws); "
          f"per fold {[round(float(r_all[folds == k].mean()), 4) for k in range(5)]}; NONE at the top {np.mean(P.argmax(1) == 0):.3f} (true {np.mean(y == 0):.3f})")
    print(f"written: artifacts/oof_{a.name}.npy (SHA-256 {sha}), artifacts/meta_{a.name}.json; row list data/{rows_file} "
          f"({len(rows_all)} rows, SHA-256 {rows_sha256(rows_all)[:16]}…"
          + ("" if before is None else ("; unchanged" if before == rows_sha256(rows_all) else f"; CHANGED, previously {before[:16]}…")) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
