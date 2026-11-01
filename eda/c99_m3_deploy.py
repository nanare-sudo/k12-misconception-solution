"""M3 channel in deployment: blend weights WITH the channel for both lines, blend files for tools/make_submission.py
--rank-only and a log with in-sample and cross-fitted values. No deployment decision is made here — only tooling and numbers.

    python -I eda/c99_m3_deploy.py [--channel m3p] [--rows <rows.npy>] [--lines best safe] [--floor-channel 0.05] [--save]
    python -I eda/c99_m3_deploy.py --variant runs/<run>/<job>/<tag> [--channel m3p] [--require-gold-only]
                                   [--allow-mismatch] [--allow-unknown]

CHANNEL: artifacts/oof_<channel>.npy (eda/a8_proto_oof.py <channel> … --rank <ranking> [--rank-partial --neutral …]): on the
rows with a ranking the exact distribution over all codes, on the other (neutral) rows the uniform distribution over the 74
possible classes. CHANNEL ROWS = rows on which the channel is NOT uniform (600 = folds 1+2 for the pilot, all 1,500 after
folds 0/3/4) — never hard-wired, checked against the meta file (rank.rows, rows_neutral); `--rows` restricts to a list
(table positions, .npy relative to the data folder; every row must carry the channel).

LINES (recipes of the earlier build scripts; channels and floors checked against their blend files; the channel is
appended last, floor --floor-channel = 0.05):
  best  primary line + channel: v1r, zs27q, think18nb (floor 0.01; weight × 1.4), pr27xm (0.05), <channel>.
        Weights as in the earlier blend-saving step: a2.fit_mix (log loss), then think18nb × 1.4. Last step in
        make_submission: margin alignment 0.75 for classes with at least three training rows (transductive, primary
        line only).
  safe  inductive line + channel: v1r, zs27t, think18sb (0.01), pr27xm (0.05), <channel>. Weights fitted in one stage
        (a2.fit_mix). Last step: inductive alignment 0.75 for classes with at least three rows — class weights from the
        blend's predictions (make_submission.oof_mixture) for THE SAME channel rows, target = the class frequencies of all
        training rows (as in make_submission).
  The weights of ALL channels are fitted ONLY on the channel rows; rows without the channel are not used. Arithmetic as in
  the build and test paths: floor without prior masking, masking after blending (make_submission). Note: a2.fit_mix is
  L-BFGS-B with bounds 0…4 (not SLSQP) — kept so that the weights match the build path.

CHECKS before anything else (abort on deviation): (1) checksums of the base OOFs (BASE_OOF_SHA256, pr27xm as deployed);
(2) the same fit WITHOUT the channel on all 1,500 rows reproduces the weights of the earlier blend files bit-identically;
(3) channel: shape, normalisation, channel rows against the meta file.

LOG (artifacts/eda/private/m3_deploy/c99_m3_deploy_<channel>[_<line>].{txt,json}; training rows only, aggregates only):
  * weights per line: with and without the channel (both on the channel rows) and the deployed ones (1,500 rows);
  * IN-SAMPLE MAP@3 on the channel rows with the last step as deployed — with channel, without channel (fitted the same
    way), deployed blend without channel. OPTIMISTIC: weights (and inductive class weights) from the same rows;
  * CROSS on the channel rows (the honest measure): per channel fold, weights only from the channel rows of the other
    channel folds, last step as deployed with the target from the rows outside the fold.
--save writes artifacts/stack_<name>.json in the format of tools/make_submission.py (channels, weights, floors; plus
fit_rows, rank_only and — inductive line — inductive_check, which make_submission recomputes). No oof_<name>.npy: the blend
exists only on the channel rows.

--variant: training regime of the OOF channel (training targets of the fold models behind its ranking) against that of the
test side (training targets of the model that computed the test ranking): via run configuration (models -> merged), merge
job (adapter_from) and sft_recipe.json of the training job (fetched; otherwise the configuration of the training job).
"teacher" = pool rows with teacher targets (sft_pool / teacher_derived), otherwise "gold-only". Exit 0 = same regime (with
--require-gold-only: both gold-only), 5 = different regime or a different system text, 6 = cannot be determined or (with
--require-gold-only) teacher targets. The inductive line requires gold-only; --allow-* do not apply there.

CHANNEL FROM SEVERAL DRAWS (eda/a9_m3_draws_oof.py, e.g. m3x2 = mean of two M3 draws): meta_<channel>.json carries
rank.source = sources of the draws joined with commas and rank.draws. The blend weights may then ONLY be fitted on rows with
ALL draws (all draws exist on the test rows) — `--rows derived/<channel>_rows.npy` (rank.rows_all_draws_file, checksum
checked); otherwise abort. --save records the number of draws and the checksum of the row list in the rank_only entry of the
blend file. --variant resolves all draws and shards of the OOF channel (ONE test draw per call; the build scripts call it
once per test draw).
--versus <channel2>: the same line with <channel2> instead of <channel> (e.g. m3p5 = one draw), fitted on the same rows and
with the same procedure — weights, in-sample and cross, paired difference <channel> − <channel2>. Report only; the blend file
(--save) remains that of the channel. Log then c99_m3_deploy_<channel>[_<line>]_vs_<channel2>.{txt,json}.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import labels  # noqa: E402
import metric  # noqa: E402

EPS = 1e-9
OUT = COMP / "artifacts" / "eda" / "private" / "m3_deploy"
ALIGN_STRENGTH, ALIGN_MIN = 0.75, 3
FLOOR_CHANNEL = 0.05
CLUSTER_DRAWS, CLUSTER_SEED = 4000, 99_001
PR27XM_SHA = "bf3d1b58abfc57eb64139418ed7646f45ba9f879791661658035171b9a2f0d03"  # artifacts/oof_pr27xm.npy (deployed sources)
BASE_OOF_SHA256 = {  # artifacts/oof_<channel>.npy of the base channels as deployed
    "v1r": "e1729c1fddad7ebb84a927902404c0be5a3a5a3bc55d57033960effc81192a55",
    "zs27q": "7a11b94751f6d458c6539724e547f28d24fdc2439bc3241831c6844373c3dd65",
    "zs27t": "54e045dfe32c7be102dff36691edc5e54ef653b67d75fdfde4702434ce709ee1",
    "think18nb": "9c3403b57d07a956d2a2932f3fd865ea560b587572f58ba26061f5fba352c79a",
    "think18sb": "275cace8a36ca7574ef7c4b93b173602af9f9d2b346c85f491976410d0cdcf11",
}
LINES = {
    # "label" and "last_label" are recorded in the blend files (note, last_step) and reports
    "best": {"label": "primary line (earlier recipe + channel)", "base": "v1r_zs27q_think18nb_pr27xm_w14",
             "channels": ["v1r", "zs27q", "think18nb", "pr27xm"], "floor": {"think18nb": 0.01, "pr27xm": 0.05},
             "think": "think18nb", "factor": 1.4, "last": "trans", "name": "v1r_zs27q_think18nb_pr27xm_{ch}_w14",
             "last_label": "margin alignment 0.75 for classes with at least 3 training rows (transductive; make_submission --align 0.75 --align-min 3)"},
    "safe": {"label": "inductive line (earlier recipe + channel)", "base": "v1r_zs27t_think18sb_pr27xm",
             "channels": ["v1r", "zs27t", "think18sb", "pr27xm"], "floor": {"think18sb": 0.01, "pr27xm": 0.05},
             "think": None, "factor": 1.0, "last": "ind", "name": "v1r_zs27t_think18sb_pr27xm_{ch}",
             "last_label": "inductive alignment 0.75 for classes with at least 3 rows (make_submission --align-inductive 0.75 --align-min 3)"},
}
GROUPS = ("NONE", "seen", "never seen", "new question")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A2 = load("c99_a2", COMP / "eda" / "a2_probe_stack.py")
C21 = load("c99_c21", COMP / "eda" / "c21_margin_align.py")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def weights_sha16(w: np.ndarray) -> str:
    """Checksum of the inductive class weights as in tools/make_submission.py (weights_sha256)."""
    return hashlib.sha256(np.ascontiguousarray(w, dtype=np.float64).tobytes()).hexdigest()[:16]


def rows_sha256(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).astype("<i8").tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ arithmetic

def logs_of(mats: list[np.ndarray], floors: list[float], seen: np.ndarray, mask_first: bool = False) -> list[np.ndarray]:
    """Channels as log probabilities. mask_first=False: as in the build path and the test path (make_submission) — floor
    without prior masking; True: as in make_submission.oof_mixture and the recipe arithmetic — mask first, then floor."""
    out = []
    for m, fl in zip(mats, floors):
        p = np.where(seen[None, :], m, 0.0) if mask_first else m
        out.append(np.log(np.clip(A2.floor_probs(p, fl, seen), EPS, 1)))
    return out


def fit_theta(logs: list[np.ndarray], y: np.ndarray, rows: np.ndarray, think_idx: int | None, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Weights on log loss ONLY from `rows` (a2.fit_mix), then the factor on the reasoning channel.
    -> (deployed weights, weights before the factor)."""
    base = A2.fit_mix([c[rows] for c in logs], None, y[rows])
    th = base.copy()
    if think_idx is not None:
        th[think_idx] *= factor
    return th, base


def mix(theta, logs: list[np.ndarray], rows: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Blend on `rows` as in the test path of make_submission: log-linear, classes without a training example zero, normalised."""
    P = A2.apply_mix(np.asarray(theta, dtype=float), [c[rows] for c in logs], None)
    P = np.where(seen[None, :], P, 0.0)
    return P / P.sum(1, keepdims=True)


def target_of(y: np.ndarray, rows: np.ndarray, K: int, min_rows: int = ALIGN_MIN) -> np.ndarray:
    """Class frequencies of the rows `rows`, classes below min_rows excluded (as in make_submission)."""
    c = np.bincount(y[rows], minlength=K)
    return np.where(c >= min_rows, c, 0) / len(rows)


def apply_last(P: np.ndarray, w: np.ndarray, seen: np.ndarray, strength: float = ALIGN_STRENGTH) -> np.ndarray:
    """Apply class weights with a strength, mask, normalise — as make_submission does after --align/--align-inductive."""
    Q = C21.apply_weights(P, w, strength)
    Q = np.where(seen[None, :], Q, 0.0)
    return Q / Q.sum(1, keepdims=True)


def last_step(kind: str, P_eval: np.ndarray, target: np.ndarray, seen: np.ndarray, P_fit: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """trans: weights on the evaluated set itself (margin alignment); ind: weights from P_fit (the blend's predictions for its
    fit rows), only applied to P_eval. -> (P after the step, class weights)."""
    if kind == "trans":
        w = C21.fit_weights(P_eval, target)
    elif kind == "ind":
        w = C21.fit_weights(P_fit, target)
    else:
        raise ValueError(f"unknown last step {kind}")
    return apply_last(P_eval, w, seen), w


# ------------------------------------------------------------------------------------------------ channel and context

def context() -> dict:
    """Training table, labels, folds and label-defined groups (without the pool)."""
    from vlm import common as C

    root = C.find_data_root()
    train = C.load_table(root, "train")
    folds = C.load_folds(len(train))
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes)
    seen = labels.seen_mask(train["root_code"], classes)
    q = train["question_id"].to_numpy()
    q_code = np.unique(q, return_inverse=True)[1]
    same_q, other = q[:, None] == q[None, :], folds[:, None] != folds[None, :]
    q_seen = (same_q & other).any(1)
    cs = (same_q & other & (y[:, None] == y[None, :])).any(1)
    group = np.where(y == 0, "NONE", np.where(cs, "seen", np.where(q_seen, "never seen", "new question")))
    return {"root": root, "train": train, "folds": folds, "classes": classes, "y": y, "seen": seen, "n": len(y), "K": len(classes),
            "q_code": q_code, "group": group}


def channel_rows(P: np.ndarray, seen: np.ndarray, meta: dict | None = None, rows: np.ndarray | None = None,
                 min_rows: int = 50) -> tuple[np.ndarray, dict]:
    """Channel rows = rows on which the channel is not uniform (checked against the meta file); with `rows` only these (every
    one must carry the channel). -> (sorted positions, statistics)."""
    n = len(P)
    if P.ndim != 2 or P.shape[1] != len(seen) or not np.isfinite(P).all():
        raise SystemExit(f"channel: shape {P.shape} or non-finite values")
    if not np.allclose(P.sum(1), 1.0, atol=1e-6):
        raise SystemExit("channel: rows do not sum to 1")
    u = seen / seen.sum()
    uniform = np.all(np.abs(P - u[None, :]) <= 1e-12, axis=1)
    have = ~uniform
    info = {"rows_table": int(n), "rows_channel": int(have.sum()), "rows_uniform": int(uniform.sum()),
            "unseen_mass_max": float(P[:, ~seen].sum(1).max()) if (~seen).any() else 0.0}
    if meta is not None:
        rk = meta.get("rank") or {}
        if rk.get("rows") is not None and int(rk["rows"]) != int(have.sum()):
            raise SystemExit(f"channel: {int(have.sum())} rows not uniform, the meta file says rank.rows = {rk['rows']}")
        if meta.get("rows_neutral") is not None and int(meta["rows_neutral"]) != int(uniform.sum()):
            raise SystemExit(f"channel: {int(uniform.sum())} uniform rows, the meta file says rows_neutral = {meta['rows_neutral']}")
        info["meta_checked"] = True
    if rows is None:
        out = np.flatnonzero(have)
    else:
        out = np.unique(np.asarray(rows, dtype=np.int64))
        if len(out) != len(np.asarray(rows)) or out.min() < 0 or out.max() >= n:
            raise SystemExit("--rows: duplicate positions or positions outside the training table")
        if not have[out].all():
            raise SystemExit(f"--rows: {int((~have[out]).sum())} rows without the channel (uniform) — only channel rows are allowed")
        info["rows_restricted"] = int(len(out))
    if len(out) < min_rows:
        raise SystemExit(f"only {len(out)} channel rows — too few for a blend (required {min_rows})")
    return out, info


def draws_check(ch_meta: dict, rows: np.ndarray, root: Path) -> dict | None:
    """Channel from several draws (eda/a9_m3_draws_oof.py; rank.source with commas, rank.draws): the blend weights may only be
    fitted on rows with ALL draws — all draws exist on the test rows, rows with fewer draws would carry a different (sharper)
    channel. Row list rank.rows_all_draws_file (relative to the data folder) with checksum.
    -> statistics, None for a channel from one draw (then unchanged as before)."""
    rk = ch_meta.get("rank") or {}
    draws = rk.get("draws") or []
    srcs = str(rk.get("source") or "").split(",")
    if not draws and len(srcs) < 2:
        return None
    if len(srcs) < 2 or len(draws) != len(srcs) or [str(d.get("source")) for d in draws] != srcs:
        raise SystemExit("channel from several draws: rank.source (comma list) and rank.draws do not match (eda/a9_m3_draws_oof.py)")
    f = rk.get("rows_all_draws_file")
    if not f:
        raise SystemExit("channel from several draws: rank.rows_all_draws_file missing in the meta file (eda/a9_m3_draws_oof.py)")
    p = Path(f) if Path(f).is_absolute() else Path(root) / f
    if not p.is_file():
        raise SystemExit(f"row list {f} (rows with all draws) missing — rerun eda/a9_m3_draws_oof.py")
    rows_all = np.load(p)
    if rows_sha256(rows_all) != rk.get("rows_all_draws_sha256"):
        raise SystemExit(f"row list {f} does not match the meta file (checksum) — artifact and row list come from different runs of a9")
    bad = ~np.isin(np.asarray(rows), rows_all)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} of {len(rows)} fit rows do not carry all {len(srcs)} draws — fit the weights only on the rows with all "
                         f"draws: --rows {f}")
    return {"draws": len(srcs), "sources": srcs, "rows_by_draws": rk.get("rows_by_draws"), "rows_all_draws": int(len(rows_all)),
            "rows_all_draws_file": str(f), "rows_all_draws_sha256": rk.get("rows_all_draws_sha256"), "fit_rows": int(len(rows))}


def cluster_se(v: np.ndarray, q_code: np.ndarray, rows: np.ndarray, draws: int = CLUSTER_DRAWS, seed: int = CLUSTER_SEED) -> dict:
    """Mean of v over rows; σ and 95 % interval bootstrapped over tasks."""
    s = np.bincount(q_code[rows], weights=v[rows], minlength=q_code.max() + 1)
    c = np.bincount(q_code[rows], minlength=q_code.max() + 1).astype(float)
    keep = c > 0
    s, c = s[keep], c[keep]
    idx = np.random.default_rng(seed).integers(0, len(s), size=(draws, len(s)))
    vals = s[idx].sum(1) / c[idx].sum(1)
    return {"mean": float(v[rows].mean()), "se": float(vals.std(ddof=1)), "ci95": [float(x) for x in np.quantile(vals, [0.025, 0.975])]}


def paired(ctx: dict, r1: np.ndarray, r0: np.ndarray, rows: np.ndarray) -> dict:
    """Paired difference on `rows`: Δ, σ over rows (metric.paired_bootstrap) and over tasks, z with the larger one, per fold
    and group."""
    d = np.zeros(ctx["n"])
    d[rows] = r1[rows] - r0[rows]
    pb = metric.paired_bootstrap(r1[rows], r0[rows])
    cq = cluster_se(d, ctx["q_code"], rows)
    se = max(pb["sigma"], cq["se"])
    folds, group = ctx["folds"], ctx["group"]
    return {"map3_new": float(r1[rows].mean()), "map3_old": float(r0[rows].mean()), "delta": float(d[rows].mean()), "se_rows": pb["sigma"],
            "se_q": cq["se"], "ci95_q": cq["ci95"], "z": float(d[rows].mean() / se) if se > 0 else 0.0,
            "per_fold": {str(k): float(d[rows][folds[rows] == k].mean()) for k in sorted(set(folds[rows].tolist()))},
            "groups": {g: {"rows": int((group[rows] == g).sum()), "delta": float(d[rows][group[rows] == g].mean())}
                       for g in GROUPS if (group[rows] == g).any()},
            "rows_better": int((d[rows] > 0).sum()), "rows_worse": int((d[rows] < 0).sum())}


def fmt_pair(r: dict) -> str:
    pf = " ".join(f"F{k} {v:+.4f}" for k, v in r["per_fold"].items())
    gr = " | ".join(f"{g} ({v['rows']}) {v['delta']:+.4f}" for g, v in r["groups"].items())
    return (f"{r['map3_old']:.4f} -> {r['map3_new']:.4f}  Δ {r['delta']:+.4f} ± {r['se_rows']:.4f} rows, ± {r['se_q']:.4f} tasks (z {r['z']:+.2f}; "
            f"95 % over tasks {r['ci95_q'][0]:+.4f} … {r['ci95_q'][1]:+.4f}); {pf}; better {r['rows_better']}, worse {r['rows_worse']}\n"
            f"        groups: {gr}")


# ------------------------------------------------------------------------------------------------ lines

def line_setup(key: str, ch: str, floor_ch: float, art: Path) -> dict:
    """Recipe of a line from the submitted blend file (channels, floors, factor checked) plus the channel."""
    L = LINES[key]
    base_file = art / f"stack_{L['base']}.json"
    base = json.loads(base_file.read_text())
    floors_base = {k: float(v) for k, v in (base.get("floor") or {}).items()}
    want_scaled = {L["think"]: L["factor"]} if L["think"] else {}
    if base["channels"] != L["channels"] or floors_base != L["floor"] or (base.get("scaled") or {}) != want_scaled:
        raise SystemExit(f"{base_file.name}: channels, floors or factor differ from the recipe — build path and c99 do not match")
    names = L["channels"] + [ch]
    floors = {**L["floor"], ch: float(floor_ch)}
    return {**L, "key": key, "names": names, "names_base": list(L["channels"]), "floors": floors, "floors_base": dict(L["floor"]),
            "theta_prod": np.asarray(base["theta"], dtype=float), "theta_prod_base": np.asarray((base.get("base") or {}).get("theta", base["theta"]), dtype=float),
            "stack_name": L["name"].format(ch=ch)}


def variant_logs(spec: dict, mats: dict[str, np.ndarray], seen: np.ndarray, with_channel: bool, mask_first: bool = False) -> tuple[list[str], list[np.ndarray], list[float], int | None]:
    names = spec["names"] if with_channel else spec["names_base"]
    fl = [spec["floors"].get(n, 0.0) for n in names]
    logs = logs_of([mats[n] for n in names], fl, seen, mask_first=mask_first)
    ti = names.index(spec["think"]) if spec["think"] else None
    return names, logs, fl, ti


def insample(ctx: dict, spec: dict, mats: dict[str, np.ndarray], rows: np.ndarray, ind_mixture=None) -> dict:
    """In-sample on the channel rows: weights with / without the channel from these rows, last step as deployed (target from
    all training rows; inductive: class weights from the blend on the same rows), plus the deployed blend without the channel
    (weights and inductive class weights from all 1,500 rows). ind_mixture(stack) -> the blend's predictions for all training
    rows (make_submission.oof_mixture); None = local computation with the same arithmetic."""
    y, seen, n, K = ctx["y"], ctx["seen"], ctx["n"], ctx["K"]
    allr = np.arange(n)
    target = target_of(y, allr, K)
    out: dict = {"rows": int(len(rows)), "theta": {}, "map3": {}, "rows_scores": {}, "weights": {}}
    for tag, with_ch in (("with", True), ("without", False)):
        names, logs, fl, ti = variant_logs(spec, mats, seen, with_ch)
        th, th_base = fit_theta(logs, y, rows, ti, spec["factor"])
        P = mix(th, logs, rows, seen)
        P_fit = None
        if spec["last"] == "ind":
            stack = {"channels": names, "theta": [float(v) for v in th], "floor": {k: v for k, v in zip(names, fl) if v}, "none": [], "probes": {}}
            P_fit = (ind_mixture(stack) if ind_mixture else mix(th, logs_of([mats[k] for k in names], fl, seen, mask_first=True), allr, seen))[rows]
        Q, w = last_step(spec["last"], P, target, seen, P_fit)
        r = np.zeros(n)
        r[rows] = metric.map3_rows_from_proba(y[rows], Q, seen)
        r_pre = np.zeros(n)
        r_pre[rows] = metric.map3_rows_from_proba(y[rows], P, seen)
        out["theta"][tag] = {"names": names, "theta": [float(v) for v in th], "theta_before_factor": [float(v) for v in th_base]}
        out["map3"][tag] = float(r[rows].mean())
        out["map3"][f"{tag}_pre_last_step"] = float(r_pre[rows].mean())
        out["rows_scores"][tag] = r
        out["weights"][tag] = w
    # deployed blend without the channel (weights from all 1,500 rows), evaluated on the same rows
    names, logs, fl, ti = variant_logs(spec, mats, seen, False)
    P = mix(spec["theta_prod"], logs, rows, seen)
    P_fit = None
    if spec["last"] == "ind":
        stack = {"channels": names, "theta": [float(v) for v in spec["theta_prod"]], "floor": {k: v for k, v in zip(names, fl) if v}, "none": [], "probes": {}}
        P_fit = ind_mixture(stack) if ind_mixture else mix(spec["theta_prod"], logs_of([mats[k] for k in names], fl, seen, mask_first=True), allr, seen)
    Q, _ = last_step(spec["last"], P, target, seen, P_fit)
    r = np.zeros(n)
    r[rows] = metric.map3_rows_from_proba(y[rows], Q, seen)
    out["map3"]["deployed_without_channel"] = float(r[rows].mean())
    out["rows_scores"]["deployed"] = r
    out["theta"]["deployed"] = {"names": names, "theta": [float(v) for v in spec["theta_prod"]]}
    out["delta_with_without"] = paired(ctx, out["rows_scores"]["with"], out["rows_scores"]["without"], rows)
    out["delta_with_deployed"] = paired(ctx, out["rows_scores"]["with"], out["rows_scores"]["deployed"], rows)
    return out


def cross(ctx: dict, spec: dict, mats: dict[str, np.ndarray], rows: np.ndarray, mask_first: bool = False) -> dict | None:
    """Cross-fit on the channel rows: per channel fold k, weights only from the channel rows of the other channel folds,
    evaluated on the channel rows of k; last step as deployed with the target from all rows outside k (margin alignment of the
    set of k, or class weights from the blend of the fit rows). Two channel folds = the inner-600 procedure.
    mask_first=True: recipe arithmetic (check), otherwise that of the build path. -> None with fewer than two folds."""
    y, seen, n, K, folds = ctx["y"], ctx["seen"], ctx["n"], ctx["K"], ctx["folds"]
    kf = sorted(set(folds[rows].tolist()))
    if len(kf) < 2:
        return None
    res: dict = {"folds": kf, "rows": int(len(rows)), "mask_first": mask_first}
    for tag, with_ch in (("with", True), ("without", False)):
        names, logs, fl, ti = variant_logs(spec, mats, seen, with_ch, mask_first=mask_first)
        logs_m = logs if mask_first else logs_of([mats[k] for k in names], fl, seen, mask_first=True)
        r, r_pre = np.zeros(n), np.zeros(n)
        for k in kf:
            F, E = rows[folds[rows] != k], rows[folds[rows] == k]
            th, _ = fit_theta(logs, y, F, ti, spec["factor"])
            P = mix(th, logs, E, seen)
            P_fit = mix(th, logs_m, F, seen) if spec["last"] == "ind" else None
            Q, _ = last_step(spec["last"], P, target_of(y, np.flatnonzero(folds != k), K), seen, P_fit)
            r[E] = metric.map3_rows_from_proba(y[E], Q, seen)
            r_pre[E] = metric.map3_rows_from_proba(y[E], P, seen)
        res[tag] = float(r[rows].mean())
        res[f"{tag}_pre_last_step"] = float(r_pre[rows].mean())
        res[f"_r_{tag}"] = r
    res["delta"] = paired(ctx, res["_r_with"], res["_r_without"], rows)
    return res


# ------------------------------------------------------------------------------------------------ training regime (--variant)

def run_dir(run: str) -> Path:
    return COMP / "artifacts" / "runs" / run / run


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def rank_constituents(spec: str) -> list[tuple[str, str, dict]]:
    """(run, job, meta) of the ranking(s) behind runs/<run>/<job>/<tag> — combined_from resolved. A channel from several draws
    (eda/a9_m3_draws_oof.py: draws separated by commas, shards by +) is resolved part by part."""
    if "," in spec or "+" in spec:
        return [c for part in spec.replace(",", "+").split("+") for c in rank_constituents(part)]
    bits = spec.split("/")
    if len(bits) != 4 or bits[0] != "runs":
        raise SystemExit(f"{spec}: expected runs/<run>/<job>/<tag>")
    _, run, job, tag = bits
    meta = read_json(run_dir(run) / job / f"{tag}_meta.json")
    if meta is None:
        raise SystemExit(f"{spec}: {tag}_meta.json missing (run not finished or not fetched)")
    if not meta.get("combined_from"):
        return [(run, job, meta)]
    out = []
    for s in meta["combined_from"]:
        r, j = s.get("run", run), s["job"]
        m = read_json(run_dir(r) / j / f"{tag}_meta.json")
        if m is None:
            raise SystemExit(f"{spec}: meta file of fold job {r}/{j} missing")
        out.append((r, j, m))
    return out


def _config_jobs(run: str) -> tuple[dict, list[dict]] | None:
    f = COMP / "configs" / "vlm" / f"{run}.yaml"
    if not f.is_file():
        return None
    import yaml

    cfg = yaml.safe_load(f.read_text()) or {}
    return cfg.get("models") or {}, [j for w in cfg.get("workers") or [] for j in w.get("jobs") or []]


def model_training(run: str, model: str) -> dict:
    """Training targets of the model `model` (key under models of the run configuration configs/vlm/<run>.yaml): merged name ->
    merge job (out_model) -> adapter_from -> sft_recipe.json of the training job (fetched), otherwise its configuration."""
    out: dict = {"run": run, "model": model, "variant": "unknown", "evidence": None}
    cj = _config_jobs(run)
    if cj is None:
        out["why"] = f"configs/vlm/{run}.yaml missing"
        return out
    models, jobs = cj
    merged = (models.get(model) or {}).get("merged")
    mj = [j for j in jobs if j.get("kind") == "merge" and merged and j.get("out_model") == merged]
    if len(mj) != 1:
        out["why"] = f"no unique merge job for model {model!r} (merged {merged!r}) in configs/vlm/{run}.yaml"
        return out
    src = str(mj[0]["adapter_from"])
    r2, j2 = src[3:].split("/", 1) if src.startswith("../") else (run, src)
    out.update(merge_job=mj[0]["name"], train_run=r2, train_job=j2)
    rec = read_json(run_dir(r2) / j2 / "sft_recipe.json")
    if rec is not None:
        teacher = bool(rec.get("teacher_derived")) or bool(rec.get("pool"))
        out.update(variant="teacher" if teacher else "gold-only", evidence=f"artifacts/runs/{r2}/{r2}/{j2}/sft_recipe.json", fold=rec.get("fold"),
                   system_sha256=rec.get("system_sha256"), notes_max_tokens=rec.get("notes_max_tokens"), rows_used=rec.get("rows_used"))
    else:
        cj2 = _config_jobs(r2)
        tj = [j for j in (cj2[1] if cj2 else []) if j.get("name") == j2]
        if len(tj) == 1:
            out.update(variant="teacher" if tj[0].get("sft_pool") else "gold-only", fold=tj[0].get("fold"),
                       evidence=f"configs/vlm/{r2}.yaml (job {j2}; sft_recipe.json not fetched)")
        else:
            out["why"] = f"neither artifacts/runs/{r2}/{r2}/{j2}/sft_recipe.json nor job {j2} in configs/vlm/{r2}.yaml"
    summ = read_json(run_dir(run) / mj[0]["name"] / "summary.json")
    ad = run_dir(r2) / j2 / "adapter" / "adapter_model.safetensors"
    want = ((summ or {}).get("merge") or {}).get("adapter_sha256")
    if want and ad.is_file():
        out["adapter_matches_merge"] = sha256_file(ad) == want  # hint only (a fetched intermediate state is possible)
    return out


def variant_check(test_spec: str, ch: str, require_gold_only: bool = False, allow_mismatch: bool = False, allow_unknown: bool = False,
                  art: Path | None = None) -> tuple[int, list[str], dict]:
    """Training regime of the OOF channel against that of the test side. -> (exit code 0/5/6, lines, statistics)."""
    art = COMP / "artifacts" if art is None else art
    if "," in test_spec or "+" in test_spec:
        raise SystemExit(f"--variant {test_spec}: ONE test draw runs/<run>/<job>/<tag> per call (with several draws call once per draw)")
    meta = read_json(art / f"meta_{ch}.json")
    if meta is None or not (meta.get("rank") or {}).get("source"):
        raise SystemExit(f"artifacts/meta_{ch}.json missing or names no ranking (rank.source)")
    oof_side = [{"rank_job": f"{r}/{j}", **model_training(r, str((m.get("recipe") or {}).get("model")))} for r, j, m in rank_constituents(meta["rank"]["source"])]
    t_run, t_job, t_meta = rank_constituents(test_spec)[0]
    if t_meta.get("table") != "test":
        raise SystemExit(f"{test_spec}: no output for test rows (table={t_meta.get('table')!r})")
    test_side = {"rank_job": f"{t_run}/{t_job}", **model_training(t_run, str((t_meta.get("recipe") or {}).get("model")))}
    lines = [f"training regime of OOF channel {ch} (ranking {meta['rank']['source']}):"]
    for s in oof_side:
        lines.append(f"   {s['rank_job']}: model {s['model']} <- {s.get('train_run')}/{s.get('train_job')}: {s['variant'].upper()} "
                     f"(evidence: {s.get('evidence') or s.get('why')}; fold {s.get('fold')}; adapter = merged one: {s.get('adapter_matches_merge', '–')})")
    lines.append(f"training regime of test side {test_spec}: model {test_side['model']} <- {test_side.get('train_run')}/{test_side.get('train_job')}: "
                 f"{test_side['variant'].upper()} (evidence: {test_side.get('evidence') or test_side.get('why')}; fold {test_side.get('fold')}; "
                 f"adapter = merged one: {test_side.get('adapter_matches_merge', '–')})")
    oof_vars = {s["variant"] for s in oof_side}
    systems = {s.get("system_sha256") for s in oof_side + [test_side] if s.get("system_sha256")}
    code, verdict = 0, "SAME TRAINING REGIME"
    if "unknown" in oof_vars or test_side["variant"] == "unknown":
        code, verdict = 6, "TRAINING REGIME CANNOT BE DETERMINED"
    elif len(oof_vars) != 1 or oof_vars != {test_side["variant"]}:
        code, verdict = 5, f"DIFFERENT TRAINING REGIME (OOF {sorted(oof_vars)}, test {test_side['variant']}) — the weights would belong to a different channel"
    elif len(systems) > 1:
        code, verdict = 5, "DIFFERENT SYSTEM TEXT between fold models and test model"
    if require_gold_only:
        if code == 0 and test_side["variant"] != "gold-only":
            code, verdict = 6, "TEACHER TARGETS — not allowed for the inductive line (gold-only required)"
    elif (code == 5 and allow_mismatch) or (code == 6 and allow_unknown):
        verdict += " — DELIBERATELY OVERRIDDEN (--allow-*; mention it in the report)"
        code = 0
    if test_side.get("fold") is not None:
        lines.append(f"   note: the test model did NOT learn fold {test_side['fold']} (no full fit on all rows)")
    lines.append(f"-> {verdict}" + (" ; the inductive line requires gold-only" if require_gold_only else "") + f" (Exit {code})")
    return code, lines, {"oof_side": oof_side, "test_side": test_side, "code": code, "verdict": verdict}


# ------------------------------------------------------------------------------------------------ saving

def stack_record(spec: dict, ch: str, ins: dict, cr: dict | None, rows: np.ndarray, folds: np.ndarray, oof_sha: dict, ch_meta: dict,
                 ind_check: dict | None) -> dict:
    """Blend file in the format of tools/make_submission.py (channels, theta, floor, none, probes) plus provenance."""
    th = ins["theta"]["with"]
    rec = {"channels": th["names"], "none": [], "probes": {}, "theta": th["theta"], "floor": {k: v for k, v in spec["floors"].items() if v},
           "rows": int(len(rows)), "folds": sorted(set(folds[rows].tolist())), "fit_rows": [int(i) for i in rows], "fit_rows_sha256": rows_sha256(rows),
           "base": {"channels": th["names"], "theta": th["theta_before_factor"]}, "scaled": {spec["think"]: spec["factor"]} if spec["think"] else {},
           "rank_only": {ch: {"floor": spec["floors"][ch], "rank_source": (ch_meta.get("rank") or {}).get("source"), "oof_sha256": oof_sha[ch],
                              "rows": int(len(rows))}},
           "cv": cr["with_pre_last_step"] if cr else None, "cv_rows": int(len(rows)) if cr else None,
           "cv_note": "cross-fit on the channel rows (weights only from the other channel folds) WITHOUT the last step; with the last step see cross",
           "cross": {k: v for k, v in cr.items() if not k.startswith("_")} if cr else None,
           "insample": {k: ins["map3"][k] for k in ins["map3"]}, "oof_sha256": oof_sha, "last_step": spec["last_label"],
           "note": f"eda/c99_m3_deploy.py: weights of ALL channels on log loss only from the {len(rows)} channel rows (a2.fit_mix"
                   + (f", then {spec['think']} × {spec['factor']:g}" if spec["think"] else "") + f"); recipe {spec['label']}"}
    if ind_check:
        rec["inductive_check"] = ind_check
    rk = ch_meta.get("rank") or {}
    if rk.get("draws"):  # channel from several draws (eda/a9_m3_draws_oof.py): make_submission checks the number and the row list
        rec["rank_only"][ch].update(draws=len(rk["draws"]), rows_all_draws_sha256=rk.get("rows_all_draws_sha256"))
    return rec


# ------------------------------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--channel", default="m3p", help="OOF artifact of the channel (artifacts/oof_<channel>.npy, meta_<channel>.json)")
    ap.add_argument("--rows", help="table positions (.npy, relative to the data folder or absolute): blend only on these channel rows")
    ap.add_argument("--lines", nargs="+", default=["best", "safe"], choices=sorted(LINES))
    ap.add_argument("--floor-channel", type=float, default=FLOOR_CHANNEL, help="floor of the channel in the blend (default 0.05)")
    ap.add_argument("--save", action="store_true", help="write blend files artifacts/stack_<name>.json")
    ap.add_argument("--variant", metavar="RANK", help="only check the training regime: test ranking runs/<run>/<job>/<tag> against the OOF channel")
    ap.add_argument("--require-gold-only", action="store_true", help="--variant: only gold-only allowed (inductive line)")
    ap.add_argument("--allow-mismatch", action="store_true", help="--variant: deliberately override a different training regime (not with --require-gold-only)")
    ap.add_argument("--allow-unknown", action="store_true", help="--variant: deliberately override an undeterminable training regime (not with --require-gold-only)")
    ap.add_argument("--versus", metavar="CHANNEL2", help="comparison: the same line with CHANNEL2 instead of --channel (e.g. m3p5), same rows and procedure; report only")
    a = ap.parse_args()
    if a.variant:
        code, lines, _ = variant_check(a.variant, a.channel, a.require_gold_only, a.allow_mismatch, a.allow_unknown)
        print("\n".join(lines))
        return code
    if a.floor_channel <= 0:
        raise SystemExit("--floor-channel must be > 0 (an exact channel without a floor acts as a veto)")
    t0 = time.time()
    L: list[str] = []

    def say(s: str = "") -> None:
        L.append(s)
        print(s, flush=True)

    ctx = context()
    art = COMP / "artifacts"
    y, seen, n, folds = ctx["y"], ctx["seen"], ctx["n"], ctx["folds"]
    ch = a.channel
    # ------------------------------------------------------------ checksums of the base OOFs
    expected = {**BASE_OOF_SHA256, "pr27xm": PR27XM_SHA}
    oof_sha: dict[str, str] = {}
    for nm, want in expected.items():
        got = sha256_file(art / f"oof_{nm}.npy")
        if got != want:
            raise SystemExit(f"artifacts/oof_{nm}.npy differs from the submitted state (SHA-256 {got[:16]}… instead of {want[:16]}…)")
        oof_sha[nm] = got[:16]
    mats = {nm: np.load(art / f"oof_{nm}.npy") for nm in expected}
    # ------------------------------------------------------------ channel
    ch_file, meta_file = art / f"oof_{ch}.npy", art / f"meta_{ch}.json"
    if not ch_file.is_file() or not meta_file.is_file():
        raise SystemExit(f"{ch_file.name} or {meta_file.name} missing")
    ch_meta = json.loads(meta_file.read_text())
    if not (ch_meta.get("rank") or {}).get("source") or ch_meta.get("rank_mean") or ch_meta.get("class_weights") is not None \
            or int((ch_meta.get("rank") or {}).get("rows_named_only") or 0):
        raise SystemExit(f"{meta_file.name}: not a pure ranking channel (eda/a8_proto_oof.py --rank without mean, debiasing or mixed form)")
    mats[ch] = np.load(ch_file)
    oof_sha[ch] = sha256_file(ch_file)[:16]
    rows_arg = None
    if a.rows:
        p = Path(a.rows)
        rows_arg = np.load(p if p.is_absolute() else ctx["root"] / p)
    rows, cinfo = channel_rows(mats[ch], seen, ch_meta, rows_arg)
    kf = {int(k): int((folds[rows] == k).sum()) for k in sorted(set(folds[rows].tolist()))}
    say(f"c99 — channel {ch}: blend weights WITH the channel on the channel rows ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})")
    say(f"channel artifacts/oof_{ch}.npy (SHA-256 {oof_sha[ch]}…), ranking {ch_meta['rank']['source']}: {cinfo['rows_channel']} rows with the channel, "
        f"{cinfo['rows_uniform']} uniform (meta file matches); blend on {len(rows)} rows, per fold {kf}; mass on classes without a "
        f"training example at most {cinfo['unseen_mass_max']:.2g}")
    say(f"base OOFs with checksums as submitted: " + ", ".join(f"{k} {v}…" for k, v in oof_sha.items() if k != ch))
    dinfo = draws_check(ch_meta, rows, ctx["root"])  # channel from several draws (a9): only rows with all draws
    if dinfo:
        say(f"channel from {dinfo['draws']} draws (eda/a9_m3_draws_oof.py): rows by number of draws {dinfo['rows_by_draws']}; "
            f"all {len(rows)} fit rows carry all draws (row list {dinfo['rows_all_draws_file']}, {dinfo['rows_all_draws']} rows, checksum matches)")
    vs = a.versus
    vs_sha = None
    if vs:
        if vs == ch:
            raise SystemExit("--versus: a channel other than --channel is required")
        vs_file, vs_meta_file = art / f"oof_{vs}.npy", art / f"meta_{vs}.json"
        if not vs_file.is_file() or not vs_meta_file.is_file():
            raise SystemExit(f"--versus: {vs_file.name} or {vs_meta_file.name} missing")
        vs_meta = json.loads(vs_meta_file.read_text())
        if not (vs_meta.get("rank") or {}).get("source") or vs_meta.get("rank_mean") or vs_meta.get("class_weights") is not None \
                or int((vs_meta.get("rank") or {}).get("rows_named_only") or 0):
            raise SystemExit(f"--versus: {vs_meta_file.name} does not describe a pure ranking channel")
        mats[vs] = np.load(vs_file)
        vs_sha = sha256_file(vs_file)[:16]
        channel_rows(mats[vs], seen, vs_meta, rows)  # every fit row must also carry the comparison channel
        say(f"comparison channel {vs}: artifacts/oof_{vs}.npy (SHA-256 {vs_sha}…), ranking {vs_meta['rank']['source']} — the same {len(rows)} rows, "
            f"the same procedure (report; the blend file remains that of {ch})")
    # ------------------------------------------------------------ check of the existing state: fit without the channel on all rows
    MS = load("c99_ms", COMP / "tools" / "make_submission.py")

    def ind_mixture(stack: dict) -> np.ndarray:
        return MS.oof_mixture(stack, seen, art, n_rows=n)

    specs = {k: line_setup(k, ch, a.floor_channel, art) for k in a.lines}
    out: dict = {"channel": ch, "oof_sha256": oof_sha, "channel_info": cinfo, "rows": int(len(rows)), "rows_per_fold": kf,
                 "rows_sha256": rows_sha256(rows), "rank_source": ch_meta["rank"]["source"], "floor_channel": a.floor_channel, "lines": {}}
    if dinfo:
        out["draws"] = dinfo
    if vs:
        out["versus"] = {"channel": vs, "oof_sha256": vs_sha}
    for k, spec in specs.items():
        names, logs, _, ti = variant_logs(spec, mats, seen, False)
        th, th_base = fit_theta(logs, y, np.arange(n), ti, spec["factor"])
        err = max(float(np.abs(th - spec["theta_prod"]).max()), float(np.abs(th_base - spec["theta_prod_base"]).max()))
        say(f"check of {spec['base']}: fit without the channel on all {n} rows -> largest deviation of the weights {err:.1e} "
            f"-> {'BIT-IDENTICAL' if err == 0 else ('EQUAL (≤1e-12)' if err <= 1e-12 else 'DEVIATION')}")
        if err > 1e-12:
            raise SystemExit(f"{spec['base']}: the fitting function does not reproduce the submitted weights — check the build path")
        out["lines"][k] = {"stored_check_max_abs": err}
    # ------------------------------------------------------------ per line
    for k, spec in specs.items():
        say("")
        say(f"{spec['label']} — last step: {spec['last_label']}; floors {spec['floors']}"
            + (f"; {spec['think']} × {spec['factor']:g}" if spec["think"] else ""))
        ins = insample(ctx, spec, mats, rows, ind_mixture)
        for tag, label in (("with", f"WITH channel ({len(rows)} rows)"), ("without", f"WITHOUT channel ({len(rows)} rows)"), ("deployed", f"deployed ({n} rows)")):
            t = ins["theta"][tag]
            say(f"   weights {label:<28} " + " · ".join(f"{nm} {v:.3f}" for nm, v in zip(t["names"], t["theta"])))
        m = ins["map3"]
        say(f"   IN-SAMPLE on the {len(rows)} channel rows (weights and last step from the same rows — OPTIMISTIC): with channel {m['with']:.4f} · "
            f"without channel {m['without']:.4f} · deployed without channel {m['deployed_without_channel']:.4f}  (without last step: with {m['with_pre_last_step']:.4f}, "
            f"without {m['without_pre_last_step']:.4f})")
        say(f"     with − without:    {fmt_pair(ins['delta_with_without'])}")
        say(f"     with − deployed:   {fmt_pair(ins['delta_with_deployed'])}")
        rec: dict = {"insample": {kk: v for kk, v in ins.items() if kk not in ("rows_scores", "weights")}}
        ind_check = None
        if spec["last"] == "ind":
            w = ins["weights"]["with"]
            target = target_of(y, np.arange(n), ctx["K"])
            used = target > 0
            ind_check = {"strength": ALIGN_STRENGTH, "min_rows": ALIGN_MIN, "weights_sha256": weights_sha16(w), "fit_rows": int(len(rows)),
                         "target_rows": int(n), "classes": int(used.sum()),
                         "weights_q10_q50_q90": [round(float(q), 3) for q in np.quantile(w[used], [0.1, 0.5, 0.9])],
                         "note": "class weights = c21.fit_weights(make_submission.oof_mixture(blend)[fit_rows], frequencies of all training rows, classes with at least 3)"}
            say(f"   inductive class weights (blend WITH the channel, {len(rows)} fit rows; target all {n} rows): {ind_check['classes']} classes, "
                f"10/50/90 % {ind_check['weights_q10_q50_q90']}, SHA {ind_check['weights_sha256']} (make_submission recomputes them)")
            rec["inductive_check"] = ind_check
        cr = cross(ctx, spec, mats, rows)
        if cr is None:
            say("   CROSS: only one channel fold — no cross value possible")
        else:
            say(f"   CROSS on the channel rows (weights only from the other channel folds {cr['folds']}; last step as deployed, target from the rows "
                f"outside the fold; arithmetic of the build path): with channel {cr['with']:.4f}, without {cr['without']:.4f} (without last step {cr['with_pre_last_step']:.4f} / "
                f"{cr['without_pre_last_step']:.4f})")
            say(f"     with − without:    {fmt_pair(cr['delta'])}")
            rec["cross"] = {kk: v for kk, v in cr.items() if not kk.startswith("_")}
        if vs:  # the same path with the comparison channel (same rows, same procedure) — report only
            spec_v = line_setup(k, vs, a.floor_channel, art)
            ins_v = insample(ctx, spec_v, mats, rows, ind_mixture)
            t = ins_v["theta"]["with"]
            say(f"   COMPARISON {ch} against {vs} (in its place; the same {len(rows)} rows, floors and last step):")
            say(f"     weights WITH {vs:<16} " + " · ".join(f"{nm} {v:.3f}" for nm, v in zip(t["names"], t["theta"])))
            d_in = paired(ctx, ins["rows_scores"]["with"], ins_v["rows_scores"]["with"], rows)
            say(f"     IN-SAMPLE (OPTIMISTIC): {ch} {ins['map3']['with']:.4f} · {vs} {ins_v['map3']['with']:.4f}  (without last step "
                f"{ins['map3']['with_pre_last_step']:.4f} / {ins_v['map3']['with_pre_last_step']:.4f})")
            say(f"       {ch} − {vs}:  {fmt_pair(d_in)}")
            vrec: dict = {"channel": vs, "theta": t, "insample": {kk: ins_v["map3"][kk] for kk in ins_v["map3"]}, "delta_insample": d_in}
            cr_v = cross(ctx, spec_v, mats, rows) if cr is not None else None
            if cr_v is not None:
                d_cr = paired(ctx, cr["_r_with"], cr_v["_r_with"], rows)
                say(f"     CROSS: {ch} {cr['with']:.4f} · {vs} {cr_v['with']:.4f}  (without last step {cr['with_pre_last_step']:.4f} / "
                    f"{cr_v['with_pre_last_step']:.4f})")
                say(f"       {ch} − {vs}:  {fmt_pair(d_cr)}")
                vrec.update(cross={kk: v for kk, v in cr_v.items() if not kk.startswith("_")}, delta_cross=d_cr)
            rec["versus"] = vrec
        if a.save:
            stack = stack_record(spec, ch, ins, cr, rows, folds, oof_sha, ch_meta, ind_check)
            if ind_check:  # the same computation as make_submission --align-inductive on the saved blend
                tgt = target_of(y, np.arange(n), ctx["K"])
                w_ms = C21.fit_weights(MS.oof_mixture(stack, seen, art, n_rows=n)[MS.stack_fit_rows(stack, n)], tgt)
                if weights_sha16(w_ms) != ind_check["weights_sha256"]:
                    raise SystemExit("inductive class weights: recomputation via make_submission differs")
            path = art / f"stack_{spec['stack_name']}.json"
            path.write_text(json.dumps(stack, indent=1, ensure_ascii=False) + "\n")
            say(f"   saved: artifacts/{path.name} (channels {stack['channels']}, {len(rows)} fit rows)")
            rec["stack"] = str(path.relative_to(COMP))
        out["lines"][k].update(rec)
    say("")
    say(f"Limits: weights from {len(rows)} rows (channel rows); in-sample values are optimistic, the cross value is the honest "
        "measure (at 600 rows 2σ ≈ 0.010); the base OOFs are not nested within the meta fold. No deployment decision is made "
        "here.")
    out["runtime_s"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{ch}" if set(a.lines) == set(LINES) else f"{ch}_{'_'.join(a.lines)}"  # one line (single-line build): its own file
    if vs:
        tag += f"_vs_{vs}"  # comparison run: its own log, the build path's log stays
    say(f"-> artifacts/eda/private/m3_deploy/c99_m3_deploy_{tag}.{{txt,json}} ({time.time() - t0:.0f} s)")
    (OUT / f"c99_m3_deploy_{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float) + "\n")
    (OUT / f"c99_m3_deploy_{tag}.txt").write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
