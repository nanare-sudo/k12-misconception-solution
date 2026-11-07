"""Tests of the second stage of the INDUCTIVE line: tools/make_submission.py (apply_stage2_safe, stage2_safe_order, switch --stage2
with --align-inductive), eda/c120_s036_safe_stage2.py and tools/builds/s036_safe_stage2.sh.

Synthetic (fast, without data): NONE guard (base top NONE -> base order), null model -> probabilities bit-identical, guard (c) aborts,
wrong deployment files are rejected, the command-line check (an inductive file requires --align-inductive as in the file; transductive
files unchanged). With real files (skipped if they are missing): c120 fit --verify = deployment file, null model, checksums in the
build script. No test writes under artifacts/."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

COMP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COMP / "src")]
import stage2 as S2  # noqa: E402

PY = [sys.executable, "-I"]
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
SAFE_CHANNELS = ["v1r", "zs27t", "think18sb", "pr27xm", "m3x3_full"]
STAGE2_FILE = COMP / "artifacts" / "stage2_s036_safe.json"
OOF_JSON = COMP / "artifacts" / "eda" / "private" / "stage2_inductive" / "c120_oof.json"
BUILD = COMP / "tools" / "builds" / "s036_safe_stage2.sh"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classes_toy() -> list[str]:
    return ["NONE", "ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00", "Dec-01", "PV-01", "ALG-06", "FRA-02"]


# ------------------------------------------------------------------------------------------------ synthetic

def test_stage2_safe_order_protects_none_tops():
    MS = _load("make_submission_safe_order", "tools/make_submission.py")
    cand = np.array([[0, 3, 4, 5, 6], [2, 0, 3, 4, 5], [0, 1, 2, 3, 4], [7, 1, 0, 2, 3]])
    nf = 2
    X = np.zeros((4 * 5, nf))
    X[:, 0] = np.tile([0, 0, 0, 0, 1.0], 4)  # feature 0 points to the fifth candidate
    model = {"w": [5.0, 0.0], "mu": [0.0, 0.0], "sd": [1.0, 1.0]}
    top_u, n_u = MS.stage2_safe_order(model, X, cand, protect=False)
    top_p, n_p = MS.stage2_safe_order(model, X, cand, protect=True)
    assert n_u == 0 and np.array_equal(top_u[:, 0], cand[:, 4])
    assert n_p == 2  # rows 0 and 2 have NONE at the base top
    assert np.array_equal(top_p[[0, 2]], cand[[0, 2]]) and np.array_equal(top_p[[1, 3]], top_u[[1, 3]])
    zero = {"w": [0.0, 0.0], "mu": [0.0, 0.0], "sd": [1.0, 1.0]}
    top_z, n_z = MS.stage2_safe_order(zero, X, cand, protect=True)
    assert n_z == 0 and np.array_equal(top_z, cand)


def _toy_safe(tmp_path, *, w_scale: float = 0.0, oof_share: float = 0.0, protect: bool = True, **over):
    """Synthetic inputs for make_submission.apply_stage2_safe (recipe of the inductive line, without notes)."""
    import pandas as pd

    cls = classes_toy()
    K, n_test, n_train = len(cls), 60, 40
    rng = np.random.default_rng(7)
    seen = np.ones(K, bool)
    seen[9] = False

    def rnd():
        P = rng.dirichlet(np.ones(K), size=n_test) * seen[None, :]
        return P / P.sum(1, keepdims=True)

    raw, draws, Q, P_pre = {nm: rnd() for nm in SAFE_CHANNELS}, [rnd(), rnd(), rnd()], rnd(), rnd()
    Q[:15, 0] += 2.0  # 15 rows with NONE at the top
    Q = Q / Q.sum(1, keepdims=True)
    stack = {"channels": list(SAFE_CHANNELS), "arith_beta": 0.4}
    stack_file = tmp_path / "stack_toy.json"
    stack_file.write_text(json.dumps(stack))
    train = pd.DataFrame({"root_code": [cls[i % 9] for i in range(n_train)], "question_id": [f"q{i % 7}" for i in range(n_train)],
                          "topic_name": ["A" if i % 2 else "B" for i in range(n_train)]})
    test = pd.DataFrame({"question_id": [f"q{i % 7}" for i in range(n_test)], "topic_name": ["A" if i % 3 else "B" for i in range(n_test)],
                         "grade": [7] * n_test})
    names = S2.feature_names(SAFE_CHANNELS, "fallback")
    w = [0.0] * len(names)
    w[names.index("f_rank")] = w_scale  # positive: candidates from the back move to the front (many new top codes)
    ranks = ["runs/r1/job/test", "runs/r2/job/test", "runs/r3/job/test"]
    s2 = {"stack_sha256": sha(stack_file), "channels": list(SAFE_CHANNELS), "arith_beta": 0.4, "align_mode": "inductive", "align_inductive": 0.75,
          "align_min": 3, "K": S2.K_CAND, "lambda": S2.LAMBDA, "null": w_scale == 0.0, "feature_set": "fallback", "protect_none": protect,
          "draws_channel": "m3x3_full", "test_draws": ranks,
          "model": {"features": names, "w": w, "mu": [0.0] * len(names), "sd": [1.0] * len(names), "oof_changed_top_share": oof_share,
                    "oof_changed_top3_share": oof_share},
          "novel_task_share_top5": 0.5, "guard_rule": {"changed_top_factor": [0.5, 2.0]}}
    s2.update(over)
    s2_file = tmp_path / "stage2_safe_toy.json"
    s2_file.write_text(json.dumps(s2))
    args = (s2_file, stack_file, stack, Q, P_pre, raw, {"m3x3_full": draws}, {"m3x3_full": ",".join(ranks)}, train, test, cls, seen, tmp_path)
    return args


def test_apply_stage2_safe_null_is_bitwise_identity(tmp_path):
    MS = _load("make_submission_safe_null", "tools/make_submission.py")
    args = _toy_safe(tmp_path)
    rep = tmp_path / "guards.json"
    Q_new, info = MS.apply_stage2_safe(*args, report_path=rep)
    assert np.array_equal(Q_new, args[3])
    assert info["null"] and info["rows_protected"] == 0 and info["guard_notes"] is None and info["feature_set"] == "fallback"
    assert json.loads(rep.read_text())["align_mode"] == "inductive"


def test_apply_stage2_safe_reorders_protects_and_guards(tmp_path):
    MS = _load("make_submission_safe_apply", "tools/make_submission.py")
    args = _toy_safe(tmp_path, w_scale=10.0, oof_share=0.7)
    Q = args[3]
    Q_new, info = MS.apply_stage2_safe(*args)
    seen = args[11]
    cand = S2.candidates(Q, seen)
    t_new = S2.candidates(Q_new, seen)
    none_top = cand[:, 0] == 0
    assert info["rows_protected"] == int(none_top.sum()) >= 15  # 15 set, more at random
    assert np.array_equal(t_new[none_top], cand[none_top])  # NONE tops guarded
    assert (t_new[~none_top, 0] == cand[~none_top, 4]).all()  # otherwise the fifth candidate at the front
    assert np.allclose(Q_new.sum(1), 1.0)
    # without the guard every top code changes
    _, info_u = MS.apply_stage2_safe(*_toy_safe(tmp_path, w_scale=10.0, oof_share=0.7, protect=False))
    assert info_u["rows_protected"] == 0 and info_u["guard_changed_top"]["test"] == pytest.approx(1.0)
    # guard (c): OOF share far below the test share -> abort without a file
    with pytest.raises(SystemExit, match="guard \\(c\\)"):
        MS.apply_stage2_safe(*_toy_safe(tmp_path, w_scale=10.0, oof_share=0.05))


@pytest.mark.parametrize("over, msg", [({"align_mode": "trans"}, "align_mode"), ({"feature_set": "main"}, "fallback"),
                                       ({"test_draws": ["runs/x/job/test"] * 3}, "test draws"), ({"channels": ["v1r"]}, "channels"),
                                       ({"stack_sha256": "0" * 64}, "blend file"), ({"protect_none": "yes"}, "protect_none"),
                                       ({"K": 10}, "K 10")])
def test_apply_stage2_safe_rejects_wrong_files(tmp_path, over, msg):
    MS = _load("make_submission_safe_reject", "tools/make_submission.py")
    with pytest.raises(SystemExit, match=msg):
        MS.apply_stage2_safe(*_toy_safe(tmp_path, **over))


def _cli(tmp_path, s2: dict, *extra) -> subprocess.CompletedProcess:
    f = tmp_path / "s2.json"
    f.write_text(json.dumps(s2))
    return subprocess.run(PY + ["tools/make_submission.py", "--name", "__missing__", "--stage2", str(f), *extra], cwd=COMP, env=ENV,
                          capture_output=True, text=True, timeout=120)


def test_cli_inductive_file_requires_align_inductive_transductive_unchanged(tmp_path):
    ind = {"align_mode": "inductive", "align_inductive": 0.75, "align_min": 3}
    for extra in (("--align", "0.75"), ("--align-inductive", "0.5"), ("--align-inductive", "0.75", "--align-min", "2"), ()):
        r = _cli(tmp_path, ind, *extra)
        assert r.returncode != 0 and "(inductive) requires --align-inductive 0.75 --align-min 3" in r.stderr
    r = _cli(tmp_path, ind, "--align-inductive", "0.75", "--align-min", "3")  # check passed; fails only on the missing blend file
    assert r.returncode != 0 and "requires" not in r.stderr and "stack___missing__.json" in r.stderr
    trans = {"align": 0.75, "align_min": 3}
    r = _cli(tmp_path, trans, "--align-inductive", "0.75", "--align-min", "3")
    assert r.returncode != 0 and "--stage2 requires --align 0.75 --align-min 3 (transductive, as in the fit)" in r.stderr
    r = _cli(tmp_path, trans, "--align", "0.75", "--align-min", "3")
    assert r.returncode != 0 and "requires" not in r.stderr and "stack___missing__.json" in r.stderr


# ------------------------------------------------------------------------------------------------ real files

need_real = pytest.mark.skipif(not (STAGE2_FILE.is_file() and OOF_JSON.is_file()), reason="deployment file or OOF reproduction missing")


@need_real
def test_c120_fit_verify_reproduces_stage2_file():
    r = subprocess.run(PY + ["eda/c120_s036_safe_stage2.py", "fit", "--verify"], cwd=COMP, env=ENV, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "VERIFIED" in r.stdout


@need_real
def test_stage2_s036_file_fields_and_null_model(tmp_path):
    M = _load("c120_test", "eda/c120_s036_safe_stage2.py")
    s2 = json.loads(STAGE2_FILE.read_text())
    assert s2["align_mode"] == "inductive" and s2["align_inductive"] == 0.75 and s2["align_min"] == 3
    assert s2["feature_set"] == "fallback" and s2["protect_none"] is True and s2["channels"] == SAFE_CHANNELS
    assert s2["model"]["features"] == S2.feature_names(SAFE_CHANNELS, "fallback") and not s2["model"]["abstain"]
    assert s2["stack_sha256"] == sha(COMP / s2["stack"]) == M.STACK_SHA
    assert s2["test_draws"] == list(M.TEST_DRAWS) and 0.0 < s2["model"]["oof_changed_top_share"] < 0.1
    oof = json.loads(OOF_JSON.read_text())
    assert oof["anchor_units"] == M.ANCHOR_UNITS and oof["deploy_units"] == M.TARGET_UNITS and oof["null_point_exact"]
    nul = M.null_stage2(STAGE2_FILE)
    assert nul["null"] and set(nul["model"]["w"]) == {0.0} and nul["model"]["features"] == s2["model"]["features"]
    assert nul["align_mode"] == "inductive" and nul["test_draws"] == s2["test_draws"]


@pytest.mark.skipif(not BUILD.is_file() or not STAGE2_FILE.is_file(), reason="build script or deployment file missing")
def test_build_script_s036_fixed_checksums_match_files():
    txt = BUILD.read_text()
    pins = {"STAGE2_SHA": STAGE2_FILE, "STACK_SHA": COMP / "artifacts" / "stack_v1r_zs27t_think18sb_pr27xm_m3x3_full_ar.json",
            "BASE_FILE_SHA": COMP / "artifacts" / "submissions" / "sub_s030n_safe.csv"}
    out = COMP / "artifacts" / "submissions" / "sub_s036_safe_stage2.csv"
    if out.is_file() and 'OUT_SHA="' in txt:
        pins["OUT_SHA"] = out
    for key, f in pins.items():
        line = next(ln for ln in txt.splitlines() if ln.strip().startswith(f"{key}="))
        want = line.split('"')[1].split(":-")[-1].rstrip("}")
        assert len(want) == 64 and want == sha(f), key
