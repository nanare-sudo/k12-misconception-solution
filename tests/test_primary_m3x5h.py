"""Tests of primary submission A primary_m3x5h: eda/c166_primary_m3x5h.py (channel with a row list, row lists, large-photo draw guard and its
positive controls, comparison form of the copy null point, exit codes), the additive path "draw with row list" in tools/make_submission.py
(rank_only_rowlist_guard, rank_only_rowlist_draw, rank_only_draws_channel with rowlist, placeholder) and the build script
tools/builds/primary_m3x5h.sh.

SYNTHETIC with files under tmp_path (runs of the ranking as dummies in an artifacts/runs tree of their own); with real training artifacts
(skipped if they are missing): channel m3x5h = channel of the evaluation, pins in the build script, positive controls on the training side.
No test row is read or printed."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

COMP = Path(__file__).resolve().parents[1]
ART = COMP / "artifacts"
sys.path[:0] = [str(COMP / "src")]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MS = _load("make_submission_primary_m3x5h_test", "tools/make_submission.py")
C166 = _load("c166_primary_m3x5h_test", "eda/c166_primary_m3x5h.py")
A9 = _load("a9_primary_m3x5h_test", "eda/a9_m3_draws_oof.py")
SH = COMP / "tools" / "builds" / "primary_m3x5h.sh"
K = 6
SEEN = np.array([True, True, True, True, False, True])


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        pytest.skip(f"artifacts missing: {missing}")


def _probs(rng, n: int) -> np.ndarray:
    P = rng.random((n, K)) * SEEN
    return P / P.sum(1, keepdims=True)


# ------------------------------------------------------------------------------------------------ channel with a row list (c166.combine)

@pytest.fixture(autouse=True)
def _a9(monkeypatch):
    monkeypatch.setattr(C166, "A9", lambda: A9)


def test_combine_equals_where_list_mean5_else_mean4():
    rng = np.random.default_rng(0)
    n = 40
    draws = [(_probs(rng, n), np.ones(n, bool)) for _ in range(5)]
    m = rng.random(n) < 0.6
    P, cnt = C166.combine(draws, {4: m}, n, K)
    mean5, _ = A9.mean_of_draws(draws, n, K)
    mean4, _ = A9.mean_of_draws(draws[:4], n, K)
    assert np.array_equal(P, np.where(m[:, None], mean5, mean4))  # bit-identical: the same sum in the same order
    assert np.array_equal(cnt, 4 + m.astype(np.int64))


def test_combine_empty_list_is_old_channel_bit_identical():
    rng = np.random.default_rng(1)
    n = 25
    draws = [(_probs(rng, n), np.ones(n, bool)) for _ in range(5)]
    P, cnt = C166.combine(draws, {4: np.zeros(n, bool)}, n, K)
    assert np.array_equal(P, A9.mean_of_draws(draws[:4], n, K)[0]) and (cnt == 4).all()


def test_combine_restriction_intersects_coverage():
    rng = np.random.default_rng(2)
    n = 10
    h = np.ones(n, bool)
    h[3] = False
    draws = [(_probs(rng, n), np.ones(n, bool)) for _ in range(4)] + [(_probs(rng, n), h)]
    m = np.zeros(n, bool)
    m[[2, 3, 4]] = True
    _, cnt = C166.combine(draws, {4: m}, n, K)
    assert cnt.tolist() == [4, 4, 5, 4, 5, 4, 4, 4, 4, 4]


# ------------------------------------------------------------------------------------------------ row lists (c166.load_list)

def test_load_list_pins_sha_count_and_order(tmp_path):
    f = tmp_path / "rows.npy"
    np.save(f, np.array([1, 4, 7], dtype=np.int64))
    rows, info = C166.load_list(tmp_path, "train", (str(f), _sha(f), 3))
    assert rows.tolist() == [1, 4, 7] and info["rows_sha256"] == A9.rows_sha256(rows) and info["rows"] == 3
    with pytest.raises(C166.TechnicalError, match="SHA-256"):
        C166.load_list(tmp_path, "train", (str(f), "0" * 64, 3))
    with pytest.raises(C166.TechnicalError, match="ascending"):
        C166.load_list(tmp_path, "train", (str(f), _sha(f), 4))
    g = tmp_path / "unsorted.npy"
    np.save(g, np.array([4, 1], dtype=np.int64))
    with pytest.raises(C166.TechnicalError, match="ascending"):
        C166.load_list(tmp_path, "train", (str(g), _sha(g), 2))
    with pytest.raises(C166.Missing):
        C166.load_list(tmp_path, "train", (str(tmp_path / "missing.npy"), "0" * 64, 0))


# ------------------------------------------------------------------------------------------------ large-photo draw guard

def _notes(rows, rng, complete=0.65, mean_chars=9800):
    return {int(r): ("complete" if rng.random() < complete else "open", "x" * int(max(0, rng.normal(mean_chars, 3000)))) for r in rows}


def _side(seed: int, n: int = 1200):
    rng = np.random.default_rng(seed)
    general = np.array([False, True, True, False, False, False])
    P = np.zeros((n, K))
    top = rng.choice(K, size=n, p=[0.35, 0.12, 0.08, 0.2, 0.0, 0.25])
    P[np.arange(n), top] = 0.95
    P[np.arange(n), (top + 1) % K] += 0.05
    m4 = P.copy()
    flip = rng.random(n) < 0.2
    m4[flip] = np.roll(m4[flip], 1, axis=1)
    rows = np.arange(n)
    return P, m4, rows, _notes(rows, rng), general


def test_rzh_identity_and_same_process_pass():
    P, m4, rows, notes, general = _side(3)
    ref = C166.rzh_stats(P, m4, rows, notes, general)
    assert C166.rzh_check(ref, ref)["ok"]
    P2, m42, rows2, notes2, _ = _side(4, n=2400)  # a second sample of the same process
    chk = C166.rzh_check(ref, C166.rzh_stats(P2, m42, rows2, notes2, general))
    assert chk["ok"], [c for c in chk["checks"] if not c["ok"]]


@pytest.mark.parametrize("control, stat", [("h_shift", "share_same_top_mean4"), ("h_uniform", "mean_top_prob"), ("notes_cut", "notes_chars_mean")])
def test_rzh_controls_fire(control, stat):
    P, m4, rows, notes, general = _side(5)
    ref = C166.rzh_stats(P, m4, rows, notes, general)
    Q, notes_c = C166.apply_control(control, P.copy(), rows, dict(notes), np.ones(K, bool))
    chk = C166.rzh_check(ref, C166.rzh_stats(Q, m4, rows, notes_c, general))
    assert not chk["ok"] and stat in [c["stat"] for c in chk["failed"]]
    assert np.array_equal(P, _side(5)[0])  # the control does not change the input


def test_rzh_floor_and_empty_and_rule_frozen():
    P, m4, rows, notes, general = _side(6)
    ref = C166.rzh_stats(P, m4, rows, notes, general)
    half = {r: v for r, v in notes.items() if r % 10}  # 10 % without a record -> below the floor 0.97
    chk = C166.rzh_check(ref, C166.rzh_stats(P, m4, rows, half, general))
    assert not chk["ok"] and [c["stat"] for c in chk["failed"]] == ["share_answered"]
    assert not C166.rzh_check(ref, C166.rzh_stats(P, m4, np.zeros(0, int), notes, general))["ok"]
    assert C166.RZH["rel"] == 0.10 and C166.RZH["z_max"] == 3.0 and C166.RZH["floors"] == {"share_answered": 0.97}
    assert set(C166.CONTROLS) == {"h_shift", "h_uniform", "notes_cut"}


# ------------------------------------------------------------------------------------------------ make_submission: draw with row list

def _rk(tmp_path, n_train=8, specs5="runs/h/h_rank/bench", rows_tr=(1, 3, 5), rows_te=(0, 2, 4, 6)):
    ftr, fte = tmp_path / "train_rows.npy", tmp_path / "test_rows.npy"
    np.save(ftr, np.array(rows_tr, dtype=np.int64))
    np.save(fte, np.array(rows_te, dtype=np.int64))
    sha_rows = hashlib.sha256(np.array(rows_tr, dtype=np.int64).astype("<i8").tobytes()).hexdigest()
    refs = [f"runs/d{j}/d{j}_rank/bench" for j in range(1, 5)] + [specs5]
    draws = [{"source": s, "rows": n_train, "rows_sha256": "x"} for s in refs[:4]] + [{"source": refs[4], "rows": len(rows_tr), "rows_sha256": sha_rows}]
    rk = {"source": ",".join(refs), "draws": draws,
          "row_list_draws": [{"draw": 5, "source": refs[4], "train": {"file": str(ftr), "sha256": _sha(ftr), "rows": len(rows_tr), "rows_sha256": sha_rows},
                              "test": {"file": str(fte), "sha256": _sha(fte), "rows": len(rows_te)}}]}
    specs = [f"runs/t{j}/t{j}_rank/test" for j in range(1, 5)] + [f"runs/ha/h_test_rank/test+runs/hb/h_test_rank/test@{fte}"]
    return rk, specs


def test_rowlist_guard_accepts_and_parses(tmp_path):
    rk, specs = _rk(tmp_path)
    out, info = MS.rank_only_rowlist_guard("m3x5h", rk, specs, tmp_path, 8)
    assert list(out) == [5] and out[5]["spec"] == "runs/ha/h_test_rank/test+runs/hb/h_test_rank/test" and out[5]["rows"].tolist() == [0, 2, 4, 6]
    assert out[5]["ref"] == rk["source"].split(",")[4] and out[5]["placeholder"] is False and info[0]["test_rows"] == 4
    assert out[5]["none_band"] is None and info[0]["none_band"] is None  # without the field: the band of the channel
    rk["row_list_draws"][0]["none_band"] = [0.27, 0.42]
    out, info = MS.rank_only_rowlist_guard("m3x5h", rk, specs, tmp_path, 8)
    assert out[5]["none_band"] == (0.27, 0.42) and info[0]["none_band"] == [0.27, 0.42]
    for bad in ([0.42, 0.27], [0.2], "0.3", [0.1, 1.2]):
        rk["row_list_draws"][0]["none_band"] = bad
        with pytest.raises(SystemExit, match="none_band"):
            MS.rank_only_rowlist_guard("m3x5h", rk, specs, tmp_path, 8)


@pytest.mark.parametrize("mutate, match", [
    (lambda rk, sp: sp.__setitem__(4, sp[4].split("@")[0]), "must read <shards>@"),
    (lambda rk, sp: sp.__setitem__(4, sp[4].split("@")[0] + "@other.npy"), "must read <shards>@"),
    (lambda rk, sp: rk["row_list_draws"][0]["test"].__setitem__("sha256", "0" * 64), "test list"),
    (lambda rk, sp: rk["row_list_draws"][0]["test"].__setitem__("rows", 5), "ascending"),
    (lambda rk, sp: rk["row_list_draws"][0]["train"].__setitem__("sha256", "0" * 64), "training list"),
    (lambda rk, sp: rk["draws"][4].__setitem__("rows", 8), "does not cover exactly the training list"),
    (lambda rk, sp: rk["draws"][0].__setitem__("rows", 7), "does not cover all 8 training rows"),
    (lambda rk, sp: sp.__setitem__(0, sp[0] + "@x.npy"), "only for the draws with a row list"),
    (lambda rk, sp: rk["row_list_draws"][0].__setitem__("draw", 4), "not a draw of the artifact"),
    (lambda rk, sp: rk["row_list_draws"].append(dict(rk["row_list_draws"][0])), "twice"),
])
def test_rowlist_guard_refuses(tmp_path, mutate, match):
    rk, specs = _rk(tmp_path)
    mutate(rk, specs)
    with pytest.raises(SystemExit, match=match):
        MS.rank_only_rowlist_guard("m3x5h", rk, specs, tmp_path, 8)


def _fake_rank(root: Path, run: str, job: str, tag: str, idx, lp, image=2048, seed=5, table="test"):
    d = root / "artifacts" / "runs" / run / run / job
    d.mkdir(parents=True, exist_ok=True)
    meta = {"table": table, "recipe": {"image": {"max_side": image, "min_side": image, "question_max_side": 448}, "beams": 74, "model": "m", "fold": None},
            "sft": {"sft_from": "gen/gen", "finish_tokens": 320}, "probes": ["proto_rank"], "dtype": "bf16", "quant": None, "adapter": False}
    (d / f"{tag}_meta.json").write_text(json.dumps(meta))
    np.save(d / f"{tag}_idx.npy", np.asarray(idx, dtype=np.int64))
    np.save(d / f"{tag}_proto_rank.npy", np.asarray(lp, dtype=np.float64))
    np.save(d / f"{tag}_proto_rank.done.npy", np.ones(len(idx), bool))
    g = root / "artifacts" / "runs" / run / run / "gen"
    g.mkdir(parents=True, exist_ok=True)
    (g / "gen_generation.json").write_text(json.dumps({"recipe": {"table": table, "image": {"max_side": image}, "settings": {"seed": seed, "max_tokens": 5200}}}))
    return f"runs/{run}/{job}/{tag}"


def _lp(rng, n):
    lp = np.log(_probs(rng, n) + 1e-12)
    lp[:, ~SEEN] = -np.inf
    return lp


@pytest.fixture
def fake_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(MS.a8, "COMP", tmp_path)
    monkeypatch.setattr(MS.c95.A8, "COMP", tmp_path)
    rng = np.random.default_rng(7)
    n_test = 12
    ref = "+".join(_fake_rank(tmp_path, "oof", f"r_f{k}", "bench", [k], _lp(rng, 1), table="train") for k in range(2))
    return tmp_path, rng, n_test, ref


def _rl(spec, rows, ref, placeholder=False):
    return {"spec": spec, "rows": np.asarray(rows, dtype=np.int64), "rows_file": "list.npy", "sha256": "s", "placeholder": placeholder, "ref": ref}


def test_rowlist_draw_exact_coverage(fake_runs):
    root, rng, n, ref = fake_runs
    lst = [1, 2, 5, 8, 9]
    la, lb = _lp(rng, 3), _lp(rng, 2)
    spec = _fake_rank(root, "ta", "rank", "test", [1, 5, 9], la) + "+" + _fake_rank(root, "tb", "rank", "test", [2, 8], lb)
    P, hr, info = MS.rank_only_rowlist_draw("k[5]", _rl(spec, lst, ref), n, SEEN, none_band=(0.0, 1.0))
    assert hr.tolist() == [i in lst for i in range(n)] and info["rows_covered"] == 5 and info["rows_covered_outside_list"] == 0
    assert info["settings_checked"] and len(info["shards"]) == 2
    assert np.allclose(P[~hr], SEEN / SEEN.sum()) and np.allclose(P[hr].sum(1), 1.0) and (P[hr][:, ~SEEN] == 0).all()


@pytest.mark.parametrize("case, match", [("more", "outside the list 1"), ("less", "missing in the list 1"), ("overlap", "overlap"),
                                         ("settings", "computed or generated differently"), ("seed", "computed or generated differently")])
def test_rowlist_draw_refuses(fake_runs, case, match):
    root, rng, n, ref = fake_runs
    lst = [1, 2, 5]
    a = {"more": [1, 2, 5, 6], "less": [1, 2], "overlap": [1, 2], "settings": [1, 2, 5], "seed": [1, 2, 5]}[case]
    kw = {"image": 1024} if case == "settings" else ({"seed": 4} if case == "seed" else {})
    spec = _fake_rank(root, f"t_{case}", "rank", "test", a, _lp(rng, len(a)), **kw)
    if case == "overlap":
        spec += "+" + _fake_rank(root, f"t_{case}2", "rank", "test", [2, 5], _lp(rng, 2))
    with pytest.raises(SystemExit, match=match):
        MS.rank_only_rowlist_draw("k[5]", _rl(spec, lst, ref), n, SEEN, none_band=(0.0, 1.0))


def test_rowlist_draw_entry_band_replaces_channel_band(fake_runs):
    root, rng, n, ref = fake_runs
    lp = _lp(rng, 4)
    lp[:, 0] = 0.0  # NONE at the top on all list rows (share 1.0)
    spec = _fake_rank(root, "tband", "rank", "test", [1, 2, 3, 4], lp)
    rl = _rl(spec, [1, 2, 3, 4], ref)
    with pytest.raises(SystemExit, match="malfunction band"):
        MS.rank_only_rowlist_draw("k[5]", rl, n, SEEN, none_band=(0.0, 0.9))
    _, _, info = MS.rank_only_rowlist_draw("k[5]", {**rl, "none_band": (0.5, 1.0)}, n, SEEN, none_band=(0.0, 0.9))  # the band of the entry applies
    assert info["none_band"] == [0.5, 1.0] and info["none_top_rank"] == 1.0
    with pytest.raises(SystemExit, match="malfunction band"):
        MS.rank_only_rowlist_draw("k[5]", {**rl, "none_band": (0.0, 0.5)}, n, SEEN, none_band=(0.0, 1.0))


def test_rowlist_draw_placeholder_restricts_and_empty_list(fake_runs):
    root, rng, n, ref = fake_runs
    full = _fake_rank(root, "tfull", "rank", "test", list(range(n)), _lp(rng, n), image=1024, seed=4)  # foreign settings, all rows
    P, hr, info = MS.rank_only_rowlist_draw("k[5]", _rl(full, [3, 4], ref, placeholder=True), n, SEEN, none_band=(0.0, 1.0))
    assert hr.sum() == 2 and info["rows_covered_outside_list"] == n - 2 and info["placeholder"] and not info["settings_checked"]
    P0, hr0, _ = MS.rank_only_rowlist_draw("k[5]", _rl(full, [], ref, placeholder=True), n, SEEN, none_band=(0.5, 0.6))  # empty list: no guard
    assert not hr0.any() and np.allclose(P0, SEEN / SEEN.sum())
    same = _fake_rank(root, "tfull2", "rank", "test", list(range(n)), _lp(rng, n))  # settings as the OOF draw, but all rows
    with pytest.raises(SystemExit, match="outside the list"):  # without placeholder: coverage must be EXACTLY the list
        MS.rank_only_rowlist_draw("k[5]", _rl(same, [3, 4], ref), n, SEEN, none_band=(0.0, 1.0))
    with pytest.raises(SystemExit, match="computed or generated differently"):  # without placeholder: foreign settings rejected
        MS.rank_only_rowlist_draw("k[5]", _rl(full, [3, 4], ref), n, SEEN, none_band=(0.0, 1.0))


def test_draws_channel_with_rowlist_equals_a9(fake_runs, monkeypatch):
    root, rng, n, ref = fake_runs
    specs = [_fake_rank(root, f"d{j}", "rank", "test", list(range(n)), _lp(rng, n)) for j in range(4)]
    hspec = _fake_rank(root, "hh", "rank", "test", [0, 3, 7], _lp(rng, 3))
    monkeypatch.setattr(MS, "rank_only_draw_differences", lambda spec, r: {})
    keep = []
    P, cnt, info = MS.rank_only_draws_channel("k", specs + [hspec + "@list.npy"], [ref] * 5, n, SEEN, none_band=(0.0, 1.0), keep=keep,
                                              rowlist={5: _rl(hspec, [0, 3, 7], ref)})
    loads = [MS.a8.load_rank(s, n, SEEN, table="test")[:2] for s in specs + [hspec]]
    want, wcnt = A9.mean_of_draws(loads, n, K)
    assert np.array_equal(P, want) and np.array_equal(cnt, wcnt) and info["rows_by_draws"] == {"4": 9, "5": 3} and len(keep) == 5
    P_old, _, _ = MS.rank_only_draws_channel("k", specs, [ref] * 4, n, SEEN, none_band=(0.0, 1.0))  # without rowlist: unchanged (4 draws)
    assert np.array_equal(P_old, A9.mean_of_draws(loads[:4], n, K)[0])


def test_make_submission_rowlist_path_is_additive():
    sig = inspect.signature(MS.rank_only_draws_channel).parameters
    assert sig["rowlist"].default is None and inspect.signature(MS.rank_only_draws_guard).parameters["rowlist_placeholder"].default is False
    g = inspect.getsource(MS.rank_only_draws_guard)
    assert g.count('if rk.get("row_list_draws"):') == 1 and g.index('if rk.get("row_list_draws"):') < g.index("outside = ~np.isin(fit, rows_all)")
    c = inspect.getsource(MS.rank_only_draws_channel)
    assert c.count("if rowlist is not None and j in rowlist:") == 1
    n = inspect.getsource(MS.rank_only_notes_guard)
    assert 'if placeholder and not rk.get("row_list_draws"):' in n
    main = inspect.getsource(MS.main)
    assert main.count("rowlist=rank_only_rowlists.get(name)") == 1 and "--rank-only-rows" not in main  # no new option


# ------------------------------------------------------------------------------------------------ c166 and build script

def test_c166_constants_exit_codes_and_normalized():
    assert C166.H_SPEC == "+".join(f"runs/{C166.H_SHARDS[k]}/m3h_rank_f{k}/bench" for k in range(5))
    assert C166.UNITS == {"base": {"line": [7592, 7542], "with": [7597, 7566]}, "add5_sm": {"line": [7584, 7542], "with": [7583, 7564]}}
    assert C166.exit_code_of(C166.Missing("x")) == 3 and C166.exit_code_of(C166.TechnicalError("x")) == 4
    assert C166.exit_code_of(C166.GuardFail("x")) == 5 and C166.exit_code_of(C166.ControlFail("x")) == 6 and C166.exit_code_of(ValueError()) is None
    a = {"channels": ["v1r", "m3x5hk"], "draws_channel": "m3x5hk", "models": {"main": {"features": ["m3x5hk_logp", "x"]}}, "stack": "a", "m3x5h_build": {},
         "extra_columns": {"channels": ["v1r", "m3x5hk"]}, "guard_build": {"oof": "a", "rule": "r"}}
    b = {"channels": ["v1r", "m3x4_full"], "draws_channel": "m3x4_full", "models": {"main": {"features": ["m3x4_full_logp", "x"]}}, "stack": "b",
         "cols_build": {"x": 1}, "extra_columns": {"channels": ["v1r", "m3x4_full"]}, "guard_build": {"oof": "b", "rule": "r"}}
    assert C166.normalized5(a, "m3x5hk") == C166.normalized5(b, "m3x4_full")
    b["guard_build"]["rule"] = "s"
    assert C166.normalized5(a, "m3x5hk") != C166.normalized5(b, "m3x4_full")


def test_build_script_call_and_pins():
    assert subprocess.run(["bash", "-n", str(SH)]).returncode == 0
    assert subprocess.run(["bash", str(SH), "x"], capture_output=True).returncode == 2
    text = SH.read_text()
    pins = {"CH_OOF": ART / "oof_m3x5h.npy", "CH_META": ART / "meta_m3x5h.json", "STACK": ART / "stack_primary_m3x5h.json",
            "STAGE2_BASE": ART / "stage2_primary_m3x5h_base.json", "STAGE2": ART / "stage2_primary_m3x5h.json"}
    empty = []
    for var, f in pins.items():
        m = re.search(rf'^{var}="[^"]+"; {var}_SHA="([0-9a-f]{{64}})?"', text, re.M)
        assert m, var
        if not m.group(1):  # written by the v0.4 code: empty until the first build
            empty.append(var)
        elif f.is_file():
            assert _sha(f) == m.group(1), f"{f.name} differs from the pin in the build script"
    for var in ("COLS_FILE_SHA", "M3X4_FILE_SHA", "LIST_TRAIN_SHA", "LIST_TEST_SHA", "TRAIN_H_SHA"):
        assert re.search(rf'{var}="[0-9a-f]{{64}}"', text), var
    m = re.search(r'^OUT="[^"]+"; OUT_SHA="([0-9a-f]{64})?"', text, re.M)
    out = ART / "submissions" / "sub_primary_m3x5h.csv"
    if m and m.group(1) and out.is_file():  # only the checksum of the file (its content is not read)
        assert _sha(out) == m.group(1)
    assert "bash tools/builds/primary_cols.sh" in text and "--rank-only-placeholder" in text and "exit 7" in text
    assert C166.LISTS["train"][1] in text and C166.LISTS["test"][1] in text
    if empty:
        pytest.skip(f"result pins empty until the first build of the v0.4 code: {empty}")


# ------------------------------------------------------------------------------------------------ real artifacts (training side only)

def test_meta_row_list_and_reference():
    f = ART / "meta_m3x5h.json"
    _need(f)
    rk = json.loads(f.read_text())["rank"]
    rl = rk["row_list_draws"]
    assert len(rl) == 1 and rl[0]["draw"] == 5 and rl[0]["source"] == C166.H_SPEC == rk["source"].split(",")[4]
    assert (rl[0]["train"]["file"], rl[0]["train"]["sha256"], rl[0]["train"]["rows"]) == C166.LISTS["train"]
    assert (rl[0]["test"]["file"], rl[0]["test"]["sha256"], rl[0]["test"]["rows"]) == C166.LISTS["test"]
    assert rk["rows_by_draws"] == {"4": 499, "5": 1001} and rk["draws"][4]["rows"] == 1001 and all(d["rows"] == 1500 for d in rk["draws"][:4])
    g = rl[0]["guard"]
    assert g["rule"] == C166.RZH["rule"] and g["rel"] == 0.1 and g["z_max"] == 3.0 and g["reference"]["rows"] == 1001
    assert rl[0]["none_band"] == list(C166.NONE_BAND_H) == [0.27, 0.42]
    assert C166.NONE_BAND_H[0] < g["reference"]["share_none_top"] < C166.NONE_BAND_H[1]


def test_channel_equals_measurement_and_controls_fire_on_train_side():
    """oof_m3x5h = a9 mean(d1–d4, h on the list) = add5_sm of the evaluation (as c166 oof); the positive controls of the large-photo draw
    guard fire on the training side against its own reference, the unchanged training side passes."""
    f = ART / "oof_m3x5h.npy"
    _need(f, ART / "meta_m3x5h.json", ART / "oof_m3x4_full.npy")
    meta = json.loads((ART / "meta_m3x5h.json").read_text())
    import labels
    from vlm import common as VC

    root = VC.find_data_root()
    train = VC.load_table(root, "train")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    seen = labels.seen_mask(train["root_code"], classes)
    n = len(train)
    specs = meta["rank"]["source"].split(",")
    draws = []
    for s in specs:
        try:
            P, h, _ = A9.load_draw(s, n, seen)
        except SystemExit as exc:
            pytest.skip(f"draw not in the tree: {exc}")
        draws.append((P, h))
    rows = np.load(root / C166.LISTS["train"][0])
    m = np.zeros(n, bool)
    m[rows] = True
    P, _ = C166.combine(draws, {4: m}, n, len(classes))
    assert np.array_equal(P, np.load(f))
    m4 = np.load(ART / "oof_m3x4_full.npy")
    general = np.isin(np.array(classes), ("ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00"))
    notes = C166.notes_of_spec(C166.H_SPEC, "train")
    ref = C166.rzh_stats(draws[4][0], m4, rows, notes, general)
    assert ref == meta["rank"]["row_list_draws"][0]["guard"]["reference"]
    assert C166.rzh_check(ref, ref)["ok"]
    for c in C166.CONTROLS:
        Q, nc = C166.apply_control(c, draws[4][0], rows, notes, seen)
        assert not C166.rzh_check(ref, C166.rzh_stats(Q, m4, rows, nc, general))["ok"], c
