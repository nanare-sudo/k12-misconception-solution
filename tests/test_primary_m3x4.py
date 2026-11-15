"""Tests of the primary candidate primary_m3x4: eda/c135_primary_m3x4.py (rule, bank statistics, placebo permutation, comparison form of the
copy null point, field draws_first in the chain build) and the additive path tools/make_submission.stage2_draws_first.

Rule and helpers SYNTHETIC; with real training artifacts (skipped if they are missing): channel m3x4_full = mean of the four draws (the first
three = m3x3_full bit-identical), blend file against primary_m3px27, deployment file against stage2_primary_m3px27.json, guard of the test
side for several draws (meta files and checksums only; no test row)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

COMP = Path(__file__).resolve().parents[1]
ART = COMP / "artifacts"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COMP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MS = _load("make_submission_primary_m3x4_test", "tools/make_submission.py")
B = _load("c135_primary_m3x4_test", "eda/c135_primary_m3x4.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        pytest.skip(f"artifacts missing: {missing}")


# ------------------------------------------------------------------------------------------------ make_submission: draws_first (additive)

def test_draws_first_without_field_is_a_noop():
    draws, specs = [1, 2, 3, 4], ["a", "b", "c", "d"]
    out = MS.stage2_draws_first({"draws_channel": "m3x3_full"}, draws, specs)
    assert out[0] is draws and out[1] is specs  # earlier deployment files: unchanged (also the objects)


def test_draws_first_takes_the_first_three():
    s2 = {"draws_channel": "m3x4_full", "draws_first": 3}
    d, s = MS.stage2_draws_first(s2, ["P1", "P2", "P3", "P4"], ["r1", "r2", "r3", "r4"])
    assert d == ["P1", "P2", "P3"] and s == ["r1", "r2", "r3"]
    d, s = MS.stage2_draws_first(s2, ["P1", "P2", "P3"], ["r1", "r2", "r3"])  # copy null point 'd4 removed': three of three
    assert d == ["P1", "P2", "P3"] and s == ["r1", "r2", "r3"]
    for bad in (2, 4, "3", True, None):
        with pytest.raises(SystemExit, match="draws_first"):
            MS.stage2_draws_first({**s2, "draws_first": bad}, ["P1", "P2", "P3", "P4"], ["r1", "r2", "r3", "r4"])
    with pytest.raises(SystemExit, match="draws_first"):
        MS.stage2_draws_first(s2, ["P1", "P2"], ["r1", "r2"])


def test_apply_stage2_calls_the_hook_once():
    src = Path(COMP / "tools" / "make_submission.py").read_text()
    body = src.split("def apply_stage2(", 1)[1].split("\ndef ", 1)[0]
    assert body.count("stage2_draws_first(s2, draws, specs)") == 1
    i_specs = body.index('specs = (rank_only.get(dch) or "").split(",")')
    i_hook = body.index("stage2_draws_first(s2, draws, specs)")
    i_check = body.index("if len(draws) != len(S2.DRAWS)")
    assert i_specs < i_hook < i_check  # after reading the test draws, before checking them against test_notes


# ------------------------------------------------------------------------------------------------ c135: statistics and rule (synthetic)

def _rec(R3, R4, copies=None):
    return {"R3": [list(u) for u in R3], "R4": [list(u) for u in R4], "copy": copies or {}}


def test_half_and_split_summary_are_exact():
    assert B.half((7500, 7520)) == Fraction(15020, 18000)
    R3 = [(7500, 7400), (7480, 7390), (7470, 7380), (7460, 7370)]
    R4 = [(7506, 7400), (7490, 7396), (7480, 7386), (7466, 7376)]
    cp = {f"{j}_{k}": (7490 + j, 7380 + k) for j in range(4) for k in range(3)}
    s = B.summarize_split(_rec(R3, R4, cp))
    r3 = [B.half(u) for u in R3]
    r4 = [B.half(u) for u in R4]
    assert s["G"] == sum(r4) / 4 - sum(r3) / 4
    assert s["G"] == Fraction(sum(a + b for a, b in R4) - sum(a + b for a, b in R3), 4 * 18000)
    assert s["G_ex"] == (r4[0] + r4[3]) / 2 - (r3[0] + r3[3]) / 2
    assert s["old_form"] == r4[0] - r3[0]
    assert s["G_copy"] == sum(B.half(u) for u in cp.values()) / 12 - sum(r3) / 4
    assert s["old_form_copy"] == sum(B.half(cp[f"0_{k}"]) for k in range(3)) / 3 - r3[0]
    assert s["G_F"] == Fraction(sum(u[0] for u in R4) - sum(u[0] for u in R3), 4 * 9000)
    assert "G_copy" not in B.summarize_split(_rec(R3, R4))  # final check without copies


def _split(G, Gc):
    return {"G": Fraction(G), "G_copy": Fraction(Gc)}


def test_decide_four_draw_rule():
    orig_ok = {"old_form": Fraction(3, 18000), "old_form_copy": Fraction(1, 18000)}
    sp = [_split(Fraction(2, 18000), Fraction(-1, 18000))] * 20
    d = B.decide(orig_ok, sp)
    assert d["verdict"] == "DEPLOY" and all(d["checks"].values())
    # (1) mean not > 0
    d = B.decide(orig_ok, [_split(0, Fraction(-1, 18000))] * 20)
    assert d["verdict"] == "NO DEPLOY" and not d["checks"]["(1) mean G > 0"]
    # (1) above 0, but not above the copy null point
    d = B.decide(orig_ok, [_split(Fraction(2, 18000), Fraction(2, 18000))] * 20)
    assert d["verdict"] == "NO DEPLOY" and not d["checks"]["(1) mean G > mean G_copy"]
    # (2) original equal to the copy null point ("above" = strictly greater)
    d = B.decide({"old_form": Fraction(1, 18000), "old_form_copy": Fraction(1, 18000)}, sp)
    assert d["verdict"] == "NO DEPLOY" and not d["checks"]["(2) original m3x4 − m3x3 > copy null point"]
    # (2) also holds if the original is negative but above the copy null point (wording of rule part (2))
    d = B.decide({"old_form": Fraction(-1, 18000), "old_form_copy": Fraction(-3, 18000)}, sp)
    assert d["verdict"] == "DEPLOY"
    with pytest.raises(ValueError):
        B.decide(orig_ok, sp[:19])


def test_rule_seeds_fixed():
    assert B.RULE_SEEDS == tuple(range(1000, 1020)) and B.FINAL_SEEDS == tuple(range(2000, 2020)) and len(B.PLACEBO_SEEDS) >= 15
    assert not set(B.RULE_SEEDS) & set(B.FINAL_SEEDS)
    assert B.TRIPLES[0] == (0, 1, 2) and {tuple(sorted(t)) for t in B.TRIPLES} == {(0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3)}
    assert B.EXCHANGEABLE == (0, 3) and B.TRIPLES[3] == (0, 1, 3)  # d1d2d3 and d1d2d4: only d3 <-> d4


def test_placebo_perm_stays_within_folds():
    folds = np.array([0, 1, 0, 2, 1, 0, 2, 2, 1, 0] * 3)
    p = B.placebo_perm(folds, 7)
    assert sorted(p.tolist()) == list(range(len(folds)))  # permutation
    assert np.array_equal(folds[p], folds)  # every row gets a row of the same fold
    assert np.array_equal(p, B.placebo_perm(folds, 7)) and not np.array_equal(p, B.placebo_perm(folds, 8))


def test_parse_seeds_and_labels():
    assert B.parse_seeds("orig,1000, 1001") == [None, 1000, 1001]
    assert [B.seed_label(s) for s in (None, 1000)] == ["orig", "1000"]


def test_units_of_rejects_non_sixths():
    assert B.units_of(np.array([1.0, 0.5, 1 / 3, 0.0])) == 6 + 3 + 2
    with pytest.raises(B.TechnicalError):
        B.units_of(np.array([0.25]))


def test_sft_log_lines_counts_only_test_lines(tmp_path):
    """Report count of the test guard: lines of the runner for all test rows (pattern c133.LOG_LINE; the recorded German wording of the
    same line is covered by the pattern test of tests/test_primary_m3px27.py)."""
    f = tmp_path / "worker0.log"
    f.write_text("[w0] 01:03:21 sft_from: notes for 3498 rows of table test, of which without a finding 3498 (findings are generated before scoring)\n"
                 "[w0] 01:03:22 sft_from: notes for 1500 rows of table train, of which without a finding 1500 (findings are generated before scoring)\n"
                 "[w0] 01:03:23 sft_from: notes for 3497 rows of table test, of which without a finding 3497 (findings are generated before scoring)\n"
                 "[w0] 01:03:24 another line\n")
    assert B.sft_log_lines([f], 3498) == 1
    assert B.sft_log_lines([f, f], 3498) == 2 and B.sft_log_lines([], 3498) == 0


# ------------------------------------------------------------------------------------------------ c135: chain build and copy comparison

def test_draws_channel_wrapper_inserts_draws_first(monkeypatch):
    C = B.c118()
    orig = C.build_stage2

    def stub(*a, **kw):
        return {"kind": "x", "draws_channel": "m3x3_full", "draw_sources": ["d1", "d2", "d3"]}, {"rep": 1}

    monkeypatch.setattr(C, "build_stage2", stub)
    with B._DrawsChannelAs("m3x4_full"):
        out, rep = C.build_stage2()
        assert list(out) == ["kind", "draws_channel", "draws_first", "draw_sources"]
        assert out["draws_channel"] == "m3x4_full" and out["draws_first"] == 3 and rep == {"rep": 1}
    assert C.build_stage2 is stub  # restored
    monkeypatch.setattr(C, "build_stage2", lambda *a, **kw: ({"draws_channel": "other"}, {}))
    with B._DrawsChannelAs("m3x4_full"), pytest.raises(B.TechnicalError):
        C.build_stage2()
    monkeypatch.setattr(C, "build_stage2", orig)


def test_normalized_compare_renames_and_drops_new_fields():
    d = {"stack": "artifacts/stack_primary_m3x4_copy.json", "channels": ["v1r", B.CH_COPY], "draws_channel": B.CH_COPY, "draws_first": 3,
         "models": {"main": {"features": [f"{B.CH_COPY}_logp", "v1r_logp"], "w": [1.0, 2.0]}}, "m3x4_build": {"kind": "copy"},
         "m3px27_build": {"x": 1}, "mm_build": {"oof": "a", "rule": "expected-value rule"}}
    r = {"stack": "artifacts/stack_primary_m3px27.json", "channels": ["v1r", B.M3], "draws_channel": B.M3,
         "models": {"main": {"features": [f"{B.M3}_logp", "v1r_logp"], "w": [1.0, 2.0]}}, "m3px27_build": {"x": 2},
         "mm_build": {"oof": "b", "rule": "expected-value rule"}}
    X = B.c133()
    assert X.diff_keys(B.normalized(d, B.CH_COPY), B.normalized(r, B.M3)) == []
    r["models"]["main"]["w"][0] = 1.5
    assert X.diff_keys(B.normalized(d, B.CH_COPY), B.normalized(r, B.M3)) == [".models.main.w"]


# ------------------------------------------------------------------------------------------------ real training artifacts

def test_m3x4_full_is_the_mean_of_four_draws():
    f = [ART / f"oof_{B.M3X4}.npy", ART / f"meta_{B.M3X4}.json", ART / f"oof_{B.M3}.npy", ART / f"meta_{B.M3}.json"]
    _need(*f)
    P4, meta4, P3, meta3 = np.load(f[0]), json.loads(f[1].read_text()), np.load(f[2]), json.loads(f[3].read_text())
    rk4, rk3 = meta4["rank"], meta3["rank"]
    specs = rk4["source"].split(",")
    assert specs[:3] == rk3["source"].split(",") and specs[3] == B.D4_SPEC and len(rk4["draws"]) == 4
    assert rk4["rows_by_draws"] == {"4": 1500} and rk4["rows_all_draws_file"] == rk3["rows_all_draws_file"]
    assert rk4["rows_all_draws_sha256"] == rk3["rows_all_draws_sha256"]
    assert meta4["checks"]["m3x3_full"]["first_three_mean_bit_identical"] is True
    assert P4.shape == P3.shape and np.allclose(P4.sum(1), 1.0) and not np.array_equal(P4, P3)
    assert meta4["folds_hash"] == meta3["folds_hash"]


def test_stack_against_primary_m3px27():
    files = [ART / f"stack_{B.STEM}.json", ART / f"stack_{B.STEM_M3PX27}.json", ART / f"oof_{B.M3X4}.npy", ART / f"meta_{B.M3X4}.json"]
    _need(*files)
    stack, ref = (json.loads(Path(p).read_text()) for p in files[:2])
    ren = lambda s: B.M3X4 if s == B.M3 else s  # noqa: E731
    assert stack["channels"] == [ren(c) for c in ref["channels"]] and stack["floor"] == {ren(k): v for k, v in ref["floor"].items()}
    for k in ("last_step", "rows", "folds", "fit_rows", "fit_rows_sha256", "scaled", "none", "probes"):
        assert stack[k] == ref[k], k
    ro = stack["rank_only"]
    assert list(ro) == [ren(k) for k in ref["rank_only"]] and ro["q397m3"] == ref["rank_only"]["q397m3"]
    m4 = json.loads(files[3].read_text())
    assert ro[B.M3X4] == {"floor": 0.05, "rank_source": m4["rank"]["source"], "oof_sha256": _sha(files[2])[:16], "rows": 1500, "draws": 4,
                          "rows_all_draws_sha256": m4["rank"]["rows_all_draws_sha256"]}
    assert stack["arith_beta"] in B.c133().c110().BETA_GRID and stack["replaces"]["by"] == B.M3X4 and stack["replaces"]["channel"] == B.M3
    assert stack["replaces"]["stack_primary_m3px27_sha256"] == B.STACK_M3PX27_SHA == _sha(files[1])


def test_make_submission_draws_guard_accepts_four_test_draws():
    files = [ART / f"stack_{B.STEM}.json", ART / f"meta_{B.M3X4}.json", ART / f"oof_{B.M3X4}.npy"]
    _need(*files)
    from vlm import common as VC

    root = VC.find_data_root()
    stack = json.loads(files[0].read_text())
    specs = [*B.TEST_RANKS3, B.D4_TEST]
    refs, info = MS.rank_only_draws_guard(B.M3X4, stack, specs, root, 1500)
    assert refs == json.loads(files[1].read_text())["rank"]["source"].split(",") and info["rows_all_draws"] == 1500
    with pytest.raises(SystemExit, match="test draw"):
        MS.rank_only_draws_guard(B.M3X4, stack, specs[:3], root, 1500)  # three test draws against an artifact of four


def test_stage2_against_primary_m3px27():
    files = [B.STAGE2, B.STAGE2_M3PX27, ART / f"stack_{B.STEM}.json"]
    _need(*files)
    s2, pm = (json.loads(Path(p).read_text()) for p in files[:2])
    want = []
    for k in pm:
        want.append(k)
        if k == "draws_channel":
            want.append("draws_first")
    assert list(s2) == [*want, "m3x4_build"]
    assert s2["draws_channel"] == B.M3X4 and s2["draws_first"] == 3 and s2["test_notes"] == pm["test_notes"]
    assert [c["rank"] for c in s2["test_notes"]] == list(B.TEST_RANKS3)
    assert s2["stack_sha256"] == _sha(files[2])
    ren = lambda s: s.replace(f"{B.M3}_", f"{B.M3X4}_", 1) if s.startswith(f"{B.M3}_") else (B.M3X4 if s == B.M3 else s)  # noqa: E731
    for fs in ("main", "fallback"):
        assert s2["models"][fs]["features"] == [ren(x) for x in pm["models"][fs]["features"]]
    m = s2["models"]["main"]
    assert m["feature_set"] == "main_guard_i1_mm" and m["protect_none"] is True and len(m["w"]) == 64 and not m["abstain"]
    sb = s2["m3x4_build"]
    assert sb["kind"] == "deploy" and sb["oof_rows"]["bit_identical"] and all(v["ok"] for v in sb["null_point_i"].values())
    assert s2["guard_rule"] == pm["guard_rule"] and s2["mech27"] == pm["mech27"]
