"""Choice of the rotation list for the test side of v1r in the build scripts of the final lines (tools/builds/orient_list.sh,
tools/orient_list.py).

Synthetic, without competition data: choice (default repro, legacy, unknown mode), rejection of the first-version list in repro (path
of the v1r source and checksum of the file), the checks of check / verify-sub / compare. Against the local files, skipped if they are
missing: the real lists. Nothing about individual test rows is printed."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

COMP = Path(__file__).resolve().parents[1]
HELPER = COMP / "tools" / "builds" / "orient_list.sh"
LEGACY_LITERAL = "v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy"  # source in all build scripts with the first-version list


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OL = _load("orient_list_t", COMP / "tools" / "orient_list.py")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _bash(script: str, cwd: Path, **env_extra: str) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", **env_extra}
    return subprocess.run(["bash", "-c", script], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)


def _helper_consts() -> dict:
    return dict(re.findall(r'^(ORIENT_[A-Z0-9_]+)="([^"]*)"$', HELPER.read_text(), flags=re.M))


# ------------------------------------------------------------------------------------------------ choice (bash, without data)

def test_helper_constants_agree_with_python():
    c = _helper_consts()
    assert c["ORIENT_V1R_RUN"] == OL.V1R_RUN
    assert (c["ORIENT_LEGACY_ROWS"], c["ORIENT_LEGACY_ROWS_SHA"]) == (OL.MODES["legacy"]["rows"], OL.MODES["legacy"]["rows_sha256"])
    assert (c["ORIENT_REPRO_ROWS"], c["ORIENT_REPRO_ROWS_SHA"]) == (OL.MODES["repro"]["rows"], OL.MODES["repro"]["rows_sha256"])
    assert (c["ORIENT_REPRO_JSON"], c["ORIENT_REPRO_JSON_SHA"]) == (OL.MODES["repro"]["json"], OL.MODES["repro"]["json_sha256"])
    assert OL.v1r_spec("legacy") == LEGACY_LITERAL  # legacy = verbatim the earlier source of the build scripts


@pytest.mark.parametrize("env, want", [({}, "repro"), ({"ORIENT": ""}, "repro"), ({"ORIENT": "legacy"}, "legacy"), ({"ORIENT": "repro"}, "repro")])
def test_choose_default_is_repro(env, want):
    r = _bash('. tools/builds/orient_list.sh; orient_choose; printf "%s|%s|%s" "$ORIENT" "$ORIENT_ROWS" "$V1R_TEST"; bash -c \'printf "|%s" "$ORIENT"\'',
              COMP, **env)
    assert r.returncode == 0, r.stderr
    mode, rows, spec, inherited = r.stdout.split("|")
    assert (mode, rows, spec, inherited) == (want, OL.MODES[want]["rows"], OL.v1r_spec(want), want)  # exported for nested build scripts
    assert OL.select_mode(env={k: v for k, v in env.items()}) == want


def test_choose_rejects_unknown_mode():
    r = _bash(". tools/builds/orient_list.sh; orient_choose; echo reached", COMP, ORIENT="new")
    assert r.returncode == 2 and "reached" not in r.stdout and "ORIENT=new" in r.stderr
    with pytest.raises(ValueError):
        OL.select_mode(env={"ORIENT": "new"})


@pytest.mark.parametrize("spec", [LEGACY_LITERAL, "v1r=v1rot_t4/rot_test/test@derived/orient/test_v80_rows.npy", "v1r=v1rot_t4/rot_test/test"])
def test_reject_legacy_by_path(spec):
    r = _bash(f'. tools/builds/orient_list.sh; orient_reject_legacy "{spec}"; echo reached', COMP)
    assert r.returncode == 4 and "reached" not in r.stdout


def _fake_comp(tmp: Path) -> Path:
    comp = tmp / "comp"
    (comp / "tools" / "builds").mkdir(parents=True)
    shutil.copy(HELPER, comp / "tools" / "builds" / "orient_list.sh")
    return comp


def test_verify_rejects_wrong_files(tmp_path):
    comp = _fake_comp(tmp_path)
    for mode in ("legacy", "repro"):
        f = comp / "data" / OL.MODES[mode]["rows"]
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, np.arange(3, dtype=np.int64))  # wrong content under the right name
    for mode in ("legacy", "repro"):
        r = _bash(". tools/builds/orient_list.sh; orient_select; echo reached", comp, ORIENT=mode)
        assert r.returncode == 4 and "reached" not in r.stdout and "differs" in r.stderr, (mode, r.stderr)


@pytest.mark.skipif(not (COMP / "data" / OL.MODES["legacy"]["rows"]).is_file(), reason="frozen rotation list missing (other machine)")
def test_repro_rejects_old_list_under_new_name(tmp_path):
    """If the first-version list lies under the name of the released list, the repro variant rejects it explicitly (not only via the checksum)."""
    comp = _fake_comp(tmp_path)
    dst = comp / "data" / OL.MODES["repro"]["rows"]
    dst.parent.mkdir(parents=True)
    shutil.copy(COMP / "data" / OL.MODES["legacy"]["rows"], dst)
    r = _bash(". tools/builds/orient_list.sh; orient_select; echo reached", comp, ORIENT="repro")
    assert r.returncode == 4 and "reached" not in r.stdout and "first-version rotation list" in r.stderr
    same_sha = _bash(". tools/builds/orient_list.sh; orient_reject_legacy \"v1r=x@derived/orient_repro/test_c80_rows.npy\"", comp)
    assert same_sha.returncode == 4  # the same checksum, also without the path of the first-version list


@pytest.mark.skipif(not all((COMP / "data" / OL.MODES[m]["rows"]).is_file() for m in OL.MODES), reason="rotation lists missing (other machine)")
def test_select_real_lists():
    for mode in ("legacy", "repro"):
        r = _bash(". tools/builds/orient_list.sh; orient_select; printf %s \"$V1R_TEST\"", COMP, ORIENT=mode)
        assert r.returncode == 0, r.stderr
        assert r.stdout == OL.v1r_spec(mode)


# ------------------------------------------------------------------------------------------------ check / verify-sub / compare (synthetic)

N_TEST = 20


def _ids():
    return [f"t{i:02d}" for i in range(N_TEST)]


def _write_list(root: Path, rel_rows: str, rel_json: str, mapping: dict, source: str) -> None:
    ids = _ids()
    rows = np.array(sorted(ids.index(k) for k in mapping), dtype=np.int64)
    (root / rel_rows).parent.mkdir(parents=True, exist_ok=True)
    np.save(root / rel_rows, rows)
    fj = root / rel_json
    fj.write_text(json.dumps(mapping, sort_keys=True))
    counts = {str(k): sum(1 for v in mapping.values() if v == k) for k in (1, 2, 3)}
    meta = {"table": "test", "rows": len(rows), "of": N_TEST, "min_conf": 0.8, "turns_ccw": counts, "source": source, "sha256": _sha(fj)}
    fj.with_name(fj.name[:-5] + ".meta.json").write_text(json.dumps(meta, indent=1))


@pytest.fixture
def fake(tmp_path):
    """Artificial data root and repository folder: first-version list {1,3,5,7}, released list {1,3,5,8}; rot_test computed {1,3,5,7,8,10} with the c50 list."""
    root, comp = tmp_path / "data", tmp_path / "comp"
    root.mkdir()
    pd.DataFrame({"trace_id": _ids(), "x": range(N_TEST)}).to_csv(root / "test.csv", index=False)
    old = {"t01": 1, "t03": 2, "t05": 3, "t07": 1}
    new = {"t01": 1, "t03": 2, "t05": 3, "t08": 1}
    _write_list(root, OL.MODES["legacy"]["rows"], OL.MODES["legacy"]["json"], old, "first version")
    _write_list(root, OL.MODES["repro"]["rows"], OL.MODES["repro"]["json"], new, OL.WEIGHTS_SOURCE)
    rot = "derived/orient/test_c50.json"
    (root / rot).write_text(json.dumps({**old, **new, "t10": 2}, sort_keys=True))
    d = comp / "artifacts" / "runs" / "v1rot_t4" / "v1rot_t4" / "rot_test"
    d.mkdir(parents=True)
    idx = np.array([1, 3, 5, 7, 8, 10], dtype=np.int64)
    np.save(d / "test_idx.npy", idx)
    np.save(d / "test_diag.done.npy", np.ones(idx.size, dtype=bool))
    (d / "test_meta.json").write_text(json.dumps({"table": "test", "n": int(idx.size), "recipe": {"image": {"rotate_from": rot}}}))
    w = comp / OL.WEIGHTS_SOURCE
    w.parent.mkdir(parents=True)
    w.write_bytes(b"weights")
    modes = {k: dict(v) for k, v in OL.MODES.items()}
    for k in modes:
        modes[k].update(rows_sha256=_sha(root / modes[k]["rows"]), json_sha256=_sha(root / modes[k]["json"]), n_rows=4)
    desc = {"weights": {"weights_sha256": _sha(w)},
            "lists_at_training": {"test_c80": {"rows": 4, "sha256_rows": modes["repro"]["rows_sha256"], "sha256_json": modes["repro"]["json_sha256"]}}}
    (comp / OL.WEIGHTS_DESC).write_text(json.dumps(desc))
    return {"root": root, "comp": comp, "modes": modes, "rot": {rot: _sha(root / rot)}, "run_dir": d}


def _check(fk, mode="repro", spec="auto"):
    if spec == "auto":
        spec = OL.v1r_spec(mode, fk["modes"])
    return OL.check(mode, fk["root"], fk["comp"], spec=spec, modes=fk["modes"], rotate_sha=fk["rot"])


def test_check_passes_and_counts(fake):
    problems, info = _check(fake)
    assert not problems, problems
    assert info["rows_in_run"] == 4 and info["rows_same_turn_as_run"] == 4 and info["turns_ccw"] == {"1": 2, "2": 1, "3": 1}
    assert info["vs_legacy"] == {"both": 3, "only_legacy": 1, "only_this": 1, "changed_rows": 2}
    problems, _ = _check(fake, mode="legacy")
    assert not problems, problems


def test_check_repro_rejects_legacy_spec(fake):
    problems, _ = _check(fake, spec=OL.v1r_spec("legacy", fake["modes"]))
    assert any("first-version rotation list" in p for p in problems)


def test_check_repro_rejects_old_list_content(fake):
    """The first-version list under a new name (and with its checksum as "expected") is rejected all the same."""
    m, root = fake["modes"], fake["root"]
    shutil.copy(root / m["legacy"]["rows"], root / m["repro"]["rows"])
    m["repro"]["rows_sha256"] = m["legacy"]["rows_sha256"]
    problems, _ = _check(fake)
    assert any("checksum of the first-version rotation list" in p for p in problems)


def test_check_detects_rows_not_in_run(fake):
    np.save(fake["run_dir"] / "test_idx.npy", np.array([1, 3, 5, 7, 10, 11], dtype=np.int64))
    problems, info = _check(fake)
    assert info["rows_in_run"] == 3 and any("not computed" in p for p in problems)


def test_check_detects_other_rotation_in_run(fake):
    rot = fake["root"] / "derived/orient/test_c50.json"
    m = json.loads(rot.read_text())
    m["t08"] = 2
    rot.write_text(json.dumps(m, sort_keys=True))
    fake["rot"] = {"derived/orient/test_c50.json": _sha(rot)}
    problems, info = _check(fake)
    assert info["rows_same_turn_as_run"] == 3 and any("another rotation or none" in p for p in problems)


def test_check_detects_json_rows_mismatch_and_provenance(fake):
    root, m = fake["root"], fake["modes"]
    fj = root / m["repro"]["json"]
    mp = json.loads(fj.read_text())
    mp["t09"] = mp.pop("t08")
    fj.write_text(json.dumps(mp, sort_keys=True))
    m["repro"]["json_sha256"] = _sha(fj)
    problems, _ = _check(fake)
    assert any("different rows" in p for p in problems)
    assert any("orient_cnn.json" in p for p in problems)  # the provenance names the old JSON checksum


def _sub_json(path: Path, rows_file: str, rows: int, n_test: int = N_TEST) -> Path:
    run = OL.V1R_RUN
    path.write_text(json.dumps({"check": {"n_rows": n_test}, "sources": {"v1r": [
        {"spec": run, "rows": 6}, {"draw": run, "rows_file": rows_file, "rows": rows}, {"fill": "v1", "rows": n_test - rows}]}}))
    return path


def test_verify_sub(tmp_path):
    modes = {k: {**v, "n_rows": 4} for k, v in OL.MODES.items()}
    new = _sub_json(tmp_path / "new.json", modes["repro"]["rows"], 4)
    old = _sub_json(tmp_path / "old.json", modes["legacy"]["rows"], 4)
    assert OL.verify_sub("repro", new, modes=modes)[0] == []
    assert OL.verify_sub("legacy", old, modes=modes)[0] == []
    problems, _ = OL.verify_sub("repro", old, modes=modes)
    assert any("first-version rotation list" in p for p in problems)
    assert OL.verify_sub("legacy", new, modes=modes)[0]
    assert OL.verify_sub("repro", _sub_json(tmp_path / "short.json", modes["repro"]["rows"], 3), modes=modes)[0]


def test_compare_counts(fake, tmp_path):
    ids = _ids()
    ref = pd.DataFrame({"trace_id": ids, "root_code": ["NONE CAL-01 ANS-01"] * N_TEST})
    new = ref.copy()
    new.loc[7, "root_code"] = "CAL-01 NONE ANS-01"   # row only in the first-version list: other top code
    new.loc[12, "root_code"] = "ANS-01 NONE CAL-01"  # other row: other top code
    new.loc[3, "root_code"] = "NONE ANS-01 CAL-01"   # row in both lists: only the order of places 2–3
    ref.to_csv(tmp_path / "ref.csv", index=False)
    new.sample(frac=1.0, random_state=0).to_csv(tmp_path / "new.csv", index=False)  # other row order
    res = OL.compare(tmp_path / "ref.csv", tmp_path / "new.csv", fake["root"], modes=fake["modes"])
    assert (res["other_top1"], res["other_top3_ordered"], res["other_top3_set"]) == (2, 3, 0)
    assert res["orient_changed"] == {"rows": 2, "other_top1": 1, "other_top3_ordered": 1, "other_top3_set": 0}
    assert res["orient_only_legacy"]["other_top1"] == 1 and res["orient_only_repro"]["other_top1"] == 0
    assert res["elsewhere"] == {"rows": N_TEST - 2, "other_top1": 1, "other_top3_ordered": 2, "other_top3_set": 0}
