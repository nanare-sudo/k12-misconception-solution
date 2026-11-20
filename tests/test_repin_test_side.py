"""tools/repin_test_side.py on a synthetic copy of the files it reads and writes (no competition data needed).

--check must write nothing; --write must back up the changed files, change exactly the pinned fields and keep the JSON
layout of each file byte for byte.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

COMP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repin_test_side_t", COMP / "tools" / "repin_test_side.py")
RP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RP)

LAYOUT_STAGE2 = {"indent": 1, "ensure_ascii": False, "end": "\n"}
LAYOUT_META = {"indent": 2, "ensure_ascii": False, "end": ""}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture()
def repo(tmp_path):
    """Minimal repository layout: the source files and the four files with pins, in the layouts of their writers."""
    for rel in RP.EXTRA:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"row": 0, "text": rel}) + "\n", encoding="utf-8")
    notes = tmp_path / "data" / RP.PX27_NOTES
    notes.parent.mkdir(parents=True)
    notes.write_text(json.dumps({"table": "test", "row": 0, "text": "n"}) + "\n", encoding="utf-8")
    art = tmp_path / "artifacts"

    def pins():
        return {rel: _sha(tmp_path / rel) for rel in RP.EXTRA}

    def stage2(extra: bool) -> dict:
        d = {"form": "≤ 20 px and é stay as written", "theta": [0.25, 1e-9], "mech27": {"test": {"file": RP.PR27_RAW, "sha256": pins()[RP.PR27_RAW]},
                                                                                "train": [{"file": "x", "sha256": "0" * 64}]}}
        if extra:
            d["extra_columns"] = {"test": {"spec": {"pr27": {"raw": [RP.PR27_RAW]}}, "files_sha256": pins()}, "names": ["a", "b"]}
        return d

    for name, extra in (("stage2_primary_m3x4.json", False), ("stage2_primary_m3x5h.json", True), ("stage2_primary_cols.json", True)):
        (art / name).write_bytes(RP.dump(stage2(extra), LAYOUT_STAGE2))
    meta = {"rank_mean": {"notes": {"test": {"px27_file": RP.PX27_NOTES, "px27_sha256": _sha(notes), "model": "m"}, "note": "x"}}, "cv": 0.8}
    (art / "meta_m3px27xm.json").write_bytes(RP.dump(meta, LAYOUT_META))
    return tmp_path


def _run(repo: Path, *args: str) -> int:
    return RP.main([*args, "--artifacts", str(repo / "artifacts"), "--data-root", str(repo / "data")])


def test_check_on_unchanged_files_writes_nothing(repo):
    before = _tree(repo)
    assert _run(repo, "--check") == 0
    assert _run(repo) == 0  # default mode is --check
    assert _tree(repo) == before


def test_write_after_new_answers(repo):
    raw = repo / RP.PR27_RAW
    raw.write_text(raw.read_text() + json.dumps({"row": 1, "text": "new"}) + "\n", encoding="utf-8")
    art = repo / "artifacts"
    old = {p.name: p.read_bytes() for p in art.glob("*.json")}
    before = _tree(repo)
    assert _run(repo, "--check") == 1
    assert _tree(repo) == before  # --check writes nothing, also when a field is new
    assert _run(repo, "--write") == 0
    assert _run(repo, "--check") == 0
    backups = list((art / "repin_backup").iterdir())
    assert len(backups) == 1
    saved = {p.name: p.read_bytes() for p in backups[0].iterdir()}
    assert set(saved) == {"stage2_primary_m3x4.json", "stage2_primary_m3x5h.json", "stage2_primary_cols.json"}
    assert all(saved[k] == old[k] for k in saved)
    assert (art / "meta_m3px27xm.json").read_bytes() == old["meta_m3px27xm.json"]  # untouched: its source did not change
    new_sha = _sha(raw)
    for name in saved:
        b, a = json.loads(old[name]), json.loads((art / name).read_bytes())
        assert a["mech27"]["test"]["sha256"] == new_sha
        changed = {k for k in RP.flatten(b).keys() | RP.flatten(a).keys() if RP.flatten(b).get(k) != RP.flatten(a).get(k)}
        want = {("mech27", "test", "sha256")} | ({("extra_columns", "test", "files_sha256", RP.PR27_RAW)} if "extra_columns" in b else set())
        assert changed == want
        assert RP.layout_of((art / name).read_bytes(), a) == LAYOUT_STAGE2
    others = {k: v for k, v in _tree(repo).items() if not k.startswith("artifacts/")}
    assert others == {k: v for k, v in before.items() if not k.startswith("artifacts/")}


def test_write_after_new_notes_changes_only_the_meta_file(repo):
    notes = repo / "data" / RP.PX27_NOTES
    notes.write_text(notes.read_text() + "\n", encoding="utf-8")
    art = repo / "artifacts"
    old = {p.name: p.read_bytes() for p in art.glob("*.json")}
    assert _run(repo, "--write") == 0
    now = {p.name: p.read_bytes() for p in art.glob("*.json")}
    assert [k for k in now if now[k] != old[k]] == ["meta_m3px27xm.json"]
    meta = json.loads(now["meta_m3px27xm.json"])
    assert meta["rank_mean"]["notes"]["test"]["px27_sha256"] == _sha(notes)
    assert not now["meta_m3px27xm.json"].endswith(b"\n")  # layout of the writer: indent 2, no final newline


def test_write_without_changes_writes_nothing(repo):
    before = _tree(repo)
    assert _run(repo, "--write") == 0
    assert _tree(repo) == before and not (repo / "artifacts" / "repin_backup").exists()


def test_unexpected_layout_is_refused(repo):
    p = repo / "artifacts" / "stage2_primary_m3x4.json"
    d = json.loads(p.read_bytes())
    d["mech27"]["test"]["file"] = "artifacts/runs/other/proto_raw.jsonl"
    p.write_bytes(RP.dump(d, LAYOUT_STAGE2))
    raw = repo / RP.PR27_RAW
    raw.write_text(raw.read_text() + "{}\n", encoding="utf-8")
    before = _tree(repo)
    assert _run(repo, "--write") == 4
    assert _tree(repo) == before


def test_unknown_json_layout_is_refused(repo):
    p = repo / "artifacts" / "stage2_primary_m3x4.json"
    p.write_bytes(json.dumps(json.loads(p.read_bytes()), indent=3).encode())
    raw = repo / RP.PR27_RAW
    raw.write_text(raw.read_text() + "{}\n", encoding="utf-8")
    before = _tree(repo)
    assert _run(repo, "--write") == 4
    assert _tree(repo) == before


def test_missing_source_file(repo):
    (repo / RP.Q397_LABELS).unlink()
    assert _run(repo, "--check") == 3
