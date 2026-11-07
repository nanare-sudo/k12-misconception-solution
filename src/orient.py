"""Orientation network for uprighting rotated photos: images → ink maps → network → orientation → rotation lists.

Reproducible version of the self-supervised network of the first orientation analysis (a development script, not included). The
first version is not repeatable: its shift augmentation drew from the unseeded global `np.random`, and the weights were never
saved — only its predictions (`artifacts/eda/private/i5_rot_cnn_pred.parquet`) and the lists derived from them
(`data/derived/orient/*_c50*`, `*_c80*`) remain.

Commands (entry point `python -I tools/orient_net.py <command>`, from the repository root):

    maps        ink and writing maps 64×64 from the images (train, test, pool sample)
    check-maps  check the maps against the statistics cache of the image analysis (counts only)
    train       train the network, save the weights (orient_cnn.pt + orient_cnn.json), write lists
    infer       inference ONLY with saved weights → orientation per photo → lists (the reproduction path; --from-images)
    robust      sensitivity of the lists to the compute path (batch size, threads, float32; counts only)
    compare     new lists against the frozen ones (counts and checksums only; training side with the MAP@3 of v1r)

What is fixed:
* Maps: the same computation as `eda/i5_01_image_stats.py` (function `expensive`, fields `ink_map`, `wr_map`). It is
  loaded from there, not re-implemented, so that the network input and the statistics cache stay byte-identical; decoding as
  in `process` (PIL, alpha channel on a white background). Order: image files sorted; pool = the fixed sample of the image
  analysis (10,000 files, seed 20261010).
* Training: architecture and recipe of the first version (12 epochs, 2 folds, 2 rounds, AdamW + OneCycle, label smoothing
  0.05, shift up to ±3 pixels, confidently rotated images from 0.8 on enter round 2 rotated). Every random source has its own
  generator with a fixed seed per (round, fold): initialisation and dropout via `torch.manual_seed`, order and rotations via
  a `torch.Generator`, shift via `np.random.default_rng` — nothing draws from a global, unseeded generator. Fixed thread
  count, `torch.use_deterministic_algorithms(True)`.
* Prediction: mean over the four test rotations, computed in float64, in fixed batches per table (batch size and thread
  count change the values at most in the last bit, not the lists — command `robust`). Test rows: mean of the fold networks
  of the last round; training and pool rows: the fold network that did not see them (fold assignment recomputed from the
  seed).
* Saved are the networks of the last round (one file, tensors per fold) and a description with hyperparameters, seed,
  environment, SHA-256 of the file, of the tensors and of the input maps.

Class r means "the upright sheet is rotated by r quarter turns counter-clockwise"; it is uprighted with (4 − r) mod 4 quarter
turns counter-clockwise (list format as in `tools/make_orient.py`).

Data use: training uses only train and pool maps; test maps are only predicted. For test rows this module only prints
counts and checksums; maps, predictions and lists live under `data/` (not versioned).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COMP = Path(__file__).resolve().parents[1]
FORMAT = "k12-orient-cnn/1"
FIT_TABLES = ("train", "unlabelled")  # order of the training rows: first train, then the pool sample
ALL_TABLES = ("train", "test", "unlabelled")
UNL_SAMPLE_SEED = 20261010  # = eda/i5_common.UNL_SAMPLE_SEED
UNL_SAMPLE_N = 10000  # = eda/i5_common.UNL_SAMPLE_N
MAP_SIDE = 64
DEFAULT_WEIGHTS = COMP / "artifacts" / "orient" / "orient_cnn.pt"
DEFAULT_OUT = COMP / "data" / "derived" / "orient_repro"
FROZEN_DIR = COMP / "data" / "derived" / "orient"  # frozen lists of the submitted files — never overwrite
STATS_CACHE = COMP / "artifacts" / "eda" / "private" / "i5_image_stats.parquet"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class TrainConfig:
    """Recipe of the first version (development script, not included), plus seed, threads and prediction batch."""

    seed: int = 0
    epochs: int = 12
    n_folds: int = 2
    rounds: int = 2
    batch: int = 128
    lr: float = 2e-3
    max_lr: float = 3e-3
    weight_decay: float = 1e-3
    label_smoothing: float = 0.05
    max_shift: int = 3
    sure_conf: float = 0.8
    threads: int = 2
    predict_batch: int = 512
    channels: tuple = (2, 24, 48, 96, 128)
    hidden: int = 128
    dropout: float = 0.3

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        d["channels"] = tuple(d["channels"])
        return cls(**d)


# ----------------------------------------------------------------------------- initial values
def fold_seed(seed: int, rnd: int, fold: int) -> int:
    """Seed for (round, fold): independent of what ran before."""
    return int(np.random.SeedSequence([int(seed), int(rnd), int(fold)]).generate_state(1)[0])


def fold_assignment(seed: int, n_folds: int, n: int) -> np.ndarray:
    """Fold per training row (train, then pool sample) — as in the first version: default_rng(seed).integers."""
    return np.random.default_rng(seed).integers(0, n_folds, n)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode() + a.tobytes()).hexdigest()


# ----------------------------------------------------------------------------- images → maps
def image_dir(data_root: Path, table: str) -> Path:
    return Path(data_root) / "competition_data" / "images" / table


def table_files(data_root: Path, table: str) -> list[str]:
    """Image files of a table, sorted (as in the file list of the image analysis); pool: its fixed sample."""
    df = pd.read_csv(Path(data_root) / f"{table}.csv", dtype=str, keep_default_na=False, usecols=["image_file"])
    files = sorted(df["image_file"])
    if table == "unlabelled":
        rng = np.random.default_rng(UNL_SAMPLE_SEED)
        idx = np.sort(rng.choice(len(files), size=min(UNL_SAMPLE_N, len(files)), replace=False))
        files = [files[i] for i in idx]
    return files


_I5 = None


def _i5():
    """eda/i5_01_image_stats.py as a module (map computation `expensive`); also sets the cv2 threads and the pixel limit there."""
    global _I5
    if _I5 is None:
        spec = importlib.util.spec_from_file_location("i5_01_image_stats", COMP / "eda" / "i5_01_image_stats.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _I5 = mod
    return _I5


def decode_rgb(data: bytes) -> np.ndarray:
    """Full decoding as in eda/i5_01_image_stats.process: PNG and images with an alpha channel on a white background, otherwise RGB."""
    from PIL import Image

    imf = Image.open(io.BytesIO(data))
    if data[:8] == PNG_MAGIC or imf.mode in ("RGBA", "LA", "P"):
        rgba = imf.convert("RGBA")
        bgw = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = np.asarray(Image.alpha_composite(bgw, rgba).convert("RGB"))
    else:
        rgb = np.asarray(imf.convert("RGB"))
    return np.ascontiguousarray(rgb)


def maps_of_rgb(rgb: np.ndarray) -> tuple[bytes, bytes]:
    o = _i5().expensive(rgb)
    return o["ink_map"], o["wr_map"]


def map_one(path: str) -> tuple[bytes, bytes] | None:
    """Maps of one image; None if it cannot be decoded (as decode_ok = False in the image analysis)."""
    try:
        return maps_of_rgb(decode_rgb(Path(path).read_bytes()))
    except Exception:
        return None


def _pool_init() -> None:
    import cv2

    cv2.setNumThreads(1)
    _i5()


def compute_maps(data_root: Path, table: str, workers: int = 4, files: list[str] | None = None) -> dict:
    """Maps of a table in a fixed order; worker processes do not change the values (every image on its own)."""
    files = table_files(data_root, table) if files is None else list(files)
    paths = [str(image_dir(data_root, table) / f) for f in files]
    _i5()
    if workers > 1:
        ctx = mp.get_context("spawn" if "torch" in sys.modules else "fork")
        with ctx.Pool(workers, initializer=_pool_init) as pool:
            res = pool.map(map_one, paths, chunksize=16)
    else:
        _pool_init()
        res = [map_one(p) for p in paths]
    n = len(files)
    ink = np.zeros((n, MAP_SIDE, MAP_SIDE), np.uint8)
    wr = np.zeros((n, MAP_SIDE, MAP_SIDE), np.uint8)
    ok = np.zeros(n, bool)
    for i, r in enumerate(res):
        if r is not None:
            ink[i] = np.frombuffer(r[0], np.uint8).reshape(MAP_SIDE, MAP_SIDE)
            wr[i] = np.frombuffer(r[1], np.uint8).reshape(MAP_SIDE, MAP_SIDE)
            ok[i] = True
    return {"table": table, "files": np.array(files), "ink": ink, "wr": wr, "ok": ok}


def maps_from_cache(stats_path: Path, table: str, files: list[str]) -> dict:
    """The same maps from the statistics cache of the image analysis (artifacts/eda/private/i5_image_stats.parquet)."""
    df = pd.read_parquet(stats_path, columns=["folder", "file", "full", "decode_ok", "ink_map", "wr_map"])
    df = df[df.folder == table].set_index("file")
    missing = [f for f in files if f not in df.index]
    if missing:
        raise SystemExit(f"{table}: {len(missing)} files missing from the cache")
    df = df.loc[list(files)]
    n = len(files)
    ink = np.zeros((n, MAP_SIDE, MAP_SIDE), np.uint8)
    wr = np.zeros((n, MAP_SIDE, MAP_SIDE), np.uint8)
    ok = (df["full"].astype(bool) & df["decode_ok"].map(lambda v: bool(v) if v is not None and v == v else False)).to_numpy()
    for i, (a, b, good) in enumerate(zip(df["ink_map"], df["wr_map"], ok)):
        if good:
            ink[i] = np.frombuffer(a, np.uint8).reshape(MAP_SIDE, MAP_SIDE)
            wr[i] = np.frombuffer(b, np.uint8).reshape(MAP_SIDE, MAP_SIDE)
    return {"table": table, "files": np.array(files), "ink": ink, "wr": wr, "ok": ok}


def maps_path(maps_dir: Path, table: str) -> Path:
    return Path(maps_dir) / f"maps_{table}.npz"


def save_maps(path: Path, maps: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, files=maps["files"], ink=maps["ink"], wr=maps["wr"], ok=maps["ok"])


def load_maps(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {"table": Path(path).stem.removeprefix("maps_"), "files": z["files"], "ink": z["ink"].copy(),
                "wr": z["wr"].copy(), "ok": z["ok"].copy()}


def maps_sha256(maps: dict) -> str:
    """Checksum over file names, decoding status and both maps (order included)."""
    h = hashlib.sha256()
    h.update("\n".join(str(f) for f in maps["files"]).encode())
    for k in ("ok", "ink", "wr"):
        h.update(np.ascontiguousarray(maps[k]).tobytes())
    return h.hexdigest()


def compare_maps(a: dict, b: dict) -> dict:
    """Counts only: how many maps of two sources are byte-identical."""
    if list(a["files"]) != list(b["files"]):
        return {"rows": int(len(a["files"])), "same_files": False}
    both = a["ok"] & b["ok"]
    ink = (a["ink"] == b["ink"]).reshape(len(both), -1).all(1)
    wr = (a["wr"] == b["wr"]).reshape(len(both), -1).all(1)
    return {"rows": int(len(both)), "same_files": True, "ok_a": int(a["ok"].sum()), "ok_b": int(b["ok"].sum()),
            "ok_both": int(both.sum()), "ink_equal": int((ink & both).sum()), "wr_equal": int((wr & both).sum()),
            "both_equal": int((ink & wr & both).sum()), "identical": bool((a["ok"] == b["ok"]).all() and (ink & wr | ~both).all())}


def net_input(maps: dict) -> tuple[np.ndarray, np.ndarray]:
    """(files, X) of the decodable images: X uint8 (n, 2, 64, 64), dynamic range compressed by a square root as in the first version."""
    ok = maps["ok"]
    X = np.stack([maps["ink"][ok], maps["wr"][ok]], axis=1)
    X = np.round(255.0 * np.sqrt(X.astype(np.float32) / 255.0)).astype(np.uint8)
    return np.asarray(maps["files"])[ok], X


# ----------------------------------------------------------------------------- network
def _torch():
    import torch

    return torch


def make_net(cfg: TrainConfig):
    """Small CNN of the first version (4 convolution stages, head with dropout); parameter names as there."""
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional
    ch = list(cfg.channels)
    side = MAP_SIDE // 2 ** (len(ch) - 1)

    class OrientNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            for i in range(len(ch) - 1):
                self.convs.append(nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]),
                                                nn.ReLU(inplace=True)))
            self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(ch[-1] * side * side, cfg.hidden),
                                      nn.ReLU(inplace=True), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, 4))

        def forward(self, x):
            for c in self.convs:
                x = F.max_pool2d(c(x), 2)
            return self.head(x.flatten(1))

    return OrientNet()


def rot_batch(xb, k):
    """Rotate every map by k[i] quarter turns counter-clockwise."""
    torch = _torch()
    out = torch.empty_like(xb)
    for r in range(4):
        m = k == r
        if m.any():
            out[m] = torch.rot90(xb[m], r, dims=(2, 3))
    return out


def predict_tta(net, X: np.ndarray, batch: int = 512, base: np.ndarray | None = None,
                dtype: str = "float64") -> tuple[np.ndarray, dict]:
    """Probability of the orientation of the image (n × 4, float64): mean over the four test rotations, mapped back.

    Computed in float64 (dtype) on a copy of the network, in fixed batches. With `base` (assumed orientation of the images)
    additionally the self-supervision as a diagnostic: share of correct single judgements over the four rotations."""
    torch = _torch()
    F = torch.nn.functional
    tdt = getattr(torch, dtype)
    m = copy.deepcopy(net).to(tdt).eval()
    P = np.zeros((len(X), 4), np.float64)
    correct = 0
    confusion = np.zeros((4, 4), np.int64)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[s:s + batch])).to(tdt) / 255.0
            acc = torch.zeros(len(xb), 4, dtype=tdt)
            for r in range(4):
                logits = m(torch.rot90(xb, r, dims=(2, 3)))
                # prediction "orientation j" on the image rotated by r means original orientation (j − r) mod 4
                acc += torch.roll(F.softmax(logits, dim=1), shifts=-r, dims=1)
                if base is not None:
                    pred = logits.argmax(1).numpy()
                    true = (base[s:s + batch] + r) % 4
                    correct += int((pred == true).sum())
                    np.add.at(confusion, (true, pred), 1)
            P[s:s + batch] = (acc / 4).numpy()
    diag = {"correct": correct, "confusion": confusion} if base is not None else {}
    return P, diag


def ensemble_predict(nets, X: np.ndarray, batch: int = 512, dtype: str = "float64") -> np.ndarray:
    """Test rows: mean of the fold networks (sum in fold order, then divided)."""
    P = np.zeros((len(X), 4), np.float64)
    for net in nets:
        P += predict_tta(net, X, batch, dtype=dtype)[0]
    return P / len(nets)


def oof_predict(nets, X: np.ndarray, fold: np.ndarray, batch: int = 512, dtype: str = "float64") -> np.ndarray:
    """Training and pool rows: the fold network that did not see the row (batches per table and fold)."""
    P = np.zeros((len(X), 4), np.float64)
    for f, net in enumerate(nets):
        idx = np.flatnonzero(fold == f)
        if len(idx):
            P[idx] = predict_tta(net, X[idx], batch, dtype=dtype)[0]
    return P


def train_fold(X: np.ndarray, base: np.ndarray, cfg: TrainConfig, seed: int, log=print, tag: str = ""):
    """One fold network: images in their assumed upright orientation, rotated and shifted at random, learn the rotation."""
    torch = _torch()
    F = torch.nn.functional
    torch.manual_seed(seed)  # initialisation and dropout
    gen = torch.Generator().manual_seed(seed)  # order and rotations
    shift_rng = np.random.default_rng(seed)  # shift (first version: global np.random.randint)
    net = make_net(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * (len(X) // cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.max_lr, total_steps=steps)
    xt = torch.from_numpy(np.ascontiguousarray(X))
    bt = torch.from_numpy(np.ascontiguousarray(base, dtype=np.int64))
    net.train()
    t0 = time.time()
    for ep in range(cfg.epochs):
        perm = torch.randperm(len(X), generator=gen)
        for s in range(0, len(X) - cfg.batch + 1, cfg.batch):
            idx = perm[s:s + cfg.batch]
            xb = xt[idx].float() / 255.0
            k = torch.randint(0, 4, (len(idx),), generator=gen)
            xb = rot_batch(xb, (k - bt[idx]) % 4)
            dx, dy = (int(v) for v in shift_rng.integers(-cfg.max_shift, cfg.max_shift + 1, 2))
            xb = torch.roll(xb, shifts=(dy, dx), dims=(2, 3))
            loss = F.cross_entropy(net(xb), k, label_smoothing=cfg.label_smoothing)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if ep in (0, cfg.epochs // 2, cfg.epochs - 1):
            log(f"    {tag} epoch {ep + 1}/{cfg.epochs}: {time.time() - t0:.0f} s, last loss {loss.item():.4f}")
    net.eval()
    return net


class _TorchState:
    """Set threads and deterministic algorithms for the duration of a run, restore them afterwards."""

    def __init__(self, threads: int):
        self.threads = threads

    def __enter__(self):
        torch = _torch()
        self.prev = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                     getattr(torch.backends.mkldnn, "deterministic", None))
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(True)
        if self.prev[2] is not None:
            torch.backends.mkldnn.deterministic = True
        return self

    def __exit__(self, *exc):
        torch = _torch()
        torch.set_num_threads(self.prev[0])
        torch.use_deterministic_algorithms(self.prev[1])
        if self.prev[2] is not None:
            torch.backends.mkldnn.deterministic = self.prev[2]
        return False


def train_nets(fit: list[tuple[str, np.ndarray]], X_test: np.ndarray | None, cfg: TrainConfig, log=print) -> dict:
    """Self-supervised training in rounds; fit = [(table, X)] in a fixed order (train, pool).

    Round 0 treats all images as upright; every further round puts the confidently rotated ones (orientation ≠ 0, confidence
    from cfg.sure_conf on, from the prediction of the network that did not see the row) into their predicted orientation."""
    with _TorchState(cfg.threads):
        sizes = [len(x) for _, x in fit]
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        X_fit = np.concatenate([x for _, x in fit])
        n_fit = len(X_fit)
        fold = fold_assignment(cfg.seed, cfg.n_folds, n_fit)
        base = np.zeros(n_fit, np.int64)
        history = []
        nets: list = []
        P_fit = np.zeros((n_fit, 4))
        for rd in range(cfg.rounds):
            nets, P_fit, heldout = [], np.zeros((n_fit, 4)), []
            t0 = time.time()
            for f in range(cfg.n_folds):
                tr = np.flatnonzero(fold != f)
                net = train_fold(X_fit[tr], base[tr], cfg, fold_seed(cfg.seed, rd, f), log, tag=f"round {rd} fold {f}")
                nets.append(net)
                correct, n_va, confusion = 0, 0, np.zeros((4, 4), np.int64)
                for (tbl, _), a, b in zip(fit, offsets[:-1], offsets[1:]):
                    idx = a + np.flatnonzero(fold[a:b] == f)
                    P, d = predict_tta(net, X_fit[idx], cfg.predict_batch, base=base[idx])
                    P_fit[idx] = P
                    correct, n_va, confusion = correct + d["correct"], n_va + len(idx), confusion + d["confusion"]
                heldout.append({"fold": f, "train_rows": int(len(tr)), "heldout_rows": int(n_va),
                                "heldout_acc_4way": correct / max(1, 4 * n_va), "confusion": confusion.tolist()})
                log(f"  round {rd} fold {f}: self-supervision, 4 orientations, held out {heldout[-1]['heldout_acc_4way']:.4f}; "
                    f"{time.time() - t0:.0f} s")
            pred, conf = P_fit.argmax(1), P_fit.max(1)
            sure = (pred != 0) & (conf >= cfg.sure_conf)
            entry = {"round": rd, "folds": heldout, "seconds": round(time.time() - t0, 1), "tables": {}}
            for (tbl, _), a, b in zip(fit, offsets[:-1], offsets[1:]):
                entry["tables"][tbl] = {"rows": int(b - a), "pred_share": {str(k): float((pred[a:b] == k).mean()) for k in range(4)},
                                        "sure_rotated": int(sure[a:b].sum())}
            history.append(entry)
            log(f"round {rd}: confidently rotated (from {cfg.sure_conf}) among the training rows {int(sure.sum())} of {n_fit}")
            base = np.where(sure, pred, 0)
        P_test = ensemble_predict(nets, X_test, cfg.predict_batch) if X_test is not None and len(X_test) else None
    return {"nets": nets, "fold": fold, "offsets": offsets, "P_fit": P_fit, "P_test": P_test, "history": history}


# ----------------------------------------------------------------------------- weights
def state_of(nets) -> dict:
    return {f"fold{f}.{k}": v.detach().clone().contiguous() for f, net in enumerate(nets) for k, v in net.state_dict().items()}


def tensor_digest(state: dict) -> str:
    """SHA-256 over name, type, shape and content of all tensors (independent of the file format)."""
    h = hashlib.sha256()
    for k in sorted(state):
        a = state[k].detach().cpu().contiguous().numpy()
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def meta_path(weights: Path) -> Path:
    return Path(weights).with_suffix(".json")


def save_weights(path: Path, nets) -> dict:
    torch = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = state_of(nets)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    n_params = sum(p.numel() for p in nets[0].parameters())
    return {"weights_file": path.name, "weights_sha256": sha256_file(path), "tensor_sha256": tensor_digest(state),
            "n_nets": len(nets), "n_params_per_net": int(n_params), "bytes": path.stat().st_size}


def load_nets(path: Path, meta: dict | None = None) -> tuple[list, dict]:
    """Networks from the weights file; checks file and tensor checksums against the description (orient_cnn.json)."""
    torch = _torch()
    path = Path(path)
    meta = json.loads(meta_path(path).read_text()) if meta is None else meta
    if meta.get("format") != FORMAT:
        raise SystemExit(f"{meta_path(path)}: format {meta.get('format')!r}, expected {FORMAT!r}")
    if sha256_file(path) != meta["weights"]["weights_sha256"]:
        raise SystemExit(f"{path}: SHA-256 of the weights file differs from {meta_path(path).name}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if tensor_digest(state) != meta["weights"]["tensor_sha256"]:
        raise SystemExit(f"{path}: tensor checksum differs from {meta_path(path).name}")
    cfg = TrainConfig.from_dict(meta["config"])
    nets = []
    for f in range(meta["weights"]["n_nets"]):
        net = make_net(cfg)
        pre = f"fold{f}."
        net.load_state_dict({k[len(pre):]: v for k, v in state.items() if k.startswith(pre)}, strict=True)
        net.eval()
        nets.append(net)
    return nets, meta


# ----------------------------------------------------------------------------- predictions → lists
def pred_frame(table: str, files: np.ndarray, P: np.ndarray) -> pd.DataFrame:
    """Orientation per photo in the schema of i5_rot_cnn_pred.parquet (folder, file, trace_id, p0 … p3, pred, conf)."""
    df = pd.DataFrame({"folder": table, "file": [str(f) for f in files]})
    df["trace_id"] = df["file"].str.replace(r"\.[A-Za-z]+$", "", regex=True)
    for k in range(4):
        df[f"p{k}"] = P[:, k]
    df["pred"] = P.argmax(1)
    df["conf"] = P.max(1)
    return df


def from_pred_frame(p: pd.DataFrame, table: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """-> (quarter turns counter-clockwise for uprighting, confidence) per row of the table (order of the CSV)."""
    p = p[p.folder == table].set_index("trace_id")
    missing = int((~df["trace_id"].isin(p.index)).sum())
    if missing:
        raise SystemExit(f"{table}: the orientation estimate is missing for {missing} rows")
    p = p.loc[df["trace_id"]]
    return ((4 - p["pred"].to_numpy()) % 4).astype(int), p["conf"].to_numpy(dtype=float)


def write_lists(out: Path, table: str, df: pd.DataFrame, turns: np.ndarray, conf: np.ndarray, min_conf: float, name: str,
                source: str) -> dict:
    """Write a rotation list (format of tools/make_orient.py): <table>_<name>.json, _rows.npy, .meta.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    flag = (turns != 0) & (conf >= min_conf)
    mapping = {str(t): int(k) for t, k, f in zip(df["trace_id"], turns, flag) if f}
    f_json = out / f"{table}_{name}.json"
    f_json.write_text(json.dumps(mapping, sort_keys=True))
    np.save(out / f"{table}_{name}_rows.npy", np.where(flag)[0].astype(np.int64))
    counts = {str(k): int(((turns == k) & flag).sum()) for k in (1, 2, 3)}
    meta = {"table": table, "rows": int(flag.sum()), "of": int(len(df)), "min_conf": min_conf, "turns_ccw": counts,
            "source": source, "sha256": hashlib.sha256(f_json.read_bytes()).hexdigest()}
    (out / f"{table}_{name}.meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def list_files(d: Path, table: str, name: str) -> tuple[Path, Path]:
    return Path(d) / f"{table}_{name}.json", Path(d) / f"{table}_{name}_rows.npy"


def write_all_lists(out: Path, data_root: Path, preds: pd.DataFrame, tables, lists, source: str, log=print) -> dict:
    from vlm import common as VC

    res = {}
    for table in tables:
        df = VC.load_table(data_root, table)
        turns, conf = from_pred_frame(preds, table, df)
        for name, thr in lists:
            meta = write_lists(out, table, df, turns, conf, float(thr), name, source)
            f_json, f_rows = list_files(out, table, name)
            res[f"{table}_{name}"] = {"rows": meta["rows"], "of": meta["of"], "min_conf": meta["min_conf"], "turns_ccw": meta["turns_ccw"],
                                      "sha256_json": meta["sha256"], "sha256_rows": sha256_file(f_rows)}
            log(f"  {table}_{name}: {meta['rows']} of {meta['of']} photos rotated {meta['turns_ccw']}; "
                f"SHA-256 json {meta['sha256'][:16]}…, rows {res[f'{table}_{name}']['sha256_rows'][:16]}…")
    return res


def environment() -> dict:
    out = {"python": platform.python_version(), "machine": platform.machine(), "numpy": np.__version__, "pandas": pd.__version__}
    try:
        torch = _torch()
        out |= {"torch": torch.__version__, "cpu_capability": torch.backends.cpu.get_cpu_capability()}
    except Exception:  # pragma: no cover
        pass
    try:
        import cv2
        import PIL
        from PIL import features

        out |= {"opencv": cv2.__version__, "pillow": PIL.__version__, "libjpeg_turbo": features.version("libjpeg_turbo")}
    except Exception:  # pragma: no cover
        pass
    return out


# ----------------------------------------------------------------------------- commands
def _parse_lists(spec: str) -> list[tuple[str, float]]:
    out = []
    for part in spec.split(","):
        name, _, thr = part.partition("=")
        out.append((name.strip(), float(thr)))
    return out


def _guard_out(out: Path, force: bool) -> None:
    if Path(out).resolve() == FROZEN_DIR.resolve() and not force:
        raise SystemExit(f"{out}: the frozen lists of the submitted files live there — choose another directory (or --force)")


def _data_root(hint):
    from vlm import common as VC

    return VC.find_data_root(hint)


def cmd_maps(a) -> int:
    root = _data_root(a.data)
    for table in a.tables.split(","):
        t0 = time.time()
        maps = compute_maps(root, table, workers=a.workers)
        save_maps(maps_path(a.out, table), maps)
        print(f"{table}: {len(maps['files'])} images, decodable {int(maps['ok'].sum())}, SHA-256 {maps_sha256(maps)[:16]}…, "
              f"{time.time() - t0:.0f} s -> {maps_path(a.out, table)}", flush=True)
    return 0


def cmd_check_maps(a) -> int:
    res = {}
    for table in a.tables.split(","):
        new = load_maps(maps_path(a.maps_dir, table))
        old = maps_from_cache(Path(a.cache), table, list(new["files"]))
        res[table] = compare_maps(new, old) | {"sha256_images": maps_sha256(new), "sha256_cache": maps_sha256(old)}
        print(f"{table}: {res[table]}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0 if all(r.get("identical") for r in res.values()) else 1


def cmd_train(a) -> int:
    weights = Path(a.weights)
    if weights.exists() and not a.force:
        raise SystemExit(f"{weights} exists — weights are not overwritten (--force)")
    _guard_out(a.out, a.force)
    root = _data_root(a.data)
    cfg = TrainConfig(seed=a.seed, threads=a.threads, epochs=a.epochs, rounds=a.rounds, n_folds=a.folds)
    t0 = time.time()
    maps = {t: load_maps(maps_path(a.maps_dir, t)) for t in ALL_TABLES}
    fit = []
    files = {}
    for t in FIT_TABLES:
        files[t], X = net_input(maps[t])
        fit.append((t, X))
    files["test"], X_test = net_input(maps["test"])
    print(f"maps: training on {sum(len(x) for _, x in fit)} ({', '.join(f'{t} {len(x)}' for t, x in fit)}), "
          f"also predicted {len(X_test)} test maps; seed {cfg.seed}, {cfg.threads} threads", flush=True)
    res = train_nets(fit, X_test, cfg, log=lambda s: print(s, flush=True))
    wmeta = save_weights(weights, res["nets"])
    fold, off = res["fold"], res["offsets"]
    fit_meta = {t: {"rows": int(len(x)), "offset": int(o), "maps_sha256": maps_sha256(maps[t]),
                    "fold_counts": np.bincount(fold[o:o + len(x)], minlength=cfg.n_folds).tolist(),
                    "fold_sha256": sha256_array(fold[o:o + len(x)])}
                for (t, x), o in zip(fit, off[:-1])}
    preds = pd.concat([pred_frame(t, files[t], res["P_fit"][o:o + len(x)]) for (t, x), o in zip(fit, off[:-1])]
                      + [pred_frame("test", files["test"], res["P_test"])], ignore_index=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / "cnn_pred.parquet", index=False)
    lists = write_all_lists(out, root, preds, ("train", "test"), _parse_lists(a.lists), str(weights))
    meta = {"format": FORMAT, "config": asdict(cfg), "weights": wmeta,
            "fit": {"tables": fit_meta, "n_fit": int(len(fold)), "fold_sha256": sha256_array(fold),
                    "unlabelled_sample": {"seed": UNL_SAMPLE_SEED, "n": UNL_SAMPLE_N}},
            "test": {"rows": int(len(X_test)), "maps_sha256": maps_sha256(maps["test"])},
            "predict": {"dtype": "float64", "batch": cfg.predict_batch, "tta": "4 rotations, mapped back",
                        "test": "mean of the fold networks of the last round", "fit": "the fold network that did not see the row"},
            "history": res["history"], "lists_at_training": lists, "environment": environment(),
            "seconds": round(time.time() - t0, 1),
            "provenance": "src/orient.py (tools/orient_net.py train); same recipe as the first orientation network of the image analysis"}
    meta_path(weights).write_text(json.dumps(meta, indent=1))
    print(f"weights: {weights} (SHA-256 {wmeta['weights_sha256'][:16]}…, tensors {wmeta['tensor_sha256'][:16]}…), "
          f"description {meta_path(weights)}; {meta['seconds']:.0f} s", flush=True)
    return 0


def predict_table(nets, meta: dict, table: str, X: np.ndarray, batch: int | None = None, dtype: str = "float64") -> np.ndarray:
    """Orientation per photo of a table as in training: test = mean of the fold networks, train/pool = the network that did not
    see the row (fold assignment recomputed from seed and row count and checked against the description)."""
    cfg = TrainConfig.from_dict(meta["config"])
    batch = cfg.predict_batch if batch is None else batch
    if table == "test":
        return ensemble_predict(nets, X, batch, dtype)
    tm = meta["fit"]["tables"][table]
    if len(X) != tm["rows"]:
        raise SystemExit(f"{table}: {len(X)} maps, {tm['rows']} at training time")
    fold = fold_assignment(cfg.seed, cfg.n_folds, meta["fit"]["n_fit"])[tm["offset"]:tm["offset"] + tm["rows"]]
    if sha256_array(fold) != tm["fold_sha256"]:
        raise SystemExit(f"{table}: fold assignment differs from the description")
    return oof_predict(nets, X, fold, batch, dtype)


def cmd_infer(a) -> int:
    _guard_out(a.out, a.force)
    root = _data_root(a.data)
    nets, meta = load_nets(Path(a.weights))
    cfg = TrainConfig.from_dict(meta["config"])
    tables = a.tables.split(",")
    frames = []
    with _TorchState(cfg.threads):
        for t in tables:
            if a.from_images:
                maps = compute_maps(root, t, workers=a.workers)
                if a.save_maps:
                    save_maps(maps_path(a.out, t), maps)
            else:
                maps = load_maps(maps_path(a.maps_dir, t))
            ref = meta["test"]["maps_sha256"] if t == "test" else meta["fit"]["tables"][t]["maps_sha256"]
            same = maps_sha256(maps) == ref
            print(f"{t}: {len(maps['files'])} images, decodable {int(maps['ok'].sum())}; maps equal to those at training time: "
                  f"{'yes' if same else 'NO'}", flush=True)
            if not same and not a.allow_new_maps:
                raise SystemExit(f"{t}: maps differ from the training maps (environment? see environment in "
                                 f"{meta_path(Path(a.weights)).name}) — compute anyway with --allow-new-maps")
            files, X = net_input(maps)
            frames.append(pred_frame(t, files, predict_table(nets, meta, t, X)))
    preds = pd.concat(frames, ignore_index=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / "cnn_pred.parquet", index=False)
    lists = write_all_lists(out, root, preds, [t for t in tables if t in ("train", "test")], _parse_lists(a.lists), str(a.weights))
    (out / "infer_summary.json").write_text(json.dumps({"weights": str(a.weights), "weights_sha256": meta["weights"]["weights_sha256"],
                                                        "from_images": bool(a.from_images), "lists": lists,
                                                        "environment": environment()}, indent=1))
    return 0


def _flags(P: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    pred = P.argmax(1)
    return (pred != 0) & (P.max(1) >= thr), pred


def cmd_robust(a) -> int:
    """How sensitive are the lists to the compute path (batch size, thread count, float32 instead of float64)? Counts only."""
    nets, meta = load_nets(Path(a.weights))
    cfg = TrainConfig.from_dict(meta["config"])
    lists = _parse_lists(a.lists)
    res = {"weights_sha256": meta["weights"]["weights_sha256"], "tables": {}}
    for t in a.tables.split(","):
        _, X = net_input(load_maps(maps_path(a.maps_dir, t)))

        def run(batch=None, threads=cfg.threads, dtype="float64"):
            with _TorchState(threads):
                return predict_table(nets, meta, t, X, batch, dtype)

        ref = run()
        conf = ref.max(1)
        gap = conf - np.sort(ref, 1)[:, -2]
        out = {"rows": int(len(X)), "top2_gap_lt_1e-6": int((gap < 1e-6).sum()),
               "margins": {name: {"within_1e-3": int((np.abs(conf - thr) < 1e-3).sum()), "within_1e-6": int((np.abs(conf - thr) < 1e-6).sum()),
                                  "min_abs": float(np.abs(conf - thr).min())} for name, thr in lists}}
        for vname, kw in (("batch_97", {"batch": 97}), ("threads_4", {"threads": 4}), ("float32", {"dtype": "float32"})):
            P = run(**kw)
            v = {"identical": bool(np.array_equal(P, ref)), "max_abs_diff": float(np.abs(P - ref).max())}
            for lname, thr in lists:
                fa, pa = _flags(ref, thr)
                fb, pb = _flags(P, thr)
                v[f"{lname}_changed_rows"] = int((fa != fb).sum() + (fa & fb & (pa != pb)).sum())
            out[vname] = v
        res["tables"][t] = out
        print(f"{t}: {out}", flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


# ----------------------------------------------------------------------------- comparison with the frozen lists
def _mapping(d: Path, table: str, name: str) -> dict:
    return {str(k): int(v) for k, v in json.loads(list_files(d, table, name)[0].read_text()).items()}


def compare_lists(new_dir: Path, old_dir: Path, table: str, name: str) -> dict:
    """Counts and checksums only: how far a new rotation list deviates from the frozen one."""
    nj, nr = list_files(new_dir, table, name)
    oj, orr = list_files(old_dir, table, name)
    new, old = _mapping(new_dir, table, name), _mapping(old_dir, table, name)
    both = new.keys() & old.keys()
    same_turn = sum(new[k] == old[k] for k in both)
    return {"sha256_json_equal": sha256_file(nj) == sha256_file(oj), "sha256_rows_equal": sha256_file(nr) == sha256_file(orr),
            "sha256_json_new": sha256_file(nj), "sha256_rows_new": sha256_file(nr),
            "sha256_json_old": sha256_file(oj), "sha256_rows_old": sha256_file(orr),
            "rows_new": len(new), "rows_old": len(old), "both": len(both), "both_same_turn": int(same_turn),
            "both_other_turn": int(len(both) - same_turn), "only_old": len(old.keys() - new.keys()), "only_new": len(new.keys() - old.keys())}


def coverage(new_dir: Path, old_dir: Path, table: str, name: str = "c80", run_list: str = "c50") -> dict:
    """Does the existing run with rotated photos (computed on the old c50 list, with its rotation) cover the new list?"""
    new, run = _mapping(new_dir, table, name), _mapping(old_dir, table, run_list)
    covered = [k for k in new if k in run and run[k] == new[k]]
    other = [k for k in new if k in run and run[k] != new[k]]
    return {"rows_new": len(new), "covered_same_turn": len(covered), "in_run_other_turn": len(other),
            "not_in_run": len(new) - len(covered) - len(other), "needs_new_rotated_rows": len(new) - len(covered)}


def v1r_map(new_dir: Path, old_dir: Path, data_root: Path, run: str, name: str = "c80") -> dict:
    """Training side: MAP@3 of v1r with the new list against the frozen one (only rows with an existing rotated measurement).

    v1r = v1 with the rows of the list measured upright (eda/a6_assemble_oof.py --mean --fill v1). Only the rows of the old c50
    list with its rotation were measured rotated (run v1rot_t4, rot_f0 … rot_f4); rows of the new list without such a
    measurement stay unchanged here (v1) — the value is a lower bound for the new list."""
    import labels
    import metric
    from vlm import common as VC

    spec = importlib.util.spec_from_file_location("a6_assemble_oof", COMP / "eda" / "a6_assemble_oof.py")
    a6 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a6)
    train = VC.load_table(data_root, "train")
    folds = VC.load_folds(len(train))
    classes = labels.load_classes(Path(data_root) / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes)
    absent = ~labels.seen_mask(train["root_code"], classes)
    oof_v1 = np.load(COMP / "artifacts" / "oof_v1.npy")
    oof_v1r = np.load(COMP / "artifacts" / "oof_v1r.npy")
    draw = "+".join(f"{run}/rot_f{k}/oof{k}" for k in range(5))
    tid = train["trace_id"].astype(str).to_numpy()
    pos = {t: i for i, t in enumerate(tid)}
    new_dir, old_dir = Path(new_dir).resolve(), Path(old_dir).resolve()
    new, run_map = _mapping(new_dir, "train", name), _mapping(old_dir, "train", "c50")
    covered = np.array(sorted(pos[k] for k in new if k in run_map and run_map[k] == new[k]), np.int64)
    work = Path(new_dir) / "compare"
    work.mkdir(parents=True, exist_ok=True)
    f_cov = work / f"train_{name}_covered_rows.npy"
    np.save(f_cov, covered)

    def assemble(rows_file: Path) -> np.ndarray:
        P = np.zeros_like(oof_v1)
        cnt = a6.mean_draws([f"{draw}@{rows_file}"], Path(data_root), None, P, folds, y, absent)
        miss = cnt == 0
        P[miss] = oof_v1[miss]
        return P

    P_old = assemble(Path(old_dir) / f"train_{name}_rows.npy")
    P_new = assemble(f_cov)
    rows = {k: metric.map3_rows_from_proba(y, P) for k, P in (("v1", oof_v1), ("v1r_frozen", oof_v1r), ("v1r_old_rebuilt", P_old), ("v1r_new", P_new))}
    return {"rows_new_list": len(new), "rows_with_rotated_measurement": int(len(covered)),
            "rows_without_rotated_measurement": int(len(new) - len(covered)),
            "rebuild_equals_frozen_oof_v1r": bool(np.array_equal(P_old, oof_v1r)),
            "max_abs_diff_rebuild": float(np.abs(P_old - oof_v1r).max()),
            "map3": {k: float(v.mean()) for k, v in rows.items()},
            "new_vs_frozen": metric.paired_bootstrap(rows["v1r_new"], rows["v1r_frozen"]),
            "new_vs_v1": metric.paired_bootstrap(rows["v1r_new"], rows["v1"]),
            "frozen_vs_v1": metric.paired_bootstrap(rows["v1r_frozen"], rows["v1"])}


def probe_agreement(new_dir: Path, old_dir: Path, data_root: Path, run: str = "orient_t4", name: str = "c80") -> dict:
    """Training side: how often does the independent orientation probe of the 9B name the same rotation as the list?"""
    from vlm import common as VC

    spec = importlib.util.spec_from_file_location("make_orient", COMP / "tools" / "make_orient.py")
    mo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mo)
    train = VC.load_table(data_root, "train")
    turns, conf = mo.from_vlm(run, "train", len(train))
    tid = train["trace_id"].astype(str).to_numpy()
    out = {}
    for label, d in (("frozen", old_dir), ("new", new_dir)):
        m = _mapping(d, "train", name)
        idx = np.array([i for i, t in enumerate(tid) if t in m], np.int64)
        lt = np.array([m[tid[i]] for i in idx], np.int64)
        agree = turns[idx] == lt
        sure = conf[idx] >= 0.8
        out[label] = {"rows": int(len(idx)), "probe_same_turn": int(agree.sum()), "probe_same_turn_share": float(agree.mean()) if len(idx) else None,
                      "probe_sure": int(sure.sum()), "probe_sure_same_turn": int((agree & sure).sum()),
                      "probe_says_upright": int((turns[idx] == 0).sum())}
    return out


def cmd_compare(a) -> int:
    root = _data_root(a.data)
    new_dir, old_dir = Path(a.new).resolve(), Path(a.old).resolve()  # absolute: a6 reads relative paths from the data folder
    res: dict = {"new": str(Path(a.new)), "old": str(Path(a.old)), "lists": {}, "coverage": {}}
    for table in a.tables.split(","):
        for name in a.names.split(","):
            r = compare_lists(new_dir, old_dir, table, name)
            res["lists"][f"{table}_{name}"] = r
            print(f"{table}_{name}: SHA equal json {r['sha256_json_equal']} / rows {r['sha256_rows_equal']}; "
                  f"new {r['rows_new']}, old {r['rows_old']}, both {r['both']} (same rotation {r['both_same_turn']}, "
                  f"other {r['both_other_turn']}), only old {r['only_old']}, only new {r['only_new']}")
        cv = coverage(new_dir, old_dir, table, "c80", "c50")
        res["coverage"][f"{table}_c80_by_old_c50_run"] = cv
        print(f"{table}: new c80 list {cv['rows_new']} rows, in the existing rotation run (old c50) with the same rotation "
              f"{cv['covered_same_turn']}, there with another rotation {cv['in_run_other_turn']}, not computed {cv['not_in_run']}")
    if a.v1r_run:
        res["train_v1r_map"] = v1r_map(new_dir, old_dir, root, a.v1r_run)
        m = res["train_v1r_map"]
        print(f"train v1r: MAP@3 {m['map3']}; new − frozen {m['new_vs_frozen']['delta']:+.4f} ± {m['new_vs_frozen']['sigma']:.4f}; "
              f"rebuild of the frozen OOF byte-identical: {m['rebuild_equals_frozen_oof_v1r']}")
    if a.probe_run:
        res["train_probe_agreement"] = probe_agreement(new_dir, old_dir, root, a.probe_run)
        print(f"train orientation probe: {res['train_probe_agreement']}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/orient_net.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("maps", help="maps from the images")
    s.add_argument("--tables", default="train,test,unlabelled")
    s.add_argument("--out", type=Path, default=DEFAULT_OUT)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--data", default=None)
    s = sub.add_parser("check-maps", help="maps against the statistics cache (counts only)")
    s.add_argument("--maps-dir", type=Path, default=DEFAULT_OUT)
    s.add_argument("--cache", type=Path, default=STATS_CACHE)
    s.add_argument("--tables", default="train,test,unlabelled")
    s.add_argument("--json", type=Path, default=None)
    s = sub.add_parser("train", help="train, save weights, write lists")
    s.add_argument("--maps-dir", type=Path, default=DEFAULT_OUT)
    s.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    s.add_argument("--out", type=Path, default=DEFAULT_OUT)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int, default=2)
    s.add_argument("--epochs", type=int, default=12)
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--folds", type=int, default=2)
    s.add_argument("--lists", default="c50=0.5,c80=0.8")
    s.add_argument("--data", default=None)
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("infer", help="inference only with saved weights → lists")
    s.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    s.add_argument("--maps-dir", type=Path, default=DEFAULT_OUT)
    s.add_argument("--from-images", action="store_true", help="compute the maps from the images instead of loading them from --maps-dir")
    s.add_argument("--save-maps", action="store_true", help="with --from-images: write the maps to --out")
    s.add_argument("--allow-new-maps", action="store_true")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--tables", default="train,test")
    s.add_argument("--out", type=Path, default=DEFAULT_OUT)
    s.add_argument("--lists", default="c50=0.5,c80=0.8")
    s.add_argument("--data", default=None)
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("robust", help="sensitivity of the lists to the compute path (counts only)")
    s.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    s.add_argument("--maps-dir", type=Path, default=DEFAULT_OUT)
    s.add_argument("--tables", default="train,test")
    s.add_argument("--lists", default="c50=0.5,c80=0.8")
    s.add_argument("--json", type=Path, default=None)
    s = sub.add_parser("compare", help="new against frozen lists (counts and checksums only)")
    s.add_argument("--new", type=Path, default=DEFAULT_OUT)
    s.add_argument("--old", type=Path, default=FROZEN_DIR)
    s.add_argument("--tables", default="train,test")
    s.add_argument("--names", default="c50,c80")
    s.add_argument("--v1r-run", default=None, help="run with rotated v1 measurements of the training rows (v1rot_t4)")
    s.add_argument("--probe-run", default=None, help="run of the orientation probe up (orient_t4)")
    s.add_argument("--json", type=Path, default=None)
    s.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    return {"maps": cmd_maps, "check-maps": cmd_check_maps, "train": cmd_train, "infer": cmd_infer, "robust": cmd_robust,
            "compare": cmd_compare}[a.cmd](a)
