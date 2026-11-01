#!/usr/bin/env python
"""i5_01: row-level image statistics, computed ONCE and cached.

Start (from the repository root):

    python -I eda/i5_01_image_stats.py            # full run (3 worker processes)
    python -I eda/i5_01_image_stats.py --bench 40 # timing on 40 training files, no writing

Output: artifacts/eda/private/i5_image_stats.parquet (one row per image file, gitignored).

Two stages:

* CHEAP (all 55,117 files): file size, SHA-1, JPEG/PNG structure (markers, quantisation tables -> quality estimate,
  sampling, metadata segments), dimensions, a decoding probe at 1/8 resolution (DCT scaling; reads the whole entropy
  stream), brightness statistics from it, uniform border bands, pixel hash, dHash 16x16 in four rotations and pHash 8x8.
* EXPENSIVE (train, test, questions completely; unlabelled: fixed random sample, see i5_common.UNL_SAMPLE_*): full
  decoding, brightness/contrast/sharpness/colour, paper and ink share, component heights, line profiles, orientation
  features, ink map 64x64.

Images with a long side > 1024 px are downscaled by area averaging to a long side of 1024 for the expensive statistics
("norm"), so that sharpness and writing height are comparable between files.

Data use: test images are only turned into statistics here; no image file is written.
EXIF: only tag names and groups, no values except the orientation number (0x0112).
"""

from __future__ import annotations

import sys
from pathlib import Path

COMP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COMP), str(COMP / "src")]

import argparse
import hashlib
import importlib.util
import io
import multiprocessing as mp
import os
import struct
import time
import warnings

_spec = importlib.util.spec_from_file_location("i5_common", Path(__file__).with_name("i5_common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["i5_common"] = C
_spec.loader.exec_module(C)

import cv2
import numpy as np
import pandas as pd
from PIL import ExifTags, Image

cv2.setNumThreads(1)
Image.MAX_IMAGE_PIXELS = 200_000_000
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

NORM_LONG = 1024

# ----------------------------------------------------------------------------- JPEG tables
STD_LUM = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.int64)
STD_CHR = np.array(
    [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
     47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32, dtype=np.int64)
ZIGZAG = np.array(
    [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,
     7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
     39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


def _ijg_tables(base: np.ndarray) -> np.ndarray:
    out = np.zeros((100, 64), dtype=np.int64)
    for q in range(1, 101):
        s = 5000 // q if q < 50 else 200 - 2 * q
        out[q - 1] = np.clip((base * s + 50) // 100, 1, 255)
    return out


LUM_TABLES = _ijg_tables(STD_LUM)
CHR_TABLES = _ijg_tables(STD_CHR)
STD_DHT_LENS = [31, 181, 31, 181]

_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _estimate_q(tab_nat: np.ndarray, tables: np.ndarray) -> tuple[int, float]:
    err = np.abs(tables - tab_nat[None, :]).mean(axis=1)
    q = int(err.argmin())
    return q + 1, float(err[q])


def _next_marker(b: bytes, pos: int) -> int:
    n = len(b)
    while True:
        pos = b.find(b"\xff", pos)
        if pos < 0 or pos + 1 >= n:
            return -1
        m = b[pos + 1]
        if m == 0x00 or 0xD0 <= m <= 0xD7:
            pos += 2
            continue
        if m == 0xFF:
            pos += 1
            continue
        return pos


def parse_jpeg(b: bytes) -> dict:
    """Marker structure of a JPEG file (without decoding)."""
    o: dict = {"soi_ok": b[:2] == b"\xff\xd8", "eoi_at_end": b[-2:] == b"\xff\xd9"}
    last_eoi = b.rfind(b"\xff\xd9")
    o["trailing_bytes"] = (len(b) - (last_eoi + 2)) if last_eoi >= 0 else -1
    segs: list[str] = []
    qt: dict[int, np.ndarray] = {}
    dht_lens: list[int] = []
    n_scans = 0
    o.update(sof_type=-1, precision=-1, ncomp=-1, subsampling="", dri=0, sof_w=-1, sof_h=-1,
             has_jfif=False, has_exif_seg=False, has_xmp=False, has_icc=False, icc_len=0, icc_md5="",
             has_com=False, com_len=0, com_md5="", has_adobe=False, adobe_transform=-1, has_mpf=False,
             has_photoshop=False, parse_ok=True, parse_note="")
    i = 2
    n = len(b)
    icc_parts: list[bytes] = []
    try:
        while i + 4 <= n:
            if b[i] != 0xFF:
                o["parse_ok"] = False
                o["parse_note"] = f"no marker at {i}"
                break
            m = b[i + 1]
            if m == 0xFF:
                i += 1
                continue
            if m == 0xD9:
                segs.append("EOI")
                break
            if m == 0xD8 or m == 0x01 or 0xD0 <= m <= 0xD7:
                i += 2
                continue
            L = struct.unpack(">H", b[i + 2:i + 4])[0]
            p = b[i + 4:i + 2 + L]
            if 0xE0 <= m <= 0xEF:
                ident = p[:40].split(b"\x00")[0][:32].decode("latin-1", "replace")
                k = m - 0xE0
                if k == 0 and ident.startswith("JFIF"):
                    o["has_jfif"] = True
                    name = "APP0:JFIF"
                elif k == 1 and ident.startswith("Exif"):
                    o["has_exif_seg"] = True
                    name = "APP1:Exif"
                elif k == 1 and "ns.adobe.com/xap" in ident:
                    o["has_xmp"] = True
                    name = "APP1:XMP"
                elif k == 2 and ident.startswith("ICC_PROFILE"):
                    o["has_icc"] = True
                    icc_parts.append(p[14:])
                    name = "APP2:ICC"
                elif k == 2 and ident.startswith("MPF"):
                    o["has_mpf"] = True
                    name = "APP2:MPF"
                elif k == 13:
                    o["has_photoshop"] = True
                    name = "APP13"
                elif k == 14 and ident.startswith("Adobe"):
                    o["has_adobe"] = True
                    o["adobe_transform"] = p[11] if len(p) > 11 else -1
                    name = "APP14:Adobe"
                else:
                    name = f"APP{k}:?"
                segs.append(name)
            elif m == 0xFE:
                o["has_com"] = True
                o["com_len"] += len(p)
                o["com_md5"] = hashlib.md5(p).hexdigest()[:12]
                segs.append("COM")
            elif m == 0xDB:
                j = 0
                while j < len(p):
                    pq, tq = p[j] >> 4, p[j] & 0x0F
                    j += 1
                    if pq == 0:
                        vals = np.frombuffer(p[j:j + 64], dtype=np.uint8).astype(np.int64)
                        j += 64
                    else:
                        vals = np.frombuffer(p[j:j + 128], dtype=">u2").astype(np.int64)
                        j += 128
                    if len(vals) == 64:
                        nat = np.zeros(64, dtype=np.int64)
                        nat[ZIGZAG] = vals
                        qt[tq] = nat
                segs.append("DQT")
            elif m in _SOF:
                o["sof_type"] = m - 0xC0
                o["precision"] = p[0]
                o["sof_h"], o["sof_w"] = struct.unpack(">HH", p[1:5])
                nc = p[5]
                o["ncomp"] = nc
                comps = [p[6 + 3 * c: 9 + 3 * c] for c in range(nc)]
                o["subsampling"] = ",".join(f"{c[1] >> 4}x{c[1] & 0x0F}" for c in comps if len(c) == 3)
                segs.append(f"SOF{m - 0xC0}")
            elif m == 0xC4:
                dht_lens.append(L)
                segs.append("DHT")
            elif m == 0xDD:
                o["dri"] = struct.unpack(">H", p[:2])[0]
                segs.append("DRI")
            elif m == 0xDA:
                n_scans += 1
                if n_scans == 1:
                    segs.append("SOS")
                nxt = _next_marker(b, i + 2 + L)
                if nxt < 0:
                    o["parse_ok"] = False
                    o["parse_note"] = "no marker after SOS"
                    break
                i = nxt
                continue
            else:
                segs.append(f"M{m:02X}")
            i += 2 + L
    except Exception as exc:  # a broken structure is counted, not swallowed
        o["parse_ok"] = False
        o["parse_note"] = f"{type(exc).__name__}"
    o["seg_sig"] = "|".join(s for s in segs if s != "EOI")
    o["n_scans"] = n_scans
    o["n_dqt_tables"] = len(qt)
    o["dht_total_len"] = int(sum(dht_lens))
    o["huff_std"] = dht_lens == STD_DHT_LENS
    if icc_parts:
        icc = b"".join(icc_parts)
        o["icc_len"] = len(icc)
        o["icc_md5"] = hashlib.md5(icc).hexdigest()[:12]
    o["q_lum"], o["q_lum_err"], o["q_chr"], o["q_chr_err"], o["qt_md5"] = -1, np.nan, -1, np.nan, ""
    if 0 in qt:
        o["q_lum"], o["q_lum_err"] = _estimate_q(qt[0], LUM_TABLES)
    if 1 in qt:
        o["q_chr"], o["q_chr_err"] = _estimate_q(qt[1], CHR_TABLES)
    if qt:
        o["qt_md5"] = hashlib.md5(b"".join(qt[k].astype(np.uint16).tobytes() for k in sorted(qt))).hexdigest()[:12]
    return o


def parse_png(b: bytes) -> dict:
    o = {"png_ok": b[:8] == b"\x89PNG\r\n\x1a\n", "png_chunks": "", "png_bitdepth": -1, "png_colortype": -1,
         "png_interlace": -1, "png_text_keys": "", "png_has_trns": False, "png_iend": False}
    chunks: list[str] = []
    keys: list[str] = []
    i = 8
    try:
        while i + 8 <= len(b):
            L = struct.unpack(">I", b[i:i + 4])[0]
            typ = b[i + 4:i + 8].decode("latin-1")
            data = b[i + 8:i + 8 + L]
            if typ == "IHDR":
                o["png_bitdepth"], o["png_colortype"], _, _, o["png_interlace"] = data[8], data[9], data[10], data[11], data[12]
            elif typ in ("tEXt", "iTXt", "zTXt"):
                keys.append(data.split(b"\x00")[0][:24].decode("latin-1", "replace"))
            elif typ == "tRNS":
                o["png_has_trns"] = True
            elif typ == "IEND":
                o["png_iend"] = True
            if not chunks or chunks[-1] != typ:
                chunks.append(typ)
            i += 12 + L
    except Exception:
        o["png_ok"] = False
    o["png_chunks"] = "|".join(chunks)
    o["png_text_keys"] = "|".join(sorted(set(keys)))
    return o


# ----------------------------------------------------------------------------- cheap stage
def _uniform_band(gray: np.ndarray, axis: int, from_end: bool, max_range: float, max_std: float) -> tuple[int, float]:
    """Number of contiguous uniform rows (axis=0) or columns (axis=1) from the border."""
    g = gray if axis == 0 else gray.T
    if from_end:
        g = g[::-1]
    n = g.shape[0]
    rng = g.max(axis=1).astype(np.float32) - g.min(axis=1).astype(np.float32)
    std = g.std(axis=1)
    ok = (rng <= max_range) & (std <= max_std)
    k = int(np.argmin(ok)) if not ok.all() else n
    if k == 0:
        return 0, float("nan")
    band = g[:k]
    # the band must also be uniform among itself (otherwise it is only a smooth area with a gradient)
    means = band.mean(axis=1)
    k2 = k
    for j in range(1, k):
        if abs(float(means[j]) - float(means[0])) > max_range:
            k2 = j
            break
    return k2, float(means[:k2].mean())


def cheap_from_small(small: np.ndarray) -> dict:
    """Statistics from the 1/8 version (RGB, uint8)."""
    o: dict = {}
    g8 = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    o["s8_w"], o["s8_h"] = int(small.shape[1]), int(small.shape[0])
    o["pix_md5"] = hashlib.md5(small.tobytes() + bytes(str(small.shape), "ascii")).hexdigest()[:16]
    gf = g8.astype(np.float32)
    qs = np.percentile(gf, [1, 5, 50, 95, 99])
    o.update(g8_mean=float(gf.mean()), g8_std=float(gf.std()), g8_p01=float(qs[0]), g8_p05=float(qs[1]),
             g8_p50=float(qs[2]), g8_p95=float(qs[3]), g8_p99=float(qs[4]))
    mx = small.max(axis=2).astype(np.float32)
    mn = small.min(axis=2).astype(np.float32)
    o["sat8_mean"] = float(((mx - mn) / np.maximum(mx, 1.0)).mean())
    m = small.reshape(-1, 3).mean(axis=0)
    o["r8_mean"], o["gch8_mean"], o["b8_mean"] = float(m[0]), float(m[1]), float(m[2])
    for name, axis, end in (("top", 0, False), ("bottom", 0, True), ("left", 1, False), ("right", 1, True)):
        k, mean = _uniform_band(g8, axis, end, max_range=6.0, max_std=1.5)
        o[f"b8_{name}"] = k
        o[f"b8_{name}_mean"] = mean
    # Wahrnehmungs-Hashes
    t64 = cv2.resize(g8, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    for k in range(4):
        r = np.ascontiguousarray(np.rot90(t64, k))
        d = cv2.resize(r, (17, 16), interpolation=cv2.INTER_AREA)
        o[f"dh{k}"] = np.packbits(d[:, 1:] > d[:, :-1]).tobytes()
    t32 = cv2.resize(g8, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(t32)[:8, :8]
    med = np.median(low.ravel()[1:])
    o["ph"] = np.packbits(low > med).tobytes()
    o["t64_std"] = float(t64.std())
    return o


# ----------------------------------------------------------------------------- expensive stage
K3 = np.ones((3, 3), np.uint8)
K7 = np.ones((7, 7), np.uint8)
K17 = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
ANGLES = np.arange(-45, 46, 3, dtype=np.float32)  # degrees, coarse skew search (rows and columns)
FINE = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)  # fine search around the coarse optimum
K_H61 = np.ones((1, 61), np.uint8)
K_V61 = np.ones((61, 1), np.uint8)
K_DV3 = np.ones((3, 1), np.uint8)
K_DH3 = np.ones((1, 3), np.uint8)


def _profile_energy(p: np.ndarray) -> float:
    """High-frequency energy of a projection profile, normalised to the square of its mean."""
    m = float(p.mean())
    if m <= 1e-6 or len(p) < 8:
        return 0.0
    d = np.diff(p)
    return float((d * d).mean() / (m * m))


def _pitch(p: np.ndarray) -> tuple[float, float]:
    """Dominant period of a profile from the autocorrelation, plus strength (0..1)."""
    n = len(p)
    if n < 40:
        return float("nan"), 0.0
    k = max(5, n // 8) | 1
    smooth = cv2.blur(p.reshape(-1, 1).astype(np.float32), (1, k)).ravel()
    x = p - smooth
    x = x - x.mean()
    v = float((x * x).sum())
    if v <= 1e-9:
        return float("nan"), 0.0
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: n // 3] / v
    neg = np.flatnonzero(ac < 0)
    start = max(4, int(neg[0]) if len(neg) else 4)
    if start >= len(ac) - 2:
        return float("nan"), 0.0
    j = start + int(np.argmax(ac[start:]))
    return float(j), float(ac[j])


def _runs(p: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    """Segments of a profile above 30 % of its 95th percentile (start, end)."""
    if len(p) < 8:
        return []
    thr = 0.3 * float(np.percentile(p, 95))
    if thr <= 0:
        return []
    d = np.diff(np.concatenate([[0], (p > thr).astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_run]


def _band_height(p: np.ndarray) -> float:
    r = _runs(p, 2)
    return float(np.median([b - a for a, b in r])) if r else float("nan")


def _align(m: np.ndarray) -> tuple[float, float, int]:
    """Spread (MAD, share of the width) of the line starts and ends over the bands of the lines of m."""
    los, his = [], []
    for a, b in _runs(m.sum(axis=1), 3):
        c = m[a:b].sum(axis=0)
        tot = float(c.sum())
        if tot < 4:
            continue
        cs = np.cumsum(c) / tot
        los.append(np.searchsorted(cs, 0.03))
        his.append(np.searchsorted(cs, 0.97))
    if len(los) < 3:
        return float("nan"), float("nan"), len(los)
    lo = np.asarray(los, np.float64) / m.shape[1]
    hi = np.asarray(his, np.float64) / m.shape[1]
    return float(np.median(np.abs(lo - np.median(lo)))), float(np.median(np.abs(hi - np.median(hi)))), len(los)


def _rot(img: np.ndarray, angle: float, interp=cv2.INTER_LINEAR) -> np.ndarray:
    if abs(angle) < 0.2:
        return img
    hh, ww = img.shape[:2]
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((ww / 2.0, hh / 2.0), float(angle), 1.0), (ww, hh), flags=interp)


def expensive(rgb: np.ndarray) -> dict:
    """Expensive statistics on the normalised version (long side <= 1024)."""
    o: dict = {}
    h0, w0 = rgb.shape[:2]
    long0 = max(h0, w0)
    gray_native = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    o["lap_var_native"] = float(cv2.Laplacian(gray_native, cv2.CV_32F, ksize=1).var())
    if long0 > NORM_LONG:
        s = NORM_LONG / long0
        rgb = cv2.resize(rgb, (max(1, round(w0 * s)), max(1, round(h0 * s))), interpolation=cv2.INTER_AREA)
    else:
        s = 1.0
    o["norm_scale"] = float(s)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    o["norm_w"], o["norm_h"] = int(w), int(h)
    npx = gray.size

    # --- brightness, contrast
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum() / npx
    lv = np.arange(256)
    mean = float((hist * lv).sum() / npx)
    std = float(np.sqrt((hist * (lv - mean) ** 2).sum() / npx))
    pct = {p: int(np.searchsorted(cdf, p / 100.0)) for p in (1, 5, 25, 50, 75, 95, 99)}
    o.update(y_mean=mean, y_std=std, y_p01=pct[1], y_p05=pct[5], y_p25=pct[25], y_p50=pct[50], y_p75=pct[75],
             y_p95=pct[95], y_p99=pct[99], frac_lt30=float(cdf[29]), frac_lt50=float(cdf[49]),
             frac_gt230=float(1 - cdf[229]), frac_gt250=float(1 - cdf[249]), frac_eq255=float(hist[255] / npx),
             frac_eq0=float(hist[0] / npx))

    # --- sharpness
    gf = gray.astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
    o["lap_var"] = float(lap.var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy) / 8.0  # grey levels per pixel
    o["tenengrad"] = float((mag * mag).mean())
    # Crete-Roffet blur measure (0 sharp .. 1 blurred)
    bver = cv2.blur(gf, (1, 9))
    bhor = cv2.blur(gf, (9, 1))
    d_fver = np.abs(gf[1:, :] - gf[:-1, :])
    d_fhor = np.abs(gf[:, 1:] - gf[:, :-1])
    d_bver = np.abs(bver[1:, :] - bver[:-1, :])
    d_bhor = np.abs(bhor[:, 1:] - bhor[:, :-1])
    s_fver, s_fhor = float(d_fver.sum()), float(d_fhor.sum())
    s_vver = float(np.maximum(0, d_fver - d_bver).sum())
    s_vhor = float(np.maximum(0, d_fhor - d_bhor).sum())
    b_ver = (s_fver - s_vver) / s_fver if s_fver > 0 else float("nan")
    b_hor = (s_fhor - s_vhor) / s_fhor if s_fhor > 0 else float("nan")
    o["crete_blur"] = float(np.nanmax([b_ver, b_hor])) if not (np.isnan(b_ver) and np.isnan(b_hor)) else float("nan")

    # --- colour (on every second pixel)
    sub = rgb[::2, ::2].astype(np.float32)
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    o["colorfulness"] = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    o["sat_mean"] = float(sat.mean())
    o["sat_frac_gt30"] = float((sat > 0.30).mean())
    o["mean_r"], o["mean_g"], o["mean_b"] = float(r.mean()), float(g.mean()), float(b.mean())
    o["chan_maxdiff_p99"] = float(np.percentile(mx - mn, 99))

    # --- paper (bright contiguous area) at 1/4 resolution
    qw, qh = max(8, w // 4), max(8, h // 4)
    small = cv2.resize(gray, (qw, qh), interpolation=cv2.INTER_AREA)
    blur = cv2.medianBlur(small, 9) if min(qw, qh) >= 9 else small
    ot, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bmask = bright > 0
    fb = float(bmask.mean())
    tot_var = float(blur.astype(np.float32).var())
    if 0 < fb < 1 and tot_var > 0:
        m1 = float(blur[bmask].mean())
        m0 = float(blur[~bmask].mean())
        eta = fb * (1 - fb) * (m1 - m0) ** 2 / tot_var
    else:
        m1 = m0 = float(blur.mean())
        eta = 0.0
    o.update(otsu_t=float(ot), otsu_m0=m0, otsu_m1=m1, otsu_eta=float(eta), bright_frac=fb)
    paper_small = np.ones_like(bmask)
    o["paper_cc_frac"], o["paper_touch_sides"] = 1.0, 4
    if bmask.any():
        ncc, lab, stats, _ = cv2.connectedComponentsWithStats(bmask.astype(np.uint8), connectivity=8)
        if ncc > 1:
            j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            comp = lab == j
            o["paper_cc_frac"] = float(comp.mean())
            x, y, cw, ch = (int(stats[j, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            o["paper_touch_sides"] = int(x == 0) + int(y == 0) + int(x + cw == qw) + int(y + ch == qh)
            o["paper_bbox_frac"] = float(cw * ch / (qw * qh))
            paper_small = comp
    o["rel_bright_frac"] = float((small > 0.7 * pct[95]).mean())
    # real paper edge? gradient of the smoothed version on the border of the bright area
    bm8 = bmask.astype(np.uint8)
    edge = (cv2.dilate(bm8, K3) > 0) & (cv2.erode(bm8, K3) == 0)
    edge[0, :] = edge[-1, :] = False
    edge[:, 0] = edge[:, -1] = False
    if edge.any():
        sgx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        sgy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        smag = cv2.magnitude(sgx, sgy) / 8.0
        o["otsu_edge_grad"] = float(np.median(smag[edge]))
        o["otsu_edge_len"] = float(edge.sum() / (qw + qh))
    else:
        o["otsu_edge_grad"], o["otsu_edge_len"] = 0.0, 0.0
    hs = cv2.blur(np.bincount(blur.ravel(), minlength=256).astype(np.float32).reshape(1, -1), (9, 1)).ravel()
    it = int(ot)
    if 2 < it < 253:
        p0, p1 = int(np.argmax(hs[: it + 1])), it + 1 + int(np.argmax(hs[it + 1:]))
        valley = float(hs[p0:p1 + 1].min())
        o["hist_valley"] = float(valley / max(1e-6, min(hs[p0], hs[p1])))
    else:
        o["hist_valley"] = 1.0
    bimodal = (m1 - m0) >= 45 and o["otsu_edge_grad"] >= 10.0
    o["paper_bimodal"] = bool(bimodal)
    o["paper_frac"] = float(o["paper_cc_frac"]) if bimodal else 1.0
    if bimodal and o["paper_cc_frac"] >= 0.15:
        # close the paper area (writing lies inside), then bring it to full size
        pm = cv2.morphologyEx(paper_small.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        paper_full = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        paper_full = None

    # --- ink: black-hat (thin dark strokes), threshold by Otsu in [12, 60]
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, K17)
    t_o, _ = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_ink = float(np.clip(t_o, 12, 60))
    m_all = bh > t_ink
    o["bh_otsu"] = float(t_o)
    o["ink_frac_all"] = float(m_all.mean())
    mask = m_all & paper_full if paper_full is not None else m_all
    o["ink_frac"] = float(mask.mean())
    o["ink_frac_of_paper"] = float(mask.sum() / max(1, paper_full.sum())) if paper_full is not None else float(mask.mean())
    m8 = mask.astype(np.uint8)
    n_ink = int(m8.sum())
    if n_ink >= 50:
        bh_ink = bh[mask]
        o["ink_contrast_p50"] = float(np.percentile(bh_ink, 50))
        o["ink_contrast_p75"] = float(np.percentile(bh_ink, 75))
        ring = (cv2.dilate(m8, K3) > 0) & (cv2.erode(m8, K3) == 0)
        ring_idx = np.flatnonzero(ring.ravel())
        mag_ring = mag.ravel()[ring_idx]
        o["edge_grad_p90"] = float(np.percentile(mag_ring, 90))
        o["edge_sharp"] = float(o["edge_grad_p90"] / max(1.0, o["ink_contrast_p75"]))
        # edge directions (gradient angle mod 180 degrees, 36 bins of 5 degrees), weighted by magnitude
        ang = np.arctan2(gy.ravel()[ring_idx], gx.ravel()[ring_idx])
        bins = (np.floor(np.mod(ang, np.pi) / np.pi * 36).astype(np.int64)) % 36
        oh = np.bincount(bins, weights=mag_ring, minlength=36).astype(np.float32)
        o["ori_hist"] = (oh / max(1e-6, float(oh.sum()))).astype(np.float32).tobytes()
        # stroke width from the distance transform
        dt = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
        ridge = (dt >= cv2.dilate(dt, K3) - 1e-4) & mask
        o["stroke_w"] = float(2.0 * np.median(dt[ridge])) if ridge.any() else float("nan")
    else:
        o.update(ink_contrast_p50=float("nan"), ink_contrast_p75=float("nan"), edge_grad_p90=float("nan"),
                 edge_sharp=float("nan"), stroke_w=float("nan"))
        o["ori_hist"] = np.zeros(36, np.float32).tobytes()
    # noise (Immerkaer) outside the ink
    bg = cv2.dilate(m8, K7) == 0
    if bg.any():
        nz = np.abs(cv2.filter2D(gf, cv2.CV_32F, NOISE_K))
        o["noise_sigma"] = float(np.sqrt(np.pi / 2) * nz[bg].mean() / 6.0)
    else:
        o["noise_sigma"] = float("nan")

    o["flat_frac"] = float((lap == 0).mean())

    # --- skew: projection profiles of the ink mask; coarse (3 degrees, quarter resolution),
    #     fine (0.5 degrees, half resolution). Row profile = sum per image row, column profile = per column.
    hw, hh = max(8, w // 2), max(8, h // 2)
    m2 = cv2.resize(m8 * 255, (hw, hh), interpolation=cv2.INTER_AREA)
    m4 = cv2.resize(m8 * 255, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
    e_row = np.zeros(len(ANGLES), np.float32)
    e_col = np.zeros(len(ANGLES), np.float32)
    for ai, a in enumerate(ANGLES):
        rot = _rot(m4, float(a))
        e_row[ai] = _profile_energy(rot.sum(axis=1, dtype=np.float32))
        e_col[ai] = _profile_energy(rot.sum(axis=0, dtype=np.float32))
    o["e_row"] = e_row.tobytes()
    o["e_col"] = e_col.tobytes()
    zi = int(np.flatnonzero(ANGLES == 0)[0])
    o["e_row_0"], o["e_col_0"] = float(e_row[zi]), float(e_col[zi])
    best = {}
    for name, e, ax in (("row", e_row, 1), ("col", e_col, 0)):
        a0 = float(ANGLES[int(e.argmax())])
        fa, fe = a0, -1.0
        for d in FINE:
            val = _profile_energy(_rot(m2, a0 + float(d)).sum(axis=ax, dtype=np.float32))
            if val > fe:
                fa, fe = a0 + float(d), val
        best[name] = (fa, fe)
        o[f"e_{name}_max"], o[f"e_{name}_angle"] = fe, fa
        o[f"e_{name}_coarse_max"] = float(e.max())
    row_dominant = best["row"][1] >= best["col"][1]
    a_frame = best["row"][0] if row_dominant else best["col"][0]
    o["a_frame"] = float(a_frame)
    o["row_dominant"] = bool(row_dominant)

    # --- remove printed lines (long straight runs) in the deskewed image -> writing mask
    mr = (_rot(m8 * 255, a_frame) > 127).astype(np.uint8)
    hl = cv2.morphologyEx(cv2.dilate(mr, K_DV3), cv2.MORPH_OPEN, K_H61) & mr
    vl = cv2.morphologyEx(cv2.dilate(mr, K_DH3), cv2.MORPH_OPEN, K_V61) & mr
    n_mr = max(1, int(mr.sum()))
    o["hline_frac"] = float(hl.sum() / n_mr)
    o["vline_frac"] = float(vl.sum() / n_mr)
    hp = hl.sum(axis=1).astype(np.float32)
    vp = vl.sum(axis=0).astype(np.float32)
    hrun = [(a, b) for a, b in zip(*[np.flatnonzero(np.diff(np.concatenate([[0], (hp > 0.25 * w).astype(np.int8), [0]])) == s_) for s_ in (1, -1)])]
    vrun = [(a, b) for a, b in zip(*[np.flatnonzero(np.diff(np.concatenate([[0], (vp > 0.25 * h).astype(np.int8), [0]])) == s_) for s_ in (1, -1)])]
    o["n_hlines"], o["n_vlines"] = len(hrun), len(vrun)
    if len(hrun) >= 3:
        o["hline_pitch"] = float(np.median(np.diff([0.5 * (a + b) for a, b in hrun])))
    if len(vrun) >= 3:
        o["vline_pitch"] = float(np.median(np.diff([0.5 * (a + b) for a, b in vrun])))
    lines = cv2.dilate(hl | vl, K3)
    wr = mr & (1 - lines)
    o["writing_frac"] = float(wr.mean())
    o["writing_of_ink"] = float(wr.sum() / n_mr)

    # --- connected components of the writing mask (character-like)
    ncc, lab, stats, cent = cv2.connectedComponentsWithStats(wr, connectivity=8)
    o["cc_n_all"] = int(ncc - 1)
    o["cc_n_keep"] = 0
    if ncc > 1:
        cw = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
        ch = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        ca = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        big = np.maximum(cw, ch)
        sml = np.minimum(cw, ch)
        keep = (big >= 8) & (big <= 160) & (sml >= 3) & (ca >= 12) & (big / np.maximum(sml, 1) <= 3.5)
        o["cc_n_keep"] = int(keep.sum())
        o["cc_big_frac_area"] = float(ca[big > 160].sum() / max(1.0, ca.sum()))
        if keep.sum() >= 5:
            q = np.percentile(big[keep], [10, 25, 50, 75, 90])
            o.update(cc_size_p10=float(q[0]), cc_size_p25=float(q[1]), cc_size_p50=float(q[2]),
                     cc_size_p75=float(q[3]), cc_size_p90=float(q[4]))
            o["cc_h_p50"] = float(np.median(ch[keep]))
            o["cc_w_p50"] = float(np.median(cw[keep]))
            o["cc_frac_tall"] = float((ch[keep] > cw[keep]).mean())
            o["cc_frac_wide"] = float((cw[keep] > ch[keep]).mean())
            o["cc_loghw_mean"] = float(np.log(ch[keep] / cw[keep]).mean())
            o["cc_size_hist"] = np.bincount(np.clip(big[keep].astype(np.int64), 0, 160), minlength=161).astype(np.uint16).tobytes()
            # area-weighted median of the character size
            order = np.argsort(big[keep])
            cs = np.cumsum(ca[keep][order])
            o["cc_size_wmed"] = float(big[keep][order][int(np.searchsorted(cs, 0.5 * cs[-1]))])
    if wr.any():
        dt = cv2.distanceTransform(wr, cv2.DIST_L2, 3)
        ridge = (dt >= cv2.dilate(dt, K3) - 1e-4) & (wr > 0)
        o["stroke_w_writing"] = float(2.0 * np.median(dt[ridge])) if ridge.any() else float("nan")

    # --- line structure of the writing (deskewed, half resolution): energy, period, band height, alignment
    w2 = cv2.resize(wr * 255, (hw, hh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    pr, pc = w2.sum(axis=1), w2.sum(axis=0)
    o["ew_row"], o["ew_col"] = _profile_energy(pr), _profile_energy(pc)
    for name, p in (("row", pr), ("col", pc)):
        pitch, strength = _pitch(p)
        o[f"wpitch_{name}"], o[f"wpitch_{name}_ac"] = pitch * 2.0, strength  # pixels of the normalised version
        o[f"wband_{name}"] = _band_height(p) * 2.0
    lo, hi, nb = _align(w2)
    o["al_row_start"], o["al_row_end"], o["al_row_n"] = lo, hi, nb
    lo, hi, nb = _align(np.ascontiguousarray(w2.T))
    o["al_col_start"], o["al_col_end"], o["al_col_n"] = lo, hi, nb
    # the same with the full ink mask (including printed lines), not deskewed
    f2 = _rot(m2, a_frame)
    f2r, f2c = f2.sum(axis=1, dtype=np.float32) / 255.0, f2.sum(axis=0, dtype=np.float32) / 255.0
    for name, p in (("row", f2r), ("col", f2c)):
        pitch, strength = _pitch(p)
        o[f"pitch_{name}"], o[f"pitch_{name}_ac"] = pitch * 2.0, strength
        o[f"band_{name}"] = _band_height(p) * 2.0

    # --- ink distribution (centroid, quantiles, asymmetry) and ink maps 64x64 (not deskewed)
    wr0 = (m8 & (1 - (_rot(lines * 255, -a_frame) > 127).astype(np.uint8))) if abs(a_frame) >= 0.2 else wr
    for tag, mm in (("ink", m8), ("wr", wr0)):
        if int(mm.sum()) < 50:
            continue
        for name, mass in (("x", mm.sum(axis=0).astype(np.float64)), ("y", mm.sum(axis=1).astype(np.float64))):
            cs = mass.cumsum() / mass.sum()
            q05, q50, q95 = (float(np.searchsorted(cs, q) / max(1, len(mass) - 1)) for q in (0.05, 0.5, 0.95))
            o[f"{tag}_{name}_q05"], o[f"{tag}_{name}_q50"], o[f"{tag}_{name}_q95"] = q05, q50, q95
            o[f"{tag}_{name}_asym"] = float((q95 + q05 - 2 * q50) / max(1e-6, q95 - q05))
    o["ink_map"] = cv2.resize(m8 * 255, (64, 64), interpolation=cv2.INTER_AREA).astype(np.uint8).tobytes()
    o["wr_map"] = cv2.resize(wr0 * 255, (64, 64), interpolation=cv2.INTER_AREA).astype(np.uint8).tobytes()

    # --- uniform border bands and image-wide straight edges at full (normalised) resolution
    for name, axis, end in (("top", 0, False), ("bottom", 0, True), ("left", 1, False), ("right", 1, True)):
        k, mval = _uniform_band(gray, axis, end, max_range=12.0, max_std=3.0)
        o[f"fb_{name}"] = k
        o[f"fb_{name}_mean"] = mval
    gi = gray.astype(np.int16)
    if h > 4 and w > 4:
        fr = (np.abs(gi[1:, :] - gi[:-1, :]) > 25).mean(axis=1)
        fc = (np.abs(gi[:, 1:] - gi[:, :-1]) > 25).mean(axis=0)
        jr, jc = int(fr.argmax()), int(fc.argmax())
        o["edge_row_max"], o["edge_row_pos"] = float(fr[jr]), float(jr / (h - 1))
        o["edge_col_max"], o["edge_col_pos"] = float(fc[jc]), float(jc / (w - 1))
    return o


# ----------------------------------------------------------------------------- per file
def process(job: tuple[str, str, bool]) -> dict:
    folder, fname, full = job
    path = C.IMG / folder / fname
    o: dict = {"folder": folder, "file": fname, "full": bool(full)}
    try:
        b = path.read_bytes()
    except Exception as exc:
        o["read_err"] = type(exc).__name__
        return o
    o["file_size"] = len(b)
    o["sha1"] = hashlib.sha1(b).hexdigest()
    is_jpeg = b[:2] == b"\xff\xd8"
    is_png = b[:8] == b"\x89PNG\r\n\x1a\n"
    o["magic"] = "JPEG" if is_jpeg else "PNG" if is_png else "?"
    if is_jpeg:
        o.update(parse_jpeg(b))
    elif is_png:
        o.update(parse_png(b))
    # --- PIL header and metadata
    try:
        im = Image.open(io.BytesIO(b))
        w, h = im.size
        o.update(width=int(w), height=int(h), mode=im.mode, pil_format=im.format, header_ok=True)
        info = im.info
        o["info_keys"] = "|".join(sorted(str(k) for k in info.keys()))
        dens = info.get("jfif_density")
        if dens:
            o["jfif_dx"], o["jfif_dy"], o["jfif_unit"] = int(dens[0]), int(dens[1]), int(info.get("jfif_unit", -1))
        o["progressive"] = bool(info.get("progressive", False) or info.get("progression", False))
        try:
            ex = im.getexif()
            names = []
            if len(ex):
                names = [ExifTags.TAGS.get(k, f"0x{k:04X}") for k in ex.keys()]
                o["exif_orientation"] = int(ex.get(0x0112, -1)) if isinstance(ex.get(0x0112, -1), int) else -1
            o["exif_n_ifd0"] = len(names)
            o["exif_tag_names"] = "|".join(sorted(names))
            o["exif_n_exififd"] = len(ex.get_ifd(0x8769)) if len(ex) else 0
            o["exif_n_gps"] = len(ex.get_ifd(0x8825)) if len(ex) else 0
        except Exception:
            o["exif_n_ifd0"] = -1
    except Exception as exc:
        o["header_ok"] = False
        o["header_err"] = type(exc).__name__
        return o
    # --- decoding probe at 1/8 resolution
    try:
        im8 = Image.open(io.BytesIO(b))
        if is_jpeg:
            im8.draft("RGB", (max(1, w // 8), max(1, h // 8)))
            small = np.asarray(im8.convert("RGB"))
        else:
            rgba = im8.convert("RGBA")
            bgw = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            arr = np.asarray(Image.alpha_composite(bgw, rgba).convert("RGB"))
            small = cv2.resize(arr, (max(1, -(-w // 8)), max(1, -(-h // 8))), interpolation=cv2.INTER_AREA)
        o["decode8_ok"] = True
        o.update(cheap_from_small(np.ascontiguousarray(small)))
    except Exception as exc:
        o["decode8_ok"] = False
        o["decode8_err"] = f"{type(exc).__name__}"
    if not full:
        return o
    # --- full decoding and expensive statistics
    try:
        imf = Image.open(io.BytesIO(b))
        if is_png or imf.mode in ("RGBA", "LA", "P"):
            rgba = imf.convert("RGBA")
            a = np.asarray(rgba)[..., 3]
            o["alpha_min"] = int(a.min())
            o["alpha_frac_lt255"] = float((a < 255).mean())
            o["alpha_frac_lt128"] = float((a < 128).mean())
            under = np.asarray(rgba)[..., :3][a < 128]
            o["alpha_under_mean"] = float(under.mean()) if under.size else float("nan")
            naive = np.asarray(imf.convert("RGB"))
            o["naive_rgb_mean"] = float(naive.mean())
            bgw = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = np.asarray(Image.alpha_composite(bgw, rgba).convert("RGB"))
            o["n_colors_64"] = int(len(np.unique((rgb // 64).reshape(-1, 3), axis=0)))
        else:
            rgb = np.asarray(imf.convert("RGB"))
        o["decode_ok"] = True
        o.update(expensive(np.ascontiguousarray(rgb)))
    except Exception as exc:
        o["decode_ok"] = False
        o["decode_err"] = f"{type(exc).__name__}"
    return o


def _init_worker():
    cv2.setNumThreads(1)


def build_jobs() -> list[tuple[str, str, bool]]:
    jobs: list[tuple[str, str, bool]] = []
    for folder in ("train", "questions", "test"):
        jobs += [(folder, f, True) for f in sorted(os.listdir(C.IMG / folder))]
    sample = set(C.unl_sample_files())
    unl = sorted(os.listdir(C.IMG / "unlabelled"))
    jobs += [("unlabelled", f, True) for f in unl if f in sample]
    jobs += [("unlabelled", f, False) for f in unl if f not in sample]
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=int, default=0, help="timing only, on N training files")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--part", type=int, default=4000, help="rows per part file")
    args = ap.parse_args()

    if args.bench:
        files = sorted(os.listdir(C.IMG / "train"))[: args.bench]
        t0 = time.time()
        rows = [process(("train", f, True)) for f in files]
        dt = time.time() - t0
        t1 = time.time()
        _ = [process(("train", f, False)) for f in files]
        dt2 = time.time() - t1
        df = pd.DataFrame(rows)
        print(f"expensive: {1000 * dt / len(files):.1f} ms/file, cheap: {1000 * dt2 / len(files):.1f} ms/file")
        num = df.select_dtypes("number")
        with pd.option_context("display.max_rows", 300, "display.width", 200):
            print(num.describe().T[["count", "mean", "min", "50%", "max"]])
        print(df["seg_sig"].value_counts().head())
        print("decode_ok", df.get("decode_ok").value_counts().to_dict(), "columns:", len(df.columns))
        return

    C.PARTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    n_parts = (len(jobs) + args.part - 1) // args.part
    print(f"{len(jobs)} files, {n_parts} parts, {args.workers} worker processes", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers, initializer=_init_worker) as pool:
        for pi in range(n_parts):
            out = C.PARTS_DIR / f"part_{pi:04d}.parquet"
            if out.exists():
                continue
            chunk = jobs[pi * args.part:(pi + 1) * args.part]
            rows = pool.map(process, chunk, chunksize=16)
            df = pd.DataFrame(rows)
            tmp = out.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(out)
            el = time.time() - t0
            print(f"part {pi + 1}/{n_parts} ({len(chunk)} files, {chunk[0][0]}..{chunk[-1][0]}) after {el:.0f} s", flush=True)
    parts = [pd.read_parquet(p) for p in sorted(C.PARTS_DIR.glob("part_*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df["trace_id"] = df["file"].str.replace(r"\.[A-Za-z]+$", "", regex=True)
    tmp = C.STATS_PATH.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(C.STATS_PATH)
    print(f"written: {C.STATS_PATH} {df.shape} after {time.time() - t0:.0f} s", flush=True)
    print(df.groupby("folder")[["full"]].agg(["size", "sum"]))


if __name__ == "__main__":
    main()
