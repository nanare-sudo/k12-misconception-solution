#!/usr/bin/env python3
"""Build the test prediction of a saved blend and write it as a checked submission.

    python tools/make_submission.py --name <blend> --test <channel>=<run>/<job>/<tag>[/<probe>][+<more shards> ...] [...]
                                    [--none-test <run>/<job>/<tag>/<probe> ...] [--out <file>]

<blend> names the blend file artifacts/stack_<blend>.json written by the blend-fitting step (e.g. eda/c99_m3_deploy.py
--save, eda/c110_q397_deploy.py stack --save): it lists the channels in a fixed order and the weights fitted on all training
rows (log loss). For every model channel the test measurement is given here (several shards joined with `+`); the channel
"prior" comes from artifacts/test_prior_question_v1.npy. Every test row must be covered by every channel exactly once. The
conversion of log probabilities is the same as when the weights were fitted (remaining mass for dropped candidates, classes
without a training example stay at 0).

Several DRAWS of the same channel (every reasoning text is one draw) are separated by commas and averaged arithmetically —
per row over the draws that cover it; `@<file.npy>` restricts a draw to the table positions in the file (path relative to the
data folder), exactly like `eda/a6_assemble_oof.py --mean` on the training rows:
    --test think3=<run>/<job>/test/think@derived/think_rows/test.npy,<run2>/…/test/think@derived/think_rows/test_tight_q.npy
    --fill think3=zs27q

MARGIN ALIGNMENT (`--align <s>`, transductive — primary line and safety line with set alignment): the split is stratified by
root_code, so the class frequencies of the test set are those of the training set. Class weights bring the mean predicted mass
per class over all test rows to the training frequency (iteratively; only classes with at least `--align-min` training rows, the
others keep weight 1) and are applied with strength s (eda/c21_margin_align.py). Only the set of predictions is used, no label and
no individual row. `--suffix` appends a suffix to the names of the output files (e.g. `_ra`) so that the version without
alignment is kept.

BLANK IMAGES (`--blank-rows derived/blank/test_rows.npy`): a black photo shows no work; the only such training row carries
the meta code REL-01 (upload does not match the task), while the channels invent a flawless solution there. For the listed
test rows (frozen list — a property of the individual image, inductive) `--blank-code` is put at rank 1; the
rest of the order stays.

TASK-AGNOSTIC CHANNEL (`--proto <channel>=runs/<run>/<job>/<tag>[+…][,<more draws>]`): answers in the teacher protocol for
the TEST rows (job proto_gen, vlm/proto.py) — the same conversion as for the OOF artifact (eda/a8_proto_oof.py): the named
codes with their probabilities, the rest spread evenly; several draws are averaged; rows without a valid answer get the
uniform distribution (no effect in the log-linear mean). The answer files themselves are never inspected; only counts are
printed. If the channel's meta file (artifacts/meta_<channel>.json) carries class weights of an inductive debiasing
(eda/a8_proto_oof.py --debias), exactly these weights are applied to the test rows with a valid answer — before the floor,
as for the training rows.

MEAN CHANNEL WITH RANKING (`--rank <channel>=runs/<run>/<job>/<tag>`): for a channel that is also listed under --proto —
the exact ranking of all codes behind the notes of the same test rows (probe proto_rank, table test) is averaged
arithmetically with the named codes, 0.5 · (named + ranking), BEFORE the floor; rows without a ranking keep the named codes.
The same function as for the OOF artifact (eda/c95_rank_mean.channel). The channel's meta file (artifacts/meta_<channel>.json,
field rank_mean) requires --rank and --rank requires it; the floor of the blend must be the one of the definition (0.05).
Technical limits: ranking on at least 98 % of the test rows (`--rank-max-missing`), median candidate mass ≥ 0.9
(`--rank-min-mass`), share of NONE at the top of the ranking within the malfunction band 0.30–0.38 fixed in advance
(`--rank-none-band`; never used to choose between variants). Only counts and shares are printed.
MEAN CHANNEL WITH THE RANKING OF A FINE-TUNED MODEL (channel m3px27xm = ½(pr27 + m3px27); field rank_mean.rank_compare = "sft" in
meta_<channel>.json, eda/c133_primary_m3px27.py): the ranking of the training rows consists of fold shards (joined with +, per fold the
adapter without that fold), the ranking of the test rows comes from the final fit. It is then compared as for --rank-only
(rank_only_source_differences against EVERY shard: only model, fold and note source may differ) instead of with
eda/c95_rank_mean.recipe_differences, and the NOTE GUARD rank_mean.notes (rank_mean_notes_guard) requires the frozen training and test
notes behind which the rankings were computed. Without the field (all earlier mean channels, e.g. pr27xm) everything stays bit-identical.
SEPARATE SOURCE OF THE NAMED CODES (field rank_mean.named in meta_<channel>.json, written by an evaluation of a protocol variant that is
not part of the final candidates; only together with rank_mean.notes): the named codes come from another protocol run than the notes behind
which the ranking was computed. The note guard (rank_mean.notes) checks ranking and notes as before; in addition rank_mean_named_guard
checks the source under --proto: rank_mean.proto_sources = [named.train.source] (the OOF artifact named these codes), every test draw is
listed in named.test.source_runs, table test with all rows, and its fingerprint (proto_source_fingerprint: naming rule, generation recipe
<tag>_generation.json without GEN_FREE_KEYS, model from summary.json) equals that of the training side (named.train). Only meta, recipe and
summary files are read. Without the field (all other channels) none of this is computed, and the sources file stays bit-identical.

RANK-ONLY CHANNEL (`--rank-only <channel>=runs/<run>/<job>/<tag>`, M3): a channel that consists ONLY of the exact ranking of
all codes (probe proto_rank, e.g. the trained 27B in sft mode) — no mean with named codes as with --rank. On the test rows
with a complete, exact ranking the distribution over all codes (eda/a8_proto_oof.load_rank, the same conversion as for the
OOF artifact `a8 … --rank`), on all other rows the uniform distribution over the classes with a training example (like the
neutral rows of the OOF artifact; no effect in the log-linear mean) — counted in the log. As for every channel, the floor
comes from the blend file (eda/c99_m3_deploy.py sets 0.05) and must be > 0. The same guards as --rank
(`--rank-max-missing`, `--rank-min-mass`, `--rank-none-band`, `--rank-partial`); the channel's meta file (meta_<channel>.json,
field rank) names the ranking of the training rows, whose settings (image, beams, finish, number format; per fold job) must
match those of the test rows — only model, fold and note source may differ. If the blend file lists the channel under
`rank_only`, --rank-only is mandatory.

SEVERAL DRAWS OF A RANK-ONLY CHANNEL (`--rank-only <channel>=runs/<run1>/<job1>/<tag>,runs/<run2>/…`): the OOF artifact
averages several draws (eda/a9_m3_draws_oof.py; meta_<channel>.json: rank.source = OOF draws separated by commas,
rank.draws). Exactly one test draw per OOF draw, in THE SAME order. Every test draw passes all guards like a single one
(coverage ≥ 98 %, mass, NONE band, partial state) and is checked against ITS OOF draw: ranking settings per fold job as above
and additionally the settings of the text generation (<tag>_generation.json behind sft.sft_from: max_tokens, temperature,
system text, image, backend) — a test draw with 5,200 tokens only matches an OOF draw with 5,200 tokens. Per test row the
mean of the probabilities over the draws with an exact ranking (the same computation as a9), rows without any ranking
uniform. The blend file must have been fitted on THIS artifact (rank_only.rank_source, oof_sha256) and only on rows with ALL
draws (fit_rows ⊆ rank.rows_all_draws_file, checksum; eda/c99_m3_deploy.py --rows). A single source without a comma for a
channel with one draw runs unchanged as above.
DRAW WITH ROW LIST (primary submission A; only if meta_<channel>.json carries the field rank.row_list_draws, eda/c166_primary_m3x5h.py —
without the field every channel runs bit-identically as above): a draw j of the channel counts only on the rows of a list — on the training
rows the list row_list_draws[].train (file = SHA-256; coverage of draw j in the artifact = exactly these rows), on the test rows the list
row_list_draws[].test. The test spec of this draw reads `<shard>[+<shard>…]@<test list>` with exactly the file of the meta file (path
relative to the data folder or absolute; SHA-256 pinned); every shard complete and exact (a8.load_rank, no overlap), its settings of ranking
AND generation = those of OOF draw j (rank_only_draw_differences per shard: image, seed, temperature, top_p, top_k, tokens, beams, adapter
type …), and the coverage of the exact ranking is EXACTLY the list (not more, not less) — otherwise abort without a file. Mass and NONE band
as for every draw, on the list rows. All other draws must cover all training rows; the blend may then be fitted on all training rows (the
same structure on both sides: list rows with all draws, the other rows without draw j). Per row the same computation as a9 (sum in the
order of the draws, division by their number). `--rank-only-placeholder <channel>` (ONLY for code checks, --out outside
artifacts/submissions, never submitted) allows a foreign test ranking for the draw with row list: settings not compared, coverage ⊇ list,
restricted to the list (stand-in and the copy null point "empty list" of the build path tools/builds/primary_m3x5h.sh).

FOLD SHARDS AS SOURCE (eda/c117_q397m3_deploy.py, channel q397m3): if meta_<channel>.json names as the ranking of the
training rows ONE draw made of several fold shards joined with + (rank.source without a comma), the settings of the test
ranking are compared with EVERY shard (rank_only_source_differences); a source without + runs unchanged.
NOTE GUARD (field rank.notes in meta_<channel>.json, eda/c117_q397m3_deploy.py; without the field — all other channels —
unchanged): if a rank-only channel ranks behind FOREIGN frozen notes (q397m3: M3 behind the notes of Qwen3.5-397B), the
settings comparison says nothing about the note source (sft_from is free there). Hence additionally (rank_only_notes_guard):
every shard of the training rows ranked behind rank.notes.train.sft_from (file = checksum in the meta file = copy in the
upload folder = the one of the OOF artifact); the test ranking ranked behind rank.notes.test.sft_from, whose file (= meta
checksum = copy in the upload folder) carries table test with all rows and one note per row, comes from one of the runs
rank.notes.test.source_runs (checksums of <tag>_raw.jsonl and <tag>_meta.json of that run) and names the model
rank.notes.test.model; the blend file was fitted on THIS artifact (rank_only.<channel>.rank_source and oof_sha256). Only
checksums and meta files (aggregates) are read.
`--rank-only-placeholder <channel>` (ONLY for code checks with a foreign test ranking, e.g. tools/builds/s029_q397m3.sh with
PLACEHOLDER=1) skips the check of the test notes, requires --out outside artifacts/submissions and records "placeholder" in
the JSON file — such a file is never submitted.
OWN NONE BAND (`--rank-none-band-for <channel>=<lo>,<hi>`, default none): a malfunction band fixed in advance for ONE
rank-only channel instead of --rank-none-band — for a channel whose share of NONE at the top already lies at the edge of the
common band on the training rows (q397m3: OOF 0.310 against 0.30–0.38; band OOF ± 0.04 like the M3 band around 0.34).
Recorded in the JSON file.

FIT ROWS (`fit_rows` in the blend file, eda/c99_m3_deploy.py): if a blend was fitted on a subset of the training rows only
(the channel exists only there), the class weights of the inductive alignment also come only from the blend's predictions
for THESE rows (the target is, as always, the class frequencies of all training rows); if the blend file carries
`inductive_check.weights_sha256`, the recomputed weights must match it. Blend files without `fit_rows` stay unchanged.

ARITHMETIC SHARE (`arith_beta` in the blend file): the blend is P = (1 − β)·log-linear + β·Σ w_i·p_i with w = θ/Σθ and p_i
the channel with its floor (normalised per row over the training classes); β = 0 is the pure log-linear blend of the
earlier blend files. The same step is part of the predictions for the training rows (`oof_mixture`) from which the inductive
alignment takes its class weights. Measurement: docs/METHOD.md §7 (the evaluation script is not included).

SHARE BY CODE NOVELTY (`arith_beta_novel` in the blend file, eda/c110_q397_deploy.py): per (row, class) β_new instead of β
for codes that were never the root among the reference rows of the same task (question_id with the same topic_name; NONE
is never novel) — reference for the test rows: all training rows (code_novelty); for the training rows in oof_mixture: the
rows of the other folds (train_novelty). Only training labels and the row's task, inductive. Without the field the
computation stays unchanged (bit-identical).

INDUCTIVE ALIGNMENT (`--align-inductive <strength>`): the same computation as the margin alignment, but the class weights
come from the saved blend's predictions for the TRAINING rows (OOF artifacts of the channels, floors and weights of the blend
file) — they are fixed before inference and use nothing about the set of test rows. For the inductive line; excludes
`--align`.

SECOND STAGE (`--stage2 artifacts/stage2_s032.json`; without the switch everything stays unchanged): after the margin
alignment (mandatory, strength and class limit as in the file) a conditional logit (src/stage2.py, weights from
eda/c118_stage2.py fit) reorders the first K = 5 candidates of every test row; only afterwards REL-01 for blank images (the
note guard and the NONE band are aborting guards before that). Features of the test rows: blend channels BEFORE the floor,
the individual M3 test draws d1–d3 (channel draws_channel, in the order of the file), the frozen M3 test notes (only a
machine regex count of the codes; file = checksum in its meta file, source = the test ranking of the same draw including its
checksum), knowledge from all training rows. Checks: blend file = the one of the deployment file (checksum, channels, β).
GUARDS (aggregates only, `--stage2-report`): (a) note statistics per test draw (mean number of codes mentioned, share
mentioning the top code, distribution of the mention count 0–3 for the first five, share of word-identical notes per pair)
within the range of the three training draws ± 10 % relative — otherwise the fallback feature set without notes is used
automatically; (b) share "novel for the task" among the first five, test against OOF (reported only); (c) share of test rows
with a new top code between ½ and 2 times the OOF share of the chosen set, otherwise abort without a file. A null model
(`null` in the file, weight 0) changes no order: the file is byte-identical to the file without --stage2 (copy null point);
(c) is not enforced there.
WIDENED RULE of the note guard (only if `guard_rule` of the deployment file names `share_z_max` AND `n_ref`, e.g.
artifacts/stage2_s034.json): a statistic passes within the old band OR — shares only (top-code mention per draw, mention
distribution per level, word-identical notes per pair, each against its own training value) — with |z| ≤ share_z_max in
the two-sample binomial test (n_ref = number of training rows, checked; n_test = number of test rows); the rule only widens.
Without the fields everything stays as above, byte-identical down to the statistics file. There is no switch to override
the guard.
MAIN_GUARD MODELS (only if a model of the deployment file names the fields, e.g. artifacts/stage2_s035.json): `feature_set`
selects the feature set of the model (main_guard = main form + general-code identifiers; a fallback model must never read the
notes), `protect_none` the NONE guard (base top NONE -> the base order of the row stays; applied before guard (c), like the OOF
share of the set). Without the fields everything stays as above, byte-identical down to the statistics file.
SECOND STAGE OF THE INDUCTIVE LINE (only if the deployment file names `align_mode` "inductive", e.g.
artifacts/stage2_s036_safe.json from eda/c120_s036_safe_stage2.py fit; without the field everything stays as above,
byte-identical): requires `--align-inductive` and `--align-min` as in the file (instead of --align) and computes
apply_stage2_safe: feature set of the file (only "fallback": no notes and therefore no note guard), knowledge from all training
rows, M3 test draws = those named in `test_draws` (in the order of the file), then the NONE guard (`protect_none`; base top NONE
-> the base order of the row stays), guard (c) on the share of new top codes after the guard (½ to 2 times the OOF share,
otherwise abort without a file), (b) reported only. A null model (`null`) changes no order: the file is byte-identical to the
file without --stage2 (copy null point).
SECOND STAGE OF THE SAFETY LINE WITH SET ALIGNMENT (only if the deployment file names `align_mode` "set", e.g.
artifacts/stage2_safety_final.json from eda/c123_safety_set.py fit): requires `--align` and `--align-min` as in the file (margin
alignment as one set over the test rows, as in the primary line) and computes the same path apply_stage2_safe (feature set "fallback"
or "fallback_i1", no notes, NONE guard, guard (c)); only in this mode also the class-mass statistic (`class_mass`: L1 distance of the
mean test mass per class from the target of the alignment — training shares of the classes with at least align_min rows —, before
and after the alignment; aggregates only). Files with `align_mode` "inductive" and files without the field stay byte-identical down
to the statistics file.
INTERACTION MODELS: a main model may name the feature set main_guard_i1 (main_guard + interactions I1; checks as for main_guard, a
fallback model with this set is rejected). Only if the main model names the feature set main_guard_i1_mm (e.g.
artifacts/stage2_primary_final.json): in addition the column minor_x_mech27 — row feature mech27 from the frozen pr27 test answers
(field mech27.test of the deployment file: file and SHA-256, the check is required), final answer = answer key compared by
src/answer_match.py (without eval); statistics as shares only (field mech27 of the report).
NOTE FEATURES OF THE SAFETY LINE (only if a deployment file with align_mode "set" names the feature set fallback_i1_nd, e.g.
artifacts/stage2_safety_nd.json from eda/c129_safety_nd.py fit): apply_stage2_safe also reads the frozen M3 test notes (test_notes, as the
primary line) and the frozen training notes (train_notes, file + SHA-256), counts the mentions (m3_men, m3_last) and computes the
example similarity (src/note_sim.py; reference: all training rows of other questions). The widened note guard and the similarity guard
(band ±10 % OR |z| ≤ 3) act ONLY as an abort without a file, never as a switch; a null model only computes them. --stage2-control
copy|permute|truncate (positive control of the build script, only with this feature set): changes the test notes in memory, NEVER writes
a file and ends with exit code 5 if a guard fires (control passed), otherwise with exit code 6.
REGEX-EXACT NOTE COLUMNS OF THE SAFETY LINE (only if a deployment file with align_mode "set" names the feature set fallback_i1_ndr, e.g.
artifacts/stage2_safety_ndr.json from eda/c130_safety_ndr.py fit): as fallback_i1_nd and in addition the six regex-exact note columns R
(src/note_regex.py) from the M3 test notes d1–d3 and the frozen pr27 test notes (field r_notes.pr27_test of the deployment file: file =
SHA-256 = checksum of the meta file, table test, all rows, source as in the file). The guard of the columns R against r_reference (pr27
mention rate per row, share of rows in which pr27 mentions the top of the base form, mean last position of the M3 mentions per draw; band
±10 % OR |z| ≤ 3) acts ONLY as an abort without a file, like the other guards. Additional positive controls: pr27_shift (pr27 notes
shifted by one row), pr27_m3d1 (pr27 replaced by the M3 notes d1); truncate also shortens the pr27 notes; exit codes 5/6 as above. All
other deployment files compute and report unchanged.
EXTRA COLUMNS (only if the deployment file carries the field extra_columns, e.g. artifacts/stage2_primary_cols.json from
eda/c151_primary_cols.py fit): the main model of the primary line reads extra columns at the end of main_guard_i1_mm (sources
extra_columns.names; src/stage2_sources.py) from frozen protocol answers of the test rows (files and SHA-256 in extra_columns.test, check
mandatory; completeness, parser cross-check against proto_labels.jsonl), from the blend channels and draws of the line and from the table
and the taxonomy (reference = all 1,500 training rows). Test guard per column and source against extra_columns.guard_reference (rule fixed
in advance: band ±15 % relative OR |z| ≤ 3; valid answers per source additionally ≥ 0.97) ONLY as an abort without a file, also for the
null model; statistics in the report (field extra_columns). --stage2-xcontrol <control> (positive control, stage2_sources.CONTROLS:
q397_drop10, q397_unformed, mech_zero, q397_rotate, fac_flip, chan_rotate, topic_unknown; only with its source in extra_columns.names):
perturbs the test sources in memory, NEVER writes a file, exit code 5 = the guard fires (passed), 6 = it does not.
Without the field every deployment file computes and reports bit-identically as before.

Written are artifacts/test_<blend>.npy, artifacts/submissions/sub_<blend>.csv and artifacts/submissions/sub_<blend>.json
(sources, SHA-256, statistics). Only aggregates over the test rows are printed, plus the same statistics on the OOF rows
for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import labels  # noqa: E402
import metric  # noqa: E402
import stage2 as S2  # noqa: E402
import note_sim as NS  # noqa: E402  (example similarity of the M3 notes, feature set fallback_i1_nd only)
import note_regex as NR  # noqa: E402  (regex-exact note columns R, feature set fallback_i1_ndr only)
import answer_match as AM  # noqa: E402  (mechanical comparison final answer = answer key, without eval)
import stage2_sources as XS  # noqa: E402  (extra second-stage columns from frozen protocol answers, only with the field extra_columns)
import submission_io  # noqa: E402
from vlm import common as C  # noqa: E402
from vlm.engine import probs_from_logp  # noqa: E402

spec = importlib.util.spec_from_file_location("a2", COMP / "eda" / "a2_probe_stack.py")
a2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a2)
spec8 = importlib.util.spec_from_file_location("a8", COMP / "eda" / "a8_proto_oof.py")
a8 = importlib.util.module_from_spec(spec8)
spec8.loader.exec_module(a8)
spec95 = importlib.util.spec_from_file_location("c95_rank_mean", COMP / "eda" / "c95_rank_mean.py")
c95 = importlib.util.module_from_spec(spec95)
spec95.loader.exec_module(c95)
EPS = 1e-9


def load_shards(specs: str, n_test: int, n_classes: int, absent: np.ndarray, width: int | None = None, partial: bool = False) -> tuple[np.ndarray, list[dict]]:
    """Assemble the shards of a channel into one matrix over all test rows (raw log values, not yet normalised).

    partial: the shards may leave out test rows (cascade, --fill); double coverage remains an error."""
    w = n_classes if width is None else width
    lp = np.full((n_test, w), -np.inf)
    seen = np.zeros(n_test, dtype=np.int64)
    metas = []
    for s in specs.split("+"):
        parts = s.split("/")
        run, job, tag = parts[:3]
        probe = parts[3] if len(parts) > 3 else "diag"
        d = COMP / "artifacts" / "runs" / run / run / job
        meta = json.loads((d / f"{tag}_meta.json").read_text())
        if meta.get("table") != "test":
            raise SystemExit(f"{s}: no output for test rows (table={meta.get('table')!r})")
        idx = np.load(d / f"{tag}_idx.npy")
        arr = np.load(d / f"{tag}_{probe}.npy")
        done = d / f"{tag}_{probe}.done.npy"
        if done.is_file() and not np.load(done).all():
            raise SystemExit(f"{s}: measurement incomplete ({int((~np.load(done)).sum())} rows missing)")
        if arr.shape != (len(idx), w):
            raise SystemExit(f"{s}: shape {arr.shape} does not match {len(idx)} rows × {w}")
        lp[idx] = arr
        seen[idx] += 1
        metas.append({"spec": s, "rows": int(len(idx)), "dtype": meta.get("dtype"), "quant": meta.get("quant"), "adapter": meta.get("adapter"),
                      "recipe": meta.get("recipe")})
    if (seen > 1).any() or (not partial and not (seen == 1).all()):
        raise SystemExit(f"{specs}: coverage incomplete — {int((seen == 0).sum())} test rows missing, {int((seen > 1).sum())} covered twice")
    return lp, metas


def load_draws(specs: str, n_test: int, n_classes: int, absent: np.ndarray, root: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Average several draws of a channel arithmetically (separated by commas, each with an optional @<rows.npy>).

    Returns: probabilities (rows without a draw stay 0), number of draws per row, sources."""
    S = np.zeros((n_test, n_classes))
    cnt = np.zeros(n_test, dtype=np.int64)
    metas: list[dict] = []
    for draw in specs.split(","):
        dspec, _, rows_file = draw.partition("@")
        lp, m = load_shards(dspec, n_test, n_classes, absent, partial=True)
        have = np.isfinite(lp).any(1)
        if rows_file:
            keep = np.zeros(n_test, dtype=bool)
            keep[np.load(rows_file if Path(rows_file).is_absolute() else root / rows_file)] = True
            have &= keep
        lp[~have] = 0.0
        S[have] += probs_from_logp(lp, absent=absent)[have]
        cnt[have] += 1
        metas += m + [{"draw": dspec, "rows_file": rows_file or None, "rows": int(have.sum())}]
    cov = cnt > 0
    S[cov] /= cnt[cov, None]
    return S, cnt, metas


def rank_mean_guard(name: str, stack: dict, rank_spec: str | None, n_draws: int, art_dir: Path | None = None) -> dict | None:
    """Check the channel form against the OOF artifact: a mean channel with ranking (meta_<channel>.json, field rank_mean,
    eda/c95_rank_mean.py) requires --rank, --rank requires such a channel; the floor of the blend and the number of draws of
    the named codes as for the training rows. -> definition from the meta file (or None)."""
    meta_file = (COMP / "artifacts" if art_dir is None else art_dir) / f"meta_{name}.json"
    ch_meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
    rm = ch_meta.get("rank_mean")
    if rm and rank_spec is None:
        raise SystemExit(f"channel {name}: the OOF artifact is a mean channel with ranking ({meta_file.name}, rank_mean) — --rank {name}=runs/… missing")
    if rank_spec is not None and not rm:
        raise SystemExit(f"--rank {name}: {meta_file.name} does not describe a mean channel (field rank_mean missing) — the weights would belong to a different channel form")
    if rm:
        fl = float((stack.get("floor") or {}).get(name, 0.0))
        if fl != float(rm["floor"]):
            raise SystemExit(f"channel {name}: floor of the blend {fl:g}, the definition of the mean channel requires {float(rm['floor']):g} (eda/c95_rank_mean.py)")
        if ch_meta.get("class_weights") is not None:
            raise SystemExit(f"channel {name}: a mean channel with debiasing is not defined")
        if n_draws != len(rm["proto_sources"]):
            raise SystemExit(f"channel {name}: {n_draws} draw(s) of the named codes, the OOF artifact has {len(rm['proto_sources'])}")
    return rm


def proto_channel(name: str, draws: list[str], rank_spec: str | None, n_test: int, classes: list[str], seen_cls: np.ndarray,
                  neutral=None, rank_partial: bool = False, max_missing: float = 0.02, min_mass: float = 0.9,
                  none_band: tuple[float, float] = (0.30, 0.38), ref_rank: str | None = None,
                  ref_compare: str = "c95") -> tuple[np.ndarray, np.ndarray, dict]:
    """Task-agnostic channel on the test rows: named codes (a8.mean_of_draws) or, with `rank_spec`, the mean channel of named
    codes and exact ranking (eda/c95_rank_mean.channel — the same function as for the OOF artifact). ref_rank: ranking of the
    training rows of the OOF artifact — both runs must have the same settings (except for the notes). ref_compare: "c95" (default, as
    before: eda/c95_rank_mean.recipe_differences) or "sft" (fold shards of a fine-tuned model against its final fit,
    rank_only_source_differences against every shard). -> (P, number of valid answers per row, ranking statistics; aggregates only)."""
    if ref_compare not in ("c95", "sft"):
        raise SystemExit(f"channel {name}: unknown comparison form of the ranking {ref_compare!r} (c95 | sft)")
    if rank_spec is None:
        P, cnt = a8.mean_of_draws(draws, n_test, classes, seen_cls, table="test", neutral=neutral)
        return P, cnt, {}
    missing = c95.rank_files_missing(rank_spec)
    if missing:
        raise SystemExit(f"channel {name}: ranking {rank_spec} — missing {', '.join(missing)} (run not finished or not fetched)")
    if ref_rank is not None:
        diff = rank_only_source_differences(rank_spec, ref_rank) if ref_compare == "sft" else c95.recipe_differences(rank_spec, ref_rank)
        if diff:
            raise SystemExit(f"channel {name}: ranking of the test rows computed differently from that of the training rows ({ref_rank}): {diff}")
    P, cnt, hr, rs = c95.channel(draws, rank_spec, n_test, classes, seen_cls, table="test", neutral=neutral, rank_partial=rank_partial)
    info = {k: rs[k] for k in ("rows_probe", "rows_done", "rows_exact", "rows_due", "rows_rank", "rows_named_only", "rows_uniform_only",
                               "rows_without_answer", "rows_rank_without_answer", "share_rank", "mass_median", "mass_p05", "mass_below_half",
                               "none_top_rank", "none_top_named", "none_top_mean", "same_top_named_rank", "idx_sha256", "partial")}
    print(f"  channel {name}: MEAN of named codes and exact ranking ({rank_spec}): ranking on {info['rows_rank']} of {info['rows_due']} test rows "
          f"({info['share_rank']:.4f}; probe {info['rows_done']}/{info['rows_probe']} computed, {info['rows_exact']} exact), {info['rows_named_only']} rows named codes only, "
          f"{info['rows_uniform_only']} with neither (uniform)")
    print(f"  channel {name}: candidate mass median {info['mass_median']:.3f}, 5 % quantile {info['mass_p05']:.3f}, below 0.5 in {info['mass_below_half']:.3f} of the rows; "
          f"NONE at the top: ranking {info['none_top_rank']:.3f}, named {info['none_top_named']:.3f}, mean {info['none_top_mean']:.3f}; "
          f"same top named/ranking {info['same_top_named_rank']:.3f}")
    if info["share_rank"] < 1.0 - max_missing:
        raise SystemExit(f"channel {name}: ranking only on {info['share_rank']:.2%} of the test rows (at least {1.0 - max_missing:.0%} required)")
    if info["mass_median"] < min_mass:
        raise SystemExit(f"channel {name}: median candidate mass {info['mass_median']:.3f} below {min_mass:g} — forced answer start is unnatural")
    if not (none_band[0] <= info["none_top_rank"] <= none_band[1]):
        raise SystemExit(f"channel {name}: share of NONE at the top of the ranking {info['none_top_rank']:.3f} outside the malfunction band fixed in advance "
                         f"{none_band[0]:.2f}–{none_band[1]:.2f} — check the run, do not deploy")
    return P, cnt, info


# The two definition strings (RANK_ONLY_DEFINITION, RANK_ONLY_DRAWS_DEFINITION) are recorded in the sidecar JSON of every
# submission.
RANK_ONLY_DEFINITION = ("exact distribution over all codes (probe proto_rank) on the rows with a complete, exact ranking; other "
                        "rows uniform over the classes with a training example; floor from the blend file "
                        "(OOF: eda/a8_proto_oof.py --rank; blend: eda/c99_m3_deploy.py)")
RANK_ONLY_FREE_KEYS = ("model", "fold", "proto_from")  # may differ between fold models (training) and the full fit (test)


def rank_only_guard(name: str, stack: dict, art_dir: Path | None = None) -> str:
    """Check the channel form against the OOF artifact: a RANK-ONLY channel (meta_<channel>.json, field rank from
    eda/a8_proto_oof.py --rank) — no mean channel, no debiasing, no mixed form with named codes; floor of the blend > 0 and
    equal to the one the blend file records for the channel. -> source of the ranking of the training rows."""
    meta_file = (COMP / "artifacts" if art_dir is None else art_dir) / f"meta_{name}.json"
    if not meta_file.is_file():
        raise SystemExit(f"--rank-only {name}: {meta_file.name} missing — the channel's OOF artifact is not described")
    meta = json.loads(meta_file.read_text())
    rk = meta.get("rank") or {}
    if not rk.get("source"):
        raise SystemExit(f"--rank-only {name}: {meta_file.name} does not describe a ranking channel (field rank.source missing; eda/a8_proto_oof.py --rank)")
    if meta.get("rank_mean"):
        raise SystemExit(f"--rank-only {name}: {meta_file.name} describes a mean channel (rank_mean) — use --proto and --rank for it")
    if meta.get("class_weights") is not None:
        raise SystemExit(f"--rank-only {name}: a ranking channel with debiasing is not defined")
    if int(rk.get("rows_named_only") or 0):
        raise SystemExit(f"--rank-only {name}: in the OOF artifact {int(rk['rows_named_only'])} rows carry named codes instead of the ranking (mixed form)")
    fl = float((stack.get("floor") or {}).get(name, 0.0))
    if fl <= 0:
        raise SystemExit(f"--rank-only {name}: the blend sets no floor for the channel — an exact channel without a floor acts as a veto")
    want = ((stack.get("rank_only") or {}).get(name) or {}).get("floor")
    if want is not None and float(want) != fl:
        raise SystemExit(f"--rank-only {name}: floor of the blend {fl:g}, the blend file records {float(want):g} for the channel")
    return str(rk["source"])


def _rank_meta(spec: str) -> dict:
    bits = spec.split("/")
    if len(bits) != 4 or bits[0] != "runs":
        raise SystemExit(f"{spec}: expected runs/<run>/<job>/<tag>")
    _, run, job, tag = bits
    f = a8.COMP / "artifacts" / "runs" / run / run / job / f"{tag}_meta.json"
    if not f.is_file():
        raise SystemExit(f"{spec}: {f.name} missing (run not finished or not fetched)")
    return json.loads(f.read_text())


def rank_only_reference(ref_source: str) -> list[dict]:
    """Meta files of the ranking(s) behind the OOF artifact: a combined output (vlm/sft.combine_rank or combine_rank_runs,
    field combined_from) is resolved into its fold jobs — each carries its own settings."""
    meta = _rank_meta(ref_source)
    if not meta.get("combined_from"):
        return [meta]
    _, run, _, tag = ref_source.split("/")
    return [_rank_meta(f"runs/{s.get('run', run)}/{s['job']}/{tag}") for s in meta["combined_from"]]


def rank_only_recipe_differences(spec: str, ref_source: str) -> dict:
    """Settings of the test-row ranking against those of the training rows (every fold job of the OOF artifact): differences
    except model (fold model vs. full fit), fold and note source (`proto_from`, `sft.sft_from`). Empty = the same computation
    (image, beams, finish, number format, probe) with a different model."""
    def comparable(m: dict) -> dict:
        out = {f"recipe.{k}": v for k, v in (m.get("recipe") or {}).items() if k not in RANK_ONLY_FREE_KEYS}
        out.update({f"sft.{k}": v for k, v in (m.get("sft") or {}).items() if k != "sft_from"})
        out.update({k: m.get(k) for k in ("probes", "dtype", "quant", "adapter")})
        return out

    test = _rank_meta(spec)
    if not test.get("recipe"):
        raise SystemExit(f"{spec}: meta file without settings (recipe) — no comparison with the training rows possible")
    a = comparable(test)
    diff = {}
    for j, ref in enumerate(rank_only_reference(ref_source)):
        if not ref.get("recipe"):
            raise SystemExit(f"{ref_source}: ranking of the training rows without settings (recipe, job {ref.get('job', j)}) — no comparison possible")
        b = comparable(ref)
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                diff[f"{k} [{ref.get('job', j)}]"] = (a.get(k), b.get(k))
    return diff


def rank_only_source_differences(spec: str, ref_source: str) -> dict:
    """rank_only_recipe_differences against EVERY shard (joined with +) of the training-row ranking (one draw made of fold
    shards without a combined output, eda/c117_q397m3_deploy.py); a source without + is exactly rank_only_recipe_differences."""
    if "+" not in str(ref_source):
        return rank_only_recipe_differences(spec, ref_source)
    diff: dict = {}
    for shard in str(ref_source).split("+"):
        diff.update(rank_only_recipe_differences(spec, shard))
    return diff


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _notes_file_check(root: Path, rel: str, upload: str | None, table: str, n_rows: int) -> tuple[dict, list[str]]:
    """Frozen note file (tools/make_q397m3_notes.py): file = checksum in its meta file = copy in the upload folder (the file the
    remote machine read); table and row count according to the meta file. Only checksums and the meta file are read, never the
    notes. -> ({file, sha256, meta}, findings)."""
    f = Path(root) / str(rel)
    mf = f.with_suffix(".meta.json")
    if not f.is_file() or not mf.is_file():
        return {"file": str(rel)}, [f"{rel} or {mf.name} missing"]
    meta = json.loads(mf.read_text())
    sha = _file_sha256(f)
    problems = []
    if meta.get("sha256") != sha:
        problems.append(f"{rel}: checksum does not match the meta file")
    if meta.get("table") != table or int(meta.get("rows", -1)) != int(n_rows):
        problems.append(f"{rel}: table {meta.get('table')!r} with {meta.get('rows')} rows (required {table}, {n_rows})")
    if upload:
        up = Path(root) / str(upload)
        if not up.is_file() or _file_sha256(up) != sha:
            problems.append(f"{upload}: copy in the upload folder missing or different (the remote machine read this file)")
    return {"file": str(rel), "sha256": sha, "meta": meta}, problems


def rank_only_notes_guard(name: str, spec: str, root: Path, n_test: int, n_train: int, stack: dict | None = None, placeholder: bool = False,
                          art_dir: Path | None = None) -> dict | None:
    """NOTE GUARD of a rank-only channel behind foreign frozen notes (field rank.notes in meta_<channel>.json,
    eda/c117_q397m3_deploy.py; see the module docstring). Without the field (all other channels) -> None, nothing is checked.
    placeholder: do not check the test notes (code check only, with a foreign test ranking). Violated -> SystemExit.
    -> checksums and source (aggregates)."""
    art = COMP / "artifacts" if art_dir is None else art_dir
    meta_file = art / f"meta_{name}.json"
    meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
    rk = meta.get("rank") or {}
    nt = rk.get("notes")
    if not nt:
        if placeholder and not rk.get("row_list_draws"):  # with a row list the placeholder applies to the draw with the row list
            raise SystemExit(f"--rank-only-placeholder {name}: the channel has no note guard (rank.notes in {meta_file.name}) — nothing to skip")
        return None
    ref_source = str(rk.get("source") or "")
    tr, te = nt.get("train") or {}, nt.get("test") or {}
    if not tr.get("sft_from") or not te.get("sft_from") or not te.get("source_runs") or not te.get("model"):
        raise SystemExit(f"--rank-only {name}: {meta_file.name} describes the note guard incompletely (rank.notes.train/test)")
    problems = []
    if stack is not None:  # the blend was fitted on THIS artifact
        ro = (stack.get("rank_only") or {}).get(name) or {}
        sha16 = _file_sha256(art / f"oof_{name}.npy")[:16]
        if ro.get("rank_source") != ref_source:
            problems.append(f"blend file fitted on a different ranking (rank_only.{name}.rank_source)")
        if ro.get("oof_sha256") != sha16:
            problems.append(f"artifacts/oof_{name}.npy ({sha16}…) is not the artifact of the blend ({ro.get('oof_sha256')}…)")
    for run, job, m in rank_only_constituents(ref_source):  # training side: every shard behind the training notes
        if (m.get("sft") or {}).get("sft_from") != tr["sft_from"]:
            problems.append(f"OOF ranking {run}/{job} ranked behind {(m.get('sft') or {}).get('sft_from')!r} instead of {tr['sft_from']}")
    info_tr, p = _notes_file_check(root, tr["sft_from"], tr.get("upload"), "train", n_train)
    problems += p
    if tr.get("sha256") and info_tr.get("sha256") != tr["sha256"]:
        problems.append(f"{tr['sft_from']}: not the note file of the OOF artifact (checksum)")
    out = {"notes_train": tr["sft_from"], "notes_train_sha256": info_tr.get("sha256"), "placeholder": bool(placeholder)}
    if placeholder:
        print(f"  channel {name}: PLACEHOLDER — test ranking {spec} is NOT the intended one, test notes not checked (code check only, never submit)")
    else:
        t_meta = _rank_meta(spec)
        if t_meta.get("table") != "test":
            problems.append(f"test ranking {spec}: table {t_meta.get('table')!r}")
        if (t_meta.get("sft") or {}).get("sft_from") != te["sft_from"]:
            problems.append(f"test ranking ranked behind {(t_meta.get('sft') or {}).get('sft_from')!r} instead of {te['sft_from']} (notes of the 397B)")
        info_te, p = _notes_file_check(root, te["sft_from"], te.get("upload"), "test", n_test)
        problems += p
        tm = info_te.get("meta") or {}
        if tm and tm.get("model") != te["model"]:
            problems.append(f"test notes from model {tm.get('model')!r} instead of {te['model']}")
        if tm and (int(tm.get("with_record", -1)) != int(n_test) or (tm.get("states") or {}).get("open") != int(n_test)):
            problems.append(f"test notes not available for every row (with record {tm.get('with_record')}, states {tm.get('states')}) — the OOF side had every row")
        src = tm.get("source")
        if tm and src not in te["source_runs"]:
            problems.append(f"test notes from {src!r}, allowed {te['source_runs']}")
        elif tm:
            _, run, job, tag = str(src).split("/")
            d = COMP / "artifacts" / "runs" / run / run / job
            for fname, key in ((f"{tag}_raw.jsonl", "source_sha256"), (f"{tag}_meta.json", "meta_sha256")):
                if not (d / fname).is_file() or _file_sha256(d / fname) != tm.get(key):
                    problems.append(f"test notes do not match run {src} ({fname}: checksum)")
        out.update(notes_test=te["sft_from"], notes_test_sha256=info_te.get("sha256"), notes_test_source=src, notes_test_model=tm.get("model"))
    if problems:
        raise SystemExit(f"--rank-only {name}: note guard violated — " + "; ".join(problems))
    return out


def rank_mean_notes_guard(name: str, spec: str, root: Path, n_test: int, n_train: int, rm_def: dict | None) -> dict | None:
    """NOTE GUARD of a mean channel whose ranking a fine-tuned model computed behind frozen notes (field rank_mean.notes in
    meta_<channel>.json, eda/c133_primary_m3px27.py; channel m3px27xm). Without the field -> None, nothing is checked (all earlier mean
    channels). Checked: every OOF shard (rank_mean.rank_source) ranked behind the training note file; training and test notes = checksum
    of their meta file = copy in the upload folder, training notes = checksum of the OOF artifact; the test ranking ranked behind the test
    note file; test notes for every row, from the named model, from an allowed run (checksums of <tag>_raw.jsonl and <tag>_meta.json) and
    character-identical with the input of the ranking of the untrained 27B (px27_notes, if required). Only meta files and checksums are
    read, never the notes. Violated -> SystemExit. -> checksums and source (aggregates)."""
    nt = (rm_def or {}).get("notes")
    if not nt:
        return None
    tr, te = nt.get("train") or {}, nt.get("test") or {}
    if not tr.get("sft_from") or not tr.get("sha256") or not te.get("sft_from") or not te.get("source_runs") or not te.get("model"):
        raise SystemExit(f"--rank {name}: meta_{name}.json describes the note guard incompletely (rank_mean.notes.train/test)")
    problems = []
    for run, job, m in rank_only_constituents(str(rm_def.get("rank_source") or "")):
        if (m.get("sft") or {}).get("sft_from") != tr["sft_from"]:
            problems.append(f"OOF ranking {run}/{job} ranked behind {(m.get('sft') or {}).get('sft_from')!r} instead of {tr['sft_from']}")
    info_tr, p = _notes_file_check(root, tr["sft_from"], tr.get("upload"), "train", n_train)
    problems += p
    if info_tr.get("sha256") != tr["sha256"]:
        problems.append(f"{tr['sft_from']}: not the note file of the OOF artifact (checksum)")
    t_meta = _rank_meta(spec)
    if t_meta.get("table") != "test":
        problems.append(f"test ranking {spec}: table {t_meta.get('table')!r}")
    if (t_meta.get("sft") or {}).get("sft_from") != te["sft_from"]:
        problems.append(f"test ranking ranked behind {(t_meta.get('sft') or {}).get('sft_from')!r} instead of {te['sft_from']}")
    info_te, p = _notes_file_check(root, te["sft_from"], te.get("upload"), "test", n_test)
    problems += p
    tm = info_te.get("meta") or {}
    if tm and tm.get("model") != te["model"]:
        problems.append(f"test notes from model {tm.get('model')!r} instead of {te['model']}")
    if tm and (int(tm.get("with_record", -1)) != int(n_test) or (tm.get("states") or {}).get("open") != int(n_test)):
        problems.append(f"test notes not available for every row (with record {tm.get('with_record')}, states {tm.get('states')}) — the OOF side had every row")
    src = tm.get("source")
    if tm and src not in te["source_runs"]:
        problems.append(f"test notes from {src!r}, allowed {te['source_runs']}")
    elif tm:
        _, run, job, tag = str(src).split("/")
        d = COMP / "artifacts" / "runs" / run / run / job
        for fname, key in ((f"{tag}_raw.jsonl", "source_sha256"), (f"{tag}_meta.json", "meta_sha256")):
            if not (d / fname).is_file() or _file_sha256(d / fname) != tm.get(key):
                problems.append(f"test notes do not match run {src} ({fname}: checksum)")
    px = tm.get("px27_notes") or {}
    if te.get("px27_sha256") and (px.get("sha256") != te["px27_sha256"] or int(px.get("identical_notes", -1)) != int(n_test)):
        problems.append(f"test notes not character-identical with the input {te.get('px27_file')} (px27_notes {px.get('identical_notes')}/{n_test})")
    if problems:
        raise SystemExit(f"--rank {name}: note guard violated — " + "; ".join(problems))
    out = {"notes_train": tr["sft_from"], "notes_train_sha256": info_tr.get("sha256"), "notes_test": te["sft_from"],
           "notes_test_sha256": info_te.get("sha256"), "notes_test_source": src, "notes_test_model": tm.get("model"),
           "notes_test_px27_identical": px.get("identical_notes")}
    print(f"  channel {name}: note guard of the ranking (rank_mean.notes): training notes {str(out['notes_train_sha256'])[:16]}…, test notes "
          f"{str(out['notes_test_sha256'])[:16]}… from {src} ({tm.get('model')}), character-identical with px27 {px.get('identical_notes')}/{n_test} — held")
    return out


PROTO_DEFAULT_RULE = "specific_first"  # = vlm/proto.DEFAULT_RULE (meta files with the default carry no field naming_rule)


def proto_source_fingerprint(spec: str) -> dict:
    """Fingerprint of a source of named codes `runs/<run>/<job>/<tag>[+…]` (job proto_gen; field rank_mean.named, see the header): table
    and rows (<tag>_meta.json), naming rule (field naming_rule; if absent: the default), generation recipe (<tag>_generation.json: checksum of
    the content recomputed, = generation_sha256 of the meta file; fingerprint = SHA-256 of the recipe without GEN_FREE_KEYS — table, model
    path, row set; None = run without a recipe file) and model (summary.json: repo, revision, dtype). Several shards (+) must agree in table,
    rule, recipe and model. Only meta, recipe and summary files are read (aggregates), never answers.
    -> {table, rows, naming_rule, generation, model, shards, problems}."""
    shards, problems = [], []
    for part in str(spec).split("+"):
        bits = part.split("/")
        if len(bits) != 4 or bits[0] != "runs":
            raise SystemExit(f"{part}: expected runs/<run>/<job>/<tag>")
        _, run, job, tag = bits
        d = COMP / "artifacts" / "runs" / run / run / job
        mf = d / f"{tag}_meta.json"
        if not mf.is_file():
            raise SystemExit(f"{part}: {tag}_meta.json missing (run not finished or not fetched)")
        meta = json.loads(mf.read_text())
        gen = None
        gf = d / f"{tag}_generation.json"
        if gf.is_file():
            g = json.loads(gf.read_text())
            rec = g.get("recipe")
            if not isinstance(rec, dict):
                problems.append(f"{part}: {gf.name} without a recipe")
            else:
                if hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest() != g.get("sha256"):
                    problems.append(f"{part}: {gf.name} — checksum does not match the recipe")
                if meta.get("generation_sha256") != g.get("sha256"):
                    problems.append(f"{part}: the meta file names the fingerprint {str(meta.get('generation_sha256'))[:16]}…, {gf.name} {str(g.get('sha256'))[:16]}…")
                if rec.get("table") != meta.get("table"):
                    problems.append(f"{part}: recipe for table {rec.get('table')!r}, meta file {meta.get('table')!r}")
                gen = hashlib.sha256(json.dumps({k: v for k, v in rec.items() if k not in GEN_FREE_KEYS}, sort_keys=True).encode()).hexdigest()
        elif meta.get("generation_sha256"):
            problems.append(f"{part}: the meta file names a fingerprint, {gf.name} is missing")
        sf = d / "summary.json"
        m = (json.loads(sf.read_text()).get("model") or {}) if sf.is_file() else {}
        model = {k: m.get(k) for k in ("repo", "revision", "dtype")} if m else None
        shards.append({"spec": part, "table": meta.get("table"), "rows": int(meta.get("rows") or 0),
                       "naming_rule": str(meta.get("naming_rule") or PROTO_DEFAULT_RULE), "generation": gen, "model": model})
    first = shards[0]
    for s in shards[1:]:
        for k in ("table", "naming_rule", "generation", "model"):
            if s[k] != first[k]:
                problems.append(f"{s['spec']}: {k} differs from the first shard")
    return {"table": first["table"], "rows": int(sum(s["rows"] for s in shards)), "naming_rule": first["naming_rule"],
            "generation": first["generation"], "model": first["model"], "shards": shards, "problems": problems}


def rank_mean_named_guard(name: str, draws: list[str], rm_def: dict | None, n_test: int) -> dict | None:
    """Source of the named codes of a mean channel that is separate from the note source of the ranking (field rank_mean.named in
    meta_<channel>.json). Without the field -> None, nothing checked (all other channels). Checked: the field is complete and stands only
    next to the note guard (rank_mean.notes); rank_mean.proto_sources = [named.train.source]; every test draw under --proto is listed in
    named.test.source_runs (none twice) and is table test with all rows; its fingerprint (proto_source_fingerprint: naming rule, generation
    recipe, model) equals named.train. Violated -> SystemExit. -> sources and fingerprints (aggregates)."""
    nm = (rm_def or {}).get("named")
    if not nm:
        return None
    tr, te = nm.get("train") or {}, nm.get("test") or {}
    runs = list(te.get("source_runs") or [])
    if not tr.get("source") or any(k not in tr for k in ("naming_rule", "generation", "model")) or not runs:
        raise SystemExit(f"--proto {name}: meta_{name}.json describes the source of the named codes incompletely (rank_mean.named.train/test)")
    if not (rm_def or {}).get("notes"):
        raise SystemExit(f"--proto {name}: rank_mean.named is defined only next to the note guard rank_mean.notes")
    if list(rm_def.get("proto_sources") or []) != [tr["source"]]:
        raise SystemExit(f"--proto {name}: the OOF artifact named codes from {rm_def.get('proto_sources')}, rank_mean.named.train.source "
                         f"{tr['source']!r}")
    problems, out = [], []
    if len(set(draws)) != len(draws):
        problems.append("the same source several times under --proto")
    for d in draws:
        if d not in runs:
            problems.append(f"named codes from {d!r}, allowed {runs} (rank_mean.named.test.source_runs)")
            continue
        fp = proto_source_fingerprint(d)
        problems += fp["problems"]
        if fp["table"] != "test" or fp["rows"] != int(n_test):
            problems.append(f"{d}: table {fp['table']!r} with {fp['rows']} rows (required: test, {n_test})")
        for k in ("naming_rule", "generation", "model"):
            if fp[k] != tr[k]:
                problems.append(f"{d}: {k} {fp[k]!r} instead of {tr[k]!r} (training side {tr['source']})")
        out.append({"proto": d, "rows": fp["rows"], "naming_rule": fp["naming_rule"], "generation": fp["generation"], "model": fp["model"]})
    if problems:
        raise SystemExit(f"--proto {name}: source of the named codes violated — " + "; ".join(problems))
    print(f"  channel {name}: source of the named codes separate from the notes (rank_mean.named): {', '.join(x['proto'] for x in out)} — rule "
          f"{out[0]['naming_rule']}, recipe {str(out[0]['generation'])[:16]}…, model {(out[0]['model'] or {}).get('repo')} as on the training side "
          f"{tr['source']} — held")
    return {"train_source": tr["source"], "test": out}


def parse_band_for(items: list[str], rank_only: dict) -> dict[str, tuple[float, float]]:
    """--rank-none-band-for <channel>=<lo>,<hi> -> {channel: (lo, hi)}; only for channels under --rank-only, at most once per
    channel, 0 ≤ lo < hi ≤ 1. Empty (default) = all channels use --rank-none-band as before."""
    out: dict[str, tuple[float, float]] = {}
    for item in items:
        name, sep, vals = str(item).partition("=")
        bits = vals.split(",")
        try:
            lo, hi = (float(bits[0]), float(bits[1])) if sep and len(bits) == 2 else (float("nan"), float("nan"))
        except ValueError:
            lo, hi = float("nan"), float("nan")
        if not (0.0 <= lo < hi <= 1.0):
            raise SystemExit(f"--rank-none-band-for {item}: expected <channel>=<lo>,<hi> with 0 ≤ lo < hi ≤ 1")
        if name not in rank_only or name in out:
            raise SystemExit(f"--rank-none-band-for {item}: {name} is not listed under --rank-only or is given twice")
        out[name] = (lo, hi)
    return out


def rank_only_channel(name: str, spec: str, n_test: int, seen_cls: np.ndarray, ref_source: str | None = None, rank_partial: bool = False,
                      max_missing: float = 0.02, min_mass: float = 0.9,
                      none_band: tuple[float, float] = (0.30, 0.38)) -> tuple[np.ndarray, np.ndarray, dict]:
    """RANK-ONLY channel on the test rows (M3: probe proto_rank in sft mode): on the rows with a complete, exact ranking the
    distribution over all codes (a8.load_rank — the same conversion as for the OOF artifact), on all other rows the uniform
    distribution over the classes with a training example (like the neutral rows of the OOF artifact).
    Guards as for --rank (coverage, mass, NONE band); ref_source: ranking of the training rows (meta_<channel>.json) — the same
    settings are required. -> (P, ranking present [n_test], statistics; aggregates only)."""
    missing = c95.rank_files_missing(spec)
    if missing:
        raise SystemExit(f"channel {name}: ranking {spec} — missing {', '.join(missing)} (run not finished or not fetched)")
    if ref_source is not None:
        diff = rank_only_source_differences(spec, ref_source)  # without + exactly rank_only_recipe_differences
        if diff:
            raise SystemExit(f"channel {name}: ranking of the test rows computed differently from that of the training rows ({ref_source}): {diff}")
    R, h, st = a8.load_rank(spec, n_test, seen_cls, table="test", allow_partial=rank_partial)
    hr = h & c95.exact_rank_rows(spec, n_test, seen_cls)
    P = np.tile(seen_cls / seen_cls.sum(), (n_test, 1)).astype(float)  # like a8.mean_of_draws on neutral rows
    P[hr] = R[hr]
    info = {"rows_table": int(n_test), "rows_probe": st["rows_probe"], "rows_done": st["rows_done"], "rows_exact": st["rows_exact"],
            "rows_rank": int(hr.sum()), "rows_uniform": int((~hr).sum()), "rows_outside_probe": int(n_test - st["rows_probe"]),
            "share_rank": float(hr.sum() / max(1, n_test)), "mass_median": st["mass_median"], "mass_p05": st["mass_p05"],
            "mass_below_half": st["mass_below_half"], "none_top_rank": c95.top_none_share(R, hr), "idx_sha256": st["idx_sha256"],
            "partial": bool(rank_partial)}
    print(f"  channel {name}: RANK-ONLY ({spec}): exact on {info['rows_rank']} of {n_test} test rows ({info['share_rank']:.4f}; probe "
          f"{info['rows_done']}/{info['rows_probe']} computed, {info['rows_exact']} exact); {info['rows_uniform']} rows without ranking -> uniform "
          f"(of which outside the probe {info['rows_outside_probe']})")
    print(f"  channel {name}: candidate mass median {info['mass_median']:.3f}, 5 % quantile {info['mass_p05']:.3f}, below 0.5 in {info['mass_below_half']:.3f} of the rows; "
          f"NONE at the top of the ranking {info['none_top_rank']:.3f}")
    if info["share_rank"] < 1.0 - max_missing:
        raise SystemExit(f"channel {name}: ranking only on {info['share_rank']:.2%} of the test rows (at least {1.0 - max_missing:.0%} required)")
    if info["mass_median"] < min_mass:
        raise SystemExit(f"channel {name}: median candidate mass {info['mass_median']:.3f} below {min_mass:g} — forced answer start is unnatural")
    if not (none_band[0] <= info["none_top_rank"] <= none_band[1]):
        raise SystemExit(f"channel {name}: share of NONE at the top of the ranking {info['none_top_rank']:.3f} outside the malfunction band fixed in advance "
                         f"{none_band[0]:.2f}–{none_band[1]:.2f} — check the run, do not deploy")
    return P, hr, info


RANK_ONLY_DRAWS_DEFINITION = ("mean of the probabilities of several draws: per draw the exact distribution over all codes (probe "
                              "proto_rank) on its rows with a complete, exact ranking; per row Σ P_j / number of draws with a ranking; "
                              "rows without any ranking uniform over the classes with a training example; floor from the blend file "
                              "(OOF: eda/a9_m3_draws_oof.py; blend: eda/c99_m3_deploy.py --rows)")
GEN_FREE_KEYS = ("table", "model_dir", "rows_sha256")  # generation: table, path of the merged model and row set may differ


def rank_only_constituents(source: str) -> list[tuple[str, str, dict]]:
    """(run, job, meta) of every ranking behind ONE draw of the OOF artifact: shards (+) and combined outputs (combined_from)
    resolved."""
    out = []
    for shard in str(source).split("+"):
        bits = shard.split("/")
        if len(bits) != 4 or bits[0] != "runs":
            raise SystemExit(f"{shard}: expected runs/<run>/<job>/<tag>")
        _, run, job, tag = bits
        meta = _rank_meta(shard)
        if not meta.get("combined_from"):
            out.append((run, job, meta))
            continue
        for s in meta["combined_from"]:
            r = s.get("run", run)
            out.append((r, s["job"], _rank_meta(f"runs/{r}/{s['job']}/{tag}")))
    return out


def gen_settings(run: str, rank_meta: dict) -> dict | None:
    """Settings of the text generation behind a ranking: sft.sft_from = <job>/<tag> in the same run -> <tag>_generation.json,
    field recipe without GEN_FREE_KEYS (max_tokens, temperature, system text, image, backend). None = missing."""
    src = str((rank_meta.get("sft") or {}).get("sft_from") or "")
    if "/" not in src:
        return None
    job, tag = src.split("/", 1)
    f = a8.COMP / "artifacts" / "runs" / run / run / job / f"{tag}_generation.json"
    if not f.is_file():
        return None
    rec = json.loads(f.read_text()).get("recipe")
    return {k: v for k, v in rec.items() if k not in GEN_FREE_KEYS} if isinstance(rec, dict) else None


def rank_only_draw_differences(spec: str, ref_draw: str) -> dict:
    """Test draw against ONE draw of the OOF artifact: ranking settings as in rank_only_recipe_differences (per shard, per fold
    job) AND settings of the text generation (gen_settings, e.g. max_tokens 3,200 against 5,200). Missing generation settings on
    one side count as a difference. Empty = the same computation and generation with a different model."""
    diff: dict = {}
    for shard in str(ref_draw).split("+"):
        diff.update(rank_only_recipe_differences(spec, shard))
    test_meta = _rank_meta(spec)
    g_test = gen_settings(spec.split("/")[1], test_meta)
    for run, job, meta in rank_only_constituents(ref_draw):
        if test_meta.get("sft_examples") != meta.get("sft_examples"):  # M3k (examples in the prompt) is a different channel than M3
            diff[f"sft_examples [{job}]"] = (test_meta.get("sft_examples"), meta.get("sft_examples"))
        g_ref = gen_settings(run, meta)
        if g_test is None or g_ref is None:
            diff[f"gen [{job}]"] = ("missing" if g_test is None else "present", "missing" if g_ref is None else "present")
            continue
        for k in sorted(set(g_test) | set(g_ref)):
            if g_test.get(k) != g_ref.get(k):
                diff[f"gen.{k} [{job}]"] = (g_test.get(k), g_ref.get(k))
    return diff


def _rows_path(root: Path, f: str) -> Path:
    return Path(f) if Path(f).is_absolute() else Path(root) / f


def rank_only_rowlist_guard(name: str, rk: dict, specs: list[str], root: Path, n_train: int, placeholder: bool = False) -> tuple[dict, list[dict]]:
    """Additive; only with the field rank.row_list_draws in meta_<channel>.json (eda/c166_primary_m3x5h.py): DRAW WITH ROW LIST (see the
    header). Per entry: draw j (1-based) with source = OOF draw j; training list = SHA-256 of the file, its rows = coverage of draw j in the
    artifact (rank.draws[j].rows_sha256); test spec `<shards>@<test list>` with exactly the file of the meta file (SHA-256, number of rows,
    ascending and unique); all other draws cover all training rows. placeholder: a placeholder of the test ranking is allowed (code checks
    only). -> ({j: data for rank_only_rowlist_draw}, statistics per entry; only aggregates and checksums)."""
    refs = str(rk.get("source") or "").split(",")
    draws = rk.get("draws") or []
    out: dict[int, dict] = {}
    info: list[dict] = []
    for e in rk.get("row_list_draws") or []:
        j = int(e.get("draw", -1)) if str(e.get("draw", "")).lstrip("-").isdigit() else -1
        if not 1 <= j <= len(refs) or j in out or str(e.get("source")) != refs[j - 1]:
            raise SystemExit(f"--rank-only {name}: rank.row_list_draws names draw {e.get('draw')!r} with source {e.get('source')!r} — not a draw of the "
                             f"artifact, or twice")
        tr, te = e.get("train") or {}, e.get("test") or {}
        if not tr.get("file") or not te.get("file") or not tr.get("sha256") or not te.get("sha256"):
            raise SystemExit(f"--rank-only {name}: draw {j} with row list without file and SHA-256 of the training and test list")
        ptr = _rows_path(root, tr["file"])
        if not ptr.is_file() or _file_sha256(ptr) != tr["sha256"]:
            raise SystemExit(f"--rank-only {name}: training list {tr['file']} of draw {j} missing or different (SHA-256)")
        rows_tr = np.asarray(np.load(ptr), dtype=np.int64)
        sha_tr = hashlib.sha256(rows_tr.astype("<i8").tobytes()).hexdigest()
        dj = draws[j - 1] if j - 1 < len(draws) else {}
        if sha_tr != tr.get("rows_sha256") or str(dj.get("rows_sha256")) != sha_tr or int(dj.get("rows", -1)) != len(rows_tr):
            raise SystemExit(f"--rank-only {name}: draw {j} does not cover exactly the training list {tr['file']} in the OOF artifact ({dj.get('rows')} rows)")
        dspec, sep, rows_file = str(specs[j - 1]).partition("@")
        if not sep or rows_file != te["file"]:
            raise SystemExit(f"--rank-only {name}: the test spec of draw {j} must read <shards>@{te['file']} (row list of the meta file), "
                             f"given {specs[j - 1]!r}")
        pte = _rows_path(root, rows_file)
        if not pte.is_file() or _file_sha256(pte) != te["sha256"]:
            raise SystemExit(f"--rank-only {name}: test list {rows_file} of draw {j} missing or different (SHA-256 pinned in the meta file)")
        rows_te = np.asarray(np.load(pte), dtype=np.int64)
        if rows_te.ndim != 1 or len(rows_te) != int(te.get("rows", -1)) or (len(rows_te) and (np.any(np.diff(rows_te) <= 0) or rows_te.min() < 0)):
            raise SystemExit(f"--rank-only {name}: test list {rows_file} is not an ascending list of unique positions with {te.get('rows')} entries")
        nb = e.get("none_band")  # malfunction band of the list rows fixed in advance (meta file); without the field the band of the channel
        if nb is not None and not (isinstance(nb, list) and len(nb) == 2 and all(isinstance(v, (int, float)) for v in nb) and 0.0 <= nb[0] < nb[1] <= 1.0):
            raise SystemExit(f"--rank-only {name}: row_list_draws[].none_band {nb!r} — expected [lo, hi] with 0 ≤ lo < hi ≤ 1")
        out[j] = {"spec": dspec, "rows_file": rows_file, "rows": rows_te, "sha256": te["sha256"], "placeholder": bool(placeholder), "ref": refs[j - 1],
                  "none_band": None if nb is None else (float(nb[0]), float(nb[1]))}
        info.append({"draw": j, "train_list": tr["file"], "train_list_sha256": tr["sha256"], "train_rows": int(len(rows_tr)), "test_list": rows_file,
                     "test_list_sha256": te["sha256"], "test_rows": int(len(rows_te)), "placeholder": bool(placeholder),
                     "none_band": None if nb is None else [float(nb[0]), float(nb[1])]})
    if not out:
        raise SystemExit(f"--rank-only {name}: rank.row_list_draws without an entry")
    for j, s in enumerate(specs, 1):
        if j not in out and "@" in str(s):
            raise SystemExit(f"--rank-only {name}: @<rows> only for the draws with a row list (draw {j} has none)")
        if j not in out and int((draws[j - 1] if j - 1 < len(draws) else {}).get("rows", -1)) != n_train:
            raise SystemExit(f"--rank-only {name}: draw {j} without a row list does not cover all {n_train} training rows")
    return out, info


def rank_only_rowlist_draw(name: str, rl: dict, n_test: int, seen_cls: np.ndarray, min_mass: float = 0.9,
                           none_band: tuple[float, float] = (0.30, 0.38)) -> tuple[np.ndarray, np.ndarray, dict]:
    """Additive: TEST DRAW WITH ROW LIST (rank_only_rowlist_guard). Shards (+) each complete and exact (a8.load_rank without partial state,
    c95.exact_rank_rows), without overlap; without placeholder the settings per shard = OOF draw (rank_only_draw_differences) and coverage =
    EXACTLY the list; with placeholder coverage ⊇ list. On the list rows the exact distribution, elsewhere the uniform distribution (the draw
    does not count there: rows without ranking). Mass (median per shard) and NONE band on the list rows — the band of the entry
    (row_list_draws[].none_band, fixed in advance in the meta file), otherwise none_band. -> (P, list rows [n_test], statistics)."""
    if rl.get("none_band") is not None:
        none_band = tuple(rl["none_band"])
    rows = np.asarray(rl["rows"], dtype=np.int64)
    if len(rows) and rows.max() >= n_test:
        raise SystemExit(f"channel {name}: test list {rl['rows_file']} contains positions outside the test table")
    want = np.zeros(n_test, bool)
    want[rows] = True
    cov, R = np.zeros(n_test, bool), np.zeros((n_test, len(seen_cls)))
    shards = []
    for s in str(rl["spec"]).split("+"):
        missing = c95.rank_files_missing(s)
        if missing:
            raise SystemExit(f"channel {name}: ranking {s} — missing {', '.join(missing)} (run not finished or not fetched)")
        if not rl["placeholder"]:
            diff = rank_only_draw_differences(s, rl["ref"])
            if diff:
                raise SystemExit(f"channel {name}: test shard {s} computed or generated differently from the OOF draw ({rl['ref']}): {diff}")
        Rs, hs, st = a8.load_rank(s, n_test, seen_cls, table="test", allow_partial=False)
        hs = hs & c95.exact_rank_rows(s, n_test, seen_cls)
        if int(hs.sum()) != int(st["rows_probe"]):
            raise SystemExit(f"channel {name}: test shard {s} — {int(hs.sum())} of {st['rows_probe']} rows of the probe exact")
        if (cov & hs).any():
            raise SystemExit(f"channel {name}: test shards overlap ({int((cov & hs).sum())} rows twice)")
        R[hs] = Rs[hs]
        cov |= hs
        shards.append({"spec": s, "rows_probe": st["rows_probe"], "rows_exact": st["rows_exact"], "mass_median": st["mass_median"],
                       "mass_p05": st["mass_p05"], "idx_sha256": st["idx_sha256"]})
    outside, lacking = int((cov & ~want).sum()), int((want & ~cov).sum())
    if lacking or (outside and not rl["placeholder"]):
        raise SystemExit(f"channel {name}: coverage of the test draw with row list {int(cov.sum())} rows (outside the list {outside}, missing in the "
                         f"list {lacking}) — required EXACTLY the list {rl['rows_file']} ({int(want.sum())} rows)")
    hr = want
    P = np.tile(seen_cls / seen_cls.sum(), (n_test, 1)).astype(float)  # as rank_only_channel: rows without ranking uniform
    P[hr] = R[hr]
    none_top = c95.top_none_share(R, hr)
    info = {"rows_table": int(n_test), "rows_list": int(hr.sum()), "rows_rank": int(hr.sum()), "rows_uniform": int((~hr).sum()),
            "rows_covered": int(cov.sum()), "rows_covered_outside_list": outside, "share_rank": float(hr.sum() / max(1, n_test)),
            "list": rl["rows_file"], "list_sha256": rl["sha256"], "none_top_rank": none_top, "shards": shards, "placeholder": bool(rl["placeholder"]),
            "settings_checked": not rl["placeholder"], "none_band": [float(none_band[0]), float(none_band[1])]}
    print(f"  channel {name}: DRAW WITH ROW LIST ({rl['spec']} @ {rl['rows_file']}): exact on {info['rows_list']} list rows of {n_test} test rows "
          f"({len(shards)} shards, coverage = list{' ⊇ (PLACEHOLDER, settings not compared)' if rl['placeholder'] else ''}); "
          f"{info['rows_uniform']} rows without this draw; NONE at the top on the list {none_top:.3f}")
    if hr.any():
        if min(s["mass_median"] for s in shards) < min_mass:
            raise SystemExit(f"channel {name}: median of the candidate mass below {min_mass:g} in a shard — forced answer start unnatural")
        if not (none_band[0] <= none_top <= none_band[1]):
            raise SystemExit(f"channel {name}: share of NONE at the top on the list rows {none_top:.3f} outside the malfunction band fixed in advance "
                             f"{none_band[0]:.2f}–{none_band[1]:.2f} — check the run, do not deploy")
    return P, hr, info


def rank_only_draws_guard(name: str, stack: dict, specs: list[str], root: Path, n_train: int, art_dir: Path | None = None,
                          rowlist_placeholder: bool = False) -> tuple[list[str], dict]:
    """Several draws of a rank-only channel (OOF: eda/a9_m3_draws_oof.py): exactly one test draw per OOF draw, in the same
    order; the blend file was fitted on THIS artifact (rank_only.rank_source and oof_sha256) and only on rows with ALL draws
    (fit_rows ⊆ rank.rows_all_draws_file, checksum). -> (source per OOF draw, statistics).
    Additive, only with the field rank.row_list_draws: draws with a row list (rank_only_rowlist_guard) — the blend is then fitted on rows
    with all OTHER draws (all training rows); the data for the test channel are stored under the key _rowlist."""
    art = COMP / "artifacts" if art_dir is None else art_dir
    meta = json.loads((art / f"meta_{name}.json").read_text())
    rk = meta.get("rank") or {}
    refs = str(rk.get("source") or "").split(",")
    draws = rk.get("draws") or []
    if len(refs) < 2 or len(draws) != len(refs) or [str(d.get("source")) for d in draws] != refs:
        raise SystemExit(f"--rank-only {name}: meta_{name}.json does not describe a channel made of several draws (rank.source with commas and rank.draws; "
                         f"eda/a9_m3_draws_oof.py) — several test draws would belong to a different channel form")
    if len(specs) != len(refs):
        raise SystemExit(f"--rank-only {name}: {len(specs)} test draw(s) given, the OOF artifact averages {len(refs)} draws ({rk['source']}) — "
                         f"one test draw per OOF draw, in the same order")
    if len(set(specs)) != len(specs):
        raise SystemExit(f"--rank-only {name}: the same test draw given more than once")
    ro = (stack.get("rank_only") or {}).get(name) or {}
    if ro.get("rank_source") != rk["source"]:
        raise SystemExit(f"--rank-only {name}: the blend file was fitted on a different channel (rank_only.rank_source {ro.get('rank_source')!r}, "
                         f"artifact {rk['source']!r}) — rerun eda/c99_m3_deploy.py --save")
    sha16 = hashlib.sha256((art / f"oof_{name}.npy").read_bytes()).hexdigest()[:16]
    if ro.get("oof_sha256") != sha16:
        raise SystemExit(f"--rank-only {name}: artifacts/oof_{name}.npy ({sha16}…) is not the artifact the blend was fitted on "
                         f"({ro.get('oof_sha256')}…) — rerun eda/c99_m3_deploy.py --save")
    f = rk.get("rows_all_draws_file")
    p = (Path(f) if Path(f).is_absolute() else Path(root) / f) if f else None
    if p is None or not p.is_file():
        raise SystemExit(f"--rank-only {name}: list of the rows with all draws ({f}) missing — eda/a9_m3_draws_oof.py")
    rows_all = np.asarray(np.load(p), dtype=np.int64)
    if hashlib.sha256(rows_all.astype("<i8").tobytes()).hexdigest() != rk.get("rows_all_draws_sha256"):
        raise SystemExit(f"--rank-only {name}: row list {f} does not match meta_{name}.json (checksum)")
    fit = stack_fit_rows(stack, n_train)
    fit = np.arange(n_train) if fit is None else fit
    if rk.get("row_list_draws"):  # additive: draws with a row list; all other draws cover all training rows
        rowlist, rl_info = rank_only_rowlist_guard(name, rk, specs, root, n_train, placeholder=rowlist_placeholder)
        return refs, {"oof_sha256": sha16, "fit_rows": int(len(fit)), "rows_all_draws": int(len(rows_all)), "rows_all_draws_file": str(f),
                      "row_list_draws": rl_info, "_rowlist": rowlist}
    outside = ~np.isin(fit, rows_all)
    if outside.any():
        raise SystemExit(f"--rank-only {name}: {int(outside.sum())} fit rows of the blend do not carry all {len(refs)} draws — "
                         f"eda/c99_m3_deploy.py --channel {name} --rows {f} --save")
    return refs, {"oof_sha256": sha16, "fit_rows": int(len(fit)), "rows_all_draws": int(len(rows_all)), "rows_all_draws_file": str(f)}


def rank_only_draws_channel(name: str, specs: list[str], refs: list[str], n_test: int, seen_cls: np.ndarray, rank_partial: bool = False,
                            max_missing: float = 0.02, min_mass: float = 0.9,
                            none_band: tuple[float, float] = (0.30, 0.38), keep: list | None = None,
                            rowlist: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rank-only channel made of SEVERAL draws on the test rows: every test draw as in rank_only_channel (guards: coverage,
    mass, NONE band, partial state) and checked against ITS OOF draw (rank_only_draw_differences); per row the mean over the
    draws with an exact ranking (the same computation as eda/a9_m3_draws_oof.mean_of_draws), rows without any ranking uniform.
    -> (P, number of draws with a ranking per row, statistics; aggregates only).
    rowlist (additive; from rank_only_draws_guard._rowlist): draws with a row list via rank_only_rowlist_draw; None = as before."""
    if len(specs) != len(refs) or len(specs) < 2:
        raise SystemExit(f"channel {name}: {len(specs)} test draw(s) against {len(refs)} OOF draw(s)")
    S, cnt = np.zeros((n_test, len(seen_cls))), np.zeros(n_test, dtype=np.int64)
    tops, per_draw = [], []
    for j, (spec, ref) in enumerate(zip(specs, refs), 1):
        if rowlist is not None and j in rowlist:  # draw with a row list: shards, coverage = list, settings per shard
            Pj, hr, info = rank_only_rowlist_draw(f"{name}[{j}]", rowlist[j], n_test, seen_cls, min_mass=min_mass, none_band=none_band)
        else:
            missing = c95.rank_files_missing(spec)
            if missing:
                raise SystemExit(f"channel {name}, draw {j}: ranking {spec} — missing {', '.join(missing)} (run not finished or not fetched)")
            diff = rank_only_draw_differences(spec, ref)
            if diff:
                raise SystemExit(f"channel {name}, draw {j}: test draw {spec} computed or generated differently from OOF draw {j} ({ref}): {diff}")
            Pj, hr, info = rank_only_channel(f"{name}[{j}]", spec, n_test, seen_cls, ref_source=None, rank_partial=rank_partial,
                                             max_missing=max_missing, min_mass=min_mass, none_band=none_band)
        S[hr] += Pj[hr]
        if keep is not None:  # second stage (--stage2): the individual draws as features (rows without a ranking uniform)
            keep.append(Pj.copy())
        cnt[hr] += 1
        tops.append(np.where(hr, Pj.argmax(1), -1))
        per_draw.append({"draw": j, "rank_only": spec, "train_rank": ref, **info})
    P = np.tile(seen_cls / seen_cls.sum(), (n_test, 1)).astype(float)  # rows without any ranking: as in rank_only_channel
    have = cnt > 0
    P[have] = S[have] / cnt[have, None]
    T = np.stack(tops)
    every = (T >= 0).all(0)
    info = {"draws": len(specs), "rows_table": int(n_test), "rows_by_draws": {str(k): int((cnt == k).sum()) for k in sorted(set(cnt.tolist()))},
            "rows_uniform": int((~have).sum()), "share_all_draws": float((cnt == len(specs)).mean()), "none_top_mean": c95.top_none_share(P, have),
            "same_top_all_draws": float((T[:, every] == T[0, every]).all(0).mean()) if every.any() else float("nan"), "per_draw": per_draw}
    print(f"  channel {name}: MEAN of {len(specs)} draws — test rows by number of draws with a ranking {info['rows_by_draws']}, "
          f"{info['rows_uniform']} without any ranking -> uniform; NONE at the top of the mean {info['none_top_mean']:.3f}; "
          f"same top in all draws {info['same_top_all_draws']:.3f}")
    return P, cnt, info


def stack_fit_rows(stack: dict, n_train: int) -> np.ndarray | None:
    """Fit rows of a blend that was fitted on a subset of the training rows only (eda/c99_m3_deploy.py: the channel exists only
    there) -> sorted table positions; None for blends over all rows."""
    rows = stack.get("fit_rows")
    if rows is None:
        return None
    r = np.asarray(rows, dtype=np.int64)
    if r.ndim != 1 or len(r) == 0 or r.min() < 0 or r.max() >= n_train or not np.all(np.diff(r) > 0):
        raise SystemExit("blend file: fit_rows is not a sorted, unique list of positions in the training table")
    want = stack.get("fit_rows_sha256")
    if want and hashlib.sha256(r.astype("<i8").tobytes()).hexdigest() != want:
        raise SystemExit("blend file: fit_rows does not match fit_rows_sha256")
    return r


def aggregates(P: np.ndarray, allowed: np.ndarray, classes: list[str]) -> dict:
    top = metric.top3_indices(P, allowed)
    first = top[:, 0]
    hist = np.bincount(first, minlength=len(classes)) / len(first)
    return {"n": int(len(P)), "share_none_rank1": round(float((first == 0).mean()), 4), "distinct_rank1": int(len(set(first.tolist()))),
            "mean_p_rank1": round(float(P[np.arange(len(P)), first].mean()), 4), "share_none_in_top3": round(float((top == 0).any(1).mean()), 4),
            "hist": hist}


def arith_beta_of(stack: dict) -> float:
    """Share of the arithmetic blend (field `arith_beta` of the blend file): 0 = purely log-linear, as in the earlier blend
    files."""
    beta = float(stack.get("arith_beta") or 0.0)
    if not 0.0 <= beta <= 1.0:
        raise SystemExit(f"blend file: arith_beta {beta} is not in [0, 1]")
    return beta


def arith_beta_novel_of(stack: dict) -> float | None:
    """Arithmetic share for NOVEL codes (field `arith_beta_novel`): None = one share β for all classes (as in the earlier blend
    files)."""
    if stack.get("arith_beta_novel") is None:
        return None
    bn = float(stack["arith_beta_novel"])
    if not 0.0 <= bn <= 1.0:
        raise SystemExit(f"blend file: arith_beta_novel {bn} is not in [0, 1]")
    return bn


def code_novelty(ref_q, ref_topic, ref_y, q, topic, n_classes: int) -> np.ndarray:
    """Novelty per (row, class) -> bool [len(q), n_classes]: True if the class was never the root among the reference rows of
    the same task (question_id and the same topic_name); NONE (class 0) is never novel. Only training labels and the row's
    task are used."""
    by_key: dict = {}
    for a, b, c in zip(ref_q, ref_topic, ref_y):
        by_key.setdefault((a, b), set()).add(int(c))
    out = np.ones((len(q), n_classes), dtype=bool)
    for i, key in enumerate(zip(q, topic)):
        known = by_key.get(key)
        if known:
            out[i, sorted(known)] = False
    out[:, 0] = False
    return out


def train_novelty(train: pd.DataFrame, folds: np.ndarray, classes: list[str]) -> np.ndarray:
    """Novelty for the training rows (oof_mixture): the reference are the rows of the other folds — like the OOF channels
    themselves."""
    y = labels.encode(train["root_code"], classes)
    q, t = train["question_id"].to_numpy(), train["topic_name"].to_numpy()
    folds = np.asarray(folds)
    out = np.zeros((len(train), len(classes)), dtype=bool)
    for k in np.unique(folds):
        ev, ref = folds == k, folds != k
        out[ev] = code_novelty(q[ref], t[ref], y[ref], q[ev], t[ev], len(classes))
    return out


def mix_channels(stack: dict, chans: list[np.ndarray], none_feats: np.ndarray | None, seen_cls: np.ndarray,
                 novel: np.ndarray | None = None) -> np.ndarray:
    """Blend of the channels (log probabilities with floor):
    P = (1 − β)·log-linear + β·Σ w_i·p_i with w = θ/Σθ and p_i the channel with its floor, normalised per row over the training
    classes; β = 0 is the pure log-linear blend (a2.apply_mix, classes without a training example zero, normalised).
    With `arith_beta_novel` in the blend file, β_new applies per (row, class) where `novel` (code_novelty) is true, otherwise β."""
    theta = np.asarray(stack["theta"], dtype=float)
    beta = arith_beta_of(stack)
    beta_new = arith_beta_novel_of(stack)
    if (beta or beta_new is not None) and (none_feats is not None or stack.get("none") or stack.get("probes")):
        raise SystemExit("The arithmetic share is only implemented for blends without answer-check signals and probes.")
    P = np.where(seen_cls[None, :], a2.apply_mix(theta, chans, none_feats), 0.0)
    P = P / P.sum(1, keepdims=True)
    if beta or beta_new is not None:
        w = theta[: len(chans)] / theta[: len(chans)].sum()
        A = np.zeros_like(P)
        for wi, lg in zip(w, chans):
            pi = np.where(seen_cls[None, :], np.exp(lg), 0.0)
            A += wi * pi / pi.sum(1, keepdims=True)
        A = np.where(seen_cls[None, :], A, 0.0)
        A = A / A.sum(1, keepdims=True)
        if beta_new is None:
            P = (1.0 - beta) * P + beta * A
        else:
            if novel is None or np.shape(novel) != P.shape:
                raise SystemExit("blend file with arith_beta_novel: the novelty per row and class is missing or does not match "
                                 "(code_novelty / train_novelty)")
            B = np.where(np.asarray(novel, dtype=bool), beta_new, beta)
            P = (1.0 - B) * P + B * A
        P = P / P.sum(1, keepdims=True)
    return P


def oof_mixture(stack: dict, seen_cls: np.ndarray, art_dir: Path, n_rows: int | None = None,
                novel: np.ndarray | None = None) -> np.ndarray:
    """Predictions of the saved blend for the training rows: OOF artifacts of the channels, floors and weights of the blend
    file — the same computation as for the test rows (without answer-check signals and probes).
    n_rows: required number of rows per artifact (training table); every artifact must be complete and normalised per row.
    novel: novelty of the training rows (train_novelty) — required only for blend files with arith_beta_novel."""
    if stack.get("none") or stack.get("probes"):
        raise SystemExit("The inductive alignment is only implemented for blends without answer-check signals and probes.")
    chans = []
    for name in stack["channels"]:
        P = np.load(art_dir / ("oof_prior_question_v1.npy" if name == "prior" else f"oof_{name}.npy"))
        if P.ndim != 2 or P.shape[1] != len(seen_cls) or (n_rows is not None and len(P) != n_rows):
            raise SystemExit(f"OOF artifact of channel {name}: shape {P.shape}, expected ({n_rows}, {len(seen_cls)})")
        if not np.isfinite(P).all() or not np.allclose(P.sum(1), 1.0, atol=1e-6):  # a gap (zero rows) would silently shift the class weights
            raise SystemExit(f"OOF artifact of channel {name}: non-finite values or rows that do not sum to 1")
        P = a2.floor_probs(np.where(seen_cls[None, :], P, 0.0), float((stack.get("floor") or {}).get(name, 0.0)), seen_cls)
        chans.append(np.log(np.clip(P, EPS, 1)))
    return mix_channels(stack, chans, None, seen_cls, novel=novel)


def _test_notes_texts(root: Path, rel_file: str, rank_spec: str, n_test: int) -> tuple[list[str], dict]:
    """Frozen M3 test notes of one draw (tools/freeze_m3_texts.py) for the second stage: file = checksum in its meta file, table
    test with every row exactly once, source = the test ranking of THIS draw (source_rank, rank_sha256). Only the field text is
    read, for machine counting; none of it is printed. -> (texts in table order, checksums)."""
    f = Path(root) / rel_file
    mf = f.with_suffix(".meta.json")
    if not f.is_file() or not mf.is_file():
        raise SystemExit(f"--stage2: {rel_file} or {mf.name} missing")
    meta = json.loads(mf.read_text())
    sha = _file_sha256(f)
    bits = rank_spec.split("/")
    if len(bits) != 4 or bits[0] != "runs":
        raise SystemExit(f"--stage2: test ranking {rank_spec} is not of the form runs/<run>/<job>/<tag>")
    _, run, job, tag = bits
    rank_file = COMP / "artifacts" / "runs" / run / run / job / f"{tag}_proto_rank.jsonl"
    problems = []
    if meta.get("sha256") != sha:
        problems.append("checksum does not match the meta file")
    if meta.get("table") != "test" or int(meta.get("rows", -1)) != n_test:
        problems.append(f"table {meta.get('table')!r} with {meta.get('rows')} rows")
    if meta.get("source_rank") != f"runs/{run}/{run}/{job}/{tag}":
        problems.append(f"source {meta.get('source_rank')!r} is not the test ranking {rank_spec}")
    elif not rank_file.is_file() or _file_sha256(rank_file) != meta.get("rank_sha256"):
        problems.append("test ranking does not match the checksum in the meta file")
    texts: dict[int, str] = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r["row"])
        if r.get("table") != "test" or i in texts or not 0 <= i < n_test:
            problems.append("rows not unique or not table test")
            break
        texts[i] = str(r.get("text") or "")
    if len(texts) != n_test:
        problems.append(f"{len(texts)} of {n_test} rows")
    if problems:
        raise SystemExit(f"--stage2: test notes {rel_file}: " + "; ".join(problems))
    return [texts[i] for i in range(n_test)], {"file": rel_file, "sha256": sha, "rank": rank_spec, "rank_sha256": meta.get("rank_sha256")}


def _stage2_model_form(model: dict, fs: str, classes: list[str]) -> tuple[str, bool]:
    """Feature set and NONE guard of a model of the deployment file. Without the fields feature_set/protect_none: (fs, False) — exactly
    as before. A fallback model must never read the notes (otherwise the note guard would have no effect); the guard requires NONE as class 0."""
    fs_feat = model.get("feature_set", fs)
    if fs_feat not in (*S2.FEATURE_SETS, S2.GUARD_SET, S2.I1_SET, S2.MM_SET):  # main_guard_i1/main_guard_i1_mm read the notes, so never as fallback
        raise SystemExit(f"--stage2: unknown feature set {fs_feat!r} in model {fs}")
    if fs == "fallback" and S2.uses_notes(fs_feat):
        raise SystemExit(f"--stage2: the fallback model names the feature set {fs_feat!r}, which reads the notes")
    protect = model.get("protect_none", False)
    if not isinstance(protect, bool):
        raise SystemExit(f"--stage2: protect_none of model {fs} is not a boolean ({protect!r})")
    if protect and classes[S2.NONE_IDX] != "NONE":
        raise SystemExit("--stage2: the NONE guard requires NONE as class 0")
    return fs_feat, protect


def _test_mech27(s2: dict, test: pd.DataFrame, classes: list[str], root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Row feature mech27 of the test side — the final answer of the frozen pr27 test answer (last readable JSON answer per row) matches
    the answer key mechanically (src/answer_match.py); file and SHA-256 from the field mech27.test of the deployment file, the meta file
    next to it must name the table test; class mask minor from taxonomy_v1.csv, checked against mech27.minor_codes. Aggregates only."""
    spec = s2.get("mech27") or {}
    ts = spec.get("test") or {}
    f = Path(str(ts.get("file") or ""))
    path = f if f.is_absolute() else COMP / f
    if not ts.get("file") or not path.is_file():
        raise SystemExit(f"--stage2: test answers for mech27 missing ({ts.get('file')})")
    sha = _file_sha256(path)
    if sha != ts.get("sha256"):
        raise SystemExit(f"--stage2: test answers {f} have SHA-256 {sha[:16]}… instead of {str(ts.get('sha256'))[:16]}… (deployment file)")
    meta_file = path.parent / "proto_meta.json"
    if meta_file.is_file() and json.loads(meta_file.read_text()).get("table") != "test":
        raise SystemExit(f"--stage2: {meta_file} does not name the table test")
    answers = AM.last_answers([path], len(test))
    mech = AM.mech_rows(answers, test["correct_answer"].tolist())
    tax = pd.read_csv(root / "competition_data" / "taxonomy_v1.csv", dtype=str, keep_default_na=False)
    minor = AM.minor_mask(classes, dict(zip(tax["id"], tax["default_severity"])))
    codes = [c for c, v in zip(classes, minor) if v]
    if codes != list(spec.get("minor_codes") or []):
        raise SystemExit(f"--stage2: minor codes of the taxonomy {codes} != deployment file {spec.get('minor_codes')}")
    info = {"file": str(f), "sha256": sha, "share_with_answer": float(np.mean([o is not None for o in answers])),
            "share_mech": float(mech.mean()), "minor_codes": len(codes), "oof_share_mech": spec.get("train_share_mech")}
    print(f"    SECOND STAGE, mech27: test answers {f.name} (SHA-256 {sha[:16]}…), answer present {info['share_with_answer']:.4f}, "
          f"mechanically correct {info['share_mech']:.4f} (training {spec.get('train_share_mech')}); {len(codes)} minor codes (aggregates only)")
    return mech, minor, info


def _stage2_extra_columns(s2: dict, train: pd.DataFrame, test: pd.DataFrame, classes: list[str], root: Path, cand: np.ndarray,
                          chans: list, draws: list, seen_cls: np.ndarray, xcontrol: str | None = None) -> tuple[np.ndarray, list[str], dict]:
    """Additive; only with the field extra_columns of the deployment file (e.g. artifacts/stage2_primary_cols.json from
    eda/c151_primary_cols.py fit): extra second-stage columns for the test rows (src/stage2_sources.py; sources extra_columns.names in this
    order) — protocol answers pr27/q397 from frozen files (extra_columns.test: files and SHA-256, check mandatory; per run table/rows/model,
    exactly one first pass per row, parser cross-check against proto_labels.jsonl), pr27 final_answer_correct (the same answer file), blend
    channels BEFORE the floor in the order of the blend and the draws d1–d3 (as Static of the second stage), table and taxonomy; task
    knowledge, task row counts and topic sections with reference = ALL training rows (as the task knowledge). Test guard per column and
    source against extra_columns.guard_reference by the rule fixed in advance stage2_sources.GUARD_RULE (other values in the file are
    rejected). xcontrol: positive control (stage2_sources.CONTROLS, in memory only). Aggregates only.
    -> (columns [n_test·K, m], feature names, statistics)."""
    spec = s2.get("extra_columns") or {}
    names = list(spec.get("names") or [])
    if not names or len(set(names)) != len(names) or any(nm not in XS.SOURCE_ORDER for nm in names):
        raise SystemExit(f"--stage2: extra_columns.names {names} — unknown, duplicate or no sources (known: {list(XS.SOURCE_ORDER)})")
    if spec.get("module") != "src/stage2_sources.py" or spec.get("version") != XS.VERSION:
        raise SystemExit(f"--stage2: extra_columns requires src/stage2_sources.py in version {spec.get('version')!r} (code: {XS.VERSION})")
    if list(spec.get("features") or []) != XS.feature_names(names):
        raise SystemExit(f"--stage2: extra_columns.features do not match the sources {names}")
    rule = spec.get("guard_rule") or {}
    if (float(rule.get("rel", -1)) != XS.GUARD_RULE["rel"] or float(rule.get("z_max", -1)) != XS.GUARD_RULE["z_max"]
            or rule.get("floors") != XS.GUARD_RULE["floors"]):
        raise SystemExit(f"--stage2: extra_columns.guard_rule differs from the rule fixed in advance (rel {rule.get('rel')}, z_max {rule.get('z_max')}, "
                         f"floors {rule.get('floors')}; required {XS.GUARD_RULE['rel']}, {XS.GUARD_RULE['z_max']}, {XS.GUARD_RULE['floors']})")
    if xcontrol is not None and XS.CONTROL_NEEDS[xcontrol] not in names:
        raise SystemExit(f"--stage2-xcontrol {xcontrol}: the source {XS.CONTROL_NEEDS[xcontrol]} is not in extra_columns.names")
    side = spec.get("test") or {}
    try:
        P, sinfo = XS.load_side(side.get("spec") or {}, "test", COMP, root, classes, [str(k) for k in test["correct_answer"].tolist()],
                                want_sha=side.get("files_sha256") or {})
    except (XS.SourceError, KeyError, FileNotFoundError) as exc:
        raise SystemExit(f"--stage2: extra columns: {exc}") from None
    tax = pd.read_csv(root / "competition_data" / "taxonomy_v1.csv", dtype=str, keep_default_na=False)
    minor = XS.minor_vector(classes, dict(zip(tax["id"], tax["default_severity"])))
    codes = [c for c, v in zip(classes, minor) if v]
    if codes != list(spec.get("minor_codes") or []):
        raise SystemExit(f"--stage2: minor codes of the taxonomy {codes} != extra_columns.minor_codes {spec.get('minor_codes')}")
    if "oth_novelty" in names and (spec.get("channels") != s2.get("channels") or len(chans) != len(spec.get("channels") or [])):
        raise SystemExit(f"--stage2: blend channels of the novelty quantities {spec.get('channels')} != channels of the deployment file {s2.get('channels')}")
    inp = {"classes": classes, "P": P, "minor": minor, "row_keys": S2.task_keys(test["question_id"].to_numpy(), test["topic_name"].to_numpy()),
           "ref_keys": S2.task_keys(train["question_id"].to_numpy(), train["topic_name"].to_numpy()), "ref_y": labels.encode(train["root_code"], classes),
           "fac": XS.fac_rows([COMP / r for r in (side.get("spec") or {})["pr27"]["raw"]], len(test)) if "fac27_rule" in names else None,
           "sev_minor": XS.sev_minor_raw(classes, tax["id"].tolist(), tax["default_severity"].tolist()), "chans": chans, "draws": draws, "seen": seen_cls,
           "row_topics": test["topic_name"].to_numpy().astype(str), "ref_topics": train["topic_name"].to_numpy().astype(str),
           "sections": XS.class_sections(classes, tax["id"].tolist(), tax["section"].tolist())}
    if xcontrol is not None:
        inp = XS.apply_control(inp, xcontrol)
    cols = XS.build_columns(names, inp)
    rows, ref_all = np.arange(len(test)), np.arange(len(train))
    E = cols.cols(rows, cand, ref_all)
    stats = XS.guard_stats(cols, E, cand, inp, rows, ref_all)
    guard = XS.guard_check(spec["guard_reference"], stats, XS.GUARD_RULE["rel"], XS.GUARD_RULE["z_max"], XS.GUARD_RULE["floors"])
    info = {"names": names, "features": cols.features, "version": XS.VERSION, "files": sinfo["files"], "control": xcontrol,
            "sources": {s: {k: v for k, v in d.items() if v is not None} for s, d in sinfo["sources"].items()}, "test": stats, "guard": guard}
    print(f"    SECOND STAGE, extra columns (frozen sources{', CONTROL ' + xcontrol if xcontrol else ''}): {len(names)} sources, {len(cols.features)} columns "
          f"(answers from {len(sinfo['files'])} frozen files, SHA-256 checked); test guard per column and source "
          f"{'held' if guard['ok'] else 'VIOLATED'} ({len(guard['checks']) - len(guard['failed'])}/{len(guard['checks'])} statistics; band "
          f"±{XS.GUARD_RULE['rel']:g} relative or |z| ≤ {XS.GUARD_RULE['z_max']:g}, valid answers ≥ {XS.GUARD_RULE['floors']['share_valid']:g}; aggregates only)")
    for c in guard["failed"]:
        print(f"    test guard of the extra columns violated: {c['scope']} {c['stat']} test {c['test']:.5f} against training {c['train']:.5f}, z {c['z']:+.2f}")
    return E, cols.features, info


def stage2_draws_first(s2: dict, draws: list, specs: list[str]) -> tuple[list, list[str]]:
    """Additive (eda/c135_primary_m3x4.py): if the deployment file names `draws_first` (= number of draws of the second stage, d1–d3), the
    second-stage features read only the FIRST draws_first test draws of the channel draws_channel — the channel itself averages more draws
    (m3x4_full = mean of d1..d4), the features stay, as on the training rows, from d1–d3. Without the field (all deployment files up to the
    v0.3 candidates) unchanged. -> (matrices, test rankings) for the check against test_notes."""
    if "draws_first" not in s2:
        return draws, specs
    k = s2["draws_first"]
    if not isinstance(k, int) or isinstance(k, bool) or k != len(S2.DRAWS) or len(draws) < k or len(specs) < k:
        raise SystemExit(f"--stage2: draws_first {k!r} requires {len(S2.DRAWS)} and at least as many test draws of the channel "
                         f"{s2.get('draws_channel')} ({len(draws)} matrices, {len(specs)} test rankings)")
    return list(draws[:k]), list(specs[:k])


def apply_stage2(path: Path, stack_file: Path, stack: dict, Q: np.ndarray, P_pre: np.ndarray, raw_chans: dict, draw_keep: dict, rank_only: dict,
                 train: pd.DataFrame, test: pd.DataFrame, classes: list[str], seen_cls: np.ndarray, root: Path,
                 report_path: Path | None = None, xcontrol: str | None = None) -> tuple[np.ndarray, dict]:
    """SECOND STAGE (see the module docstring): conditional logit over the first K candidates of the aligned test blend Q.
    P_pre = blend before the alignment. Guards (a)–(c), aggregates only. -> (Q with a new order of the first K, statistics)."""
    s2 = json.loads(Path(path).read_text())
    n_test, K = len(test), len(classes)
    null = bool(s2.get("null"))
    problems = []
    if s2.get("stack_sha256") != _file_sha256(stack_file):
        problems.append(f"blend file {stack_file.name} is not the one of the deployment file ({s2.get('stack')})")
    if s2.get("channels") != list(stack["channels"]) or float(s2.get("arith_beta", -1)) != arith_beta_of(stack):
        problems.append("channels or β of the deployment file do not match the blend file")
    if int(s2.get("K", -1)) != S2.K_CAND or float(s2.get("lambda", -1)) != S2.LAMBDA:
        problems.append(f"K {s2.get('K')} / λ {s2.get('lambda')} instead of {S2.K_CAND} / {S2.LAMBDA}")
    dch = s2.get("draws_channel")
    draws = draw_keep.get(dch) or []
    notes_cfg = s2.get("test_notes") or []
    specs = (rank_only.get(dch) or "").split(",")
    draws, specs = stage2_draws_first(s2, draws, specs)  # only with the field draws_first (otherwise unchanged)
    if len(draws) != len(S2.DRAWS) or len(notes_cfg) != len(S2.DRAWS) or specs != [c["rank"] for c in notes_cfg]:
        problems.append(f"M3 test draws of channel {dch}: {len(draws)} matrices, test rankings {specs} against the deployment file "
                        f"{[c.get('rank') for c in notes_cfg]}")
    missing = [nm for nm in stack["channels"] if nm not in raw_chans]
    if missing:
        problems.append(f"channels without a test matrix {missing}")
    if problems:
        raise SystemExit("--stage2: " + "; ".join(problems))
    texts, notes_src = [], []
    for c in notes_cfg:
        t, src = _test_notes_texts(root, c["file"], c["rank"], n_test)
        texts.append(t)
        notes_src.append(src)
    notes = S2.notes_features(texts, classes)
    st = S2.Static(stack["channels"], raw_chans, draws, test["grade"].to_numpy().astype(float), seen_cls, classes, notes)
    y_tr = labels.encode(train["root_code"], classes)
    kn = S2.knowledge(S2.task_keys(train["question_id"].to_numpy(), train["topic_name"].to_numpy()),
                      np.asarray(train["topic_name"].to_numpy()).astype(str), y_tr,
                      S2.task_keys(test["question_id"].to_numpy(), test["topic_name"].to_numpy()),
                      np.asarray(test["topic_name"].to_numpy()).astype(str), K)
    cand0 = S2.candidates(Q, seen_cls)
    tstats = S2.note_stats(notes["men_per_draw"], notes["m3_men"], cand0, cand0[:, 0])
    tstats["identical"] = {"d1-d2": S2.identical_share(texts[0], texts[1]), "d1-d3": S2.identical_share(texts[0], texts[2]),
                           "d2-d3": S2.identical_share(texts[1], texts[2])}
    rule = s2.get("guard_rule") or {}
    rel_tol = float(rule.get("notes_rel_tolerance", 0.10))
    widened_args = {}  # widened note-guard rule only if the deployment file names it; without the fields exactly as before
    if "share_z_max" in rule or "n_ref" in rule:
        if "share_z_max" not in rule or "n_ref" not in rule:
            raise SystemExit("--stage2: guard_rule names share_z_max and n_ref only together (widened rule)")
        if int(rule["n_ref"]) != len(train):
            raise SystemExit(f"--stage2: guard_rule n_ref {rule['n_ref']} is not the number of training rows ({len(train)})")
        # The rule was fixed in advance: only |z| ≤ 3 and ±10 % — a deployment file with other values would widen the guard
        # beyond the fixed rule and is rejected.
        if float(rule["share_z_max"]) != 3.0 or rel_tol != 0.10:
            raise SystemExit(f"--stage2: guard_rule differs from the widened rule (share_z_max {rule['share_z_max']}, notes_rel_tolerance {rel_tol}; "
                             "required 3 and 0.10)")
        widened_args = {"share_z_max": float(rule["share_z_max"]), "n_ref": int(rule["n_ref"]), "n_test": n_test}
    ga = S2.guard_notes(s2["guard_reference"], tstats, rel_tol, **widened_args)
    fs = "main" if ga["ok"] else "fallback"
    model = s2["models"][fs]
    fs_feat, protect = _stage2_model_form(model, fs, classes)  # feature set and NONE guard of the model itself; without the fields = fs, no guard
    mm_info = None
    if fs_feat == S2.MM_SET:  # row feature mech27 from the frozen pr27 test answers of the deployment file
        mech, minor, mm_info = _test_mech27(s2, test, classes, root)
        S2.attach_mech(st, mech, minor)
    rows = np.arange(n_test)
    X, cand, names = S2.pair_features(st, rows, Q, P_pre, kn, fs_feat)
    x_info = None
    if "extra_columns" in s2:  # additive: extra columns at the end of the main model; without the field everything unchanged
        Ex, x_names, x_info = _stage2_extra_columns(s2, train, test, classes, root, cand, [raw_chans[nm] for nm in stack["channels"]],
                                                    draws, seen_cls, xcontrol)
        if xcontrol is not None:  # positive control: never a file; exit code 5 = the test guard fires (passed), 6 = it does not
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps({"control": xcontrol, "extra_columns": x_info}, indent=1, ensure_ascii=False, default=float) + "\n")
            print(f"  CONTROL {xcontrol}: {'the test guard fires — control passed' if not x_info['guard']['ok'] else 'the test guard does NOT fire — control FAILED'}; "
                  "no file")
            raise SystemExit(5 if not x_info["guard"]["ok"] else 6)
        if fs == "main":
            X = np.concatenate([X, Ex], 1)
            names = names + x_names
    elif xcontrol is not None:
        raise SystemExit("--stage2-xcontrol only with a deployment file that carries the field extra_columns")
    if names != model["features"] or not np.array_equal(cand, cand0):
        raise SystemExit(f"--stage2: features of the test side ({len(names)}) do not match the deployment file ({len(model['features'])})")
    top = S2.reorder(cand, S2.scores(model, X))
    n_protected = 0
    if protect:  # NONE guard: base top NONE -> the base order of the row stays (before guard (c), like the OOF share of the set)
        top_unprotected = top
        top = S2.protect_none(top, cand)
        n_protected = int((top_unprotected != top).any(1).sum())
    changed = float((top[:, 0] != cand[:, 0]).mean())
    oof_share = float(model.get("oof_changed_top_share", 0.0))
    lo_f, hi_f = (s2.get("guard_rule") or {}).get("changed_top_factor", [0.5, 2.0])
    gc_ok = bool(lo_f * oof_share <= changed <= hi_f * oof_share)
    novel = float(X[:, names.index("novel_task")].mean())
    Q_new, ties = S2.apply_to_probs(Q, cand, top, seen_cls)
    t0, t1 = metric.top3_indices(Q, seen_cls), metric.top3_indices(Q_new, seen_cls)
    info = {"file": str(path), "sha256": _file_sha256(Path(path)), "null": null, "feature_set": fs,
            "fallback": bool(fs == "fallback"), "n_features": len(names), "K": S2.K_CAND, "lambda": S2.LAMBDA,
            "guard_notes": {"ok": ga["ok"], "rel": rel_tol, "test": tstats, "reference": s2["guard_reference"], "checks": ga["checks"]},
            "guard_novel_task_top5": {"test": novel, "oof": float(s2.get("novel_task_share_top5", float("nan")))},
            "guard_changed_top": {"test": changed, "oof": oof_share, "factor": [lo_f, hi_f], "ok": gc_ok, "enforced": not null},
            "share_top3_changed": float((t0 != t1).any(1).mean()), "share_top3_changed_oof": model.get("oof_changed_top3_share"),
            "rows_tie_fixed": int(ties), "test_notes": notes_src,
            "order": "blend -> margin alignment -> second stage -> REL-01"}
    if "feature_set" in model or "protect_none" in model:  # only with the main_guard fields (otherwise the statistics file stays unchanged)
        info.update({"model_feature_set": fs_feat, "protect_none": protect, "rows_protected": n_protected})
    if mm_info is not None:  # only with main_guard_i1_mm
        info["mech27"] = mm_info
    if x_info is not None:  # only with the field extra_columns
        info["extra_columns"] = {**x_info, "used_by_model": fs == "main"}
    if widened_args:  # only with the widened rule: rule and sample sizes in the statistics (otherwise the statistics file stays unchanged)
        info["guard_notes"].update({"rule": str(rule.get("rule", "widened")), **widened_args})
    print(f"  SECOND STAGE ({Path(path).name}{', NULL MODEL' if null else ''}): note guard{' (widened rule: band or |z| ≤ ' + format(widened_args['share_z_max'], 'g') + ')' if widened_args else ''} {'passed' if ga['ok'] else 'FAILED'} -> feature set {fs} "
          f"({len(names)}); new top code in {changed:.4f} of the test rows (OOF {oof_share:.4f}, band ×{lo_f:g}…×{hi_f:g} -> "
          f"{'ok' if gc_ok else 'VIOLATED'}); first three changed {info['share_top3_changed']:.4f} (OOF {float(info['share_top3_changed_oof'] or 0.0):.4f}); \"novel for the task\" among the first 5: test "
          f"{novel:.4f}, OOF {info['guard_novel_task_top5']['oof']:.4f}")
    if fs_feat != fs or protect:  # only with the main_guard fields
        print(f"    SECOND STAGE, model {fs}: feature set {fs_feat} ({len(names)}), NONE guard {'on' if protect else 'off'} -> "
              f"{n_protected} test rows keep the base order (count only)")
    for c in ga["checks"]:
        if not c["ok"]:
            print(f"    note guard failed: {c['stat']} test {c['test']:.4f} outside [{c['lo']:.4f}, {c['hi']:.4f}]"
                  + (f", |z| {abs(c['z']):.2f} > {widened_args['share_z_max']:g} (training value {c['ref']:.4f})" if widened_args and c.get("z") is not None else ""))
        elif widened_args and c.get("passed_by") == "z":
            print(f"    note guard (widened rule): {c['stat']} test {c['test']:.4f} outside [{c['lo']:.4f}, {c['hi']:.4f}], passed via |z| {abs(c['z']):.2f} "
                  f"≤ {widened_args['share_z_max']:g} (training value {c['ref']:.4f})")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(info, indent=1, ensure_ascii=False, default=float) + "\n")
    if x_info is not None and not x_info["guard"]["ok"]:  # test guard per column — abort without a file (also for the null model), no dropping
        raise SystemExit(f"--stage2: test guard of the extra columns violated in {len(x_info['guard']['failed'])} statistics "
                         f"({', '.join(c['scope'] + ' ' + c['stat'] for c in x_info['guard']['failed'][:6])}); no file written")
    if not null and not gc_ok:
        raise SystemExit(f"--stage2: guard (c) violated — new top code in {changed:.4f} of the test rows, required {lo_f * oof_share:.4f}…{hi_f * oof_share:.4f} "
                         f"(OOF {oof_share:.4f}); no file written")
    return Q_new, info


def class_mass_stats(Q: np.ndarray, P_pre: np.ndarray, y_train: np.ndarray, min_rows: int, classes: list[str]) -> dict:
    """L1 distance of the mean class mass of the test rows from the target of the alignment (training shares of the classes with at
    least min_rows rows; sum over the classes with a target) after (Q) and before (P_pre) the alignment, plus the NONE mass and the
    mass of the general codes against their targets. Aggregates only."""
    K = len(classes)
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=K)
    target = np.where(counts >= min_rows, counts, 0) / len(y_train)
    used = target > 0
    mq, mp = np.asarray(Q, np.float64).mean(0), np.asarray(P_pre, np.float64).mean(0)
    gen = [classes.index(c) for c in S2.GENERAL_CODES]
    return {"l1_after": float(np.abs(mq - target)[used].sum()), "l1_before": float(np.abs(mp - target)[used].sum()), "classes": int(used.sum()),
            "min_rows": int(min_rows), "none_after": float(mq[0]), "none_before": float(mp[0]), "none_target": float(target[0]),
            "general_after": float(mq[gen].sum()), "general_target": float(target[gen].sum()), "mass_without_target_after": float(mq[~used].sum())}


def _train_notes_texts(root: Path, rel_file: str, want_sha: str, n_train: int) -> tuple[list[str], dict]:
    """Frozen M3 notes of the training rows of one draw (eda/c129_safety_nd.py export-notes): file = SHA-256 of the deployment file =
    checksum of its meta file, table train, every row exactly once. Only the field text, only for the machine similarity."""
    f = Path(root) / rel_file
    mf = f.with_suffix(".meta.json")
    if not f.is_file() or not mf.is_file():
        raise SystemExit(f"--stage2: {rel_file} or {mf.name} missing")
    sha = _file_sha256(f)
    meta = json.loads(mf.read_text())
    if sha != want_sha or meta.get("sha256") != sha or meta.get("table") != "train" or int(meta.get("rows", -1)) != n_train:
        raise SystemExit(f"--stage2: training notes {rel_file}: checksum {sha[:16]}… / meta file do not match the deployment file")
    texts: dict[int, str] = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            i = int(r["row"])
            if r.get("table") != "train" or i in texts or not 0 <= i < n_train:
                raise SystemExit(f"--stage2: training notes {rel_file}: rows not unique or not table train")
            texts[i] = str(r.get("text") or "")
    if len(texts) != n_train:
        raise SystemExit(f"--stage2: training notes {rel_file}: {len(texts)} of {n_train} rows")
    return [texts[i] for i in range(n_train)], {"file": rel_file, "sha256": sha}


STAGE2_CONTROLS = ("copy", "permute", "truncate", "pr27_shift", "pr27_m3d1", "mech_flip")
M3_CONTROLS = ("copy", "permute", "truncate")  # change the M3 test notes
R_CONTROLS = ("pr27_shift", "pr27_m3d1")  # change only the pr27 test notes (feature set fallback_i1_ndr only)
MECH_CONTROLS = ("mech_flip",)  # inverts mech27 of the test rows (feature set fallback_i1_ndr_mmid only; notes unchanged)
MECH_RULE = {"mech_rule": "mech27 guard", "mech_rel": 0.10, "mech_z_max": 3.0}  # guard of mech27: band ±10 % OR |z| ≤ 3 (as the widened rule), abort only


def mech_guard(train_share: float, n_train: int, test_share: float, n_test: int, rel: float = 0.10, z_max: float = 3.0) -> dict:
    """Share "mechanically correct" (mech27) of the test rows against that of the training rows (pure pr27 reading, 0.290): passes inside the
    band ±rel relative OR at |z| ≤ z_max (two-sided two-sample binomial test, stage2.share_z) — like the widened rule, widening only.
    Aggregates only."""
    lo, hi = float(train_share) * (1.0 - rel), float(train_share) * (1.0 + rel)
    z = S2.share_z(float(train_share), int(n_train), float(test_share), int(n_test))
    ok_band, ok_z = bool(lo <= test_share <= hi), bool(abs(z) <= z_max)
    return {"stat": "share_mech", "test": float(test_share), "ref": float(train_share), "lo": lo, "hi": hi, "z": z, "ok_band": ok_band, "ok_z": ok_z,
            "ok": bool(ok_band or ok_z), "passed_by": "band" if ok_band else ("z" if ok_z else None), "rel": rel, "z_max": z_max,
            "n_ref": int(n_train), "n_test": int(n_test)}


def _control_texts(texts: list[list[str]], control: str) -> list[list[str]]:
    """Positive control (in memory only): copy = draw d1 for all three; permute = rows of every draw permuted by a fixed permutation
    (default_rng(0)); truncate = every note cut to its first 300 characters."""
    if control == "copy":
        return [list(texts[0]) for _ in texts]
    if control == "permute":
        perm = np.random.default_rng(0).permutation(len(texts[0]))
        return [[t[int(i)] for i in perm] for t in texts]
    if control == "truncate":
        return [[(x or "")[:300] for x in t] for t in texts]
    raise SystemExit(f"--stage2-control {control!r}")


def _control_pr27(pr_texts: list[str], m3_texts: list[list[str]], control: str) -> list[str]:
    """Positive control of the pr27 notes (in memory only): pr27_shift = shifted by one row (row i gets the note of row i + 1);
    pr27_m3d1 = replaced by the (unchanged) M3 notes d1; truncate = cut to the first 300 characters (as the M3 notes); copy and permute
    change only the M3 notes."""
    if control == "pr27_shift":
        return list(pr_texts[1:]) + list(pr_texts[:1])
    if control == "pr27_m3d1":
        return list(m3_texts[0])
    if control == "truncate":
        return [(x or "")[:300] for x in pr_texts]
    if control in ("copy", "permute", "mech_flip"):  # mech_flip changes only mech27
        return list(pr_texts)
    raise SystemExit(f"--stage2-control {control!r}")


def _pr27_notes_texts(root: Path, cfg: dict, n_test: int) -> tuple[list[str], dict]:
    """Frozen pr27 test notes (tools/export_proto_notes.py --allow-test: reasoning of the first pass) for the columns R. Field
    r_notes.pr27_test of the deployment file: file = SHA-256 of the deployment file = checksum of the meta file, table test with every row
    exactly once, source of the meta file = source of the deployment file. Rows are split at "\\n" (never splitlines(): U+2028 and similar
    characters stand raw in the texts). Only the field text is read, for machine counting; none of it is printed. -> (texts in table
    order, checksums)."""
    ts = (cfg or {}).get("pr27_test") or {}
    rel = str(ts.get("file") or "")
    f = Path(root) / rel
    mf = f.with_suffix(".meta.json")
    if not rel or not f.is_file() or not mf.is_file():
        raise SystemExit(f"--stage2: pr27 test notes {rel or '(r_notes.pr27_test missing)'} or their meta file missing")
    sha = _file_sha256(f)
    meta = json.loads(mf.read_text())
    problems = []
    if sha != ts.get("sha256") or meta.get("sha256") != sha:
        problems.append(f"checksum {sha[:16]}… does not match the deployment file or the meta file")
    if meta.get("table") != "test" or int(meta.get("rows", -1)) != n_test:
        problems.append(f"table {meta.get('table')!r} with {meta.get('rows')} rows")
    if [s.get("source") for s in meta.get("sources") or []] != [ts.get("source")]:
        problems.append(f"source of the meta file is not {ts.get('source')!r}")
    texts: dict[int, str] = {}
    for line in f.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r["row"])
        if r.get("table") != "test" or i in texts or not 0 <= i < n_test:
            problems.append("rows not unique or not table test")
            break
        texts[i] = str(r.get("text") or "")
    if len(texts) != n_test:
        problems.append(f"{len(texts)} of {n_test} rows")
    if problems:
        raise SystemExit(f"--stage2: pr27 test notes {rel}: " + "; ".join(problems))
    return [texts[i] for i in range(n_test)], {"file": rel, "sha256": sha, "source": ts.get("source")}


def stage2_safe_order(model: dict, X: np.ndarray, cand: np.ndarray, protect: bool) -> tuple[np.ndarray, int]:
    """New order of the first K candidates (stage2.reorder) and then — with protect — the NONE guard: if the base order has NONE (class 0)
    at the top and the model wants another top code, the base order of the whole row stays.
    -> (order [R, K], number of guarded rows)."""
    top = S2.reorder(cand, S2.scores(model, X))
    if not protect:
        return top, 0
    m = (cand[:, 0] == 0) & (top[:, 0] != 0)
    top = top.copy()
    top[m] = cand[m]
    return top, int(m.sum())


def apply_stage2_safe(path: Path, stack_file: Path, stack: dict, Q: np.ndarray, P_pre: np.ndarray, raw_chans: dict, draw_keep: dict, rank_only: dict,
                      train: pd.DataFrame, test: pd.DataFrame, classes: list[str], seen_cls: np.ndarray, root: Path,
                      report_path: Path | None = None, control: str | None = None) -> tuple[np.ndarray, dict]:
    """SECOND STAGE of the SAFETY line (deployment file with align_mode "inductive" or "set"; see the module docstring). Q = test blend
    after the alignment (inductive, or as one set over the test rows), P_pre = the blend before it. Feature set "fallback" or
    "fallback_i1" without notes (no note guard), NONE guard as in the file, guard (c) on the share of new top codes after the guard;
    with "set" also the class masses (class_mass). Aggregates only. -> (Q with a new order of the first K, statistics)."""
    s2 = json.loads(Path(path).read_text())
    n_test, K = len(test), len(classes)
    null = bool(s2.get("null"))
    model = s2.get("model") or {}
    fs = s2.get("feature_set")
    protect = s2.get("protect_none", False)
    mode = s2.get("align_mode")
    trans = mode == "set"  # margin alignment as one set over the test rows; otherwise inductive as before
    lab = "transductive" if trans else "inductive"
    problems = []
    if mode not in ("inductive", "set"):
        problems.append(f"align_mode {mode!r} instead of 'inductive' (or 'set')")
    if s2.get("stack_sha256") != _file_sha256(stack_file):
        problems.append(f"blend file {stack_file.name} is not the one of the deployment file ({s2.get('stack')})")
    if s2.get("channels") != list(stack["channels"]) or float(s2.get("arith_beta", -1)) != arith_beta_of(stack):
        problems.append("channels or β of the deployment file do not match the blend file")
    if int(s2.get("K", -1)) != S2.K_CAND or float(s2.get("lambda", -1)) != S2.LAMBDA:
        problems.append(f"K {s2.get('K')} / λ {s2.get('lambda')} instead of {S2.K_CAND} / {S2.LAMBDA}")
    if fs not in ("fallback", S2.FB_I1_SET, S2.ND_SET, S2.NDR_SET, S2.NDR_MMID_SET):  # fallback_i1 = fallback + interactions I1 (without notes); + ND; + R; + mm + identifiers
        problems.append(f"feature set {fs!r}: the safety line only knows \"fallback\", \"fallback_i1\", \"fallback_i1_nd\", \"fallback_i1_ndr\" and "
                        f"\"fallback_i1_ndr_mmid\"")
    mmid = fs == S2.NDR_MMID_SET  # computes the path of fallback_i1_ndr and in addition minor_x_mech27 + identifiers of the general codes
    nd = fs in (S2.ND_SET, S2.NDR_SET, S2.NDR_MMID_SET)  # fallback_i1_ndr computes the path of fallback_i1_nd and in addition the columns R
    ndr = fs in (S2.NDR_SET, S2.NDR_MMID_SET)
    if nd and mode != "set":
        problems.append(f"feature set {fs} only with align_mode \"set\"")
    if control is not None and not nd:
        problems.append(f"--stage2-control only with the feature set {S2.ND_SET}")
    elif control in R_CONTROLS and not ndr:
        problems.append(f"--stage2-control {control} only with the feature set {S2.NDR_SET}")
    elif control in MECH_CONTROLS and not mmid:
        problems.append(f"--stage2-control {control} only with the feature set {S2.NDR_MMID_SET}")
    if not isinstance(protect, bool) or (protect and classes[0] != "NONE"):
        problems.append(f"protect_none {protect!r} (boolean; the guard requires NONE as class 0)")
    dch = s2.get("draws_channel")
    draws = draw_keep.get(dch) or []
    specs = (rank_only.get(dch) or "").split(",")
    if "draws_first" in s2:  # channel made of more draws (m3x4_full = d1..d4), second-stage features from d1–d3 (without the field unchanged)
        if specs != list(s2.get("test_draws") or []):
            problems.append(f"M3 test draws of channel {dch}: test rankings {specs} against the deployment file {s2.get('test_draws')}")
        draws, specs = stage2_draws_first(s2, draws, specs)
    elif len(draws) != len(S2.DRAWS) or specs != list(s2.get("test_draws") or []):
        problems.append(f"M3 test draws of channel {dch}: {len(draws)} matrices, test rankings {specs} against the deployment file {s2.get('test_draws')}")
    missing = [nm for nm in stack["channels"] if nm not in raw_chans]
    if missing:
        problems.append(f"channels without a test matrix {missing}")
    if problems:
        raise SystemExit(f"--stage2 ({lab}): " + "; ".join(problems))
    y_tr = labels.encode(train["root_code"], classes)
    train_keys = S2.task_keys(train["question_id"].to_numpy(), train["topic_name"].to_numpy())
    test_keys = S2.task_keys(test["question_id"].to_numpy(), test["topic_name"].to_numpy())
    kn = S2.knowledge(train_keys, np.asarray(train["topic_name"].to_numpy()).astype(str), y_tr,
                      test_keys, np.asarray(test["topic_name"].to_numpy()).astype(str), K)
    cand0 = S2.candidates(Q, seen_cls)
    nd_info = None
    notes = None
    if nd:  # notes of the test rows, reference of the training notes, guards (abort only)
        notes_cfg = s2.get("test_notes") or []
        tr_cfg = s2.get("train_notes") or []
        if [c.get("rank") for c in notes_cfg] != specs or len(tr_cfg) != len(S2.DRAWS):
            raise SystemExit(f"--stage2 ({lab}): test notes {[c.get('rank') for c in notes_cfg]} do not match the test draws {specs} "
                             f"or {len(tr_cfg)} training note files instead of {len(S2.DRAWS)}")
        rule = s2.get("guard_rule") or {}
        if (float(rule.get("share_z_max", -1)) != 3.0 or float(rule.get("notes_rel_tolerance", -1)) != 0.10 or int(rule.get("n_ref", -1)) != len(train)
                or float(rule.get("sim_rel", -1)) != NS.SIM_REL or float(rule.get("sim_z_max", -1)) != NS.SIM_Z_MAX or rule.get("abort_only") is not True):
            raise SystemExit(f"--stage2 ({lab}): guard_rule differs from the rule of the feature set (widened 3 / 0.10 / n_ref {len(train)}, "
                             "similarity 0.10 / 3, abort only)")
        rcfg = s2.get("r_notes") or {}
        if ndr and (rule.get("r_rule") != "safety-ndr rule (6)" or float(rule.get("r_rel", -1)) != NR.REL or float(rule.get("r_z_max", -1)) != NR.Z_MAX
                    or not s2.get("r_reference") or rcfg.get("extractor") != NR.VERSION or tuple(rcfg.get("features") or ()) != S2.R_FEATS):
            raise SystemExit(f"--stage2 ({lab}): guard_rule, r_reference or r_notes differ from the rule of the feature set (guard of the columns R "
                             f"{NR.REL:g} / {NR.Z_MAX:g}, extractor {NR.VERSION}, columns {', '.join(S2.R_FEATS)})")
        if mmid and ({k: rule.get(k) for k in MECH_RULE} != MECH_RULE or (s2.get("mech27") or {}).get("train_share_mech") is None):
            raise SystemExit(f"--stage2 ({lab}): guard_rule or mech27 differ from the rule of the feature set (guard of mech27 {MECH_RULE}, train_share_mech)")
        texts, notes_src = [], []
        for c in notes_cfg:
            tx, src = _test_notes_texts(root, c["file"], c["rank"], n_test)
            texts.append(tx)
            notes_src.append(src)
        tr_texts, tr_src = [], []
        for c in tr_cfg:
            tx, src = _train_notes_texts(root, c["file"], c["sha256"], len(train))
            tr_texts.append(tx)
            tr_src.append(src)
        pr_texts, pr_src = _pr27_notes_texts(root, rcfg, n_test) if ndr else (None, None)  # pr27 test notes of the columns R
        if control is not None:
            m3_plain = texts
            if control in M3_CONTROLS:
                texts = _control_texts(texts, control)
            if ndr:
                pr_texts = _control_pr27(pr_texts, m3_plain, control)
        notes = S2.notes_features(texts, classes)
        tstats = S2.note_stats(notes["men_per_draw"], notes["m3_men"], cand0, cand0[:, 0])
        tstats["identical"] = {"d1-d2": S2.identical_share(texts[0], texts[1]), "d1-d3": S2.identical_share(texts[0], texts[2]),
                               "d2-d3": S2.identical_share(texts[1], texts[2])}
        ga = S2.guard_notes(s2["guard_reference"], tstats, 0.10, share_z_max=3.0, n_ref=len(train), n_test=n_test)
        Ssim = NS.mask_same_task(NS.similarity(texts, tr_texts), test_keys, train_keys)
        sg = NS.sim_guard(s2["sim_reference"], NS.sim_stats(NS.top_sim(Ssim)), NS.SIM_REL, NS.SIM_Z_MAX)
        kn["demo_sim"] = NS.demo_sim(Ssim, y_tr, K)
        rg, R_test = None, None
        if ndr:  # columns R of the test rows and their guard (aggregates only, abort only)
            m3_st = [NR.mention_stats(t, classes) for t in texts]
            pr_st = NR.mention_stats(pr_texts, classes)
            R_test = NR.r_features(m3_st, pr_st)
            rstats = NR.guard_stats(m3_st, pr_st, cand0)
            rg = {**NR.guard(s2["r_reference"], rstats, NR.REL, NR.Z_MAX), "test": rstats, "reference": s2["r_reference"], "enforced": not null}
        mg, mech_test, mm_info = None, None, None
        if mmid:  # mech27 of the test rows (frozen pr27 test answers, pure pr27 reading) and its guard (aggregates only, abort only)
            mech_test, minor_cls, mm_info = _test_mech27(s2, test, classes, root)
            if control == "mech_flip":
                mech_test = 1.0 - mech_test
            mg = {**mech_guard(float(s2["mech27"]["train_share_mech"]), len(train), float(mech_test.mean()), n_test, MECH_RULE["mech_rel"],
                               MECH_RULE["mech_z_max"]), "enforced": not null}
            mm_info["guard"] = mg
        nd_info = {"guard_notes": {"ok": ga["ok"], "rule": "widened", "test": tstats, "reference": s2["guard_reference"], "checks": ga["checks"],
                                   "enforced": not null},
                   "guard_sim": {**sg, "enforced": not null}, "test_notes": notes_src, "train_notes": tr_src, "control": control}
        if ndr:
            nd_info.update({"guard_r": rg, "r_notes": pr_src})
        if mmid:
            nd_info["guard_mech"] = mg
        bad = [c for c in ga["checks"] if not c["ok"]]
        print(f"  SECOND STAGE, SAFETY LINE (note features{', CONTROL ' + control if control else ''}): note guard (widened rule) {'held' if ga['ok'] else 'FAILED'}"
              f" ({len(ga['checks']) - len(bad)}/{len(ga['checks'])}); similarity guard top similarity test {sg['test']['mean']:.4f} against training "
              f"{sg['reference']['mean']:.4f} (band {sg['band'][0]:.4f}…{sg['band'][1]:.4f}, z {sg['z']:+.2f}) {'held' if sg['ok'] else 'FAILED'} (aggregates only)")
        for c in bad:
            print(f"    note guard failed: {c['stat']} test {c['test']:.4f} outside [{c['lo']:.4f}, {c['hi']:.4f}]"
                  + (f", |z| {abs(c['z']):.2f} > 3 (training value {c['ref']:.4f})" if c.get("z") is not None else ""))
        if ndr:
            rbad = [c for c in rg["checks"] if not c["ok"]]
            print(f"  SECOND STAGE, SAFETY LINE (columns R{', CONTROL ' + control if control else ''}): guard R {'held' if rg['ok'] else 'FAILED'}"
                  f" ({len(rg['checks']) - len(rbad)}/{len(rg['checks'])}); " + "; ".join(
                      f"{c['stat']} test {c['test']:.4f} against {c['ref']:.4f} (z {c['z']:+.2f}, {c['passed_by'] or 'FAILED'})" for c in rg["checks"])
                  + " (aggregates only)")
            for c in rbad:
                print(f"    guard R failed: {c['stat']} test {c['test']:.4f} outside [{c['lo']:.4f}, {c['hi']:.4f}], |z| {abs(c['z']):.2f} > {NR.Z_MAX:g} "
                      f"(training value {c['ref']:.4f})")
        if mmid:
            print(f"  SECOND STAGE, SAFETY LINE (mech27 guard{', CONTROL ' + control if control else ''}): guard of mech27 "
                  f"{'held' if mg['ok'] else 'FAILED'} — share test {mg['test']:.4f} against training {mg['ref']:.4f} (band {mg['lo']:.4f}…{mg['hi']:.4f}, "
                  f"z {mg['z']:+.2f}, {mg['passed_by'] or 'FAILED'}) (aggregates only)")
        fired = not (ga["ok"] and sg["ok"] and (rg is None or rg["ok"]) and (mg is None or mg["ok"]))
        if control is not None:  # positive control: never a file; exit code 5 = a guard fires (passed), 6 = none fires
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(nd_info, indent=1, ensure_ascii=False, default=float) + "\n")
            print(f"  CONTROL {control}: {'a guard fires — control passed' if fired else 'NO guard fires — control FAILED'}; no file")
            raise SystemExit(5 if fired else 6)
        if fired and not null:
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(nd_info, indent=1, ensure_ascii=False, default=float) + "\n")
            if mmid:
                raise SystemExit(f"--stage2 ({lab}): note, similarity, R or mech27 guard failed (abort only) — no file written")
            if ndr:
                raise SystemExit(f"--stage2 ({lab}): note, similarity or R guard failed (abort only) — no file written")
            raise SystemExit(f"--stage2 ({lab}): note or similarity guard failed (abort only) — no file written")
    st = S2.Static(stack["channels"], raw_chans, draws, test["grade"].to_numpy().astype(float), seen_cls, classes, notes)
    if ndr:  # columns R per (test row, class)
        S2.attach_r(st, R_test)
    if mmid:  # row feature mech27 and class mask minor (the identifiers of the general codes are in Static)
        S2.attach_mech(st, mech_test, minor_cls)
    X, cand, names = S2.pair_features(st, np.arange(n_test), Q, P_pre, kn, fs)
    if names != model.get("features") or not np.array_equal(cand, cand0):
        raise SystemExit(f"--stage2 ({lab}): features of the test side ({len(names)}) do not match the deployment file ({len(model.get('features') or [])})")
    top, n_protected = stage2_safe_order(model, X, cand, protect)
    changed = float((top[:, 0] != cand[:, 0]).mean())
    oof_share = float(model.get("oof_changed_top_share", 0.0))
    lo_f, hi_f = (s2.get("guard_rule") or {}).get("changed_top_factor", [0.5, 2.0])
    gc_ok = bool(lo_f * oof_share <= changed <= hi_f * oof_share)
    novel = float(X[:, names.index("novel_task")].mean())
    Q_new, ties = S2.apply_to_probs(Q, cand, top, seen_cls)
    t0, t1 = metric.top3_indices(Q, seen_cls), metric.top3_indices(Q_new, seen_cls)
    info = {"file": str(path), "sha256": _file_sha256(Path(path)), "null": null, "align_mode": mode, "feature_set": fs, "fallback": True,
            "n_features": len(names), "K": S2.K_CAND, "lambda": S2.LAMBDA, "protect_none": protect, "rows_protected": n_protected,
            "guard_notes": None, "guard_novel_task_top5": {"test": novel, "oof": float(s2.get("novel_task_share_top5", float("nan")))},
            "guard_changed_top": {"test": changed, "oof": oof_share, "factor": [lo_f, hi_f], "ok": gc_ok, "enforced": not null},
            "share_top3_changed": float((t0 != t1).any(1).mean()), "share_top3_changed_oof": model.get("oof_changed_top3_share"),
            "rows_tie_fixed": int(ties), "test_draws": specs,
            "order": ("blend -> margin alignment (one set over the test rows) -> second stage -> NONE guard -> REL-01" if trans
                      else "blend -> inductive alignment -> second stage -> NONE guard -> REL-01")}
    if trans:  # class masses after the alignment against their target (aggregates only; only in this mode, otherwise the report stays byte-identical)
        info["class_mass"] = class_mass_stats(Q, P_pre, y_tr, int(s2.get("align_min", 3)), classes)
    if nd_info is not None:  # only with fallback_i1_nd / _ndr (otherwise the report stays byte-identical)
        info.update({"guard_notes": nd_info["guard_notes"], "guard_sim": nd_info["guard_sim"], "test_notes": nd_info["test_notes"],
                     "train_notes": nd_info["train_notes"], "fallback": False})
    if ndr:  # only with fallback_i1_ndr (otherwise the report stays byte-identical)
        info.update({"guard_r": nd_info["guard_r"], "r_notes": nd_info["r_notes"]})
    if mmid:  # only with fallback_i1_ndr_mmid (otherwise the report stays byte-identical)
        info.update({"guard_mech": nd_info["guard_mech"], "mech27": mm_info})
    print(f"  SECOND STAGE, SAFETY LINE ({Path(path).name}{', NULL MODEL' if null else ''}): feature set {fs} ({len(names)}), "
          f"{('notes d1–d3 + example similarity' + (' + columns R (M3 d1–d3, pr27)' if ndr else '') + (' + minor_x_mech27 + identifiers' if mmid else '')) if nd else 'no notes'}; "
          f"NONE guard {'on' if protect else 'off'} ({n_protected} test rows keep the base order); new top code in {changed:.4f} of the test rows "
          f"(OOF {oof_share:.4f}, band ×{lo_f:g}…×{hi_f:g} -> {'ok' if gc_ok else 'VIOLATED'}); first three changed {info['share_top3_changed']:.4f} "
          f"(OOF {float(info['share_top3_changed_oof'] or 0.0):.4f}); \"novel for the task\" among the first 5: test {novel:.4f}, OOF {info['guard_novel_task_top5']['oof']:.4f}")
    if trans:
        cm = info["class_mass"]
        print(f"    margin alignment as one set: class masses of the test rows against the target ({cm['classes']} classes with at least {cm['min_rows']} "
              f"training rows) L1 {cm['l1_before']:.4f} before -> {cm['l1_after']:.4f} after the alignment; NONE {cm['none_after']:.4f} (target "
              f"{cm['none_target']:.4f}), general codes {cm['general_after']:.4f} (target {cm['general_target']:.4f}) (aggregates only)")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(info, indent=1, ensure_ascii=False, default=float) + "\n")
    if not null and not gc_ok:
        raise SystemExit(f"--stage2 ({lab}): guard (c) violated — new top code in {changed:.4f} of the test rows, required {lo_f * oof_share:.4f}…"
                         f"{hi_f * oof_share:.4f} (OOF {oof_share:.4f}); no file written")
    return Q_new, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--test", action="append", default=[], help="<channel>=<shard>[+<shard>...]")
    ap.add_argument("--none-test", action="append", default=[])
    ap.add_argument("--fill", action="append", default=[], help="<channel>=<channel>: test rows without a measurement of the first channel take the probabilities of the second (cascade)")
    ap.add_argument("--proto", action="append", default=[], help="<channel>=runs/<run>/<job>/<tag>[+…][,<more draws>]: channel from answers in the teacher protocol for the test rows (proto_gen)")
    ap.add_argument("--proto-max-missing", type=float, default=0.03, help="maximum share of test rows without a valid answer of a protocol channel")
    ap.add_argument("--proto-neutral", action="append", default=[], help="<channel>=<rows.npy> (relative to the data folder): test rows on which the protocol channel counts as not computed "
                                                                         "(gate; uniform there) — the same list as for the OOF artifact (eda/a8_proto_oof.py --neutral)")
    ap.add_argument("--rank", action="append", default=[], help="<channel>=runs/<run>/<job>/<tag>: exact ranking behind the notes of the test rows (probe proto_rank) — "
                                                                "the channel (also under --proto) becomes the mean channel 0.5·(named + ranking) like its OOF artifact (eda/c95_rank_mean.py)")
    ap.add_argument("--rank-max-missing", type=float, default=0.02, help="maximum share of test rows without an exact ranking (at least 98 %% required)")
    ap.add_argument("--rank-min-mass", type=float, default=0.9, help="smallest allowed median of the candidate mass of the ranking")
    ap.add_argument("--rank-none-band", nargs=2, type=float, default=[0.30, 0.38], help="malfunction band fixed in advance for the share of NONE at the top of the ranking")
    ap.add_argument("--rank-partial", action="store_true", help="deliberately accept an incomplete ranking run: rows without an exact ranking keep the named codes")
    ap.add_argument("--rank-only", action="append", default=[], help="<channel>=runs/<run>/<job>/<tag>: RANK-ONLY channel (probe proto_rank, e.g. M3 in sft mode) — "
                                                                     "exact distribution over all codes, rows without a ranking uniform; floor from the blend file; "
                                                                     "guards as for --rank (with --rank-partial rows without an exact ranking keep the uniform distribution); "
                                                                     "several draws separated by commas (OOF artifact from eda/a9_m3_draws_oof.py, same order): "
                                                                     "per row the mean of the probabilities, guards and settings comparison per draw")
    ap.add_argument("--rank-only-placeholder", action="append", default=[], help="<channel>: CODE CHECK ONLY with a foreign test ranking — the note guard "
                                                                                 "(rank.notes) skips the test notes; requires --out outside "
                                                                                 "artifacts/submissions; the file is never submitted")
    ap.add_argument("--rank-none-band-for", action="append", default=[], help="<channel>=<lo>,<hi>: own malfunction band fixed in advance for the share of NONE at "
                                                                              "the top of the ranking of ONE channel under --rank-only (instead of --rank-none-band)")
    ap.add_argument("--out")
    ap.add_argument("--align", type=float, default=0.0, help="margin alignment to the training class frequencies with this strength (0 = off; transductive: the test rows as one set)")
    ap.add_argument("--align-min", type=int, default=3, help="margin alignment only for classes with at least this many training rows")
    ap.add_argument("--align-inductive", type=float, default=0.0, help="inductive alignment with this strength: class weights from the blend's predictions for the training rows "
                                                                       "(0 = off; for the inductive line; not together with --align)")
    ap.add_argument("--suffix", default="", help="suffix for the names of the output files")
    ap.add_argument("--blank-rows", help="table positions of the test rows with a blank image (frozen list; path relative to the data folder): --blank-code is put at rank 1 there")
    ap.add_argument("--blank-code", default="REL-01", help="code for blank images (meta code of the taxonomy: upload does not match the task)")
    ap.add_argument("--stage2", help="deployment file of the second stage (eda/c118_stage2.py fit): conditional logit over the first K candidates "
                                     "after the margin alignment; requires --align as in the file; guards on the test notes and the change rate")
    ap.add_argument("--stage2-control", choices=STAGE2_CONTROLS, help="positive control (feature sets fallback_i1_nd/_ndr only; pr27_shift and "
                    "pr27_m3d1 only fallback_i1_ndr): change the test notes in memory; never writes a file, exit code 5 = a guard fires "
                    "(passed), 6 = none fires")
    ap.add_argument("--stage2-report", help="file for the guard statistics of the second stage (aggregates only; e.g. artifacts/eda/private/stage2/guards.json)")
    ap.add_argument("--stage2-xcontrol", choices=XS.CONTROLS, help="positive control of the test guard of the extra columns (only with a deployment file with "
                    "the field extra_columns): perturb the test sources in memory; never writes a file, exit code 5 = the guard fires (passed), 6 = it does not")
    a = ap.parse_args()
    if a.align < 0 or a.align_inductive < 0:
        raise SystemExit("--align / --align-inductive: the strength must not be negative")
    if a.align and a.align_inductive:
        raise SystemExit("--align-inductive and --align are mutually exclusive (the inductive alignment is meant for the inductive line).")
    if a.stage2_report and not a.stage2:
        raise SystemExit("--stage2-report only with --stage2")
    if a.stage2_control and not a.stage2:
        raise SystemExit("--stage2-control only with --stage2")
    if a.stage2_xcontrol and not a.stage2:
        raise SystemExit("--stage2-xcontrol only with --stage2")
    if a.stage2:  # the second stage was fitted on the alignment over the whole set: the same strength and class limit are required
        s2j = json.loads(Path(a.stage2).read_text())
        if s2j.get("align_mode") == "inductive":  # inductive line: inductive alignment as in the fit (per fold)
            if a.align or float(a.align_inductive) != float(s2j.get("align_inductive", -1)) or int(a.align_min) != int(s2j.get("align_min", -1)):
                raise SystemExit(f"--stage2 (inductive) requires --align-inductive {s2j.get('align_inductive')} --align-min {s2j.get('align_min')} "
                                 "(as in the fit) and no --align")
        elif a.align_inductive or float(a.align) != float(s2j.get("align", -1)) or int(a.align_min) != int(s2j.get("align_min", -1)):
            raise SystemExit(f"--stage2 requires --align {s2j.get('align')} --align-min {s2j.get('align_min')} (transductive, as in the fit)")

    stack = json.loads((COMP / "artifacts" / f"stack_{a.name}.json").read_text())
    if stack.get("probes"):
        raise SystemExit("Blends with probes on hidden states cannot yet be applied to test rows here.")
    root = C.find_data_root()
    train = C.load_table(root, "train")
    test = C.load_table(root, "test")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    seen_cls = labels.seen_mask(train["root_code"], classes)
    absent = ~seen_cls
    n_test, K = len(test), len(classes)
    given = dict(t.split("=", 1) for t in a.test)
    fill = dict(t.split("=", 1) for t in a.fill)
    proto = dict(t.split("=", 1) for t in a.proto)
    rank = dict(t.split("=", 1) for t in a.rank)
    for k in proto:
        if k not in stack["channels"] or k in given:
            raise SystemExit(f"--proto {k}: must be a channel of the blend and must not also be listed under --test")
    for k in rank:
        if k not in proto:
            raise SystemExit(f"--rank {k}: the channel must also be listed under --proto (named codes of the same notes)")
    rank_ref = {}
    rank_cmp: dict[str, str] = {}  # comparison form of the ranking (rank_mean.rank_compare); without the field empty = as before
    rank_notes: dict[str, dict] = {}  # note guard of the mean channel (rank_mean.notes); without the field empty = as before
    rank_named: dict[str, dict] = {}  # separate source of the named codes (rank_mean.named); without the field empty = as before
    for k in proto:
        rm_def = rank_mean_guard(k, stack, rank.get(k), len(proto[k].split(",")))
        if rm_def:
            if not rm_def.get("rank_source"):
                raise SystemExit(f"channel {k}: meta_{k}.json names no ranking of the training rows (rank_mean.rank_source)")
            rank_ref[k] = rm_def["rank_source"]
            if "rank_compare" in rm_def:  # fold shards of a fine-tuned model (without the field as before)
                rank_cmp[k] = str(rm_def["rank_compare"])
                if rank_cmp[k] not in ("c95", "sft"):
                    raise SystemExit(f"channel {k}: rank_mean.rank_compare {rank_cmp[k]!r} unknown (c95 | sft)")
            ninfo = rank_mean_notes_guard(k, rank[k], root, n_test, len(train), rm_def)
            if ninfo is not None:
                rank_notes[k] = ninfo
            named_info = rank_mean_named_guard(k, proto[k].split(","), rm_def, n_test)
            if named_info is not None:
                rank_named[k] = named_info
    for k, v in fill.items():
        if k not in stack["channels"] or (v not in given and v != "prior"):
            raise SystemExit(f"--fill {k}={v}: {k} must be a channel of the blend, {v} needs its own entry under --test")
    rank_only = dict(t.split("=", 1) for t in a.rank_only)
    for k in rank_only:
        if k not in stack["channels"] or k in given or k in proto or k in rank or k in fill:
            raise SystemExit(f"--rank-only {k}: must be a channel of the blend and must not also be listed under --test, --proto, --rank or --fill")
    for k in stack.get("rank_only") or {}:
        if k not in rank_only:
            raise SystemExit(f"channel {k}: the blend file lists it as a rank-only channel (eda/c99_m3_deploy.py) — --rank-only {k}=runs/… missing")
    rank_only_ref = {k: rank_only_guard(k, stack) for k in rank_only}
    rank_only_draws = {}  # several draws (eda/a9_m3_draws_oof.py): a comma in the test spec or in the source of the OOF artifact
    for k in rank_only:
        if "," in rank_only[k] or "," in rank_only_ref[k]:
            rank_only_draws[k] = rank_only_draws_guard(k, stack, rank_only[k].split(","), root, len(train),
                                                       rowlist_placeholder=k in set(a.rank_only_placeholder))
    rank_only_rowlists = {k: gi.pop("_rowlist") for k, (_, gi) in rank_only_draws.items() if "_rowlist" in gi}  # draws with a row list; otherwise empty
    placeholder = sorted(set(a.rank_only_placeholder))
    for k in placeholder:
        if k not in rank_only:
            raise SystemExit(f"--rank-only-placeholder {k}: no such channel under --rank-only")
    if placeholder and (not a.out or Path(a.out).resolve().is_relative_to((COMP / "artifacts" / "submissions").resolve())):
        raise SystemExit("--rank-only-placeholder: only with --out outside artifacts/submissions (a placeholder file is never submitted)")
    rank_only_notes = {}  # note guard (field rank.notes, eda/c117_q397m3_deploy.py); without the field None — all other channels
    for k in rank_only:
        ninfo = rank_only_notes_guard(k, rank_only[k], root, n_test, len(train), stack=stack, placeholder=k in placeholder)
        if ninfo is not None:
            rank_only_notes[k] = ninfo
    band_for = parse_band_for(a.rank_none_band_for, rank_only)  # empty (default): --rank-none-band for all channels as before

    def full_probs(name: str) -> np.ndarray:
        lp, _ = load_shards(given[name], n_test, K, absent)
        return probs_from_logp(lp, absent=absent)

    chans, sources = [], {}
    raw_chans: dict[str, np.ndarray] = {}  # second stage (--stage2): channels BEFORE the floor, like the OOF artifacts
    draw_keep: dict[str, list] = {k: [] for k in rank_only_draws}
    for name in stack["channels"]:
        if name == "prior":
            P = np.load(COMP / "artifacts" / "test_prior_question_v1.npy")
            if P.shape != (n_test, K):
                raise SystemExit(f"Prior: Form {P.shape}")
            sources[name] = [{"spec": "artifacts/test_prior_question_v1.npy"}]
        elif name in proto:  # task-agnostic channel: answers in the teacher protocol for the test rows
            draws = proto[name].split(",")
            neutral_file = dict(t.split("=", 1) for t in a.proto_neutral).get(name)
            neutral = np.load(root / neutral_file) if neutral_file else None
            P, cnt, rinfo = proto_channel(name, draws, rank.get(name), n_test, classes, seen_cls, neutral=neutral, rank_partial=a.rank_partial,
                                          max_missing=a.rank_max_missing, min_mass=a.rank_min_mass, none_band=tuple(a.rank_none_band),
                                          ref_rank=rank_ref.get(name), ref_compare=rank_cmp.get(name, "c95"))
            miss, n_due = cnt == 0, int((cnt >= 0).sum())
            print(f"  channel {name}: protocol answers, rows by number of valid draws {dict(sorted((int(k), int((cnt == k).sum())) for k in set(cnt.tolist())))}"
                  f" ({int(miss.sum())} without an answer get the uniform distribution" + (f"; −1 = {int((cnt < 0).sum())} rows of the gate {neutral_file}" if neutral_file else "") + ")")
            if miss.sum() > a.proto_max_missing * max(1, n_due):
                raise SystemExit(f"channel {name}: {int(miss.sum())} of {n_due} test rows to compute without a valid answer (above {a.proto_max_missing:.0%})")
            sources[name] = [{"proto": d} for d in draws] + [{"rows_without_answer": int(miss.sum()), "neutral_rows": neutral_file, "rows_neutral": int((cnt < 0).sum())}]
            if rinfo:  # mean channel with ranking: source and statistics of the ranking
                sources[name].append({"rank": rank[name], "definition": c95.DEFINITION, **rinfo})
            if name in rank_notes:  # only with rank_mean.notes; without the field the record of the sources stays unchanged
                sources[name].append({"rank_compare": rank_cmp.get(name, "c95"), "notes": rank_notes[name]})
            if name in rank_named:  # only with rank_mean.named; without the field the sources file stays unchanged
                sources[name].append({"named": rank_named[name]})
            meta_file = COMP / "artifacts" / f"meta_{name}.json"
            ch_meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
            if ch_meta.get("class_weights") is not None:  # inductive debiasing: the same weights as for the channel's OOF artifact
                cw, deb = np.array(ch_meta["class_weights"], dtype=float), ch_meta["debias"]
                if len(cw) != K:
                    raise SystemExit(f"channel {name}: {len(cw)} class weights in {meta_file.name}, the taxonomy has {K} classes")
                P = a8.apply_debias(P, cnt > 0, cw, float(deb["strength"]))
                sources[name].append({"class_weights_from": f"artifacts/{meta_file.name}", "strength": float(deb["strength"]), "min_rows": int(deb["min_rows"]),
                                      "classes": int((cw != 1).sum()), "rows": int((cnt > 0).sum())})
                print(f"  channel {name}: debiasing of the class shares (inductive; weights from the training rows, {meta_file.name}): strength {float(deb['strength']):g}, "
                      f"{int((cw != 1).sum())} classes, on {int((cnt > 0).sum())} test rows with a valid answer")
        elif name in rank_only_draws:  # several draws of a rank-only channel (a9): per row the mean over the draws
            refs_d, ginfo = rank_only_draws[name]
            P, _, rinfo = rank_only_draws_channel(name, rank_only[name].split(","), refs_d, n_test, seen_cls, rank_partial=a.rank_partial,
                                                  max_missing=a.rank_max_missing, min_mass=a.rank_min_mass,
                                                  none_band=band_for.get(name, tuple(a.rank_none_band)), keep=draw_keep[name],
                                                  rowlist=rank_only_rowlists.get(name))
            sources[name] = [{"rank_only": rank_only[name], "definition": RANK_ONLY_DRAWS_DEFINITION, "train_rank": rank_only_ref[name], **ginfo, **rinfo,
                              **({"none_band": list(band_for[name])} if name in band_for else {})}]
        elif name in rank_only:  # rank-only channel (M3, eda/c99_m3_deploy.py): exact distribution, rows without a ranking uniform
            P, _, rinfo = rank_only_channel(name, rank_only[name], n_test, seen_cls, ref_source=rank_only_ref[name], rank_partial=a.rank_partial,
                                            max_missing=a.rank_max_missing, min_mass=a.rank_min_mass,
                                            none_band=band_for.get(name, tuple(a.rank_none_band)))
            sources[name] = [{"rank_only": rank_only[name], "definition": RANK_ONLY_DEFINITION, "train_rank": rank_only_ref[name], **rinfo,
                              **({"notes": rank_only_notes[name]} if name in rank_only_notes else {}),
                              **({"none_band": list(band_for[name])} if name in band_for else {})}]
        else:
            if name not in given:
                raise SystemExit(f"--test {name}=<shards> missing for channel {name!r}")
            if "," in given[name] or "@" in given[name]:  # several draws of the same channel: arithmetic mean
                P, cnt, sources[name] = load_draws(given[name], n_test, K, absent, root)
                miss = cnt == 0
                print(f"  channel {name}: rows by number of averaged draws {dict(sorted((int(k), int((cnt == k).sum())) for k in set(cnt.tolist())))}")
                if miss.any() and name not in fill:
                    raise SystemExit(f"channel {name}: {int(miss.sum())} test rows without a draw — give --fill {name}=<channel>")
            else:
                lp, sources[name] = load_shards(given[name], n_test, K, absent, partial=name in fill)
                miss = ~np.isfinite(lp).any(1)
                if miss.any() and name not in fill:
                    raise SystemExit(f"channel {name}: {int(miss.sum())} test rows without a finite candidate")
                lp[miss] = 0.0
                P = probs_from_logp(lp, absent=absent)
            if miss.any():  # cascade: the channel did not compute these rows
                P[miss] = (np.load(COMP / "artifacts" / "test_prior_question_v1.npy") if fill[name] == "prior" else full_probs(fill[name]))[miss]
                sources[name].append({"fill": fill[name], "rows": int(miss.sum())})
                print(f"  channel {name}: {int(miss.sum())} of {n_test} test rows taken from channel {fill[name]}")
        raw_chans[name] = np.array(P, dtype=np.float64, copy=True)
        eps_floor = float((stack.get("floor") or {}).get(name, 0.0))
        if eps_floor:  # the same floor as when the weights were fitted (field floor of the blend file)
            P = a2.floor_probs(P, eps_floor, seen_cls)
            sources[name].append({"floor": eps_floor})
            print(f"  channel {name}: floor {eps_floor:g} (mixed with the uniform distribution)")
        chans.append(np.log(np.clip(P, EPS, 1)))
    none_feats = None
    if stack.get("none"):
        if len(a.none_test) != len(stack["none"]):
            raise SystemExit(f"The blend uses {len(stack['none'])} answer-check signals; --none-test names {len(a.none_test)}")
        cols = []
        for s in a.none_test:
            lp, src = load_shards(s, n_test, K, absent, width=2)
            m = lp[:, 0] - lp[:, 1]
            cols.append(np.where(np.isfinite(m), m, 0.0))
            sources[f"none:{s}"] = src
        none_feats = np.column_stack(cols)

    beta_new = arith_beta_novel_of(stack)
    novel_test = None
    if beta_new is not None:  # novelty of the test rows: reference = all training rows (only training labels and the row's task)
        novel_test = code_novelty(train["question_id"].to_numpy(), train["topic_name"].to_numpy(), labels.encode(train["root_code"], classes),
                                  test["question_id"].to_numpy(), test["topic_name"].to_numpy(), K)
    P = mix_channels(stack, chans, none_feats, seen_cls, novel=novel_test)
    if arith_beta_of(stack):
        print(f"  blend: log-linear with arithmetic share β = {arith_beta_of(stack):g} (field arith_beta of the blend file)")
    if beta_new is not None:
        print(f"  blend: share for novel codes β_new = {beta_new:g} (field arith_beta_novel) — novel are "
              f"{float(novel_test[:, seen_cls].mean()):.3f} of the pairs (test row, class with a training example)")
    P_pre_align = P  # second stage: blend before the alignment (features p_logp, p_rank)
    align_info, align_w = None, None
    ind_info, ind_w = None, None
    if a.align_inductive:
        spec_c21 = importlib.util.spec_from_file_location("c21", COMP / "eda" / "c21_margin_align.py")
        c21 = importlib.util.module_from_spec(spec_c21)
        spec_c21.loader.exec_module(c21)
        counts = np.bincount(labels.encode(train["root_code"], classes), minlength=K)
        target = np.where(counts >= a.align_min, counts, 0) / len(train)
        P_train = oof_mixture(stack, seen_cls, COMP / "artifacts", n_rows=len(train),
                              novel=train_novelty(train, C.load_folds(len(train)), classes) if beta_new is not None else None)
        fit_rows = stack_fit_rows(stack, len(train))
        if fit_rows is not None:  # blend fitted on these training rows only (eda/c99_m3_deploy.py): class weights from the same rows
            P_train = P_train[fit_rows]
        ind_w = c21.fit_weights(P_train, target)  # training rows only: fixed before inference
        want_w = (stack.get("inductive_check") or {}).get("weights_sha256")
        if want_w and hashlib.sha256(np.ascontiguousarray(ind_w, dtype=np.float64).tobytes()).hexdigest()[:16] != want_w:
            raise SystemExit("Inductive class weights differ from the blend file (inductive_check) — different --align-min or OOF artifacts changed "
                             "since eda/c99_m3_deploy.py --save; save the blend again")
        P_before = P
        P = c21.apply_weights(P, ind_w, a.align_inductive)
        P = np.where(seen_cls[None, :], P, 0.0)
        P = P / P.sum(1, keepdims=True)
        t0, t1 = metric.top3_indices(P_before, seen_cls), metric.top3_indices(P, seen_cls)
        used = target > 0
        ind_info = {"strength": a.align_inductive, "min_rows": a.align_min, "classes": int(used.sum()), "weights_from": "OOF predictions of the blend on the training rows",
                    "weights_q10_q50_q90": [round(float(q), 3) for q in np.quantile(ind_w[used], [0.1, 0.5, 0.9])],
                    "weights_sha256": hashlib.sha256(np.ascontiguousarray(ind_w, dtype=np.float64).tobytes()).hexdigest()[:16],
                    "weights": [float(v) for v in ind_w],  # per class in the order of labels.load_classes; from training rows only (part of the solution package)
                    "cv_note": "field cv applies to the blend WITHOUT the alignment; nested value of the recipe: eda/c54_inductive_align.py",
                    "share_rank1_changed": round(float((t0[:, 0] != t1[:, 0]).mean()), 4), "share_top3_changed": round(float((t0 != t1).any(1).mean()), 4),
                    "max_abs_gap_train_before": round(float(np.abs(P_train.mean(0) - target)[used].max()), 4),
                    "oof_sha256": {nm: hashlib.sha256((COMP / "artifacts" / ("oof_prior_question_v1.npy" if nm == "prior" else f"oof_{nm}.npy")).read_bytes()).hexdigest()[:16]
                                   for nm in stack["channels"]}}
        if fit_rows is not None:
            ind_info["fit_rows"] = {"rows": int(len(fit_rows)), "sha256": hashlib.sha256(fit_rows.astype("<i8").tobytes()).hexdigest()[:16],
                                    "note": "class weights only from the predictions of the blend on its fit rows (eda/c99_m3_deploy.py)"}
        print(f"  inductive alignment (class weights from training rows only): strength {a.align_inductive:g}, {ind_info['classes']} classes with at least {a.align_min} training rows; "
              f"weights 10/50/90 % {ind_info['weights_q10_q50_q90']}; rank 1 changed in {ind_info['share_rank1_changed']:.3f} of the test rows, the first three in {ind_info['share_top3_changed']:.3f}")
    if a.align:
        spec_c21 = importlib.util.spec_from_file_location("c21", COMP / "eda" / "c21_margin_align.py")
        c21 = importlib.util.module_from_spec(spec_c21)
        spec_c21.loader.exec_module(c21)
        counts = np.bincount(labels.encode(train["root_code"], classes), minlength=K)
        target = np.where(counts >= a.align_min, counts, 0) / len(train)
        align_w = c21.fit_weights(P, target)
        P_before = P
        P = c21.apply_weights(P, align_w, a.align)
        P = np.where(seen_cls[None, :], P, 0.0)
        P = P / P.sum(1, keepdims=True)
        t0, t1 = metric.top3_indices(P_before, seen_cls), metric.top3_indices(P, seen_cls)
        used = target > 0
        align_info = {"strength": a.align, "min_rows": a.align_min, "classes": int(used.sum()), "rows_of_those_classes_in_train": int(counts[used].sum()),
                      "weights_q10_q50_q90": [round(float(q), 3) for q in np.quantile(align_w[used], [0.1, 0.5, 0.9])],
                      "share_rank1_changed": round(float((t0[:, 0] != t1[:, 0]).mean()), 4), "share_top3_changed": round(float((t0 != t1).any(1).mean()), 4),
                      "max_abs_gap_before": round(float(np.abs(P_before.mean(0) - target)[used].max()), 4),
                      "max_abs_gap_after_full_strength": round(float(np.abs(c21.apply_weights(P_before, align_w, 1.0).mean(0) - target)[used].max()), 6)}
        print(f"  margin alignment (transductive, the test rows as one set): strength {a.align:g}, {align_info['classes']} classes with at least {a.align_min} training rows; "
              f"weights 10/50/90 % {align_info['weights_q10_q50_q90']}; rank 1 changed in {align_info['share_rank1_changed']:.3f} of the test rows, "
              f"the first three in {align_info['share_top3_changed']:.3f}")
    stage2_info = None
    if a.stage2:  # second stage: reranker over the first K candidates AFTER the alignment, BEFORE REL-01
        apply_fn = apply_stage2_safe if s2j.get("align_mode") in ("inductive", "set") else apply_stage2  # safety line: inductive / as one set
        if a.stage2_control and apply_fn is not apply_stage2_safe:
            raise SystemExit("--stage2-control only with a deployment file of the safety line")
        extra = {"control": a.stage2_control} if a.stage2_control else {}
        if a.stage2_xcontrol:  # only on the path of the primary line with the field extra_columns
            if apply_fn is not apply_stage2 or "extra_columns" not in s2j:
                raise SystemExit("--stage2-xcontrol only with a deployment file of the primary line with the field extra_columns")
            extra["xcontrol"] = a.stage2_xcontrol
        P, stage2_info = apply_fn(Path(a.stage2), COMP / "artifacts" / f"stack_{a.name}.json", stack, P, P_pre_align, raw_chans, draw_keep,
                                      rank_only, train, test, classes, seen_cls, root, Path(a.stage2_report) if a.stage2_report else None, **extra)
    blank_info = None
    if a.blank_rows:  # blank images (black photo): the meta code at rank 1, the rest of the order stays (inductive, per row)
        rows = np.load(root / a.blank_rows).astype(np.int64)
        if a.blank_code not in classes or not seen_cls[classes.index(a.blank_code)]:
            raise SystemExit(f"--blank-code {a.blank_code}: not a code with a training example")
        if len(rows) and (rows.min() < 0 or rows.max() >= n_test):
            raise SystemExit("--blank-rows: position outside the test table")
        j = classes.index(a.blank_code)
        t_before = metric.top3_indices(P, seen_cls)[rows]
        P[rows, j] = P[rows].max(1) + 1.0
        P[rows] = P[rows] / P[rows].sum(1, keepdims=True)
        blank_info = {"rows": int(len(rows)), "code": a.blank_code, "list": a.blank_rows, "code_was_rank1": int((t_before[:, 0] == j).sum()),
                      "code_was_in_top3": int((t_before == j).any(1).sum()), "none_was_rank1": int((t_before[:, 0] == 0).sum())}
        print(f"  blank images: {blank_info['rows']} test rows get {a.blank_code} at rank 1 (previously at rank 1: {blank_info['code_was_rank1']}, "
              f"among the first three: {blank_info['code_was_in_top3']}; NONE previously at rank 1: {blank_info['none_was_rank1']})")
    top = metric.top3_indices(P, seen_cls)
    sub = submission_io.build_submission(test["trace_id"].tolist(), top, classes)
    sample = pd.read_csv(root / "competition_data" / "sample_submission.csv", dtype=str, keep_default_na=False)
    sub = sub.set_index("trace_id").loc[sample["trace_id"]].reset_index()  # order of the sample file
    never_seen = [c for c, s in zip(classes, seen_cls) if not s]
    stats = submission_io.check_submission(sub, sample, classes, never_seen=never_seen, require_three=True)

    agg_t = aggregates(P, seen_cls, classes)
    line = {"Test": agg_t}
    oof_file = COMP / "artifacts" / f"oof_{a.name}.npy"
    if oof_file.is_file() or a.align_inductive:
        P_oof = np.load(oof_file) if oof_file.is_file() else P_train
        if a.align_inductive:  # comparison row: the same class weights on the predictions for the training rows
            P_oof = np.clip(P_oof, 1e-12, 1)
            P_oof = c21.apply_weights(P_oof / P_oof.sum(1, keepdims=True), ind_w, a.align_inductive)
        if a.align:  # comparison row with the same alignment on the OOF rows (as one set)
            P_oof = np.clip(P_oof, 1e-12, 1)
            P_oof = P_oof / P_oof.sum(1, keepdims=True)
            P_oof = c21.apply_weights(P_oof, c21.fit_weights(P_oof, target), a.align)
        line["OOF"] = aggregates(P_oof, seen_cls, classes)
    out = Path(a.out) if a.out else COMP / "artifacts" / "submissions" / f"sub_{a.name}{a.suffix}.csv"  # written only after everything has been computed
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    np.save(COMP / "artifacts" / f"test_{a.name}{a.suffix}.npy", P)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    for k, v in line.items():
        print(f"  {k:<5} rows {v['n']:5d} | NONE at rank 1 {v['share_none_rank1']:.3f} | NONE among the first three {v['share_none_in_top3']:.3f} | "
              f"distinct codes at rank 1 {v['distinct_rank1']} | mean probability at rank 1 {v['mean_p_rank1']:.3f}")
    if "OOF" in line:
        tv = 0.5 * float(np.abs(line["Test"]["hist"] - line["OOF"]["hist"]).sum())
        print(f"  distance of the rank-1 distributions test vs. OOF (half L1 sum): {tv:.3f}")
    for v in line.values():
        v.pop("hist")
    shown = out.resolve().relative_to(COMP) if out.resolve().is_relative_to(COMP) else out  # --out may lie outside the folder
    info = {"name": a.name + a.suffix, "file": str(shown), "sha256": sha, "channels": stack["channels"], "theta": stack["theta"],
            **({"arith_beta": arith_beta_of(stack)} if arith_beta_of(stack) else {}),  # only with a share: older JSON files stay bit-identical
            **({"arith_beta_novel": beta_new} if beta_new is not None else {}),
            "cv": stack.get("cv"), "sources": sources, "check": stats, "aggregates": line}
    if align_info:
        info["align"] = align_info
        info["kind"] = "transductive (margin alignment over the set of test predictions)"
    if ind_info:
        info["align_inductive"] = ind_info
        info["kind"] = "inductive (class weights only from the predictions of the blend on the training rows; fixed before inference)"
    if stage2_info:
        info["stage2"] = stage2_info
    if blank_info:
        info["blank"] = blank_info
    if placeholder:  # code check only (--rank-only-placeholder): never submit
        info["placeholder"] = {"channels": placeholder, "note": "test ranking is NOT the intended one (notes not checked) — never submit"}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1, ensure_ascii=False, default=str))
    print(f"written: {shown} (SHA-256 {sha[:16]}…), check {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
