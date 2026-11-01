"""Generate reasoning texts fast: a local vLLM server runs thinking mode for many rows at once.

The in-house generation path (engine.generate) needs 13 s per row in thinking mode (bf16, 96 GB) up to 40 s (4-bit,
A40), which is too expensive for all rows. vLLM serves the same open model with continuous batching several times
faster. Only the TEXT is produced here (the reasoning up to the line "Root:"); the ranking of all codes behind this
text is computed afterwards by the usual path (runner, probe think with `think_from`), exactly and with the same prompts.

Flow of a job of kind `think_gen` (called by the runner):
  1. Start the server:  <python of the vLLM environment> -m vllm.entrypoints.openai.api_server --model <local folder> ...
     The server listens only on 127.0.0.1; there is no outbound call (the solution runs offline).
  2. Build the same sample per row as the in-house path (SampleBuilder, task think) and send it as a chat request:
     system text, text parts and images (lossless PNG) in the same order, thinking mode on, greedy selection.
  3. Write the raw answer text per row to <target>/<tag>_texts.jsonl ({"row": table position, "text": …}),
     plus statistics (end of the answer, tokens). Interrupted runs resume.
  4. Stop the server (the GPU is needed for scoring).

For test rows only aggregates are logged; the texts are row-level derivatives and stay under artifacts/runs/
(gitignored: *.jsonl); they are not inspected.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import prompts as P


def to_messages(sample: P.Sample, system_role: bool = True) -> list[dict]:
    """Sample -> chat messages in OpenAI format; order of the parts as in engine.chat_text.

    system_role=False: the system text becomes the first text part of the user message, for models of a different
    architecture whose chat template has no system role."""
    content = [] if system_role else [{"type": "text", "text": P.SYSTEM + "\n\n"}]
    for kind, value in sample.parts:
        if kind == "text":
            content.append({"type": "text", "text": value})
        else:
            buf = io.BytesIO()
            value.convert("RGB").save(buf, format="PNG")  # lossless: the same pixels as in the in-house path
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}})
    if not system_role:
        return [{"role": "user", "content": content}]
    return [{"role": "system", "content": [{"type": "text", "text": P.SYSTEM}]}, {"role": "user", "content": content}]


def request_body(sample: P.Sample, model: str, max_tokens: int, thinking: bool = True, temperature: float = 0.0, seed: int | None = None,
                 top_p: float | None = None, top_k: int | None = None, system_role: bool = True, template_extra: dict | None = None) -> dict:
    """Chat request for one sample. Default is greedy (temperature 0); with temperature > 0 and a seed the result is a
    repeatable random draw (several reasoning texts of the same row for averaging the rankings). top_p and top_k
    restrict sampling (the Qwen3.8 model card recommends 1.0 / 0.95 / 20 for thinking mode)."""
    if sample.prefix:
        raise ValueError("think_vllm: samples with a forced answer prefix are not supported")
    template = {"enable_thinking": bool(thinking)}
    if thinking and sample.meta.get("effort"):  # reasoning effort; engine.chat_text reads the same field when scoring
        template["reasoning_effort"] = sample.meta["effort"]
    template.update(template_extra or {})  # extra arguments for the chat template of a different model (job field template_kwargs)
    body = {"model": model, "messages": to_messages(sample, system_role), "max_tokens": int(max_tokens), "temperature": float(temperature),
            "chat_template_kwargs": template, "skip_special_tokens": False}
    if seed is not None:
        body["seed"] = int(seed)
    if top_p is not None:
        body["top_p"] = float(top_p)
    if top_k is not None:
        body["top_k"] = int(top_k)  # extra field of the vLLM server
    return body


def raw_text(choice: dict) -> str:
    """Raw answer text. If the server splits off the reasoning (reasoning_content), it is prepended again;
    scoring needs the text exactly as the model wrote it behind the prompt."""
    msg = choice.get("message", {})
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if reasoning:
        text = reasoning + ("\n</think>\n\n" if "</think>" not in reasoning else "") + text
    return text


def _post(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: int) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited during start-up with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM server not ready after {timeout_s} s")


def start_server(model_dir: Path, spec: dict, log_path: Path) -> subprocess.Popen:
    py = spec.get("python") or os.environ.get("K12_VLLM_PYTHON") or sys.executable
    cmd = [py, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model_dir), "--served-model-name", "k12",
           "--host", "127.0.0.1", "--port", str(int(spec.get("port", 8000))), "--dtype", spec.get("dtype", "bfloat16"),
           "--max-model-len", str(int(spec.get("max_model_len", 24000))),
           "--gpu-memory-utilization", str(float(spec.get("gpu_memory_utilization", 0.92))),
           "--limit-mm-per-prompt", json.dumps({"image": int(spec.get("images_per_prompt", 2))}),
           "--max-num-seqs", str(int(spec.get("max_num_seqs", 32)))] + [str(x) for x in spec.get("extra_args", [])]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")  # the server does not download anything
    # Extra environment variables from the job, e.g. VLLM_USE_FLASHINFER_SAMPLER: "0". On cards without a prebuilt
    # flashinfer kernel (A40) the sampler tries to compile with ninja during warm-up and the server exits.
    env.update({str(k): str(v) for k, v in (spec.get("env") or {}).items()})
    if spec.get("bin_on_path"):
        # flashinfer compiles at start-up (all-reduce with tensor parallelism) with `ninja`; the package lives in the vLLM
        # environment, but its bin folder is not on the search path. Only behind a switch, so that other runs stay unchanged.
        env["PATH"] = str(Path(py).parent) + os.pathsep + env.get("PATH", "")
    return subprocess.Popen(cmd, stdout=open(log_path, "a"), stderr=subprocess.STDOUT, env=env)


def generate(job: dict, model_dir: Path, builder_for_row, table, rows: np.ndarray, split: str, out_dir: Path, tag: str, log=print) -> dict:
    """Generate reasoning texts for `rows` (table positions). `builder_for_row(i)` returns the finished sample."""
    spec = job.get("server", {})
    port = int(spec.get("port", 8000))
    out_dir.mkdir(parents=True, exist_ok=True)
    f_txt = out_dir / f"{tag}_texts.jsonl"
    have: dict[int, dict] = {}
    if f_txt.is_file():
        for line in f_txt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                have[int(r["row"])] = r
    todo = [int(i) for i in rows if int(i) not in have]
    stats = {"rows": int(len(rows)), "done_before": int(len(rows) - len(todo))}
    if todo:
        proc = start_server(model_dir, spec, out_dir / f"{tag}_server.log")
        try:
            wait_ready(port, proc, int(spec.get("startup_s", 1500)))
            log(f"vLLM server ready; {len(todo)} rows ({split}), up to {job.get('max_think_tokens', 2000)} tokens, {job.get('concurrency', 24)} concurrent")
            url = f"http://127.0.0.1:{port}/v1/chat/completions"
            max_tokens = int(job.get("max_think_tokens", 2000))
            temperature = float(job.get("temperature", 0.0))
            seed = job.get("seed")  # seed + table position per row: repeatable and different between rows

            def one(i: int) -> dict:
                body = request_body(builder_for_row(i), "k12", max_tokens, temperature=temperature, seed=None if seed is None else int(seed) * 1_000_003 + i,
                                    top_p=job.get("top_p"), top_k=job.get("top_k"), thinking=bool(job.get("thinking", True)),
                                    system_role=bool(job.get("system_role", True)), template_extra=job.get("template_kwargs"))
                for attempt in range(3):
                    try:
                        out = _post(url, body, timeout=int(job.get("request_timeout_s", 1800)))
                        ch = out["choices"][0]
                        return {"row": i, "text": raw_text(ch), "end": ch.get("finish_reason"), "out_tokens": int(out.get("usage", {}).get("completion_tokens", 0)),
                                "in_tokens": int(out.get("usage", {}).get("prompt_tokens", 0))}
                    except Exception as exc:  # noqa: BLE001 - the row stays open and is counted at the end
                        err = f"{type(exc).__name__}: {exc}"[:200]
                        time.sleep(3 * (attempt + 1))
                return {"row": i, "error": err}

            t0 = time.time()
            n_err = 0
            with ThreadPoolExecutor(max_workers=int(job.get("concurrency", 24))) as pool, open(f_txt, "a") as fh:
                for k, rec in enumerate(pool.map(one, todo)):
                    if "error" in rec:
                        n_err += 1
                        continue
                    have[rec["row"]] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    if (k + 1) % 100 == 0:
                        el = time.time() - t0
                        log(f"think_gen: {k + 1} of {len(todo)} rows, {el / (k + 1):.2f} s per row")
            stats.update(seconds=round(time.time() - t0, 1), s_per_row=round((time.time() - t0) / max(1, len(todo)), 3), failed=n_err)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
    got = [have[int(i)] for i in rows if int(i) in have]
    ends: dict = {}
    for r in got:
        ends[r.get("end")] = ends.get(r.get("end"), 0) + 1
    toks = [r.get("out_tokens", 0) for r in got]
    stats.update(texts=len(got), ends=ends, out_tokens_median=int(np.median(toks)) if toks else 0, out_tokens_max=int(max(toks)) if toks else 0,
                 closed_think=float(np.mean(["</think>" in r["text"] for r in got])) if got else 0.0,
                 has_root_line=float(np.mean(["Root:" in r["text"] for r in got])) if got else 0.0,
                 out_tokens_mean=float(np.mean(toks)) if toks else 0.0,
                 sampling={k: job.get(k) for k in ("temperature", "seed", "top_p", "top_k") if job.get(k) is not None})
    (out_dir / f"{tag}_texts_meta.json").write_text(json.dumps({"table": split, **stats}, indent=1))
    return stats


def load_texts(path: Path, cut_tokens: int | None = None) -> dict[int, str]:
    """Reasoning texts per table position. `cut_tokens` truncates longer texts afterwards to roughly this many tokens
    (character share = token share), to measure how much reasoning the ranking needs without generating again."""
    out: dict[int, str] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            text, n_tok = r["text"], int(r.get("out_tokens") or 0)
            if cut_tokens and n_tok > int(cut_tokens):
                text = text[: int(len(text) * int(cut_tokens) / n_tok)]
            out[int(r["row"])] = text
    return out
