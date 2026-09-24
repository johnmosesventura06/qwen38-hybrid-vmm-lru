#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""bench-3k500 — The standing one-shot serving bench: 3k-token prefill, 500-token decode.

Measures (streaming, API-observed):
  pp   = prompt tokens / TTFT           (prefill throughput)
  tg   = reciprocal-mean TPOT over the 500 streamed tokens (decode)
  TTFT = time to first chunk
Usage: bench-3k500.py [--port 8000] [--n 5] [--warmup 1] [--in 3072] [--out 500] [--model-len-check]
Prints one JSON line per run + a mean summary; compares against the stored
baseline file $HOME/notes/w4a16-baseline.json when present
(--save-baseline writes it, --set to change path).
No dependencies beyond stdlib.
"""
import argparse, json, statistics, sys, time, urllib.request

def count_prompt_tokens(port, model, text):
    # server-side truth via chat usage: one-token completion costs a prefill only
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "max_tokens": 1, "temperature": 0.0, "stream": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["usage"]["prompt_tokens"]

def build_prompt(target_tokens, port, model, salt):
    # NOTE: the tuning pass prefills the prompt into the prefix cache. The
    # caller must swap a FRESH numeric salt (1 token, same length) before
    # measuring, so the measured request is cold while the token count holds.
    unit = ("The quick brown fox jumps over the lazy dog while reviewing "
            "engineering notes about paged attention, sparse selection, and "
            "KV budgets. ") * 8
    text = f"[bench-salt-{salt}] " + unit
    n = count_prompt_tokens(port, model, text)
    text = text * max(1, int(target_tokens / max(n, 1))) + unit
    for _ in range(6):
        n = count_prompt_tokens(port, model, text)
        if abs(n - target_tokens) <= 8:
            break
        text = text[: int(len(text) * target_tokens / n)] if n > target_tokens else text + unit
    return text, n

def one_run(port, model, prompt, out_tokens, temperature=0.0):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": out_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True}
    # note: prompt is raw user text; chat template adds a small fixed overhead
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; tok_times = []; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: chunk = json.loads(payload)
            except json.JSONDecodeError: continue
            if chunk.get("usage"): usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices: continue
            delta = choices[0].get("delta", {}) or {}
            # count any emitted token — reasoning lanes stream under
            # reasoning_content instead of content
            piece = (delta.get("content") or delta.get("reasoning")
                     or delta.get("reasoning_content")
                     or choices[0].get("text"))
            if piece is not None:
                now = time.perf_counter()
                if ttft is None: ttft = now - t0
                tok_times.append(now - t0)
    n_out = usage.get("completion_tokens") or len(tok_times)
    tpots = [tok_times[i] - tok_times[i-1] for i in range(1, len(tok_times))]
    tg = (1.0 / statistics.mean(tpots)) if tpots else float("nan")
    return {"ttft_s": round(ttft, 3), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": n_out,
            "pp_tps": round((usage.get("prompt_tokens") or 0) / ttft, 1) if ttft else None,
            "tg_tps": round(tg, 2)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--in", dest="in_tok", type=int, default=3072)
    ap.add_argument("--out", type=int, default=500)
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--baseline", default="$HOME/notes/w4a16-baseline.json")
    a = ap.parse_args()

    # model metadata (max_model_len) so the report says what lane it measured
    with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=30) as r:
        models = json.loads(r.read())
    max_len = models["data"][0].get("max_model_len")

    runs = []
    for i in range(a.warmup + a.n):
        prompt, _ = build_prompt(a.in_tok, a.port, a.model, "999999999")
        # same-width fresh numeric salt: shifts only the (single-token) salt
        # slot, killing the prefix-cache hit the tuning pass created
        prompt = prompt.replace("bench-salt-999999999",
                                f"bench-salt-{time.time_ns() % 10**9:09d}", 1)
        res = one_run(a.port, a.model, prompt, a.out)
        res["run"] = i
        runs.append(res)
        print(json.dumps(res), flush=True)
    meas = runs[a.warmup:]
    n_in = round(statistics.mean(r["prompt_tokens"] or a.in_tok for r in meas))
    summary = {"max_model_len": max_len, "in": n_in, "out_target": a.out,
               "runs": a.n,
               "pp_mean": round(statistics.mean(r["pp_tps"] for r in meas), 1),
               "tg_mean": round(statistics.mean(r["tg_tps"] for r in meas), 2),
               "tg_recip_mean": round(1.0 / statistics.mean(1.0 / r["tg_tps"] for r in meas), 2),
               "ttft_mean_s": round(statistics.mean(r["ttft_s"] for r in meas), 2),
               "ts": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    print("SUMMARY " + json.dumps(summary))

    if a.save_baseline:
        with open(a.baseline, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"baseline saved -> {a.baseline}")
    else:
        try:
            with open(a.baseline) as f:
                base = json.load(f)
            dp = 100 * (summary["pp_mean"] / base["pp_mean"] - 1)
            dt = 100 * (summary["tg_recip_mean"] / base["tg_recip_mean"] - 1)
            print(f"VS BASELINE: pp {dp:+.1f}%  tg {dt:+.1f}%  "
                  f"(base {base['pp_mean']}/{base['tg_recip_mean']} @ mml {base.get('max_model_len')})")
        except FileNotFoundError:
            print("no baseline file; run with --save-baseline on the pristine lane")

if __name__ == "__main__":
    main()
