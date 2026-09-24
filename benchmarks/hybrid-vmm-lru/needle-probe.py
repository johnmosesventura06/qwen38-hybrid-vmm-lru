#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""niah-probe — needle-in-a-haystack recall + prefill/decode timing for a lane.

Purpose-built for the hybrid arm: unlike a plain speed probe, it forces decode
steps whose routing reaches NON-hot experts, which is the condition that turns
the v1 hot-expert data-loss flaw into visible damage. Recall is the gate.

Usage: niah-probe.py --port 8001 --tokens 20000 --out 64 [--depth 0.05] [--runs 2]
"""
import argparse, json, random, string, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8001)
ap.add_argument("--model", default="Qwen3.8-Flash-Next")
ap.add_argument("--tokens", type=int, default=20000, help="approx prompt token count")
ap.add_argument("--out", type=int, default=64)
ap.add_argument("--depth", type=float, default=0.05, help="needle position as fraction of prompt")
ap.add_argument("--runs", type=int, default=2)
A = ap.parse_args()

CHARS_PER_TOK = 3.6  # conservative for this tokenizer on the filler text
FILLER = ("The maintenance log for the regolith conveyor lists inspections, torque values, "
          "dust seals, bearing temperatures and the usual arguments about shift handover. ")

def build(depth):
    code = "QX-" + "".join(random.choices(string.digits, k=4))
    target_chars = int(A.tokens * CHARS_PER_TOK)
    needle = f"The vault access code is {code}. Remember it. "
    pre = int(target_chars * depth)
    head = (FILLER * (pre // len(FILLER) + 1))[:pre]
    tail = (FILLER * ((target_chars - pre) // len(FILLER) + 1))[: target_chars - pre]
    prompt = (head + "\n" + needle + "\n" + tail +
              "\n\nQuestion: what is the vault access code? Answer with the code only.")
    return code, prompt

def run(prompt):
    body = {"model": A.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": A.out, "temperature": 0.0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{A.port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    chunks = []
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or ""
                if delta.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.time() - t0
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    chunks.append(piece)
    total = time.time() - t0
    text = "".join(chunks)
    pt = usage.get("prompt_tokens") or 0
    ct = usage.get("completion_tokens") or len(chunks)
    pp = pt / ttft if (ttft and pt) else 0.0
    tg = ct / max(total - (ttft or 0), 1e-6)
    return code, text, pt, ct, ttft, total, pp, tg

print(f"=== niah-probe port={A.port} tokens~{A.tokens} out={A.out} runs={A.runs} ===")
fails = 0
for i in range(A.runs):
    code, prompt = build(A.depth)
    code, text, pt, ct, ttft, total, pp, tg = run(prompt)
    hit = code in text
    if not hit:
        fails += 1
    print(f"run{i+1}: prompt_tokens={pt} hit={hit} code={code} ttft={ttft:.2f}s "
          f"pp={pp:.0f} tok/s out={ct} tg={tg:.1f} tok/s total={total:.1f}s")
    print(f"        answer={text.strip()[:120]!r}")
print(f"RESULT: {'PASS' if fails == 0 else 'FAIL'}  ({A.runs - fails}/{A.runs} recall)")