# Running Qwen3.8-Flash-Next on 4× RTX 5060 Ti (16 GB)

### ~56 tok/s decode · 1,745 tok/s prefill @3k · 2,053 @32k · 495K context

A 125B-class MoE does not fit on 4×16 GB consumer cards: 62.8 GB of int4 experts have to
live in host RAM, and only about 44 of 512 experts per layer can sit in VRAM at once. The
usual consequence is picking one speed — fast prefill *or* fast decode. This lane gets both
by giving **one expert tier two access paths**: a contiguous CUDA-VMM view for prefill, and a
dynamic LRU mirror for decode, sharing the same 44 VRAM rows instead of pinning two copies.

*(Internal name: golden-v2.0 hybrid tier.)*

**Reproducibility note:** everything that runs the lane is in this repo except **one file** —
the FlashInfer GDN prefill gate for SM12x, which comes from a recipe whose repository publishes
no license, so we link to it instead of copying it. It is optional: the launcher mounts whatever
exists, nothing imports it, and the lane boots and serves without it. See
**[Optional: one file to pull](#1a-optional-one-file-to-pull)** below. The PCIe custom all-reduce and
all-gather wiring, by contrast, **are** shipped — verified additive over the pinned vLLM baseline.

**Status:** promoted to production on our lane 2026-09-24 and running this exact config.
This tree is the publishable cut: paths are written as `$MODELS_DIR` / `$HOME` / placeholders,
every knob is documented in [`.env.example`](.env.example), and
[`docs/RUNNING.md`](docs/RUNNING.md) takes you from a clean machine to a verified lane.
Numbers below are from one machine — treat them as a measured example, not a promise for
your hardware. Paths that were hard-coded to one host have been made configurable, and that
change is in the diff of `scripts/`.

**Quick start:** `docker build -f docker/Dockerfile -t qwen38-flash-next-2x3090:locked .` →
get the checkpoint and the PLE table ([docs/RUNNING.md](docs/RUNNING.md) steps 2–3) →
`cp .env.example .env` and edit → `./local-start.sh` → run the four gates in step 5.

---

## 1. The numbers, measured in one window

Expert weights are the problem: ~63 GiB of int4 experts cannot live in 4×16 GB of VRAM,
yet routing traffic is not uniform. Two mechanisms existed for choosing what stays hot,
and each paid for the other's win. This work merges them into a **single tier** that serves
both access paths.

All numbers below were measured on the same rig, in the same window, with cold boots, using
the same two scripts (`bench-3k500.py`, `needle-probe.py`). The alternative would have been
to compare numbers from different days and different bench shapes, which is how people end
up publishing fake wins.

| shape | arm 0 — host-RAM + VRAM mirror + dynamic LRU | arm 1 — contiguous VMM view, no mirror | **v2.0 — hybrid (this work)** |
|---|---|---|---|
| decode, 3k-token probe | 55.41 tok/s | 37.59 tok/s | **55.95 tok/s** |
| decode, sustained 4k | 61.25 tok/s | 27.74 tok/s | **59.66 tok/s** |
| prefill 3k | 1,258.0 tok/s | 1,670.0 tok/s | **1,727.2 tok/s** |
| prefill 32k | 1,411.8 tok/s | 1,838.9 tok/s | **1,917.5 tok/s** |
| prefill 190k | 1,345 tok/s | 1,731–1,775 tok/s | **1,770–1,822 tok/s** |
| needle recall @190k | pass | pass | **pass** |
| post-heavy-traffic sanity | — | — | **pass** |

Read: **prefill +33…37% over arm 0 with decode at arm 0 parity.** Against arm 1 (which is
what the lane ran the day before) the same story mirrored: prefill parity, decode +49%
(probe) / +115% (sustained).

Live production numbers after promotion (`bench-3k500.py` on the serving port): prefill
1,745.3 tok/s @3k and 2,052.9 tok/s @32k, decode 55.11 tok/s probe / 61.3 tok/s sustained,
math gate exact, recall @190k pass.

Two measurement notes that matter when reading any of these tables, ours or anyone else's:
the widely-quoted "58.35 tok/s" golden decode bar is a **warm** session number — measured
cold, the same configuration reads 55.41 — and prefill numbers below ~30k tokens are not a
gate (a table-format change once passed a 3k probe and failed real agent TTFT).

---

## 1a. Optional: one file to pull

| | |
|---|---|
| **what** | `qwen_gdn_linear_attn.py` carrying the FlashInfer GDN prefill gate for SM12x |
| **from** | recipe **r10** in [abtraore/QWEN-PEDIA](https://github.com/abtraore/QWEN-PEDIA) — her repo, her terms; we omit it only because it publishes no license file |
| **put it at** | `fn-ext/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (the path under `FN_EXT_DIR` mirrors the layout inside the `vllm` package) |
| **activates on** | cold boot — the launcher bind-mounts every file under `FN_EXT_DIR` over the installed package, no image rebuild |
| **witness** | boot log line `Using FlashInfer GDN prefill kernel (head_k_dim=128)`; prefill should then move toward the numbers in §1 |
| **if you skip it** | nothing errors. The mount loop is `find $FN_EXT_DIR -type f`, no other overlay imports this module, and vLLM uses its own shipped prefill kernel. You keep the hybrid tier and the all-reduce win; you lose part of the prefill delta |

If you would rather not depend on her repo at all: the fork's own **Apache-2.0** copy of that
file is already in this repo at
`runtime/vllm-overlay/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, and it enables
the same FlashInfer kernel on **SM10x** — widening that condition to SM12x is a two-line change
that is entirely yours to make and ship. (That is the route to prefer if you want a tree with no
third-party licensing question anywhere in it.)

## 2. The machine, the model, the engine

| Piece | What it is |
|---|---|
| GPUs | 4× RTX 5060 Ti 16 GB, all four on CPU-direct PCIe (x8/x4/x8/x4 gen5), P2P enabled via a patched driver |
| CPU / RAM | 32-core, 128 GB DDR5 (4×32, ~120 GiB usable) — the box is deliberately RAM-rich, VRAM-poor |
| Model | Qwen3.8-Flash-Next, 125B-class MoE (~A6B), W4A16 int4 experts + FP8 PLE + an n-gram (PLE) table |
| Engine | vLLM (private vendor build `0.1.dev20073+g8e685d198`, torch 2.13.0+cu130, flashinfer 0.6.17, humming-kernels 0.1.12) |
| Parallelism | TP4 + EP4, np=1 serving, batch-2 CUDA graphs, MTP off, vision tower mounted |
| Context | 495K tokens (YaRN), 517,858-token KV pool (1.05×) |

Expert layout per layer per rank: 128 local experts; the top 44 by an offline traffic
ranking are treated as "hot". Everything about this work is about where those rows live.

---

## 3. Why the two mechanisms could not simply be turned on together

**Mechanism A — mirror + LRU (the original):** the full expert set is pinned in host RAM
(UVA offload). A small VRAM buffer holds a *mirror* of the 44 hot rows. Decode looks experts
up in the mirror; when a token routes to an unmapped expert, the least-recently-used slot is
evicted and the new expert's row is copied in. Prefill reads whatever it needs straight from
host RAM — every read a PCIe trip.

**Mechanism B — contiguous VMM view:** each layer's expert tensor is one contiguous address
range built with CUDA VMM: the first 44 rows are backed by VRAM, the remainder by host memory,
and the rows are pre-permuted so hot experts come first. Prefill is fast (one gather over a
contiguous tensor, hot rows already in VRAM, zero cache bookkeeping). Decode has no mirror at
all, so cold-expert routing always pays a PCIe read.

The two were mutually exclusive in the codebase for a concrete reason: the LRU's bookkeeping
was built against mechanism A's *separate* mirror tensors, and mechanism B's initialization was
gated off whenever the LRU was enabled.

**The observation that unlocks the merge:** mechanism B's device prefix and mechanism A's mirror
hold *exactly the same thing* — the same 44 experts, in the same ranking order, both derived from
the same rankings file. So the VMM device prefix **is** the mirror. No second VRAM tier is needed
(a 44-row tier costs ~5 GiB per card; there is no headroom for two), and no extra VRAM copy exists
to keep coherent.

**The catch that made it non-trivial:** in mechanism A every expert keeps a host-resident row —
the mirror is a *copy*, the original never leaves RAM. In mechanism B the 44 hot rows existed
*only* in VRAM. An LRU eviction would therefore overwrite an expert's only copy and remove it
from the model: not a crash, not an error, just a model that slowly loses routing capacity.
The fix is to give every local expert a permanent host home inside the same tensor:

```
        VRAM per card (device-mapped prefix)        host RAM (host-mapped region)
   ┌────────────────────────────────────┐    ┌──────────────────────────────────────┐
   │ rows [0, capacity)                 │    │ rows [capacity, capacity + N)        │
   │ = the 44 ranked-hot experts        │    │ = the full permuted expert set; its  │
   │ = the decode mirror AND the fast   │    │   first `capacity` rows duplicate    │
   │   prefill rows                     │    │   the device prefix                  │
   └────────────────────────────────────┘    └──────────────────────────────────────┘
```

Two fills in one boot-time kernel launch lay both regions; a boot canary compares the
duplicate's bytes against the prefix and refuses to serve if they differ.

---

## 4. The mechanism at runtime

The forward pass already forks on query size, which is what makes the merge cheap:

| batch | path | reads |
|---|---|---|
| **> 16 tokens (prefill)** | plain MoE over the whole tensor with a shared dynamic map | hot/promoted experts from device rows, everyone else from host rows |
| **≤ 16 tokens (decode)** | LRU: check slots, copy misses up, evict oldest, then MoE over the `capacity`-row prefix | misses copied from their host row into a prefix slot |

Both paths read **one shared map** (`rot_map`: expert → the row that currently holds its bytes).
A promotion points the promoted expert at its slot and the victim back at its host row, so a
prefill step and a decode step can never disagree about where an expert lives. That map is the
seam that makes "one tier, two access paths" correct rather than merely clever.

Correctness invariants (each enforced, not assumed):
1. promotion sources are host rows (`≥ capacity`), destinations are slots (`< capacity`) — never
   overlapping, for hot and cold experts alike;
2. all expert tensors (weights, scales, zero-points, g_idx, sort) share one row space, so one map
   addresses them all;
3. the prefix content equals the ranking order equals the LRU's initial slot table (no fill needed);
4. host rows are written once at boot and never overwritten;
5. the dynamic map starts equal to mechanism B's static permutation, so prefill before any decode
   step is byte-identical to mechanism B;
6. a byte-level boot canary (prefix vs duplicate) — because map-arithmetic assertions cannot see a
   fill-order mistake.

---

## 5. What went wrong on the way (the useful part)

Three builds, each caught by a different mechanism. We think this is worth publishing precisely
because the second failure is invisible to normal testing.

**v1 — design flaw in the first spec.** The hot rows had no host home: evictions silently deleted
experts from both maps. The failure mode *made the lane faster* (less MoE work), which is why its
numbers — 81.3 tok/s sustained and 2,753 tok/s prefill — were never banked. Lesson: a correctness
gate must run before any timing is believed.

**v2 — fill order.** The duplicate rows were written at the tensor's *tail* while every map
addressed them at `capacity`, so every cold expert served a **different expert's weights from
token zero**. Symptoms: a math gate returning 434 instead of 437, empty replies, and a needle probe
that answered by continuing the prompt's filler text. Every install-time assertion passed, because
they all checked map arithmetic and none compared bytes. Caught by an independent reviewer reading
the fill vector against the addressing (verdict: REJECT, with a two-line fix).

**v3 — the shipped build.** Argument order fixed at both fill sites, plus the byte canary that
would have caught v2 at boot, plus one latent regression the reviewer found in a static-mirror code
path that our arms never reach.

---

## 6. Patch inventory and credits

This work is a fork of **DominikBucko/qwen38-flash-next-2x3090** and inherits an entire serving
stack. Nothing below is ours unless marked. Attribution is by artifact, with the honest caveat that
some upstream line-level provenance deserves a second pass before public release.

| Artifact | Origin | What it does |
|---|---|---|
| `runtime/vllm-overlay/**` (baked into the vendor image) | **DominikBucko**, over vLLM project files (Apache-2.0) | The 2×3090 serving stack: int4 WNA16/Marlin-class MoE with hot-cache + UVA offload, **the mixed-VMM allocator (`_allocate_mixed_vmm_tensor`) that this work builds on**, MTP, PLE offload, scheduler patches, QSA/cache layout |
| `fn-ext/.../compressed_tensors_moe/compressed_tensors_moe_wna16.py` | base: **DominikBucko**; **ours: the hybrid** | Adds the hybrid init, the `capacity+N` row layout, the shared dynamic map, the `HYBRID` LRU kernel path, the byte canary |
| `fn-ext/.../quantization/auto_gptq.py` | base: **DominikBucko**; **ours: the VMM shim** | Borrows the VMM implementation for the AutoGPTQ/AutoRound class and re-aliases kernel param names post-permutation — without which the VMM arm silently computes on stale tensors |
| `fn-ext/distributed/device_communicators/custom_all_reduce.py`, `cuda_communicator.py` | vLLM Apache-2.0 baseline + **our** wiring; **concept pointer** from **QWEN-PEDIA (abtraore)** r10 | Keeps custom all-reduce enabled on 4 PCIe-only GPUs (`VLLM_CUSTOM_AR_ALLOW_PCIE=1`, worth ~+7% decode here) and routes the ~2 MB logits all-gather through vLLM's own `custom_all_gather()` API. Shipped because the diff against the pinned baseline is additive-only: the recipe told us the switch was worth flipping, the code is vLLM's API plus ~40 lines of ours. |
| `fn-ext/.../mamba/gdn/qwen_gdn_linear_attn.py` — **not redistributed** | **QWEN-PEDIA (abtraore)**, recipe r10 | FlashInfer GDN prefill gate for SM12x. That repo publishes no license file, so we link instead of copying: [abtraore/QWEN-PEDIA](https://github.com/abtraore/QWEN-PEDIA). Measured with it installed; the fused input-projection arm inside it was never enabled. Alternative: widen the FlashInfer condition in the fork's own Apache-2.0 copy of that file, which already covers SM10x. |
| `fn-ext/.../nvidia/qsa.py`, `ops/qsa.py` | **vLLM PR #55557 (semerandre)**, hand-ported by us | fp8_e4m3 main-KV support in the QSA attention path |
| `fn-ext/.../nvidia/ple_layer.py`, `ple-ext/nvfp4/**` | base **DominikBucko**; **ours: the disk/bf16 table tiers** | PLE (n-gram memory table) offload: we serve a 95.4 GiB bf16 or 47.7 GiB fp8 table from an NVMe mmap so it stays out of the VRAM/RAM budget |
| `fn-ext/v1/worker/gpu/{cudagraph_utils,model_runner}.py` | **DominikBucko (#10/#11)**, vendored by us | Batch-2 CUDA graphs on this hybrid attention/MoE stack |
| GPU driver patch | **aikitoria** (615.71.09-p2p) | Enables P2P on consumer 5060 Ti-class cards (kernel patch, rebuilt per kernel bump) |
| Checkpoint | **Intel** AutoRound W4A16 int4 target, **albucino** FP8-PLE assembly, **RadixArk** NVFP4 PLE, base **Qwen** | The quantized weights this whole exercise serves |
| Kernels / APIs | **flashinfer**, **humming-kernels**, **Marlin**, **NVIDIA CUDA VMM** (`cuMemCreate`/`cuMemMap`/`cuMemSetAccess`) | The primitives everything above calls into |
| Upstream vLLM PRs we track or port | **#55557** (fp8 QSA), **#54743** (offload group scope), **#56273** (NVFP4 PLE), **#56177** (expert pool); Bucko’s **#10–#17** | Directions we read, port, or wait on |

Our own 48 commits on top of the fork cover: the hybrid tier (this work), the PLE disk/bf16
table tiers, the KV disk tier, the fp8-KV graft port, the toggle/verify/rollback machinery and
the lane runbooks.

---

## 7. Reproducing and operating

```bash
# bench shapes used for every number in this document (shipped in this repo)
python3 benchmarks/hybrid-vmm-lru/bench-3k500.py --port <port> --n 3 --warmup 1                     # decode 3k probe
python3 benchmarks/hybrid-vmm-lru/bench-3k500.py --port <port> --n 3 --warmup 1 --in 128 --out 4096 # sustained decode
python3 benchmarks/hybrid-vmm-lru/bench-3k500.py --port <port> --n 2 --warmup 0 --in 32768 --out 32 # deep prefill (the gate)
python3 benchmarks/hybrid-vmm-lru/needle-probe.py --port <port> --tokens 300000 --out 200 --runs 2  # recall at depth
```

Raw numbers: [`benchmarks/hybrid-vmm-lru/results-20260924.json`](benchmarks/hybrid-vmm-lru/results-20260924.json).

Gates before trusting a build: boot witnesses (per-layer hybrid init lines, byte canary silent,
mount set), health, a math gate, recall at depth, **recall again after heavy traffic**, then the
prefill/decode shapes above.

Operation is one line in `.env`:

| value | mode | use |
|---|---|---|
| `PREFILL_VMM_ARM=2` | **hybrid (this work)** | default: prefill headroom and golden-parity decode |
| `PREFILL_VMM_ARM=0` | mirror + dynamic LRU | decode-first fallback |
| `PREFILL_VMM_ARM=1` | contiguous VMM view, no mirror | long-context ingest where decode does not matter |

Rollback is that value plus a cold boot. The patched module stays mounted and is inert for 0/1.

**Cost:** +19–22 GiB host-pinned RAM (the duplicate rows; free RAM 63 → 41 GiB, which does squeeze
the page cache that keeps the PLE table fast), +~240 MiB VRAM per card, +~34% one-time boot copy.
No per-step cost.

---

## 8. What we would try next

The long-standing "hot slot count 32–44 is flat" law was measured for the **static** mirror, where
extra slots buy nothing because the same rows are pinned forever. Under the LRU every extra slot
removes PCIe miss copies from the decode path, so 44 → 52–56 slots (funded from the remaining VRAM
headroom or a small KV-pool cut) is the natural next arm — a VRAM trade, not a RAM one.

## 9. Licensing, attribution, and what you may not do

**This repository's code** is Apache-2.0 (`LICENSE`). Upstream vLLM copyright and SPDX
headers are preserved in every overlay file, and [`NOTICE`](NOTICE) lists exactly which
files we modified relative to the fork we build on.

**No model weights are published here.** The lane consumes separately licensed artifacts,
and those terms bind *you* directly, not us:

- **`Qwen/Qwen3.8-Flash-Next`** — **Qwen Community License 1.0**. It grants use,
  modification, distribution and commercial deployment, subject to two conditions worth
  reading twice: (1) the copyright and permission notice must accompany copies or
  substantial portions, and a product above 100M monthly active users or US$20M monthly
  revenue must display the model name in its UI; (2) if you or an affiliate operate a
  **Model-as-a-Service or AI-work-assistant business**, you need a separate license from
  Qwen before using the model *or any derivative of it* commercially — internal use is
  exempt only while you do not expose the model, its outputs, or its capabilities to third
  parties. Quantizing, merging or re-serving a checkpoint does not change that.
- **Intel's AutoRound checkpoint** and **RadixArk's NVFP4 table** — their own model cards,
  plus the Qwen terms they inherit.
- The assembly scripts copy the upstream model license into the output tree. If you publish
  an assembled checkpoint, keep that file.

**Third-party code we deliberately do not ship:** the GDN-prefill and custom-all-reduce
overlays derive from recipe r10 of QWEN-PEDIA (abtraore), whose repository carries no
license file. Public code with no license is *all rights reserved* by default — credit does
not create a right to redistribute — so we link to it and describe what it is worth, and you
fetch it from the author. The same rule applies to anything you port from someone's repo.

**Names and affiliation.** Qwen and Alibaba are their owners' marks; Intel, RadixArk, vLLM
and DominikBucko likewise. This is an unofficial fork of
`DominikBucko/qwen38-flash-next-2x3090`, published by us, with no involvement or endorsement
from any of them. And this is engineering documentation, not legal advice: read the current
license texts before you ship anything built on top of this.

If you build on the VMM idea, credit DominikBucko — the allocator and the hot-cache/LRU
machinery are his, and his field notes are what pointed at the split between static ranking
and dynamic decode.
