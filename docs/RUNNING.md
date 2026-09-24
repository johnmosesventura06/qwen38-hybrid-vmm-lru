# Running this lane from a clean machine

What a stranger needs: the image, the checkpoint, the PLE table, one `.env`, and a
verdict. Steps 1-4 are one-time; step 5 is the daily loop.

Assumed hardware, because every number in this repo is from it: 4 GPUs with 16 GB
VRAM each on CPU-direct PCIe links, ~128 GB host RAM, and NVMe with ~200 GB free
for the PLE table. Nothing here requires those exact parts, but the VRAM-coupled
knobs (`.env` marks each one) must be re-derived if you differ.

---

## 1. Build the image

```bash
docker build -f docker/Dockerfile -t qwen38-flash-next-2x3090:locked .
```

The Dockerfile layers `runtime/vllm-overlay/` (Bucko's patched vLLM tree — the
source of record for the MoE offload, hot-cache/LRU and VMM machinery) onto the
pinned vendor image recorded in `repro.lock.json`. Verify the base digest matches
`repro.lock.json` before trusting any benchmark comparison; the runtime is a
private build and small version drift changes results.

`FN_EXT_DIR` (step 4) bind-mounts *our* overlays over the installed package at
runtime, so most changes in this repo need no rebuild — only a restart.

**One optional file to pull.** This repo ships every overlay the lane needs except the FlashInfer
GDN prefill gate for SM12x, which derives from recipe r10 of
[abtraore/QWEN-PEDIA](https://github.com/abtraore/QWEN-PEDIA) — that repository publishes no
license file, so we link rather than copy (see [`NOTICE`](../NOTICE)). To take the prefill win:

```bash
# 1. obtain qwen_gdn_linear_attn.py from her recipe r10 (subject to her terms), then
# 2. place it at the same package-relative path inside your FN_EXT_DIR:
$EDITOR fn-ext/model_executor/layers/mamba/gdn/          # create the directory
# 3. cold boot; the launcher mounts whatever files exist under FN_EXT_DIR
```

Skip it and nothing breaks: the mount loop is `find "$FN_EXT_DIR" -type f`, no other shipped
overlay imports that module, and vLLM falls back to its own prefill kernel. The absence of the
`Using FlashInfer GDN prefill kernel` line is the expected state, not a fault. Alternatively — and
this keeps your tree free of any third-party licensing question — widen the FlashInfer condition
in the fork's own Apache-2.0 copy at
`runtime/vllm-overlay/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, which already
enables that kernel on SM10x.

The all-reduce and all-gather wiring **is** included, so on any PCIe TP>2 box with
`VLLM_CUSTOM_AR_ALLOW_PCIE=1` you should see the `Custom allreduce kept ENABLED ...` witness.

## 2. Get the checkpoint

The lane serves the W4A16 int4-expert + FP8-PLE assembly:

- **target**: `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (AutoRound GPTQ, symmetric INT4 group-128)
- **PLE**: FP8 E4M3 table, per `repro.lock.json`
- **assembly**: `scripts/assemble_hf_repo.sh` / `scripts/build_intel_fp8ple_hybrid.py`
  stitch target tensors + FP8 PLE + the compact MTP draft into one HF repo, and
  `scripts/validate_hybrid.py` checks tensor counts against `repro.lock.json`.

Point `MODEL_DIR` at the result. Witness: the dir must contain
`model.safetensors.index.json` (the launcher refuses otherwise).

Do **not** requantize parts of it: the packing, the FP8 PLE scale and the BF16
excludes are all load-bearing.

**Licensing, before you publish anything you assemble here.** The base checkpoint is
under the **Qwen Community License 1.0**, which permits commercial use but (a) requires
its notice to accompany copies and substantial portions, and (b) requires a separate
license from Qwen if you run a Model-as-a-Service or AI-work-assistant business and use
the model or a derivative of it commercially — a private lane on your own box is fine,
putting it behind a hosted endpoint for other people is a licensing question, not a
serving question. The assembly scripts copy the upstream `LICENSE` into the output tree;
leave it there. Intel's and RadixArk's quantizations add their own terms on top.

## 3. Generate the PLE (n-gram memory) table

The table is 320,001,536 x 160 elements. Serving it from an NVMe `mmap` keeps it
out of both the VRAM budget and the pinned-RAM budget. Pick a dtype — this is the
only difference between the two tiers:

| tier | flag | size | note |
|---|---|---|---|
| bf16 (default) | `PLE_BF16_TABLE=1` | 95.4 GiB | official weights, zero quantization error |
| fp8 | `PLE_BF16_TABLE=0` | 47.7 GiB | ~2% per-element error (e4m3); measured cost is ~3% prefill at TP2, flat at TP1 |

**bf16 (built by us, from the official public checkpoint):**

```bash
# 3a. download the 33 shards that carry the n-gram table. Derive the list from
#     the index — never hand-pick (docs/bf16-ple-table.md has the snippet):
mkdir -p "$MODELS_DIR/ple-bf16-src"
# ... copy model-000NN-of-00131.safetensors files that contain
#     ngram_embedding.shard_* tensors into that dir

# 3b. assemble (~102.4 GB scratch; verifies all 128 shards and 10 random rows byte-for-byte)
PLE_BF16_SRC="$MODELS_DIR/ple-bf16-src" python3 scripts/ple-bf16-assemble.py

# 3c. finish: scale-aware cross-check against the fp8 table, NaN census,
#     then an ATOMIC deploy into the tier directory
PLE_FP8_DIR="$MODELS_DIR/ple-disk8" python3 scripts/ple-bf16-finish.py
```

Result: `$MODELS_DIR/ple-disk8-bf16/language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight.bin`
plus a `.done.json` marker. The marker matters: its absence means the next boot
pays a first-boot write of the whole table.

**fp8:** same target filename, in `$MODELS_DIR/ple-disk8/` instead. It comes from
the checkpoint's own FP8 PLE (or the published quantized sidecar).

**Rules that will bite you:**
- The table directory is **derived from `PLE_BF16_TABLE`**; a contradictory explicit
  `PLE_DISK8_DIR` is ignored with a warning (a half-toggled pair used to produce a
  refused boot). One flag, one tier.
- The launcher **refuses to truncate an existing table** — if the dtype and the
  on-disk file disagree, it stops rather than silently rewriting 100 GB.
- Disk must be NVMe. A USB/PCIe-x1 enclosure on this path costs seconds per prefill.

## 4. Configure

```bash
cp .env.example .env
$EDITOR .env          # every flag is documented there: what it does, what it costs
```

Minimum edits: `MODEL_DIR`, `LANE_HOME`/`MODELS_DIR`, `FN_EXT_DIR`, `TP_SIZE` +
`CUDA_VISIBLE_DEVICES`, and the context/pool pair. If your GPUs are 24 GB or your
RAM is smaller than 128 GB, re-derive `VLLM_WNA16_STATIC_HOT_CACHE_SIZE`,
`KV_CACHE_MEMORY_BYTES` and `CPU_OFFLOAD_GB` before first boot — the boot banner
(slide 5) tells you whether they agree.

## 5. Boot, verify, bench

```bash
./local-start.sh                 # sources .env, runs preflight, starts the container
docker logs -f qwen38-flash-next
```

Healthy boot reads, in order (any missing line is a finding, not a nuisance):

| witness | means |
|---|---|
| `[lane] expert-view mode: HYBRID ...` | the tier you asked for is the tier you got |
| `[hybrid] VMM+LRU cache: layer=... tensor_rows=... device_prefix=... host_duplicate=...` | per layer-rank; count = layers x ranks |
| *no* `host duplicate does not mirror the device prefix` | the byte-level canary passed (weights are where the maps say) |
| `Custom allreduce kept ENABLED on 4 PCIe-only GPUs` | our collective wiring is live (needs `VLLM_CUSTOM_AR_ALLOW_PCIE=1`) |
| `Using FlashInfer GDN prefill kernel` | only if you supplied the GDN overlay — intentionally absent from this repo |
| `GPU KV cache size: N tokens ... Maximum concurrency: Mx` | **N must be >= MAX_MODEL_LEN** or your pool/ctx pair is wrong |
| `Capturing CUDA graphs (FULL): 2/2` | batch-1 and batch-2 capture survived (the failure mode is a hang here) |
| `Application startup complete` + `curl -sf localhost:PORT/health` | serving |

Then the four gates we will not skip, in this order — correctness before speed:

```bash
# math gate (temperature 0): must be exact
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
 '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"19*23 = ? answer with the number only"}],"temperature":0}'

# recall at depth — the gate that catches silent expert-weight errors
python3 benchmarks/hybrid-vmm-lru/needle-probe.py --port 8000 --tokens 300000 --out 200 --runs 2

# prefill and decode (32k is the real prefill gate; sub-30k probes miss table/tier regressions)
python3 benchmarks/hybrid-vmm-lru/bench-3k500.py --port 8000 --n 3 --warmup 1
python3 benchmarks/hybrid-vmm-lru/bench-3k500.py --port 8000 --n 2 --warmup 0 --in 32768 --out 32
python3 benchmarks/hybrid-vmm-lru/bench-3k500.py --port 8000 --n 3 --warmup 1 --in 128 --out 4096
```

Reference numbers for our box: [docs/benchmarks-hybrid-vmm-lru-20260924.json](benchmarks-hybrid-vmm-lru-20260924.json).
Expect cold boots to read 10-25% below published "warm session" bars; that is a
measurement condition, not a regression — re-bank only after a sustained run.

## 6. Changing an arm

Every arm change is **cold boot** — stop the container, edit `.env`, start. Warm
restarts on a lane with a swap-warm PLE table are a known freeze trap.

| to change | set | verify with |
|---|---|---|
| expert tier mechanism | `PREFILL_VMM_ARM=0\|1\|2` | `expert-view mode` line + the bench deltas in README |
| PLE dtype | `PLE_BF16_TABLE=1\|0` | the mounted table dir name + `GPU KV cache size` unchanged |
| KV dtype | `KV_CACHE_DTYPE=fp8\|bfloat16` | pool banner (fp8 frees ~1 GiB/card) |
| context length | `MAX_MODEL_LEN` + `HF_OVERRIDES_JSON` + `KV_CACHE_MEMORY_BYTES` | pool N >= MAX_MODEL_LEN, then a needle past the trained window |
| all-reduce posture | `DISABLE_CUSTOM_ALL_REDUCE` + `VLLM_CUSTOM_AR_ALLOW_PCIE` + `expandable_segments` | the launcher enforces the pairing; all three or none |

## 7. systemd (optional, auto-start)

`systemd/qwen38-serve.service.example` is a template: copy, substitute your paths
and user, then `systemctl --user daemon-reload && systemctl --user enable --now`.
It must run after `docker.service` and after your model mounts exist
(`RequiresMountsFor=`), and note `Restart=on-failure` will re-exec `ExecStart`.

## 8. When it doesn't come up

| symptom | first thing to check |
|---|---|
| boot OOM in the first prefill | `VLLM_WNA16_STATIC_HOT_CACHE_SIZE` and `MAX_NUM_BATCHED_TOKENS` are both on-card budgets; lower one |
| `Maximum concurrency` < 1.0x or N < MAX_MODEL_LEN | `KV_CACHE_MEMORY_BYTES` vs `MAX_MODEL_LEN` disagree |
| hang at `Capturing CUDA graphs 2/2` | batch-2 capture; pin `cudagraph_capture_sizes:[1]` or run np=1 |
| `custom all-reduce` refused / pairing error | `DISABLE_CUSTOM_ALL_REDUCE=0` requires `expandable_segments:False` |
| rope/YaRN "worked" but no effect | the `hf_overrides` line must appear in the banner's non-default args; JSON needs single quotes |
| PLE workers `waiting for 4 registrations` forever | table path/marker missing — see the `.done.json` rule in step 3 |
| fast but wrong answers, or wrong recall | run the canary check: a fill/order bug is invisible to asserts. See README §5 |
| VRAM creep over days | `nvidia-smi` ratchet after first traffic is expected (~+500-800 MiB/card with async scheduling); *idle* growth is a leak |
