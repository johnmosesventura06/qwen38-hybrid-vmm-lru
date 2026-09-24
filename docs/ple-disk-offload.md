# PLE disk8 — NVMe-mmap table offload: how the patch works
_2026-09-22 · golden-11 mechanism, golden-12 tiers · our box_

Companion to `docs/bf16-ple-table.md` (which covers the bf16 *table* and the
four dtype/scale gates). This document is about the **offload machinery**
itself: what the fork does with the PLE table, and exactly what disk8 changed.

## 0. Who owns what (fork PLE-offload architecture)
- One **CPU `PleOffloadWorker` process** (spawned by the executor before model
  load) owns the n-gram table. GPU workers own *nothing* of it — after load
  they keep only the global `weight_scale` buffer (fp8 tier).
- GPU workers register, over ZMQ (`ipc:///tmp/...`): shared **CPU output
  buffers** (+ `ready`/`consumed` flags) per layer, and TP-rank-0 shared
  **input buffers** (`input_ids`, `query_start_loc`, `ngram_context`) per DP
  group. One registration per (dp_rank, layer).
- Per forward: the GPU worker ships ids via shared memory, the worker's
  `busy_loop` runs `layer.forward_impl` → trigram hashing → `index_select`
  over the table → writes gathered rows into the registered CPU output buffer
  in `get_offload_output_dtype()` → GPU consumes + dequantizes into model
  dtype. Wire dtype: **fp8 bytes** in the fp8 tier (half the bytes — the
  reason `get_offload_output_dtype` deliberately keeps quantized storage
  dtype); bf16 passthrough in bf16 tier (see companion doc §4.2).
- This is why serving the table from NVMe is *free of GPU graph impact*: the
  table was never touched inside CUDA graphs — only by the CPU worker.

## 1. Pre-disk8: the RAM regime
Stock disk-offload tier materialized the table as an **anonymous host-RAM
tensor**: 47.7 GiB (fp8) that had to be *resident and pinned-ish* forever.
On the 120 GiB box that meant: 62.8G experts + 47.7G PLE ≈ 115G used, ~5G
available, 49G swap resident, the "never restart swap-warm" hazard, and 10-30
min cold loads (the 47.7G memcpy + swap churn).

## 2. The disk8 patch (vendored in `ple-ext/nvfp4/worker.py`)
Three functions + a load-filter, all inert unless `VLLM_PLE_DISK_OFFLOAD_DIR`
is set by the serve block:

- **`_ple_disk_attach(layer, disk_dir)`** — for every offload layer whose
  largest parameter is ≥ 1 GiB (the table): expected file =
  `<dir>/<layer_name>.<param_name>.bin` + sibling `.done.json`
  `{shape, dtype}` contract.
  - *Complete* (done.json matches shape+dtype **and** file size ==
    numel×itemsize): maps the file **copy-on-write** (`np.memmap mode="c"`),
    `MADV_RANDOM` (gather order is random; defeats readahead that evicts hot
    pages). Parameter data replaced in place — module identity, loaders, and
    the whole gather path untouched. Log: `reusing finished file`.
  - *Not complete*: **refuse-to-truncate guard** (09-22, after a foreign
    95.4 GiB bf16 file was fp8-ified by the old recovery): existing file with
    a different size aborts with dtype/shape truth. Otherwise
    `truncate(expected)` + map **shared read-write** → first boot writes the
    table through to file as `load_weights` fills the parameter.
- **`_ple_disk_finalize(...)`** — end of load: `flush()` the mapping, write
  `.done.json`, **remap copy-on-write**. From second boot on, nothing is ever
  written to the file.
- **`offload_only_iter` (the load filter)** — checkpoint tensors whose mapped
  name resolves into a *completed* table are dropped before `load_weights`
  (the sibling `.weight_scale` is the documented exception: fp8 tier passes it
  through — a real parameter; bf16 tier drops it; both gated, see companion
  doc §4.3). Witness line: `matched N checkpoint tensors, loaded K entries,
  verified M/J materialized` — **fp8-first-boot: 2/2; fp8-steady: 1/2;
  bf16-steady: 0/1**. Read these; they're the only hard proof of which tier
  actually booted.
- Consequence on disk: page-cache file, **reclaimable** under pressure (kernel
  drops cold table pages instead of swapping), hot rows stay cached. RAM
  drops 115→91 with identical throughput; boot memcpy gone (49 s weights);
  no swap regime; safe to reboot swap-clean forever.

## 3. The serve block (`scripts/docker_serve.sh`, gated on `PLE_ARM_DISK8=1`)
Mounts **only** the superset worker (0 deletions vs the image's
`v1/ple_offload/worker.py`) at the baked path + the table dir **rw**; sets
`VLLM_PLE_DISK_OFFLOAD_DIR=/ple_disk`; blanks the g10 sidecar env
(`VLLM_PLE_QUANT_DIR=`, `VLLM_PLE_PACKED=`) so quant-sidecar mode can't hijack
the boot; forwards `PLE_BF16_TABLE` for the companion-doc gates. No
`ple_layer` overlay on the fp8 tier (none needed; bf16 adds one — §4.1 of the
companion doc).

## 4. Table inventory on this box
| dir | tier | dtype | bytes | selected by |
|---|---|---|---|---|
| `$MODELS_DIR/ple-disk8` | fp8 (golden-11) | float8_e4m3fn | 51,200,245,760 | `PLE_BF16_TABLE=0` (default) |
| `$MODELS_DIR/ple-disk8-bf16` | bf16 (golden-12 boot default) | bfloat16 | 102,400,491,520 | `PLE_BF16_TABLE=1` + `PLE_DISK8_DIR=...` |

## 5. What this is NOT: g6/g10 sidecars
`PLE_ARM_INT4`/`PLE_ARM_NVFP4` mount a *packed* table + a GPU-side unpack
kernel (`ple_layer` overlay) — cheap on disk, expensive at the GPU: both died
of the deep-prefill unpack tax (g6 1300→741; g10 owner −500). disk8 has **no
unpack anywhere**: the file's bytes ARE the model's storage-format bytes; the
GPU-side fp8→bf16 cast is the one the RAM tier always did. That's why disk8
passed the ≥30k deep-prefill gate the first try and the packed tiers never
would.

## 6. Ops
- Boot witness order (any tier): `PLE-disk8 arm active` → `PLE disk offload:
  ... (95.4 GiB, reusing finished file)` → `verified 0/1 materialized` (bf16)
  or `1/2` (fp8) → `Loading weights took ~49s` → pool `517,858 @ 1.05x`.
- Rollbacks: `PLE_BF16_TABLE=0` = the fp8 disk tier; `PLE_ARM_DISK8=0` plus
  `PLE_ARM_NVFP4=1` = the packed page-cache tier that came before it. Cold boot for each.
- First-ever boot on a fresh dir = write-through (~2-3 min extra) — that's the
  one boot where the worker opens the file shared-rw. It refuses to clobber
  foreign files; only absent/complete tables accepted.

*Written from the vendored code (line-verified 09-22) and boot witnesses of
the golden-11/12 sessions.*
