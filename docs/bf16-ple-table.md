# BF16 PLE table for Qwen3.8-Flash-Next — assembly + vLLM integration
_2026-09-22 · golden-12 · 4x RTX 5060 Ti_

## What we built
A drop-in **bf16 n-gram (PLE) table** for the W4A16 lane, served from NVMe mmap
instead of RAM, replacing the baked fp8 table. Official weights, zero
quantization error, exactly what the model shipped with.

| | fp8 table (golden-11) | bf16 table (golden-12) |
|---|---|---|
| shape | 320,001,536 rows x 160 | same |
| bytes | 51,200,245,760 (47.7 GiB) | 102,400,491,520 (95.4 GiB) |
| per-element error | ~2% (e4m3; measured median 2.05% vs bf16) | 0 (source of truth) |
| carried in | host NVMe mmap (disk8 tier) | host NVMe mmap (same tier) |

## 1. Where the bytes come from
- Source: **`Qwen/Qwen3.8-Flash-Next`** (official BF16, public, ungated).
- The table is stored as **128 tensors**:
  `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight`,
  each `[2,500,012 x 160]` BF16 (128 x 2,500,012 = 320,001,536 — matches the
  model's padded vocab). Two small friends ride along (`ngram_heads_offsets`,
  `ngram_heads_vocab_sizes`) — they are NOT part of the .bin.
- All 128 shards live in files **`model-00005-of-00131.safetensors` ..
  `model-00037-of-00131.safetensors`** (33 files, ~89G). Derivable from the
  repo's `model.safetensors.index.json` — never hand-pick:

```python
import json, urllib.request
idx = json.load(open("/tmp/official-index.json"))  # curl -L .../resolve/main/model.safetensors.index.json
files = sorted({v for k, v in idx["weight_map"].items() if "ngram_embedding.shard_" in k})
```

- Download only those 33 (took 36 min at ~500 Mbps):

```python
snapshot_download("Qwen/Qwen3.8-Flash-Next", allow_patterns=files,
                  local_dir="ple-bf16-src", max_workers=4)
```

## 2. Assembly rules (the parts that bite)
1. **Row order = numeric shard order**, `shard_0 .. shard_127`, each block
   appended contiguously. `shard_10` comes AFTER `shard_2` — lexicographic
   sort corrupts 90% of the table while looking completely fine.
2. Raw-byte copy: a safetensors tensor is contiguous little-endian at
   `8 + header_len + data_start` — read/seek, no dtype conversions anywhere.
3. The target path must not pre-exist; assemble to a temp name, verify, then
   rename. (See §5 for why this rule exists.)

## 3. Verification gates before the table is allowed to serve (5)
- map all 128 shards from the files' own headers: shape/dtype/off-size asserts
- final size `== 102,400,491,520` bytes, exact
- 10 random rows byte-for-byte vs their source shard
- **scale-aware fp8 cross-check**: for 120 random rows, the fp8 table's bytes
  dequantized by the checkpoint's global `weight_scale` (= **1.993e-4**, read
  from `...ngram_embedding.weight_scale` in the W4A16 ckpt) must match the
  bf16 row within ~e4m3 noise → passes at ≤0.025, explodes on wrong ordering.
  This is the gate that proves ordering+content against something already
  serving — do NOT compare raw magnitudes (fp8 floats are stored pre-scale and
  look ~5000x larger; that false alarm cost a debug cycle).
- NaN census over the whole file, chunked (a full memmap expression on 95G
  allocates a 95G temp — chunk at 2^26 elements)

Then write `*.done.json` = `{"shape":[320001536,160],"dtype":"torch.bfloat16",
"source":"Qwen/Qwen3.8-Flash-Next"}` and rename into place.

## 4. Making vLLM run it (golden-12 code, all committed)
The disk8 tier (golden-11) already serves an fp8 `.bin` via
`VLLM_PLE_DISK_OFFLOAD_DIR` + a superset `ple_offload` worker. bf16 needed
four gates, all keyed on **`PLE_BF16_TABLE=1`** (unset/0 = byte-identical
golden-11 behavior):

1. **PLE method selection** — `fn-ext/.../ple_layer.py` (new overlay):
   `_get_ple_embedding_quant_method()` returns `None` (unquantized,
   params_dtype=bf16) when the env is on. The parameter is then bf16, the
   worker's dtype/nbytes check matches the file, `done.json` says complete →
   **reuse, no write-through**. Note: `--hf-overrides` `ple_embedding_dtype:
   null` does NOT work (vLLM ignores null overrides) — env gate only.
2. **Offload IPC dtype** — the fork's `get_offload_output_dtype()` fallback
   hardcoded `float8_e4m3fn`; the CPU worker then did
   `index_select(bf16 table → fp8 buffer)` → silent per-request
   "same scalar type" error (no crash — the tell). Fixed to return
   `default_dtype` under the toggle.
3. **Checkpoint weight_scale** — dropped in the worker's load filter under the
   toggle (a bf16 table has no scale; loading one would double-scale).
4. **serve forwarding** — `docker_serve.sh` passes `-e PLE_BF16_TABLE=1`
   through to the container.

And the guard born from §5: worker `_ple_disk_attach` now **refuses to
truncate** any existing `.bin` whose size != expected and aborts with the full
dtype/shape truth.

Toggle surface: `.env` gains
`PLE_BF16_TABLE=1` + `PLE_DISK8_DIR=$MODELS_DIR/ple-disk8-bf16`;
`0` + default dir = fp8 tier. Boot default is simply whatever `.env` last said; the
boot witnesses read the toggle and check that table's `done.json`
(see docs/RUNNING.md step 5).

## 5. Post-mortem: the first table we lost
First attempt steered the fp8-typed model at the bf16 file via hf-overrides
(see §4.1 — didn't apply). The disk tier saw "size != fp8-expected", entered
its first-boot recovery — `truncate()` + write-through — and **overwrote the
downloaded artifact** with an fp8 table (and rewrote its done.json) while the
loader bar showed an innocent 100%. The kill order was after the finalize;
the bytes were already gone. Lessons encoded: trust env gates over override
dicts; never point a self-repairing scratch path at a downloaded artifact
until a refuse-truncate guard exists (it does now); a "successful" boot log
(2/2→0/1 materialized, "reusing finished file") is the only dtype evidence
worth reading.

## 6. Measured cost/benefit (same box, same everything else)
np1 58.6 vs 59.5 (flat); np2 pair 86.1/wall 96.4 vs 89.4/100.1 (−3.7%);
prefill 8k 1,250 vs 1,286, 64k 1,071 vs 1,110 (−3%); KV pool/VRAM identical
(517,858 @ 1.05x); boot ~4 min both. Quality: needles 5/5 both ways — bf16 is
precision insurance, not a speed or visible-quality win, until someone finds a
task where 2% embedding noise matters. One transient allocator-OOM during
capture on the 0.956 budget — recovered; watch item.

## 7. Reproduce from scratch (e.g. on a fresh box)
```
# 1. index -> file list (33 files); 2. download;
# 3. python3 scripts/ple-bf16-assemble.py   (map+assemble+spot-check, from our box)
# 4. docker run --rm --entrypoint python3 -v $HOME/models:/m -v $PWD/scripts/ple-bf16-finish.py:/tmp/f pool-arm:fp8kv-v2 /tmp/f
#    (scale-aware cross-check + NaN census + atomic deploy)  [needs torch -> container]
# 5. table now live at ~/models/ple-disk8-bf16; toggle/profile per docs/bf16-ple-table.md §4.
# 6. boot with PLE_BF16_TABLE=1 (cold boot).
```

*Written from the live session logs; incident numbers verified against
the container boot log and the KV-pool banner witnesses.*
