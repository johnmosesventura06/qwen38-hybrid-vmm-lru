# Independent review — wna16_hybrid_v2.py (VMM+LRU hybrid, v2 hot-duplicate fix)

Reviewer: Hermes subagent (independent verification pass), 2026-09-24.
Artifacts verified by sha256 before review:
- `wna16_hybrid_v2.py` 2465 lines, `12e4d2b9...f354` ✓ matches assignment; `py_compile` clean.
- `wna16_hybrid_v2.patch` `a0bcf39d...fce7` ✓ matches notes §0.
- `image_module.py` (base) `aee847ab...c099` ✓; v1 fn-ext file `7ab7d6c4...7abe` ✓.
All `v2:NNNN` references below are line numbers in `the pre-release module`
(sha-identical to the copy reviewed); `image:NNNN` = `image_module.py`; `v1:NNNN` =
`fn-ext/.../compressed_tensors_moe_wna16.py`.
v1→v2 delta re-diffed locally: exactly 20 hunks, none touching `apply()` or the triton
kernels — the notes' §2 scope claim is accurate.

## VERDICT: REJECT

One correctness blocker (F1): the boot-time fill lays the duplicate rows in the **wrong
end of the tensor** relative to every map that addresses them. The v1 lossy-eviction flaw
is genuinely fixed at the map-algebra level (the algebra is verified sound below), but
the v2 fix as coded serves the **wrong expert's weights for every cold expert from
token zero**, and every one of the new install-time asserts passes on the broken build
because they check map arithmetic only — never row content. The fix is a two-site
argument swap (F1 fix, one line each); re-review after it lands plus the content canary
suggested in F3. F2 is a second, un-mentioned regression in the golden static arm.

---

## Findings by severity

### F1 — BLOCKER: fill index order contradicts the host-row addressing (silent wrong-expert serving)

The permute kernel writes output row `r` from source row `F[r]` (gather-by-index):

```
v2:129-135   new_row = tl.program_id(0) ...
             old_row = tl.load(new_to_old_ptr + new_row)
             values  = tl.load(source_ptr + old_row*row_size + ...)
             tl.store(output_ptr + new_row*row_size + ..., values)
```

But the fill vector is `[full permutation, hot-prefix copy]` — the duplicate sits at the
**tail** (rows `[N, N+capacity)`):

```
v2:1219  fill_new_to_old = torch.cat([new_to_old, new_to_old[:hot_experts]])   # VMM tensors
v2:1361  permute_index  = torch.cat([new_to_old, new_to_old[:capacity]])       # <64MiB tensors
```

Every consumer addresses the duplicate as if it started at `capacity`:

- docstring contract, v2:1080-1085: “rows `[capacity, capacity + N)`: HOST-mapped copy of
  the FULL permutation, i.e. row `capacity + k` holds the expert at permuted position `k`”
- `_served_row`, v2:1339-1345: `capacity + new_row` for cold experts → installed `new_map`
  (and therefore `rot_map` initial, v2:1699 `state["new_map"].clone()`);
- `cold_map[global_id] = capacity + old_to_new[local_id]`, v2:1673, the fix itself.

Under the code's actual fill, row `capacity + p` holds expert `new_to_old[capacity + p]`
(a *different* expert) whenever `capacity+p < N`, or `new_to_old[p - (N - capacity)]`
when `capacity+p ≥ N`. With the live geometry (capacity=44, N=128) this means:
all 84 cold experts (perm positions 44..127) read a wrong expert's bytes at the FIRST
prefill, before any decode step has run — directly refuting spec invariant 5
(“prefill before any decode step must be bit-identical to arm 1”, `hybrid-vmm-lru.md:109-110`)
and notes §3c. Every decode gather (`miss_local_ids` stores the `cold_map` row value,
v2:244 via `source_map_ptr = cache.cold_map` at v2:2294; the gather kernel reads
`source_ptr + miss_local_ids[i]*row_size`, v2:269-272) copies the wrong expert into a
slot — consistently across all 10 tensors (weights, scales, zp, g_idx, sort all built
from the same misordered index), so the GEMM is structurally valid and numerically
plausible: **silent garbage, not a crash**.

Crucially, all install asserts (v2:1741-1765) are range/arithmetic checks on the maps;
the broken build satisfies every one of them (`cold ∈ [capacity, capacity+N)`, uniformity
1b, `rot_map == new_map`, dtype). Nothing in the boot path compares a row's bytes to the
expert the maps claim lives there.

Fix (design-faithful, minimal — swap the cat args so the host region starts at
`capacity` with the full permutation):

```python
v2:1219  fill_new_to_old = torch.cat([new_to_old[:hot_experts], new_to_old])
v2:1361  permute_index   = torch.cat([new_to_old[:capacity], new_to_old])
```

(The alternative — keep the tail layout and redefine homes as `p if p>=capacity else N+p`
— contradicts the docstring/spec/design and touches more code; not recommended.)
After the swap, the row-content claims in the v2:1213-1218 comment, the docstring,
`_served_row`, `cold_map`, and notes §3b all become true. Note that notes §3b itself
contains the same algebra error (`row capacity+k ← cat(new_to_old, new_to_old[:cap])[k]`
is only true if the kernel wrote row `capacity+k` from index `k`; it writes row `r` from
index `r`), so the fix must be matched with corrected wording there.

Not mentioned anywhere in the author's notes — their §7.4 simulation modeled “the exact
map algebra + rotation semantics” but axiomatized the row-content invariant instead of
simulating the fill vector; the bug is invisible to that method.

### F2 — HIGH (latent): `cold_out` tail changed for the golden STATIC mirror arm → hot experts double-counted

Patch hunk (`wna16_hybrid_v2.patch:826-833`) rewrites the base's cold-out call:

```
image:1979   expert_map=cache.cold_map     (base)
v2:2443      expert_map=plain_map          (v2)
```

For the hybrid arm the tail is unreachable (`cache.dynamic_lru` returns at v2:2345), so
the change buys nothing for v2; for golden **dynamic LRU** it is also unreachable. But
for the immutable static mirror arm (`DYNAMIC_LRU=0`, `STATIC_HOT_CACHE_SIZE>0`, no
staging — staging is opt-in, v2:421 defaults to 0), `cache.rot_map is None` makes
`plain_map = layer.expert_map` (v2:2250-2252), which — unlike the static `cold_map`
(image:1443-1448 masks hot experts to -1 only when `not dynamic_lru`) — does **not**
mask the hot experts. `cold_out + hot_out` (v2:2435-2461) then adds every hot expert
twice: wrong output for the static arm. This contradicts spec §3.2/notes §2.5 claims
that “MIXED_VMM=1 + LRU=1 + hybrid flag off” and golden generally “behave exactly as
today” (true for the arms this lane uses, false for this arm of the same file).
Fix: `expert_map=plain_map if cache.rot_map is not None else cache.cold_map` at v2:2443
(or make the tail hybrid-aware). Not mentioned in the author's notes.

### F3 — MEDIUM: install-time asserts cannot catch F1-class errors (task step 5 answer)

Existing (v2:1741-1769): cold range `-1 or [capacity, capacity+N)`; uniformity 1b
(every local `cold ≥ capacity` — the v1-flaw canary, correct and needed);
`old_to_new[hot[i]] == i`; `rot_map.equal(new_map)` (tautological — `rot_map` *is* a
clone of that tensor, v2:1699); range `-1 or < total_rows`; dtype equality. All are
map-arithmetic; none touches tensor bytes, so F1 boots clean. Asserts I would ADD:

1. **Content canary (catches F1 exactly), per installed expert tensor at hybrid init:**
   `assert torch.equal(t[capacity:2*capacity][slices], t[:capacity][slices])` — true iff
   the host region starts with the duplicate of the device prefix (i.e. iff the fill
   order matches the addressing); a cheap few-MiB slice compare would have failed the
   delivered build at boot. Better: one sampled row equality against the still-live
   source, `row(capacity+p) == source[new_to_old[p]]`, before `empty_host_cache()`
   releases it (v2:1450-1451).
2. `assert allocation.gpu_bytes >= capacity * row_bytes` per VMM tensor (the
   “prefix fully device-backed” claim, v2:1087-1088, is formula-true but undocumented
   at the assert layer; the 2 MiB pad overhang into `row capacity`'s leading bytes is
   genuinely benign — write and read hit the same VA mapping — author's notes §2.2
   states this correctly).
3. Injectivity canary: `rot_map` values over local experts must be distinct (holds in
   both layouts; guards future map-algebra regressions, including the CUDA-graph
   rotation path).
4. Beware `PYTHONOPTIMIZE` silencing all of these (v1 §6 carried this over; keep the
   boot witness line v2:1772-1780 printing `tensor_rows=172` so an -O launch is still
   auditable — it is: the `[hybrid]` witness prints total rows, dup MiB, ids).

### F4 — MEDIUM→acceptable: Humming `num_experts`/`num_local_experts` rebind to 172 (task step 3 adjudication)

Rebind sites: v2:1438 (`self.moe = replace(self.moe, num_local_experts=total_rows)`,
permanent for the shim's second `_setup_kernel`) and v2:1441-1444
(`layer.humming_configs` entries → `num_experts=total_rows`), both before `_setup_kernel`
(v2:1445). Independent audit of the container image (`qwen38-flash-next-2x3090:locked`,
read-only `docker run --rm --entrypoint bash`, no container ops against the live lane):

- Hard asserts satisfied: `fused_humming_moe.py:327-330` `moe_problem_size` —
  `meta1.num_experts` comes from `quant_config.w1_humming_config` ← the replaced
  `layer.humming_configs` (172 main / 44 mirror via `_make_hot_cache_kernel`'s own
  replace, v2:1514-1516, applied over the 172 configs → still 44 ✓); `w1.size(0)` is the
  installed 172-row tensor ✓. `assert meta1.num_experts == meta2.num_experts` ✓ both
  replaced.
- `humming_utils.py:1003` `num_experts = layer.moe_config.num_local_experts` is inside
  `process_humming_weights_after_loading` — runs BEFORE the VMM swap and builds the
  original 128 configs; not affected by the rebind, and `get_humming_moe_quant_config`
  (humming_utils:604-651) takes the passed `humming_configs` verbatim, never re-derives
  the count ✓.
- Runtime experts object: `fused_humming_moe.py:109-110` takes
  `moe_config.num_local_experts` (172) / `num_experts` (global 512, untouched) ✓;
  `MoEPermuteScratch` (fused_humming_moe.py:193-204) and grouped `moe_permute`
  (fused_humming_moe.py:779-786) are fed the same 172 ✓ self-consistent (grouped path
  not selected anyway; lane log witness says indexed gemm).
- `moe_align_block_size` (prepare_humming_moe_kwargs, fused_humming_moe.py:591-610) is
  keyed by **global** `num_experts=self.global_num_experts` (512); mapped values up to
  171 are block-table row ids into 172-row tensors ✓ no OOB. `locks = torch.zeros(1024)`
  sized independently ✓. Workspace/buffer shapes on the indexed path derive from M/K/topk,
  not expert count (`get_buffer_metas` non-batched branch, fused_humming_moe.py:380-384) ✓.
- Tree sweep for size-128 tables indexed by `expert_map` values: none found
  (`layer.py:346`/`expert_map_manager.py:77` are construction-time;
  `modular_kernel.py:1474` derives 172 from `w1.shape[0]`;
  `topk_weight_and_reduce.py:148` uses the output buffer's dim0, not weights) ✓.
- Tuning: `estimate_local_valid_shape_m` (fused_humming_moe.py:217-219) inflates
  `valid_shape_m` by 172/128 = 1.344×. In `humming/tune/base.py:61-72` the heuristic
  consumes `shape_m / num_experts`, and the estimate and the config count scale by the
  same factor → **exact cancellation vs arm 1**. Residual divergence:
  `base.py:221-227` `estimate_num_blocks_m = min(shape_m, num_experts)` (occupancy term
  → 172-capped vs 128-capped) and the bucket list from `get_configs` are built with
  num_experts=172 → at some M values the selected `moe_block_size` / write-split config
  may differ from arm 1. That changes alignment granularity / tile config only —
  block shape does not change per-element math, and no path sizes a lookup from the
  count inconsistently. On our SM89-era cards the sm90 occupancy sampling
  (`tune/sm90.py:136-142`) is not reached; `sm8x.py` has zero `num_experts` consumers.
- **Adjudication: 172 is safe for correctness at run time; R3 is real but perf-only.**
  The prefill acceptance gate must re-measure pp3k/pp32k rather than assume bit-for-bit
  parity with arm 1 (author's own R3 position — confirmed, and no additional hidden
  dependency found beyond what §5 of the notes lists). The mechanism (bind-time
  `replace`) is already proven live at 44 for the mirror kernel since the v1 boot.

### F5 — LOW: `host_duplicate=` witness overstates host pinning

`vmm_dup_bytes` mixes VMM host-backed duplicate bytes with the `<64MiB` regular tensors'
duplicate rows, which are plain VRAM (added at v2:1409-1412 but accumulated into the
same counter printed as `host_duplicate=%.2f MiB` at v2:1772-1780). Memory *arithmetic*
in notes §4 is right (and the R4/R5 headroom flags are the honest caveats); only the
per-layer witness line is misleading when cross-checking against `/proc/meminfo`.

### F6 — info, verified-correct: flag-off parity (task step 4)

- Arm 1 (`MIXED_VMM=1`, `DYNAMIC_LRU=0`): `host_duplicate = self._dynamic_lru_enabled and
  ...` = False (v2:1329-1336) → `total_rows == N` (v2:1337), `_served_row` identity
  (v2:1339-1345), `permute_index = new_to_old` (v2:1362-1363), `_allocate_mixed_vmm_tensor`
  with `host_duplicate=False` reproduces base arithmetic exactly (v2:1144-1150:
  `original_bytes = total_rows*row_bytes` equals base's `source.numel()*element_size`
  for the checked-contiguous source, v2:1094), rebind blocks skipped (v2:1427), base log
  line untouched (v2:1470-1478). Gate: v2:1791 `if mixed and not lru: → arm-1 → return` ✓.
- Hybrid flag off on the gate: v2:1793-1808 — for WNA16 self, `self._vmm_lru_hybrid`
  False (v2:432); for borrowed AutoGPTQ self (no such attr), the env fallback re-reads
  `VLLM_WNA16_VMM_LRU_HYBRID=="0"` → falls through to golden init, which never sets
  `rot_map` (default None, v2:328) → `plain_map = layer.expert_map` everywhere
  (v2:2250-2252), the two LRU guards identical (v2:2277-2290), update kernel with
  `HYBRID=False` (v2:2306) compiles the rot branches out and the 1-element
  `lru_dummy` (v2:2304-2305, never dereferenced under `HYBRID=False`) keeps the
  signature compatible. Behaviorally identical to base for both dynamic-LRU arms.
  **Caveat: F2 falsifies the blanket “identical to today” claim for the static mirror
  arm**, which shares the file but not these two flags.
- Gate ordering hazard checked: the hybrid init clears `_mixed_vmm_state` (v2:1598)
  before the instance-level `maybe_init_mixed_vmm_hot_cache` call (v2:1601, which
  resolves to the AutoGPTQ shim `_mixvmm_init_on_autogptq`,
  `fn-ext/auto_gptq.py:988-1008`: orig init → re-alias → second `_setup_kernel`), and
  raises if the VMM init declined the duplicate layout (v2:1605-1613) ✓.

---

## Step-1 trace (capacity=4, N=8, G=8, identity expert_map; ranking [5,7,2,0,6,3,1,4])

`new_to_old = [5,7,2,0,1,3,4,6]`, `old_to_new = [3,4,2,5,6,0,7,1]`, total_rows=12.
Maps (identical in DESIGN and CODE — arithmetic verified from v2:1339-1350, 1668-1675, 1699):

```
hot_map (slot)   {5:0, 7:1, 2:2, 0:3}                        else -1
cold_map (row)   {5:4, 7:5, 2:6, 0:7, 1:8, 3:9, 4:10, 6:11}  (=capacity+old_to_new, all locals)
rot_map t=0      {5:0, 7:1, 2:2, 0:3, 1:8, 3:9, 4:10, 6:11}  (=installed new_map)
slot_global_ids  [5,7,2,0]   slot_ages [4,3,2,1]   clock [4]
```

Row byte content after boot (`row:design-expert | code-expert`), F_design=`n2o[:4]+n2o`,
F_code=`n2o+n2o[:4]` — the layouts differ in the host region only:

```
row            0  1  2  3 │ 4  5  6  7  8  9 10 11
DESIGN holds   5  7  2  0 │ 5  7  2  0  1  3  4  6   (dup at capacity+k ✓ matches maps)
CODE   holds   5  7  2  0 │ 1  3  4  6  5  7  2  0   (dup at tail ✗ off-by-capacity)
```

Steps (each cell: “expert g served by rot row r → bytes of expert X”):

| step | maps after (Δ only) | DESIGN: who serves whom | CODE: who serves whom |
|---|---|---|---|
| t0 prefill | init above | all 8 experts served correctly | e1→row8=**e5**, e3→row9=**e7**, e4→row10=**e2**, e6→row11=**e0** — 4/8 wrong at token 0 |
| B hit e7 | rot/ages unchanged (e7: slot1) | e7←slot1=e7 ✓ | same ✓ (slot rows always correct) |
| C miss e1, evicts slot3 (e0) | hot: e0→-1, e1→3; rot: e1→3, e0→7 | gather src row8=e1→slot3 ✓; e0 falls back to host row7=e0 ✓ — no loss | gather src row8=**e5**→slot3 (e1 now computes e5); e0→host row7=**e6** |
| D miss e3, evicts slot2 (e2) | hot: e2→-1, e3→2; rot: e3→2, e2→6 | src row9=e3 ✓; e2 host row6=e2 ✓ | src row9=**e7** (e3 computes e7); e2→row6=**e4** |
| E re-request evicted e0, evicts slot0 (e5) | hot: e5→-1, e0→0; rot: e0→0, e5→4 | src row7=e0 ✓ lossless re-promotion; e5 host row4=e5 ✓ | src row7=**e6**; e5→row4=**e1** |
| F miss e4, evicts slot1 (e7) | hot: e7→-1, e4→1; rot: e4→1, e7→5 | src row10=e4 ✓; e7 host row5=e7 ✓ (incl. a slot previously holding a hit-refreshed hot) | src row10=**e2**; e7→row5=**e3** |
| final prefill check | | **0 wrong** | **8/8 wrong** |
| prefill checks at each step | | `[]` always | `[1,3,4,6] → [0,1,3,4,6] → [0,1,2,3,4,6] → [0..6] → [0..7]` |

Confirmed independently (emulation mirrors the kernel: `local_id` loaded from
`cold_map` → `miss_local_ids` rows, argmin-with-protection victim, hit/miss/evict
rot stores, v2:179-249):
- **No expert ever unreadable** — TRUE in the map algebra for both layouts (every
  local keeps a host row; the bijection-in-the-maps survives every rotation).
- **Gather source == destination aliasing** — FALSE (impossible) in both layouts:
  sources are `cold_map` values ≥ capacity (v2:1673 + uniformity assert v2:1747-1753),
  destinations are slots < capacity (argmin over slot offsets, v2:195-196), and rows
  ≥ capacity are never written after boot (only writers: boot permute v2:1223-1230,
  gather into mirror views v2:2314-2344). The v1 lossy hole is genuinely closed.
- **Prefill map always points at the expert's own bytes** — TRUE for the DESIGN
  layout, **REFUTED for the code as delivered** (F1): rows exist, but hold the wrong
  expert's data.
Live-geometry consequence of F1: at capacity=44, N=128, prefill of every routed cold
expert (positions 44..127) reads `row 44+p` holding expert `new_to_old[44+p]` (or, for
p ≥ 84, `new_to_old[p-84]`) — different on every layer where the permutation is
non-degenerate, i.e. always.

## Step-2 fill audit (grid/size math) — otherwise clean

`grid = (total_rows, cdiv(row_size,1024))` with mask `offsets < row_size` (v2:1222-1230,
kernel v2:121-135): every row of `[0, total_rows)` written exactly once, no unwritten or
double-written rows, no OOB index load (`len(fill_new_to_old) == total_rows ==
grid dim0`; `fill_new_to_old[r] ∈ [0, N)` always a valid source row). `new_to_old` is
`torch.long` on the right device ✓. Byte geometry: `total_rows*row_bytes` ≤
`mapped_bytes` ✓; `gpu_bytes = align_up(capacity*row_bytes) ≥ capacity*row_bytes` → the
prefix rows are always entirely device-backed; the up-to-2 MiB pad physically covering
the leading bytes of row `capacity` is a tier artifact only (same VA reads what it wrote)
✓ notes §2.2 correct on this point. Regular tensors (`index_select`, v2:1404-1407) share
the exact same (mis)ordering → consistently mis-mapped across all 10 tensors, which is
why it reads as a plausible model rather than a crash.

## Things NOT mentioned in the author's notes (highest-value items)

1. **F1 — the fill/cat order bug.** The notes (§1, §3b) *assert* the capacity+k
   invariant as fact and their §7.4 simulation assumed it axiomatically instead of
   modeling `_permute_expert_rows_kernel` + the actual cat order; the delivered code
   contradicts its own docstring (v2:1080-1085). Every install assert passes on the
   broken build.
2. **F2 — the `cold_out` tail change (v2:2443 vs image:1979) regresses the golden
   static (non-LRU, non-staged) mirror arm** with double-counted hot experts.
   Contradicts the blanket “golden unchanged” claims, though not for the three VMM arms.
3. **F3 — the assert suite's blind spot**: `rot_map.equal(state["new_map"])` (v2:1763)
   is tautological, and no assert anywhere compares bytes to maps; the notes present
   the assert list as covering the invariants without noting that all checks are
   content-blind.
4. F4 tuning detail beyond R3's one-liner: the `shape_m/num_experts` heuristic
   cancellation is exact (base.py:69/72), so the realistic R3 exposure is narrower than
   stated — only the occupancy term (`min(shape_m, num_experts)`, base.py:221-227) and
   bucket-list shift can change block selection; no correctness path found where 172
   misleads the kernel.
5. F5 witness-mixing nit.

## Gate guidance if the fixed module is booted

All v2:1741-1765 asserts + the added content canary (F3-1) should pass on a fixed
build; the math gate (19*23) and the ≥90K needle probe from spec §5 are the arbiter for
F1-class regressions — note the *first* needle probe at sufficient depth is enough to
catch F1 on the unfixed build (cold-expert attention heads compute wrong weights from
step 0), but pp-perf gates alone (fast + garbage) would not.

Reproduction: the trace tables above come from a row-faithful Python emulation of the
update/gather/permute semantics read off v2:121-273 (script kept at
`the review workspace/sim_trace.py` on the review box);
container greps via `docker run --rm --entrypoint bash qwen38-flash-next-2x3090:locked`
(read-only; the live :8001 lane was not touched).
