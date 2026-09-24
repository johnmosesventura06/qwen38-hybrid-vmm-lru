# PREFILL_VMM_ARM — expert-view toggle (09-23; updated 09-24 for golden-v2.0)

One knob in `.env`: `PREFILL_VMM_ARM=0|1|2`. Cold boot to flip (warm restart on a swap-warm PLE table is a known freeze trap).

| Mode | Mechanism | Measured (same rig, cold boots; 09-24 window, all rows in one window) | Use when |
|---|---|---|---|
| 2 (BOOT DEFAULT, golden-v2.0) | HYBRID: VMM contiguous expert view + the dynamic-LRU mirror riding on the VMM device prefix (one tier, two access paths) | pp 3k 1727 / 32k 1917 / 190K 1770-1822; tg 55.95 probe, 59.66 sustained; recall 190K pass | everything — prefill headroom AND decode at golden parity |
| 0 | Golden: full expert set pinned in host RAM (UVA) + static hot-cache mirror 44 rows + dynamic LRU gather | pp 3k 1258 / 32k 1412 / 190K 1345; tg 55.41 probe, 61.25 sustained | decode-first tier; rollback target |
| 1 | Mixed VMM: ONE contiguous expert tensor, VRAM prefix (hot-44 pages) + host suffix page tables; LRU/mirror OFF | pp 3k 1670 / 32k 1839 / 190K 1731; tg 37.59 probe, 27.74 sustained | long-context ingest where decode does not matter |

Mechanism, code changes, invariants and the failure modes met on the way:
`docs/hybrid-vmm-lru-mechanism.md`. Rollback: set the value plus a cold boot. Measured deltas per arm: README.md §1.

Bucko #16 provenance: the mechanism lives in the baked image (`_allocate_mixed_vmm_tensor`)
but was unreachable on our checkpoint — the AutoGPTQ loader shim calls the VMM init
that exists only on the compressed-tensors class (AttributeError with LRU=0). The
`fn-ext/.../auto_gptq.py` overlay borrows the VMM impl and re-aliases the AutoRound
kernel names (w13_qweight/w13_scales...) onto the VMM tensors post-permutation —
WITHOUT the re-alias the arm silently computes on stale tensors. With LRU=1 and the
hybrid flag absent the overlay is inert (verified: golden boot shows mirror lines).

Arm 2 adds `fn-ext/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py`
(the hybrid module, sha256 8f24042bb81edf441f849f4ce9babe7d2fc8b437349d54e89718cd4e8d031180)
and exports `VLLM_WNA16_VMM_LRU_HYBRID=1` from `serve-container.sh`. That module is inert
for arms 0/1 (flag absent), so one overlay serves all three arms.

Bucko ran VMM at hot-128 on 24GB cards; our 16GB cards cap near hot-44..64 before the
KV pool pays. Arm 1's decode regression at 44 was NOT the tier being too small — it was
the missing mirror: arm 2 rides the same 44 rows and measures 55.95 (probe) / 59.66
(sustained), i.e. golden parity with arm-1 prefill. The old "hot 32-44 is flat" law was
measured for the STATIC mirror and does not bound the LRU, so extra slots are the next
lever to test (VRAM-funded) — see the mechanism doc §1.

witnesses: the container log prints the expert-view mode versus .env, the arm-2 hybrid
module mount, and the overlay mount.
rollback of the TOGGLE itself = set PREFILL_VMM_ARM=0|1 + cold boot (git revert only if
the module/scripts come out with it).