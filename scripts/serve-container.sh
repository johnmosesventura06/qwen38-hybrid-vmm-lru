#!/usr/bin/env bash
set -euo pipefail

profile=/opt/qwen38/configs/2x3090-128gb.env
[[ -f "$profile" ]] || { echo "missing runtime profile: $profile" >&2; exit 2; }
# shellcheck source=/dev/null
source "$profile"

case "$DISABLE_CUSTOM_ALL_REDUCE" in
  0) custom_all_reduce_arg= ;;
  1) custom_all_reduce_arg=--disable-custom-all-reduce ;;
  *)
    echo "DISABLE_CUSTOM_ALL_REDUCE must be 0 or 1" >&2
    exit 2
    ;;
esac

allocator_config=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
expandable_segments=
IFS=',' read -r -a allocator_options <<< "$allocator_config"
for option in "${allocator_options[@]}"; do
  compact_option=${option//[[:space:]]/}
  case "$compact_option" in
    expandable_segments:True|expandable_segments:true)
      expandable_segments=True
      ;;
    expandable_segments:False|expandable_segments:false)
      expandable_segments=False
      ;;
  esac
done
if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == 0 && "$expandable_segments" == True ]]; then
  echo "DISABLE_CUSTOM_ALL_REDUCE=0 is incompatible with expandable_segments:True in this pinned runtime; set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False or keep DISABLE_CUSTOM_ALL_REDUCE=1" >&2
  exit 2
fi

model=${1:-/model}
mtp_model=${2:-"$model/runtime/mtp-int4-g32"}
rankings=/workspace/static_hot_cache_rankings.json

[[ -f "$model/model.safetensors.index.json" ]] || {
  echo "model checkpoint not found at $model" >&2
  exit 2
}
[[ -f "$rankings" ]] || { echo "missing hot-cache rankings: $rankings" >&2; exit 2; }
[[ -f "$mtp_model/model.safetensors.index.json" ]] || {
  echo "compact MTP checkpoint not found at $mtp_model" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTORCH_CUDA_ALLOC_CONF=$allocator_config
export VLLM_PLE_CPU_OFFLOAD=1
# mixvmm toggle (09-23): PREFILL_VMM_ARM=1 -> VMM contiguous expert view
# (measured: prefill +34%, decode -35%; cold boot to flip). 0/unset = golden
# static hot-cache mirror + dynamic LRU (the banked bars). See docs/mixvmm-toggle.md.
# 2 = HYBRID (wp/hybrid-vmm-lru): VMM contiguous view + LRU mirror riding on the
# same VMM prefix (one tier, two access paths). See docs/hybrid-vmm-lru.md.
case "${PREFILL_VMM_ARM:-0}" in
  2)
    export VLLM_WNA16_DYNAMIC_LRU=1
    export VLLM_WNA16_VMM_LRU_HYBRID=1
    lane_view="HYBRID (VMM contiguous view + LRU mirror on the prefix)"
    ;;
  1)
    export VLLM_WNA16_DYNAMIC_LRU=0
    lane_view="VMM PREFILL ARM (contiguous hot+host, LRU off)"
    ;;
  *)
    export VLLM_WNA16_DYNAMIC_LRU=1
    lane_view="GOLDEN (mirror pin + dynamic LRU)"
    ;;
esac
echo "[lane] expert-view mode: $lane_view"
export VLLM_WNA16_STATIC_HOT_CACHE_FILE=$rankings
export VLLM_WNA16_MIXED_VMM_HOT_CACHE=1
export VLLM_FORCE_DYNAMIC_SPEC_SCHEDULING=1

# lane-local: MTP_DEPTH=0 turns speculative decoding off entirely
spec_arg=()
if [[ "${MTP_DEPTH:-3}" != "0" ]]; then
spec_arg=(--speculative-config
  "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_DEPTH,\"use_local_argmax_reduction\":true,\"model\":\"$mtp_model\"}")
fi

# lane-local: VISION=1 loads the multimodal tower (default off = upstream
# --language-model-only profile). Processor transforms always run on CPU.
mm_arg=(--mm-processor-device cpu)
if [[ "${VISION:-0}" != "1" ]]; then
  mm_arg+=(--language-model-only)
fi

# lane-local: HF_OVERRIDES_JSON (deep dict, e.g. YaRN rope on text_config)
# forwarded as vLLM's --hf-overrides. Verified on this build: dict values are
# applied recursively onto nested PretrainedConfigs (ModelConfig.
# _apply_dict_overrides), so text_config.rope_parameters edits reach the QSA
# layers without touching the read-only checkpoint.
hf_arg=()
if [[ -n "${HF_OVERRIDES_JSON:-}" ]]; then
  hf_arg=(--hf-overrides "$HF_OVERRIDES_JSON")
fi

# lane KV tier arm (09-23 owner, golden-13): KV_TIER_JSON carries the full
# --kv-transfer-config doc (shipped form: SimpleCPUOffloadConnector disk mode —
# evicted prefix blocks stream to NVMe and reload from there instead of being
# re-prefilled). INERT when unset -> golden-12 boot byte-identical.
kv_transfer_args=()
if [[ -n "${KV_TIER_JSON:-}" ]]; then
  kv_transfer_args=(--kv-transfer-config "$KV_TIER_JSON")
  echo "[qwen38-serve] KV tier arm active"
fi

# lane knob: ASYNC_SCHED=1 enables vLLM async scheduling (default keeps Bucko's off)
if [[ "${ASYNC_SCHED:-0}" == 1 ]]; then async_scheduling_arg=--async-scheduling; else async_scheduling_arg=--no-async-scheduling; fi

if [[ -z "${COMPILATION_CONFIG:-}" ]]; then COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'; fi
export COMPILATION_CONFIG_DEFAULT="$COMPILATION_CONFIG"

exec vllm serve "$model" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "${TP_SIZE:-2}" \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --moe-backend humming \
  --dtype bfloat16 \
  "${mm_arg[@]}" \
  --load-format safetensors \
  --safetensors-load-strategy lazy \
  --max-parallel-loading-workers "$MAX_PARALLEL_LOADING_WORKERS" \
  --offload-backend uva \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --cpu-offload-params experts \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}" \
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
  "${hf_arg[@]}" \
  "${kv_transfer_args[@]}" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  "${async_scheduling_arg}" \
  ${custom_all_reduce_arg:+"$custom_all_reduce_arg"} \
  --compilation-config "$COMPILATION_CONFIG_DEFAULT" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  "${spec_arg[@]}"
