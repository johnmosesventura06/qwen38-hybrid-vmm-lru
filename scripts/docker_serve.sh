#!/usr/bin/env bash
set -euo pipefail

# --- Host asset roots -------------------------------------------------------
# Everything below resolves from these three, so the repo is portable: set them
# in .env (or the environment) to wherever your checkpoint / PLE tables live.
# Previously these paths were hard-coded to a single host.
LANE_HOME=${LANE_HOME:-$HOME}
MODELS_DIR=${MODELS_DIR:-$LANE_HOME/models}
PLEQUANT_DIR=${PLEQUANT_DIR:-$LANE_HOME/plequant}
# ---------------------------------------------------------------------------

image=${IMAGE:-qwen38-flash-next-2x3090:locked}
model_dir=${MODEL_DIR:?Set MODEL_DIR to the assembled/downloaded model directory}
port=${PORT:-8000}
qsa_exact=${VLLM_QSA_EXACT_TOPK:-0}

if [[ ${DISABLE_CUSTOM_ALL_REDUCE+x} ]]; then
  case "$DISABLE_CUSTOM_ALL_REDUCE" in
    0|1) ;;
    *)
      echo "DISABLE_CUSTOM_ALL_REDUCE must be 0 or 1" >&2
      exit 2
      ;;
  esac
fi

docker_env=(
  -e "PORT=$port"
  -e "VLLM_QSA_EXACT_TOPK=$qsa_exact"
  -e "TP_SIZE=${TP_SIZE:-2}"
  -e "TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor"
  -e "VLLM_CUSTOM_AR_ALLOW_PCIE=${VLLM_CUSTOM_AR_ALLOW_PCIE:-0}"
)
for name in \
  CUDA_VISIBLE_DEVICES \
  NCCL_P2P_LEVEL \
  NCCL_PROTO \
  VLLM_CUSTOM_ALL_GATHER \
  VISION \
  SERVED_MODEL_NAME \
  MAX_MODEL_LEN \
  MAX_NUM_BATCHED_TOKENS \
  MAX_PARALLEL_LOADING_WORKERS \
  MAX_NUM_SEQS \
  ASYNC_SCHED \
  COMPILATION_CONFIG \
  VLLM_CUSTOM_AR_ALLOW_PCIE \
  FN_EXT_DIR \
  KV_CACHE_MEMORY_BYTES \
  CPU_OFFLOAD_GB \
  VLLM_PLE_OFFLOAD_READY_TIMEOUT \
  VLLM_WNA16_STATIC_HOT_CACHE_SIZE \
  VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS \
  PREFILL_VMM_ARM \
  VLLM_PREFIX_CACHE_RETENTION_INTERVAL \
  DISABLE_CUSTOM_ALL_REDUCE \
  VLLM_PLE_QUANT_DIR \
  VLLM_PLE_PACKED \
  PYTORCH_CUDA_ALLOC_CONF \
  HF_OVERRIDES_JSON \
  KV_CACHE_DTYPE \
  VLLM_ALLOW_LONG_MAX_MODEL_LEN \
  MTP_DEPTH \
  KV_TIER_JSON \
  PYTHONHASHSEED
do
  if declare -p "$name" &>/dev/null; then
    docker_env+=(-e "$name=${!name}")
  fi
done

model_dir=$(cd -- "$model_dir" && pwd -P)
[[ -f "$model_dir/model.safetensors.index.json" ]] || {
  echo "MODEL_DIR is not a model checkpoint: $model_dir" >&2
  exit 2
}

# lane extension hook: set FN_EXT_DIR in .env to bind-mount per-file vLLM
# overlays (tree layout under $FN_EXT/vllm/...) onto the installed package,
# e.g. QWEN-PEDIA r10's custom_all_reduce.py PCIe gate lift. Fully inert when
# unset - golden boots are untouched. Build args OUTSIDE the continued block.
ext_args=()
if [[ -n "${FN_EXT_DIR:-}" && -d "${FN_EXT_DIR:-}" ]]; then
  while IFS= read -r -d '' f; do
    rel=${f#"$FN_EXT_DIR"/}
    ext_args+=( -v "$f:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro" )
  done < <(find "$FN_EXT_DIR" -type f -print0)
  echo "[docker_serve] FN_EXT overlay mounts: ${#ext_args[@]} arg(s)"
fi

# lane PLE-int4 arm (09-13): PLE_ARM_INT4=1 in .env swaps the locked image's
# PLE worker + ple_layer for the merged primitive-ai quantization graft (DominikBucko CPU
# transport + int4 group-16 mmap sidecar, 32 G on disk vs 51 G fp8 anonymous).
# Inert when unset; rollback = PLE_ARM_INT4=0 (cold boot).
ple_arm_args=()
if [[ "${PLE_ARM_INT4:-0}" == "1" ]]; then
  ARM=${PLE_ARM_DIR:-$PLEQUANT_DIR/arm}
  { [[ -f "$ARM/worker.py" && -f "$ARM/ple_layer.py" && -d $MODELS_DIR/ples_int4 ]]; } || {
    echo "PLE_ARM_INT4=1 but graft files or table dir missing" >&2; exit 2; }
  ple_arm_args+=( -v "$ARM/worker.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py:ro" )
  ple_arm_args+=( -v "$ARM/ple_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro" )
  ple_arm_args+=( -v $MODELS_DIR/ples_int4:/ples_int4:ro )
  echo "[docker_serve] PLE-int4 ARM active"
fi

# lane PLE-nvfp4 arm (09-20): PLE_ARM_NVFP4=1 mounts the packed-e2m1 graft
# (~/plequant/nvfp4/arm worker.py + ple_layer.py) + the 28.8 G primitive-ai
# sidecar (group16 e2m1 codes, e4m3 scales, per-shard global scale on-wire).
# Page-cache/swap regime like the fp8 default — NO pinned tier (owner spec).
# INERT when unset: lane boots fp8 exactly as golden-9. Mutually exclusive with
# PLE_ARM_INT4 (both bind-mount the same two overlay paths).
# lane PLE-disk8 tier (golden-11, 09-22): PLE_ARM_DISK8=1 serves the fp8
# PLE table from an NVMe mmap FILE instead of anonymous host RAM. Stages the
# superset ple-ext worker.py (0 removals vs the image's v1/ple_offload) + the
# rw table dir; the worker writes it through on first boot, then every later
# boot skips its 132 checkpoint tensors entirely (49 s weights, CoW page cache,
# reclaimable -> no swap regime). Clears the packed-sidecar env so disk wins.
# Inert when unset. Rollback: PLE_ARM_DISK8=0, optionally PLE_ARM_NVFP4=1; cold boot.
if [[ "${PLE_ARM_DISK8:-0}" == "1" ]]; then
  REPO_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  DISK8_WORKER=${PLE_DISK8_WORKER:-"$REPO_DIR/ple-ext/nvfp4/worker.py"}
  # PLE_BF16_TABLE is the single knob: 1 -> bf16 95.4G table dir, 0 -> fp8 47.7G
  # table dir. (09-23 lesson: a hand-toggled flag left PLE_DISK8_DIR pointing at
  # the other tier and the refuse-to-truncate guard killed the boot. The flag
  # now wins over any contradicting explicit dir, with a loud warning.)
  if [[ "${PLE_BF16_TABLE:-0}" == "1" ]]; then
    DERIVED_DISK8_DIR=$MODELS_DIR/ple-disk8-bf16
  else
    DERIVED_DISK8_DIR=$MODELS_DIR/ple-disk8
  fi
  if [[ -n "${PLE_DISK8_DIR:-}" && "${PLE_DISK8_DIR}" != "${DERIVED_DISK8_DIR}" ]]; then
    echo "[docker_serve] WARNING: PLE_BF16_TABLE=${PLE_BF16_TABLE:-0} selects ${DERIVED_DISK8_DIR}, ignoring contradicting PLE_DISK8_DIR=${PLE_DISK8_DIR} (remove the stale .env line)" >&2
  fi
  DISK8_DIR=${DERIVED_DISK8_DIR}
  [[ -f "$DISK8_WORKER" ]] || { echo "PLE_ARM_DISK8=1 but worker missing: $DISK8_WORKER" >&2; exit 2; }
  mkdir -p "$DISK8_DIR"
  ple_arm_args+=( -v "$DISK8_WORKER:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py:ro" )
  ple_arm_args+=( -v "$DISK8_DIR:/ple_disk" )
  docker_env+=( -e "VLLM_PLE_DISK_OFFLOAD_DIR=/ple_disk" -e "VLLM_PLE_QUANT_DIR=" -e "VLLM_PLE_PACKED=" )
  [[ "${PLE_BF16_TABLE:-0}" == "1" ]] && docker_env+=( -e "PLE_BF16_TABLE=1" ) && echo "[docker_serve] PLE bf16 table mode"
  echo "[docker_serve] PLE-disk8 arm active (worker=$DISK8_WORKER dir=$DISK8_DIR)"
fi

if [[ "${PLE_ARM_NVFP4:-0}" == "1" ]]; then
  if [[ "${PLE_ARM_INT4:-0}" == "1" ]]; then
    echo "PLE_ARM_NVFP4 and PLE_ARM_INT4 are mutually exclusive" >&2; exit 2; fi
  ARMN=${PLE_ARM_NVFP4_DIR:-$PLEQUANT_DIR/nvfp4/arm}
  { [[ -f "$ARMN/worker.py" && -f "$ARMN/ple_layer.py" && -d $MODELS_DIR/ples_nvfp4 ]]; } || {
    echo "PLE_ARM_NVFP4=1 but graft files or table dir missing" >&2; exit 2; }
  ple_arm_args+=( -v "$ARMN/worker.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py:ro" )
  ple_arm_args+=( -v "$ARMN/ple_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro" )
  ple_arm_args+=( -v $MODELS_DIR/ples_nvfp4:/ples_nvfp4:ro )
  echo "[docker_serve] PLE-nvfp4 ARM active"
fi

# lane KV tier arm (09-23 owner): bind-mount the host dir behind the fs
# secondary tier when KV_TIER_JSON is set (container sees it at /kv_tier; the
# root_dir inside the JSON must use that path). Host dir = KV_TIER_DIR,
# default ~/models/kv-tier on the 990 PRO (09-23 measurement: lane traffic
# moves the 990 <=5 MB/s — the PLE hot-set is RAM-cached, so no contention;
# device speed is the only differentiator). Fully inert when unset.
kv_tier_args=()
if [[ -n "${KV_TIER_JSON:-}" ]]; then
  kv_tier_dir=${KV_TIER_DIR:-$MODELS_DIR/kv-tier}
  mkdir -p "$kv_tier_dir"
  kv_tier_args+=( -v "$kv_tier_dir:/kv_tier" )
  echo "[docker_serve] KV tier arm active (dir=$kv_tier_dir)"
fi

# lane deviation: the two qwen38-vllm cache mounts below persist JIT/compile
# caches across the rm -f self-restart, so the batch-2 (MAX_NUM_SEQS=2)
# autotune+capture sweep is paid once, not every boot. Comments stay OUTSIDE
# the continued docker run block (see 08-19 + 09-12 incidents).
exec docker run --rm \
  --name qwen38-flash-next \
  --gpus all \
  --ipc host \
  --cap-add SYS_PTRACE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "0.0.0.0:$port:$port" \
  "${docker_env[@]}" \
  -v "$model_dir:/model:ro" \
  -v "$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve-container.sh:/usr/local/bin/qwen38-serve:ro" \
  "${ext_args[@]}" \
  "${ple_arm_args[@]}" \
  "${kv_tier_args[@]}" \
  -v "$HOME/.cache/qwen38-vllm/vllm:/root/.cache/vllm" \
  -v "$HOME/.cache/qwen38-vllm/flashinfer:/root/.cache/flashinfer" \
  "$image" /model
