#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# TP4 boot for the qwen38-flash-next W4A16-FP8PLE lane.
# Preflight (upstream script, unmodified) then serve via upstream docker_serve.sh
# with our .env profile (TP_SIZE=4, CPU_OFFLOAD_GB=16, hot-cache 44, NCCL_P2P_LEVEL=SYS).
set -euo pipefail
cd "$(dirname "$0")"

set -a; . ./.env; set +a

./scripts/preflight.sh

export IMAGE=qwen38-flash-next-2x3090:locked
exec ./scripts/docker_serve.sh
