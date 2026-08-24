#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_policy_server.sh <ckpt_path> [gpu_id] [port]" >&2
    echo "Optional env:" >&2
    echo "  ROBOTWIN_USE_BF16=0|1" >&2
    echo "  ROBOTWIN_SERVER_CONFIG_OVERRIDES='KEY=VAL KEY2=VAL2'  (space-separated OmegaConf dotlist)" >&2
    echo "  STARVLA_BASE_VLM=/path/to/Qwen3.5-4B  (shorthand for framework.qwenvl.base_vlm=...)" >&2
    exit 1
fi

your_ckpt="$1"
gpu_id="${2:-${ROBOTWIN_SERVER_GPU:-0}}"
port="${3:-${ROBOTWIN_SERVER_PORT:-5694}}"
star_vla_python="${STARVLA_PYTHON:-${star_vla_python:-python}}"

use_bf16_flag=()
if [[ "${ROBOTWIN_USE_BF16:-1}" != "0" ]]; then
    use_bf16_flag+=(--use_bf16)
fi

config_override_flags=()
# Shorthand: fix base_vlm when ckpt sidecar points at a missing playground path.
if [[ -n "${STARVLA_BASE_VLM:-}" ]]; then
    config_override_flags+=(--config_override "framework.qwenvl.base_vlm=${STARVLA_BASE_VLM}")
fi
# Extra overrides, e.g. datasets.vla_data.data_mix=stack_bowls_three_unified80
if [[ -n "${ROBOTWIN_SERVER_CONFIG_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    override_items=(${ROBOTWIN_SERVER_CONFIG_OVERRIDES})
    for item in "${override_items[@]}"; do
        config_override_flags+=(--config_override "${item}")
    done
fi

echo "[INFO] Starting RoboTwin policy server"
echo "[INFO] checkpoint: ${your_ckpt}"
echo "[INFO] gpu: ${gpu_id}"
echo "[INFO] port: ${port}"
if ((${#config_override_flags[@]} > 0)); then
    echo "[INFO] config overrides: ${config_override_flags[*]}"
fi

exec env CUDA_VISIBLE_DEVICES="${gpu_id}" "${star_vla_python}" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    "${use_bf16_flag[@]}" \
    "${config_override_flags[@]}"
