#!/usr/bin/env bash
# Eval a unified80 Robotwin checkpoint with the SAME protocol as
# b/Robotwin/eval_files/run_robotwin_eval.sh (fair train-recipe comparison).
#
# Example:
#   bash b/Robotwin/unified_80_eval/run_unified80_eval.sh
#   CKPT_PATH=/path/to/steps_10000_pytorch_model.pt \
#     bash b/Robotwin/unified_80_eval/run_unified80_eval.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_FILES_DIR="$(cd "${SCRIPT_DIR}/../eval_files" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

###########################################################################################
# === Please modify the following paths according to your environment ===
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/home/luogang/share/zwy/Projects/RoboTwin}"
export ROBOTWIN_STARVLA_ENV="${ROBOTWIN_STARVLA_ENV:-starVLA}"
export ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"

CKPT_PATH="${CKPT_PATH:-/home/luogang/share/zwy/CKPT/0721_stack_bowls_threegcl/checkpoints/steps_5000_pytorch_model.pt}"
MODE="${MODE:-demo_clean}"          # demo_clean | demo_randomized
# -n is only the result/tag name in start_eval; the Python module is ROBOTWIN_POLICY_NAME.
POLICY_TAG="${POLICY_NAME:-0721_stack_bowls_three_8d_steps15k_gcl_5000steps}"
TASKS=(${TASKS:-stack_bowls_three})

# VLM weights (ckpt sidecar often points at a missing playground/ path).
export STARVLA_BASE_VLM="${STARVLA_BASE_VLM:-/home/luogang/share/zwy/CKPT/Qwen3.5-4B}"
# Server-side unnorm must use unified80 DataConfig (80-D stats).
export ROBOTWIN_SERVER_CONFIG_OVERRIDES="${ROBOTWIN_SERVER_CONFIG_OVERRIDES:-datasets.vla_data.data_mix=stack_bowls_three_unified80}"

# Wire this folder's client + deploy yml (same knobs as eval_files).
export ROBOTWIN_EXTRA_POLICY_PATH="${SCRIPT_DIR}"
export DEPLOY_POLICY_TEMPLATE_PATH="${SCRIPT_DIR}/deploy_policy.yml"
export ROBOTWIN_POLICY_NAME="${ROBOTWIN_POLICY_NAME:-unified80_robotwin_interface}"
# === End of environment variable configuration ===
###########################################################################################

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint file not found: ${CKPT_PATH}" >&2
    echo "Tip: pass a steps_*_pytorch_model.pt (or final_model/pytorch_model.pt)." >&2
    exit 1
fi
if [[ ! -d "${STARVLA_BASE_VLM}" ]]; then
    echo "STARVLA_BASE_VLM does not exist: ${STARVLA_BASE_VLM}" >&2
    exit 1
fi

echo "[INFO] Fair-compare mode: eval protocol == b/Robotwin/eval_files"
echo "[INFO] repo=${REPO_ROOT}"
echo "[INFO] ckpt=${CKPT_PATH}"
echo "[INFO] base_vlm=${STARVLA_BASE_VLM}"
echo "[INFO] policy_module=${ROBOTWIN_POLICY_NAME}"
echo "[INFO] deploy_yml=${DEPLOY_POLICY_TEMPLATE_PATH}"

bash "${EVAL_FILES_DIR}/start_eval.sh" \
    -m "${MODE}" \
    -n "${POLICY_TAG}" \
    -c "${CKPT_PATH}" \
    "${TASKS[@]}"
