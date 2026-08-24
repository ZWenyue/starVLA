#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STARVLA_BASE_VLM=/home/luogang/share/zwy/CKPT/Qwen3.5-0.8B 
export ROBOTWIN_SERVER_CONFIG_OVERRIDES='datasets.vla_data.data_mix=robotwin_stack_bowls_three_50' 

###########################################################################################
# === Please modify the following paths according to your environment ===
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/home/luogang/share/zwy/Projects/RoboTwin}"
export ROBOTWIN_STARVLA_ENV="${ROBOTWIN_STARVLA_ENV:-starVLA}"
export ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CKPT_PATH="${CKPT_PATH:-/home/luogang/share/zwy/CKPT/0812_stack_bowls_three_14d_qwen35_08b_gr00t_robotwin_egodex_train/checkpoints/steps_2000_pytorch_model.pt}"
MODE="${MODE:-demo_clean}"          # demo_clean | demo_randomized
POLICY_NAME="${POLICY_NAME:-0812_stack_bowls_three_14d_qwen35_08b_gr00t_robotwin_egodex_train}"
# Override via: TASKS="stack_bowls_three stack_bowls_two" bash run_robotwin_eval.sh
TASKS=(${TASKS:-stack_bowls_three})
# === End of environment variable configuration ===
###########################################################################################

bash "${SCRIPT_DIR}/start_eval.sh" \
    -m "${MODE}" \
    -n "${POLICY_NAME}" \
    -c "${CKPT_PATH}" \
    "${TASKS[@]}"
