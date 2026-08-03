#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root (b/Robotwin/train_files_dim14)

export CUDA_VISIBLE_DEVICES=4,5,6,7

# This host has no bond0/InfiniBand; use the available Ethernet NIC and disable IB.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp0s19}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1000}

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
# Must contain "Qwen3.5"; needs transformers>=5.2
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./b/Robotwin/train_files_dim14/starvla_cotrain_robotwin_abs.yaml
# Native 14-D Robotwin / Agilex LeRobot data (packed 14-D, no action_dim_mask).
# NOTE: /mnt/r/DATA/pre_train_v1/post_train/stack_bowls_three meta still says 14-D,
# but its parquet was converted to 80-D — do NOT use that path for 14-D training.
data_root_dir=/mnt/r/DATA/RoboTwin-Clean
# Discovered via examples/simBenchmarks/Robotwin/train_files/data_registry
data_mix=stack_bowls_three_14d
# 80-D pretrain is OK: trainer soft-loads and skips mismatched action-head keys
# pretrained_checkpoint=${PRETRAINED_CHECKPOINT:-./results/Checkpoints/0719_unified80_pretrain_qwen35_gr00t_fixed/checkpoints}
run_root_dir=./results/Checkpoints
run_id=0728_${data_mix}_qwen35_08b_gr00t_robotwin_pad80_train
num_processes=${NUM_PROCESSES:-4}
# === End of environment variable configuration ===
###########################################################################################

# Ensure dataset has native 14-D modality.json (left/right joints+gripper, cam_* keys)
TEMPLATE_DIR=./b/Robotwin/train_files_dim14
src="${TEMPLATE_DIR}/modality.json"
dst="${data_root_dir}/stack_bowls_three/meta/modality.json"
if [[ ! -f "${dst}" ]] || ! cmp -s "${src}" "${dst}"; then
  echo "[info] Installing 14-D modality.json -> ${dst}"
  mkdir -p "$(dirname "${dst}")"
  cp "${src}" "${dst}"
fi

# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${config_yaml}" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port 29501 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.qwenvl.vl_hidden_dim 2560 \
  --framework.action_model.action_dim 14 \
  --framework.action_model.state_dim 14 \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.include_action_mask false \
  --datasets.vla_data.use_embodiment_prompt false \
  --datasets.vla_data.per_device_batch_size 32 \
  --datasets.vla_data.num_workers 8 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 5000 \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 100 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Robotwin \
  --wandb_entity your_wandb_entity
  # --is_debug True
